import abc

import torch
from torch import inference_mode
from tqdm import tqdm

"""
Inversion code taken from: 
1. The official implementation of Edit-Friendly DDPM Inversion: https://github.com/inbarhub/DDPM_inversion
2. The LEDITS demo: https://huggingface.co/spaces/editing-images/ledits/tree/main
"""

LOW_RESOURCE = True


def invert(x0, pipe, prompt_src="", num_diffusion_steps=100, cfg_scale_src=3.5, eta=1, scaling_factor=0.18215, generator=None):
    #  inverts a real image according to Algorithm 1 in https://arxiv.org/pdf/2304.06140.pdf,
    #  based on the code in https://github.com/inbarhub/DDPM_inversion
    #  returns wt, zs, wts:
    #  wt - inverted latent
    #  wts - intermediate inverted latents
    #  zs - noise maps
    #generator = generator if generator is not None else torch.Generator('cuda').manual_seed(42)
    pipe.scheduler.set_timesteps(num_diffusion_steps)
    with inference_mode():
        #print(x0.shape)
        w0 = (pipe.vae.encode(x0).latent_dist.mode() * scaling_factor).float()
        #w0_ = (pipe.vae.encode(x0).latent_dist.sample(generator) * scaling_factor).float()
        #w0__ = (pipe.vae.encode(x0).latents * scaling_factor).float()
        #print(w0.shape)
        #print(w0_.shape)
        #print(w0__.shape)
    wt, zs, wts = inversion_forward_process(pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src,
                                            prog_bar=True, num_inference_steps=num_diffusion_steps)
    return zs, wts


def inversion_forward_process(model, x0,
                              etas=None,
                              prog_bar=False,
                              prompt="",
                              cfg_scale=3.5,
                              num_inference_steps=50, eps=None
                              ):
    
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size)
    #print(model.vae_scale_factor)
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    
    # PREPARING CONDITIONS FOR SDXL
    if model.unet.config.addition_embed_type == "text_time":
        height = model.default_sample_size * model.vae_scale_factor
        width = model.default_sample_size * model.vae_scale_factor

        original_size = (height, width)
        target_size = (height, width)
        
        (       prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
        ) = model.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=model.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
                negative_prompt_2="",
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
            )
     
        add_time_ids = model._get_add_time_ids(
            original_size,
            (0,0),
            target_size,
            dtype=prompt_embeds.dtype,
        )

        prompt_embeds = prompt_embeds.to(model.device)
        negative_prompt_embeds = negative_prompt_embeds.to(model.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(model.device)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(model.device)
        add_time_ids = add_time_ids.to(model.device).repeat(1, 1)
            
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        added_uncond_kwargs = {"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_time_ids}
    else:
        # PREPARING CONDITIONS FOR SD
        if not prompt == "":
            prompt_embeds = encode_text(model, prompt)
        negative_prompt_embeds = encode_text(model, "")
        added_cond_kwargs = None
        added_uncond_kwargs = None
    
    for t in op:
        idx = t_to_idx[int(t)]
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx][None]

        with torch.no_grad():
            out = model.unet.forward(xt, 
                                     timestep=t, 
                                     encoder_hidden_states=negative_prompt_embeds,
                                     added_cond_kwargs=added_uncond_kwargs)
            if not prompt == "":
                cond_out = model.unet.forward(xt, 
                                              timestep=t, 
                                              encoder_hidden_states=prompt_embeds,
                                              added_cond_kwargs=added_cond_kwargs)

        if not prompt == "":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            xtm1 = xts[idx + 1][None]
            # pred of x0
            pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx + 1] = xtm1

    if not zs is None:
        zs[-1] = torch.zeros_like(zs[-1])

    return xt, zs, xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    #print("x0", x0.shape)
    #print("unet_sample_size", model.unet.sample_size)
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size)

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device)
    #print("xts", xts.shape)
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    xts = torch.cat([xts, x0], dim=0)

    return xts


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                            model_output,
                                            torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

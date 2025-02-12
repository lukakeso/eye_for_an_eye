import torch
from diffusers import DDIMScheduler
from diffusers.models import ControlNetModel

from models.stable_diffusion_final import (
    CrossImageAttentionStableDiffusionPipeline, 
    CrossImageAttentionStableDiffusionControlPipeline,
    CrossImageAttentionStableDiffusionXLPipeline, 
    CrossImageAttentionStableDiffusionXLControlPipeline)
from models.unet_2d_condition import FreeUUNet2DConditionModel

def get_stable_diffusion_model(huggingface_model, controlnet_model):
    print("Loading Stable Diffusion model...")
    # try:
    #     with open('./TOKEN', 'r') as f:
    #         token = f.read().replace('\n', '') # remove the last \n!
    #         print(f'[INFO] loaded hugging face access token from ./TOKEN!')
    # except FileNotFoundError as e:
    #     token = True
    #     print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
    
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("SD Model:", huggingface_model, "ControlNet Model:", controlnet_model)
    if "xl" in str(huggingface_model):
        if controlnet_model == None or controlnet_model == "" or controlnet_model == "None":
            pipe = CrossImageAttentionStableDiffusionXLPipeline.from_pretrained(huggingface_model,
                                                                            safety_checker=None).to(device)
        else:
            controlnet = ControlNetModel.from_pretrained(controlnet_model).to(device)
            pipe = CrossImageAttentionStableDiffusionXLControlPipeline.from_pretrained(huggingface_model,
                                                                            controlnet=controlnet,
                                                                            safety_checker=None).to(device)
            print("Loaded with ControlNet")
    else:
        if controlnet_model == None or controlnet_model == "" or controlnet_model == "None":
            pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained(huggingface_model,
                                                                            safety_checker=None).to(device)
        else:
            controlnet = ControlNetModel.from_pretrained(controlnet_model).to(device)
            pipe = CrossImageAttentionStableDiffusionControlPipeline.from_pretrained(huggingface_model,
                                                                            controlnet=controlnet,
                                                                            safety_checker=None).to(device)
            print("Loaded with ControlNet")
            
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained(huggingface_model, subfolder="unet").to(device)
    pipe.scheduler = DDIMScheduler.from_config(huggingface_model, subfolder="scheduler")
    
    if "xl" in str(huggingface_model):
        print("Making SDXL more efficient")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
    print("Done.")
    return pipe
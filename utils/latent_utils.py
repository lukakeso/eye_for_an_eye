from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from appearance_transfer_model_final import AppearanceTransferModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion import invert


def load_latents_or_invert_images(model: AppearanceTransferModel, cfg: RunConfig):
    if cfg.load_latents:
        if cfg.app_latent_save_path.exists():
            print("Loading existing latents...")
            latents_app = load_single_latent(cfg.app_latent_save_path)
            noise_app = load_single_noise(cfg.app_latent_save_path)
        else:
            print("Inverting app image...")
            latents_app, noise_app = invert_single_image(sd_model=model.pipe, image_path=cfg.app_image_path, cfg=cfg)

        if cfg.struct_latent_save_path.exists():
            print("Loading existing latents...")
            latents_struct = load_single_latent(cfg.struct_latent_save_path)
            noise_struct = load_single_noise(cfg.struct_latent_save_path)
        else:
            print("Inverting struct image...")
            latents_struct, noise_struct = invert_single_image(sd_model=model.pipe, image_path=cfg.struct_image_path, cfg=cfg)

        print("Done Loading.")
    else:
        print("Inverting images...")
        app_image, struct_image = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents_app, latents_struct, noise_app, noise_struct = invert_images(app_image=app_image,
                                                                             struct_image=struct_image,
                                                                             sd_model=model.pipe,
                                                                             cfg=cfg)
        model.enable_edit = True
        print("Done Loading.")
    return latents_app, latents_struct, noise_app, noise_struct

def load_single_latent(latent_save_path: Path) -> torch.Tensor:
    latents = torch.load(latent_save_path)
    if type(latents) == list:
        latents = [l.to("cuda") for l in latents]
    else:
        latents = latents.to("cuda")
    return latents

def load_latents(app_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_app = torch.load(app_latent_save_path)
    latents_struct = torch.load(struct_latent_save_path)
    if type(latents_struct) == list:
        latents_app = [l.to("cuda") for l in latents_app]
        latents_struct = [l.to("cuda") for l in latents_struct]
    else:
        latents_app = latents_app.to("cuda")
        latents_struct = latents_struct.to("cuda")
    return latents_app, latents_struct

def load_single_noise(latent_save_path: Path) -> torch.Tensor:
    latents = torch.load(latent_save_path.parent / (latent_save_path.stem + "_ddpm_noise.pt"))
    latents = latents.to("cuda")
    return latents

def load_noise(app_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_app = torch.load(app_latent_save_path.parent / (app_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_struct = torch.load(struct_latent_save_path.parent / (struct_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_app = latents_app.to("cuda")
    latents_struct = latents_struct.to("cuda")
    return latents_app, latents_struct

def invert_single_image(sd_model: AppearanceTransferModel, image_path: Path, cfg: RunConfig):
    image = image_utils.load_size(image_path, size=cfg.image_size)
    Image.fromarray(image).save(cfg.root_output_path / f"{image_path.stem}.png")
    input_im = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
    #generator = torch.Generator('cuda').manual_seed(cfg.seed)
    zs, latents = invert(x0=input_im.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5,
                                 scaling_factor=cfg.scaling_factor)
  
    # Save the inverted latents and noises
    torch.save(latents, cfg.latents_path / f"{image_path.stem}.pt")
    torch.save(zs, cfg.latents_path / f"{image_path.stem}_ddpm_noise.pt")
    return latents, zs

def invert_images(sd_model: AppearanceTransferModel, app_image: Image.Image, struct_image: Image.Image, cfg: RunConfig):
    input_app = torch.from_numpy(np.array(app_image)).float() / 127.5 - 1.0
    input_struct = torch.from_numpy(np.array(struct_image)).float() / 127.5 - 1.0
    #generator = torch.Generator('cuda').manual_seed(cfg.seed)
    zs_app, latents_app = invert(x0=input_app.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5,
                                 scaling_factor=cfg.scaling_factor)
    zs_struct, latents_struct = invert(x0=input_struct.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                       pipe=sd_model,
                                       prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5,
                                       scaling_factor=cfg.scaling_factor)
    # Save the inverted latents and noises
    torch.save(latents_app, cfg.latents_path / f"{cfg.app_image_path.stem}.pt")
    torch.save(latents_struct, cfg.latents_path / f"{cfg.struct_image_path.stem}.pt")
    torch.save(zs_app, cfg.latents_path / f"{cfg.app_image_path.stem}_ddpm_noise.pt")
    torch.save(zs_struct, cfg.latents_path / f"{cfg.struct_image_path.stem}_ddpm_noise.pt")
    return latents_app, latents_struct, zs_app, zs_struct


def get_init_latents_and_noises(model: AppearanceTransferModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_struct.dim() == 4 and model.latents_app.dim() == 4 and model.latents_app.shape[0] > 1:
        model.latents_struct = model.latents_struct[cfg.skip_steps]
        model.latents_app = model.latents_app[cfg.skip_steps]
    init_latents = torch.stack([model.latents_struct, model.latents_app, model.latents_struct])
    init_zs = [model.zs_struct[cfg.skip_steps:], model.zs_app[cfg.skip_steps:], model.zs_struct[cfg.skip_steps:]]
    return init_latents, init_zs

def match_coord(cfg: RunConfig):
    try:
        data = np.load(cfg.match_coord)
        trg_lst = data['trg_lst']
    except Exception as e:
        trg_lst = None
    return trg_lst
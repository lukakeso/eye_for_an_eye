import pathlib
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from config import RunConfig


def load_images(cfg: RunConfig, save_path: Optional[pathlib.Path] = None) -> Tuple[Image.Image, Image.Image]:
    image_style = load_size(cfg.app_image_path, size=cfg.image_size)
    image_struct = load_size(cfg.struct_image_path, size=cfg.image_size)
    if save_path is not None:
        Image.fromarray(image_style).save(save_path / f"in_style.png")
        Image.fromarray(image_struct).save(save_path / f"in_struct.png")
    return image_style, image_struct

def load_multi_images(cfg: RunConfig, save_path: Optional[pathlib.Path] = None) -> Tuple[Image.Image, Image.Image]:
    if cfg.app_num==2:
        style_2 = load_size(cfg.app_image_path2, size=cfg.image_size)
        if save_path is not None:
            Image.style_2(style_2).save(save_path / f"style_extra_2.png")
    elif cfg.app_num==3:
        style_2 = load_size(cfg.app_image_path2, size=cfg.image_size)
        style_3 = load_size(cfg.app_image_path3, size=cfg.image_size)
        if save_path is not None:
            Image.fromarray(style_2).save(save_path / f"style_extra_2.png")
            Image.fromarray(style_3).save(save_path / f"style_extra_3.png")
    if cfg.app_num==2:
        return style_2
    elif cfg.app_num==3:
        return style_2, style_3


def load_size(image_path: pathlib.Path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512) -> Image.Image:
    if isinstance(image_path, (str, pathlib.Path)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))  
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    h, w, c = image.shape
    # offset
    """
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    """
    image = np.array(Image.fromarray(image).resize((size, size)))
    return image


def save_generated_masks(model, cfg: RunConfig):
    tensor2im(model.image_app_mask_32).save(cfg.output_path / f"mask_style_32.png")
    tensor2im(model.image_struct_mask_32).save(cfg.output_path / f"mask_struct_32.png")
    tensor2im(model.image_app_mask_64).save(cfg.output_path / f"mask_style_64.png")
    tensor2im(model.image_struct_mask_64).save(cfg.output_path / f"mask_struct_64.png")


def tensor2im(x) -> Image.Image:
    return Image.fromarray(x.cpu().numpy().astype(np.uint8) * 255)
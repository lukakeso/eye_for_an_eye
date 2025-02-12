from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional
import os
import json
from itertools import permutations

class Range(NamedTuple):
    start: int
    end: int


@dataclass
class RunConfig:
    # Appearance image path
    app_image_path: Optional[Path] = None
    # Struct image path
    struct_image_path: Optional[Path] = None
    # Domain name (e.g., buildings, animals)
    domain_name: Optional[str] = None
    # Output path
    output_path: Path = Path('./output')
    # Random seed
    seed: int = 42
    # Input prompt for inversion (will use domain name as default)
    prompt: Optional[str] = None
    # Number of timesteps
    num_timesteps: int = 100
    # Whether to use a binary mask for performing AdaIN
    use_masked_adain: bool = True
    use_adain: bool = True
    # Timesteps to apply cross-attention on 64x64 layers
    cross_attn_64_range: Range = Range(start=50, end=100) #10,90
    # Timesteps to apply cross-attention on 32x32 layers
    cross_attn_32_range: Range = Range(start=50, end=100) #10,70
    feat_range: Range = Range(start=0, end=100)
    # Timesteps to apply AdaIn
    adain_range: Range = Range(start=85, end=100) #20,100 #ours 50,100
    # Swap guidance scale
    swap_guidance_scale: float = 3.5
    # classifier guidance scale
    guidance_scale: float = 1.0
    # Attention contrasting strength
    contrast_strength: float = 1.67
    # Object nouns to use for self-segmentation (will use the domain name as default)
    object_noun: Optional[str] = None
    # Whether to load previously saved inverted latent codes
    load_latents: bool = True
    # Number of steps to skip in the denoising process (used value from original edit-friendly DDPM paper)
    skip_steps: int = 32 
    match_coord: Path = None
    sam_path: Path = Path('sam_hq_vit_l.pth')
    mask_use: bool = False
    check_attn_map: bool = False
    check_argmax: bool = False
    do_v_swap: bool = False
    bbox_path: Path = None
    struct_bbox_path: Path = None
    app_bbox_path: Path = None
    do_cross_mask: bool = False
    cross_thres: float = 0.1
    # stabilityai/stable-diffusion-xl-base-1.0
    # runwayml/stable-diffusion-v1-5
    controlnet_path: Path = None
    model_path: Path = Path("runwayml/stable-diffusion-v1-5")
    image_size: int = 512
    transfer_app_pattern: bool = False
    
    select_only_material: bool = False
    dataset_domain_images_path: Path = None
    selected_images_root: Path = Path("/d/hpc/home/lk6760/FRI_HOME/DATASETS/vitonhd/test/cloth")
    reverse_order: bool = False
    
    def __post_init__(self):
        self.controlnet_path = str(self.controlnet_path)
        self.model_path = str(self.model_path)
        if "xl" in self.model_path:
            self.domain_name = str(self.domain_name) + "_SDXL"
        else:
            self.domain_name = str(self.domain_name) + "_SD"
        
        self.root_output_path = self.output_path / self.domain_name
        # Define the paths to store the inverted latents to
        self.latents_path = Path(self.root_output_path) / "latents"
        self.latents_path.mkdir(parents=True, exist_ok=True)
        
        if self.struct_image_path != None and self.app_image_path != None:
            save_name = f'app={self.app_image_path.stem}-struct={self.struct_image_path.stem}-feature'
            self.output_path = self.output_path / self.domain_name / save_name
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            self.app_latent_save_path = self.latents_path / f"{self.app_image_path.stem}.pt"
            self.struct_latent_save_path = self.latents_path / f"{self.struct_image_path.stem}.pt"
  
        elif self.dataset_domain_images_path != None:
            with open(self.dataset_domain_images_path, "r") as file:
                data = json.load(file)
            
            self.selected_images = {}
            
            for k, v in data.items():
                for i, image_path in enumerate(v):
                    self.selected_images[f'{k}_{i}'] = image_path
                    
            self.image_pairs = list(permutations(self.selected_images, 2))
            if self.select_only_material == True:
                temp = []
                for pair in self.image_pairs:
                    if "Materials" in pair[0] or "Materials" in pair[1]:
                        temp.append(pair)
                self.image_pairs = temp
                
            if self.reverse_order:
                self.image_pairs.reverse()
        else:
            raise NotImplementedError("please define images!")

        # files = os.listdir(self.output_path)
        # image_paths = [os.path.join(self.output_path,file) for file in files if file.endswith(".png") and file.startswith("out_transfer")]    
        # if len(image_paths) == 21:
        #     self.completed = True
        # else:
        self.completed = False
        
        if "xl" in self.model_path:
            self.scaling_factor = 0.13025
            self.image_size = 1024
        else:
            self.scaling_factor = 0.18215
            self.image_size = 512
        
        # Handle the domain name, prompt, and object nouns used for masking, etc.
        if self.use_masked_adain and self.domain_name is None:
            raise ValueError("Must provide --domain_name and --prompt when using masked AdaIN")
        if not self.use_masked_adain and self.domain_name is None:
            self.domain_name = "object"
        if self.prompt is None:
            self.prompt = f"A photo of a {self.domain_name}"
        if self.object_noun is None:
            self.object_noun = self.domain_name
        self.rgb_check_path = Path(self.root_output_path) / "rgb_check"
        self.rgb_check_path.mkdir(parents=True, exist_ok=True)

              
    def set_images(self, domain_im1, domain_im2):
        
        im1 = self.selected_images[domain_im1]
        im2 = self.selected_images[domain_im2]
        self.app_image_path = self.selected_images_root / im1
        self.struct_image_path = self.selected_images_root / im2
        save_name = f'app={domain_im1}-struct={domain_im2}-feature'
        self.output_path = self.root_output_path / "results" / save_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.app_latent_save_path = self.latents_path / f"{self.app_image_path.stem}.pt"
        self.struct_latent_save_path = self.latents_path / f"{self.struct_image_path.stem}.pt"
        
        self.rgb_check_path = self.output_path

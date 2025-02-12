import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
import gc
import torch.nn as nn
import math
import cv2
import os
import re
import matplotlib.pyplot as plt
import json
from itertools import combinations

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model_final import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images
from sam_hq.segment_anything import sam_model_registry, SamPredictor

from controlnet_aux import AnylineDetector #, TEEDdetector, MidasDetector


@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)

def max_spanning_rectangle(mask):
    # Get the dimensions of the mask
    rows, cols = mask.shape

    # Create an array to store the heights of the histogram for each row
    heights = np.zeros(cols, dtype=int)

    # Variable to store the maximum area and the coordinates of the rectangle
    max_area = 0
    top_left = (-1, -1)
    bottom_right = (-1, -1)

    # Helper function to calculate the largest rectangle in a histogram
    def largest_rectangle_area(heights):
        stack = []
        max_area = 0
        left = right = -1
        heights.append(0)  # Add a zero height to flush out the remaining bars
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    right = i - 1
                    left = right - w + 1
            stack.append(i)
        heights.pop()  # Remove the added zero height
        return max_area, left, right

    # Loop through each row in the mask
    for i in range(rows):
        for j in range(cols):
            # Update the histogram heights
            if mask[i, j] == 1:
                heights[j] += 1
            else:
                heights[j] = 0

        # Calculate the largest rectangle for the current row's histogram
        area, left, right = largest_rectangle_area(list(heights))

        # If the area is larger than the previously found maximum area, update it
        if area > max_area:
            max_area = area
            top_left = (i - heights[left] + 1, left)  # Top-left corner
            bottom_right = (i, right)  # Bottom-right corner

    return max_area, top_left, bottom_right

def create_mask_from_rectangle(rows, cols, top_left, bottom_right):
    # Create an empty mask of zeros
    new_mask = np.zeros((rows, cols), dtype=np.uint8)

    # Get the coordinates of the top-left and bottom-right corners
    top_row, left_col = top_left
    bottom_row, right_col = bottom_right

    # Set the region inside the rectangle to 1
    new_mask[top_row:bottom_row+1, left_col:right_col+1] = 1

    return new_mask

def apply_erosion(binary_mask, kernel_size=3, iterations=1):
    # Create a kernel for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply erosion using the kernel
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=iterations)
    
    return eroded_mask

def fill_large_holes(binary_mask, kernel_size=15):
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        mask = binary_mask.astype(np.uint8) * 255
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return filled_mask//255

def resize_and_center_align(mask1, mask2, box1_points, box2_points):
    # Calculate width and height of the bounding boxes
    (x1_min, y1_min), (x1_max, y1_max) = box1_points  # Points for mask1 bounding box
    (x2_min, y2_min), (x2_max, y2_max) = box2_points  # Points for mask2 bounding box

    width1, height1 = x1_max - x1_min, y1_max - y1_min
    width2, height2 = x2_max - x2_min, y2_max - y2_min

    # Calculate the scaling factor to fit mask1 into mask2's bounding box
    scale = min(width2 / width1, height2 / height1)
    
    # Resize mask1 while maintaining aspect ratio
    new_size = (int(width1 * scale), int(height1 * scale))
    resized_mask1 = cv2.resize(mask1[y1_min:y1_max, x1_min:x1_max], new_size, interpolation=cv2.INTER_NEAREST)
    
    # Create an empty mask2-sized image to hold the centered mask1
    output = np.zeros_like(mask2)
    
    # Calculate top-left coordinates to center resized mask1 in mask2's bounding box
    x_offset = x2_min + (width2 - new_size[0]) // 2
    y_offset = y2_min + (height2 - new_size[1]) // 2

    # Place resized_mask1 at the calculated center position in output
    output[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized_mask1
    
    return output

def swap_coords(coords):
        top_left = coords[0]
        bottom_right = coords[1]
        return ((top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
 
    
def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        if isinstance(mask, np.ndarray):
            show_mask(mask, plt.gca(), random_color=True)
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()
        
def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def get_file_name(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    number = numbers[0]
    return number
    
def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    if cfg.completed == True:
        print("Already completed")
        exit()
    set_seed(cfg.seed)
    model = AppearanceTransferModel(cfg)
    latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cfg)
    model.set_latents(latents_app, latents_struct)
    model.set_noise(noise_app, noise_struct)
    print("Running appearance transfer...")
    images = run_appearance_transfer(model=model, cfg=cfg)
    print("Done.")
    return images


def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)    
    
    mask_lst=None
    if cfg.mask_use: #cfg.sam_path
        image_size = cfg.image_size
        if cfg.bbox_path:
            bbox=load_json(cfg.bbox_path)
        model_type = "vit_l"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=cfg.sam_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        app_image = cv2.imread(str(cfg.app_image_path))
        app_image = cv2.resize(app_image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        app_image = cv2.cvtColor(app_image, cv2.COLOR_BGR2RGB)
        
        struct_image = cv2.imread(str(cfg.struct_image_path))
        struct_image = cv2.resize(struct_image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        struct_image = cv2.cvtColor(struct_image, cv2.COLOR_BGR2RGB)
        imgs = [app_image, struct_image]
        imgs_names = [str(cfg.app_image_path.stem), str(cfg.struct_image_path.stem)]
        
        mask_lst=[]
        for i in range(0, len(imgs)):
            predictor.set_image(imgs[i])
            hq_token_only = True

            input_box = np.array([[0, 0, image_size, image_size]])
            if cfg.bbox_path:
                if i==0:
                    app_name = get_file_name(cfg.app_image_path)
                    input_box = bbox.get(app_name, [])
                else:
                    struct_name = get_file_name(cfg.struct_image_path)
                    input_box = bbox.get(struct_name, [])
                input_box = np.array([input_box])
            else:
                input_box = np.array([[0, 0, image_size, image_size]]) # Cover Full Image

            offset = image_size//32
            input_point = np.array([[image_size//2,image_size//2],[offset,offset],[offset,image_size-offset],[image_size-offset,offset],[image_size-offset,image_size-offset]])
            input_label = np.array([1,0,0,0,0])
            result_path = str(cfg.rgb_check_path)
            masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box = input_box,
                    multimask_output=False,
                    hq_token_only=hq_token_only, 
                )
            show_res(masks,scores,input_point, input_label, input_box, result_path + '/mask_' + imgs_names[i], imgs[i])
            mask_lst.append(masks) #0:style, 1:struct
        
        model.sam_app_mask = mask_lst[0]
        model.sam_struct_mask = mask_lst[1]
        
        sam.to(device="cpu")
        del sam
        del predictor
        
    if cfg.controlnet_path:
        
        #line_processor = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")

        line_processor = AnylineDetector.from_pretrained(
            "TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline"
        )
        edge_lst = []
        for i in range(0, len(imgs)):
            edge = line_processor(imgs[i], detect_resolution=1024)       # or 2048 for fine details  
            edge = edge.convert('RGB')
            edge = np.array(edge)
            edge = cv2.resize(edge, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            edge_lst.append(edge) #0:style/app, 1:struct
            show_res([None], [1.0], None, None, None, result_path + '/edges_'+imgs_names[i]+"_"+str(i), edge)

        masked_coords = []
        masked_edges = []
        fill_size = image_size//32
        for i, (mask, edge) in enumerate(zip(mask_lst, edge_lst)):
            print(mask.shape)
            filled_mask = fill_large_holes(mask[0], kernel_size=fill_size)
            max_area, top_left, bottom_right = max_spanning_rectangle(filled_mask) # finds biggest rectange in the clothing mask
            rectangle_mask = create_mask_from_rectangle(image_size, image_size, top_left, bottom_right)
            eroded_mask = apply_erosion(rectangle_mask, kernel_size=5, iterations=5)
            mask_3d = np.repeat(eroded_mask[...,None], 3, axis=2)
            masked_edge = edge * mask_3d
            masked_edges.append(masked_edge) #0:style/app, 1:struct
            masked_coords.append((top_left, bottom_right))        

        if cfg.transfer_app_pattern:
            i = 1
            controlnet_image = resize_and_center_align(masked_edges[0].astype(np.uint8),masked_edges[1].astype(np.uint8),\
                                                        swap_coords(masked_coords[0]), swap_coords(masked_coords[1]) )
            
            treshold = int(controlnet_image.max() * 0.8)
            controlnet_image[controlnet_image <= treshold] = 0 # thresholding
            print(controlnet_image.min(), controlnet_image.max(), controlnet_image.shape, treshold)
            #model.sam_app_mask = mask_lst[0]
            #model.sam_struct_mask = mask_lst[1]
        else:
            i = 0
            controlnet_image = masked_edges[i].astype(np.uint8)
            #model.sam_app_mask = mask_lst[0]
            #model.sam_struct_mask = mask_lst[1]
            # Image.fromarray(masked_edges[i].astype(np.uint8))
            # controlnet_image = controlnet_image.convert('RGB')
            
        
        show_res(mask_lst[i], [1.0], np.array(swap_coords(masked_coords[i])), np.array([1,1]), None, result_path + '/edges_'+imgs_names[i]+"_mask+control", controlnet_image)
        show_res([None], [1.0], np.array(swap_coords(masked_coords[i])), np.array([1,1]), None, result_path + '/edges_'+imgs_names[i]+"_control", controlnet_image)

        
        controlnet_image = Image.fromarray(controlnet_image.astype(np.uint8))
        controlnet_image = controlnet_image.convert('RGB')
        
        del line_processor
        
    torch.cuda.empty_cache()
    
    transfer_type_single = ["color", "material", "pattern"]
    
    comb_1 = list(combinations(transfer_type_single, 1))
    # All unique combinations of 2 elements
    comb_2 = list(combinations(transfer_type_single, 2))
    # All unique combinations of 3 elements
    comb_3 = list(combinations(transfer_type_single, 3))
    
    transfer_types = comb_1 + comb_2 + comb_3
    
    for t in transfer_types:
        
        if "shape" in t:
            # just have a text guided image generation
            # or in case we have a garment we have additionally a depth ControlNet guided generation
            pass
            
        # if "pattern" in t: #texture/pattern
        #     cfg.adain_range = Range(start=0, end=100)
        #     model.config.adain_range = Range(start=0, end=100)
        #     cfg.feat_range = Range(start=0, end=100)
        #     model.config.feat_range = Range(start=0, end=100)
            
        if "color" in t:
            cfg.adain_range = Range(start=0, end=0)
            model.config.adain_range = Range(start=0, end=0)
            cfg.feat_range = Range(start=0, end=0)
            model.config.feat_range = Range(start=0, end=0)
            
        if "material" in t:
            cfg.adain_range = Range(start=90, end=100)
            model.config.adain_range = Range(start=90, end=100)
            cfg.feat_range = Range(start=0, end=0)
            model.config.feat_range = Range(start=0, end=0)
            
        
        images = model.pipe(
            prompt=[cfg.prompt] * 3,
            image=controlnet_image,
            latents=init_latents,
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_timesteps,
            swap_guidance_scale=cfg.swap_guidance_scale,
            callback=model.get_adain_callback(),
            eta=1,
            zs=init_zs,
            generator=torch.Generator('cuda').manual_seed(cfg.seed),
            cross_image_attention_range=cfg.cross_attn_64_range,
            mask_lst = mask_lst,
            do_v_swap = cfg.do_v_swap,
            output_type = "latent"
        ).images
        
        image = model.pipe.vae.decode(images / model.pipe.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        images = model.pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        # Save images
        images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}_{t}.png")
        
    #images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}.png")
    #images[1].save(cfg.output_path / f"out_style---seed_{cfg.seed}.png")
    #images[2].save(cfg.output_path / f"out_struct---seed_{cfg.seed}.png")
    #joined_images = np.concatenate(images[::-1], axis=1)
    #Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    
    return images


if __name__ == '__main__':
    main()
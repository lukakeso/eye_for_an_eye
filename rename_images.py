import json
from itertools import permutations
import os

dataset_domain_images_path = "/d/hpc/home/lk6760/FRI_HOME/DATASETS/vitonhd/test_subset_selection.json"
images_dir = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/results/selected_SD"


with open(dataset_domain_images_path, "r") as file:
    data = json.load(file)
    
    selected_images = {}
    
    for k, v in data.items():
        for i, image_path in enumerate(v):
            selected_images[image_path] = f'{k}_{i}'

for im_name_old, im_name_new in selected_images.items():
    
    old_path = os.path.join(images_dir, im_name_old.replace(".jpg", ".png"))
    new_path = os.path.join(images_dir, im_name_new+".png")
    os.rename(old_path, new_path)
import os
import shutil
from tqdm import tqdm

prepared_results = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/results_set"
root_results = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/results/selected_material_SD/selected_material_SD"
app_garment = "Blouse"
sturct_garment = "Blouse"
transfer_types = ["color", "material", "style"]

app_dir_list = []
struct_dir_list = []
for d in os.listdir(root_results):
    if f'app={app_garment}' in d:
        app_dir_list.append(d)
        
    if f'struct={sturct_garment}' in d:
        struct_dir_list.append(d)


for d in tqdm(app_dir_list, desc="Appearance list"):
    for f in [fs for fs in os.listdir(os.path.join(root_results, d)) if fs.startswith("out_transfer")]:
        transfer_type = f[33:-4]
        old_path = os.path.join(root_results, d, f)
        new_path = os.path.join(prepared_results, d.split("_")[0].replace("=", "_"), transfer_type) #d+".png"
        
        # print(old_path)
        # print(new_path)
        
        # if os.path.isfile(old_path):
        #     print("Old file exists")
        # if os.path.exists(new_path):
        #     print("New folder exists")
        
        if os.path.isfile(old_path) and os.path.exists(new_path):
            shutil.copy(old_path, os.path.join(new_path, d+".png"))
        else:
            print("error with", old_path)
            
for d in tqdm(struct_dir_list, desc="Structure list"):
    for f in [fs for fs in os.listdir(os.path.join(root_results, d)) if fs.startswith("out_transfer")]:
        transfer_type = f[33:-4]
        old_path = os.path.join(root_results, d, f)
        new_path = os.path.join(prepared_results, d.split("_")[1][2:].replace("=", "_"), transfer_type) #d+".png"
        
        # print(old_path)
        # print(new_path)
        
        # if os.path.isfile(old_path):
        #     print("Old file exists")
        # if os.path.exists(new_path):
        #     print("New folder exists")

        if os.path.isfile(old_path) and os.path.exists(new_path):
            shutil.copy(old_path, os.path.join(new_path, d+".png"))
        else:
            print("error with", old_path)
            
print("Done!")
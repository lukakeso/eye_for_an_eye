import torch
#import ChamferDistancePytorch.chamfer2D.dist_chamfer_2D as ChamDist
#import ChamferDistancePytorch.fscore as fscore
import numpy as np
from sam_hq.segment_anything import sam_model_registry, SamPredictor
#from controlnet_aux import AnylineDetector, TEEDdetector, CannyDetector
import cv2
from PIL import Image
#import torch_fidelity
from tabulate import tabulate

import matplotlib.pyplot as plt

#from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.feature import graycomatrix, graycoprops
from skimage import color

def save_image(np_arr, file_name):
    im = Image.fromarray(np_arr)
    im.save("metrics_testing/"+file_name)

def load_image(image_path, image_size=512):
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class StructurePreservation:
    """ 
    Structure Preservation (SP): we utilize the marquee interaction mode of
    SAM [23] for selecting areas to obtain binary masks corresponding to 
    structure images and their respective output images. Then, we compute their
    Intersection over Union (IoU) results as a measure of Structure Preservation.
    """
    def __init__(self, sam_path: str = "sam_hq_vit_l.pth", 
                       model_type: str = "vit_l", 
                       device: str = "cuda"):

        sam_model = sam_model_registry[model_type](checkpoint=sam_path)
        sam_model.to(device=device)
        self.SAM = SamPredictor(sam_model)
    
    def __call__(self, struct_im, result_im, image_size: int = 512):
        masks = []
        for img in [struct_im, result_im]:
            mask = self.get_mask(img, image_size)
            masks.append(mask)
        
        #print(masks[0].shape, masks[1].shape)
        save_image(masks[0], "mask0.png")
        save_image(masks[1], "mask1.png")    
            
        struct_iou = self.IoU(masks[0], masks[1])

        return struct_iou
    
    def IoU(self, mask1, mask2):
        """
        Calculate Intersection over Union (IoU) for two binary masks.
        
        Args:
            mask1 (np.ndarray): First binary mask.
            mask2 (np.ndarray): Second binary mask.
            
        Returns:
            float: IoU value between 0 and 1.
        """
        # Ensure the masks are binary
        mask1 = mask1 > 0
        mask2 = mask2 > 0
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        # Avoid division by zero
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_mask(self, img, image_size: int = 512):
        self.SAM.set_image(img)
        input_box = np.array([[0, 0, image_size, image_size]])
        offset = image_size//32
        input_point = np.array([[image_size//2,image_size//2],[offset,offset],[offset,image_size-offset],[image_size-offset,offset],[image_size-offset,image_size-offset]])
        input_label = np.array([1,0,0,0,0])
        mask, score, _ = self.SAM.predict(
            point_coords=input_point,
            point_labels=input_label,
            box = input_box,
            multimask_output=False,
            hq_token_only=True, 
        )
        return mask[0]

class ChamferDistance:
    """
    Chamfer Distance (CD): we first extract the line drawings of the structure
    and generated images, and then filter out redundant details using the Canny
    operator. The high and low thresholds used by the Canny operator are set to
    150 and 50, respectively. Finally, we calculate the chamfer distance between
    the line drawings as a measure of the gap between the sets of edge points in
    the two images. A smaller value indicates a higher degree of match between
    the shape or edge features in the structure and generated images.
    """
    def __init__(self, detector_name: str = "anyline"):
        
        self.pixel_crop = 6
        self.detector_name = detector_name
        self.line_filter = CannyDetector()
        self.chamLoss = ChamDist.chamfer_2DDist()
        
        # LINE DETECTORS
        if detector_name == "teed":
            self.line_detector = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
        elif detector_name == "anyline":
            self.line_detector = AnylineDetector.from_pretrained(
                "TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline"
            )
        else:
            raise NotImplementedError(f'Please speficy a valid line detector name; {detector_name} is not a valid name')

    def __call__(self, struct_im, result_im, resolution: int = 512):
        
        struct_processed_image = self.line_detector(struct_im, detect_resolution=resolution, output_type="np")
        result_processed_image = self.line_detector(result_im, detect_resolution=resolution, output_type="np")
        
        struct_processed_image = self.correct_borders(struct_processed_image)
        result_processed_image = self.correct_borders(result_processed_image)
        
        save_image(struct_processed_image, "struct_processed_image.png")
        save_image(result_processed_image, "result_processed_image.png")
        
        struct_filtered_image = self.line_filter(input_image=struct_processed_image, low_threshold=50, high_threshold=150, detect_resolution=resolution, image_resolution=resolution, output_type="np")
        result_filtered_image = self.line_filter(input_image=result_processed_image, low_threshold=50, high_threshold=150, detect_resolution=resolution, image_resolution=resolution, output_type="np")
        
        save_image(struct_filtered_image, "struct_filtered_image.png")
        save_image(result_filtered_image, "result_filtered_image.png")
        
        struct_points = self.get_indexes(struct_filtered_image)
        result_points = self.get_indexes(result_filtered_image)
        
        points1 = torch.from_numpy(struct_points).float().cuda()
        points2 = torch.from_numpy(result_points).float().requires_grad_().cuda()
        dist1, dist2, _, _ = self.chamLoss(points1, points2)
        dist1 = dist1.detach().cpu().numpy()
        dist2 = dist2.detach().cpu().numpy()

        total_dist = (np.sum(dist1)+np.sum(dist2))/(len(dist1[0])+len(dist2[0]))
        
        return np.round(total_dist, decimals=3)
    
    def correct_borders(self, line_image, border_width: int = 8):
        line_image[:border_width, :, :] = 0  # Top border
        line_image[-border_width:, :, :] = 0  # Bottom border
        line_image[:, :border_width, :] = 0  # Left border
        line_image[:, -border_width:, :] = 0  # Right border
        return line_image
    
    def get_indexes(self, canny_image):
        if canny_image.ndim == 3:
            non_black_indexes = np.argwhere(canny_image[:,:,0] > 0)
        else:
            non_black_indexes = np.argwhere(canny_image > 0)
            
        return non_black_indexes[None, ...]

def FID_KID(input_dir_path, results_dir_path):
    """
    Fréchet Inception Distance (FID): we calculate the FID score between the
    structure image and the generated image to quantify the extent to which the
    two images align in terms of their structural features.
    """
    """
    Kernel Inception Distance (KID): Enhanced Similarity Measurement.
    In addition to FID, KID uses a different kernel function the 
    squared MMD distance with the rational quadratic kernel.
    """
    
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=input_dir_path, 
        input2=results_dir_path, 
        cuda=True, 
        fid=True, 
        kid=True,
        kid_subset_size=10,
        verbose=False,
    )
    
    # Define a mapping of old keys to new keys
    key_mapping = {"frechet_inception_distance": "FID", 
                   "kernel_inception_distance_mean": "KID_mean", 
                   "kernel_inception_distance_std": "KID_std"}

    # Create a new dictionary with renamed keys
    metrics_dict = {key_mapping.get(k, k): v for k, v in metrics_dict.items()}
    
    return metrics_dict

def PSNR(image1, image2):
    """
    Peak Signal-to-Noise Ratio (PSNR): it is used to measure 
    how well the generated image preserves the low-level features of appearance.
    """
    #psnr1 = sk_psnr(image1, image2)
    #psnr2 = cv2.PSNR(image1, image2)
    #print(psnr1, psnr2)
    
    return cv2.PSNR(image1, image2)

def GLCM(image1, image2, distances=[1], angles=[0]):
    """
    Gray-Level Co-occurrence Matrix (GLCM): it is used to calculate the loss
    value of texture features between the appearance image and the generated image.
    """
    image1 = (255 * color.rgb2gray(image1)).astype(np.uint8)
    image2 = (255 * color.rgb2gray(image2)).astype(np.uint8)
    print(image1.max(), image2.max())

    # Compute GLCMs for both images
    glcm1 = graycomatrix(image1, distances=distances, angles=angles, symmetric=True, normed=True)
    glcm2 = graycomatrix(image2, distances=distances, angles=angles, symmetric=True, normed=True)

    glcm_loss_abs_sum = np.sum(np.abs((glcm1 - glcm2)))
    glcm_loss_abs_mean = np.mean(np.abs((glcm1 - glcm2)))
    glcm_loss_sq_sum = np.sum((glcm1 - glcm2) ** 2)
    glcm_loss_sq_mean = np.mean((glcm1 - glcm2) ** 2)
    
    print(f"GLCM Matrix L1: sum = {glcm_loss_abs_sum}, mean = {glcm_loss_abs_mean}")
    print(f"GLCM Matrix L2: sum = {glcm_loss_sq_sum}, mean = {glcm_loss_sq_mean}")

    # Extract GLCM features
    features = ['contrast', 'correlation', 'energy', 'homogeneity']
    loss = 0
    for feature in features:
        feature1 = graycoprops(glcm1, feature)
        feature2 = graycoprops(glcm2, feature)
        
        # Mean Squared Error for this feature
        loss_abs_sum = np.sum(np.abs((feature1 - feature2)))
        loss_abs_mean = np.mean(np.abs((feature1 - feature2)))
        loss_sq_sum = np.sum((feature1 - feature2) ** 2)
        loss_sq_mean = np.mean((feature1 - feature2) ** 2)
        
        print(f"{feature}-L1: {loss_abs_sum}, {loss_abs_mean}")
        print(f"{feature}-L2: {loss_sq_sum}, {loss_sq_mean}")
    
    return loss

def ColorHistogramCorrelation(image1, image2, mask1=None, mask2=None, channels=[0,1,2]):
    """
    Color Histogram Correlation (CHC): it is used to calculate the 
    color similarity between the generated image and the appearance image. Among them,
    we use mask to cover the background of the image.
    """
    
    def plot_hist(hist, name):
        # Compute histograms for each channel
        colors = ('red', 'green', 'blue')  # RGB channels
        channel_names = ('R', 'G', 'B')
        plt.figure(figsize=(10, 5))

        for i in range(hist.shape[0]):
            plt.plot(hist[i], color=colors[i], label=f'{channel_names[i]} channel')  # Plot histogram
            plt.xlim([0, 256])
            
        plt.savefig("metrics_testing/"+name, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
        plt.close() 
        
    #image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    #image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)

    if isinstance(mask1, np.ndarray) and isinstance(mask2, np.ndarray):
        mask1 = np.uint8(mask1) * 255
        mask2 = np.uint8(mask2) * 255
        #print(cv2.countNonZero(mask1))
        #print(cv2.countNonZero(mask2))
        save_image(cv2.bitwise_and(image1, image1, mask=mask1), "masked1.png")
        save_image(cv2.bitwise_and(image2, image2, mask=mask2), "masked2.png")
    
    #print(mask1.min(), mask1.max())
    # Compute histograms for both images (use Hue channel for example)
    hist1, hist2 = [], []
    for ch in channels:
        ch_hist1 = cv2.calcHist([image1], [ch], mask1, [256], [0, 256])
        ch_hist2 = cv2.calcHist([image2], [ch], mask2, [256], [0, 256])
        hist1.append(ch_hist1.ravel())  # Flatten the histogram to 1D
        hist2.append(ch_hist2.ravel())  # Flatten the histogram to 1D

    # Stack histograms into a single array of shape (channels, bins)
    hist1 = np.vstack(hist1)
    hist2 = np.vstack(hist2)
    
    plot_hist(hist1, "hist1_plot.png")
    plot_hist(hist2, "hist2_plot.png")
    
    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)
    
    #hist1 = np.swapaxes(hist1, 0, 1)
    #hist2 = np.swapaxes(hist2, 0, 1)
     
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


if __name__ == '__main__':
    
    def get_variable_name(variable, scope):
        return [name for name, value in scope.items() if value is variable][0]

    app_dir_input = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/app_set"
    app_dir_color = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/results_set/app_Blouse/color"
    app_dir_material = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/results_set/app_Blouse/material"
    app_dir_style = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/results_set/app_Blouse/style"
    
    
    struct_dir_input = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/struct_set"
    struct_dir_color = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/results_set/struct_Blouse/color"
    struct_dir_material = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/results_set/struct_Blouse/material"
    struct_dir_style = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/metrics_testing/mini_dataset/results_set/struct_Blouse/style"


    
    im1_path = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/results/selected_material_SD/selected_material_SD/app=Blouse_0-struct=Materials_3-feature/out_transfer---seed_42__TRANSFER_color.png"
    im2_path = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/results/selected_material_SD/selected_material_SD/app=Blouse_0-struct=Materials_3-feature/out_transfer---seed_42__TRANSFER_style.png"
    #im1_path = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/eye_for_an_eye/results/selected_SD/Blouse_0.png"

    im1 = load_image(im1_path)
    im2 = load_image(im2_path)
    
    save_image(im1, "im1.png")
    save_image(im2, "im2.png")
    
    SP = StructurePreservation()
    #sp = SP(im1,im2)
    #print(sp)
    
    #CD = ChamferDistance()
    #cd = CD(im1,im2)
    #print(cd)
    
    """
    from here below implement also masked versions of metrics
    """
    
    if False:
        print("Calculating FID and KID")
        fid_kid_base = FID_KID(app_dir_input, struct_dir_input)
        fid_kid_base["type"] = get_variable_name(fid_kid_base, locals()).split("fid_kid_")[1]
        
        fid_kid_app_color = FID_KID(app_dir_input, app_dir_color)
        fid_kid_app_color["type"] = get_variable_name(fid_kid_app_color, locals()).split("fid_kid_")[1]
        
        fid_kid_app_material = FID_KID(app_dir_input, app_dir_material)
        fid_kid_app_material["type"] = get_variable_name(fid_kid_app_material, locals()).split("fid_kid_")[1]
        
        fid_kid_app_style = FID_KID(app_dir_input, app_dir_style)
        fid_kid_app_style["type"] = get_variable_name(fid_kid_app_style, locals()).split("fid_kid_")[1]
        
        fid_kid_struct_color = FID_KID(struct_dir_input, struct_dir_color)
        fid_kid_struct_color["type"] = get_variable_name(fid_kid_struct_color, locals()).split("fid_kid_")[1]

        fid_kid_struct_material = FID_KID(struct_dir_input, struct_dir_material)
        fid_kid_struct_material["type"] = get_variable_name(fid_kid_struct_material, locals()).split("fid_kid_")[1]

        fid_kid_struct_style = FID_KID(struct_dir_input, struct_dir_style)
        fid_kid_struct_style["type"] = get_variable_name(fid_kid_struct_style, locals()).split("fid_kid_")[1]

        
        

        # Convert dictionary to list of key-value pairs
        data = [
            fid_kid_base,
            fid_kid_app_color,
            fid_kid_app_material,
            fid_kid_app_style,
            fid_kid_struct_color,
            fid_kid_struct_material,
            fid_kid_struct_style
        ]
            # Specify new column order
        new_order = ["type", "FID", "KID_mean", "KID_std"]

        # Reorder the data according to the new order of keys
        reordered_data = [{key: row[key] for key in new_order} for row in data]
        
        # Print the table
        print(tabulate(reordered_data, headers="keys", tablefmt="grid"))
    
    
    #psnr = PSNR(im1, im2)
    #print(psnr)
    
    distances=[1, 3, 5]
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = GLCM(im1, im2)
    print("-------------------------------")
    
    
    mask1 = SP.get_mask(im1)
    mask2 = SP.get_mask(im2)
    chc = ColorHistogramCorrelation(im1, im2, mask1, mask2)
    print(chc)
 

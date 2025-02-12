import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2lab, label2rgb
from skimage import filters

from sam_hq.segment_anything import sam_model_registry, SamPredictor

from run import fill_large_holes, max_spanning_rectangle, create_mask_from_rectangle, apply_erosion, show_res



image_size = 256

# Load image and resize for faster processing
#img_path = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/IDM-VTON/reports/raw_clothes/02532_00_cloth_raw.jpg"    #sleeveless shit
#img_path = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/IDM-VTON/reports/raw_clothes/00654_00_cloth_raw.jpg"    #alien shirt
img_path = "/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/IDM-VTON/reports/raw_clothes/01985_00_cloth_raw.jpg"    #palm shirt

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_shape = image.shape
small_image = cv2.resize(image, (image_size, image_size))  # Resize for faster computation

# Convert to LAB color space for better color clustering
lab_image = cv2.cvtColor(small_image, cv2.COLOR_RGB2LAB)
hsv_image = cv2.cvtColor(small_image, cv2.COLOR_RGB2HSV)

image_formats = [lab_image, hsv_image, lab_image, hsv_image]

for j, image in enumerate(image_formats):
    
    
    channel_1, channel_2, channel_3 = cv2.split(image)
    extra_channel = np.full_like(channel_1, fill_value=150)
    if j == 0:
        image = np.dstack((extra_channel, channel_2, channel_3))
    elif j == 1:
        image = np.dstack((channel_1, channel_2, extra_channel))
        
    n_channels = image.shape[-1]
    reshaped_image = image.reshape((-1, n_channels))

    #--- K-means Segmentation ---
    print("K-means Segmentation")
    # Number of clusters (adjust based on desired segments)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reshaped_image)
    kmeans_labels = kmeans.labels_

    # Reshape clustered labels back to the image shape
    kmeans_segmented = kmeans_labels.reshape((small_image.shape[:2]))

    # Map clusters to average colors
    kmeans_output = np.zeros_like(small_image)
    for i in range(k):
        kmeans_output[kmeans_segmented == i] = kmeans.cluster_centers_[i]
    kmeans_output = cv2.resize(kmeans_output, (original_shape[1], original_shape[0]))

    if j % 2 == 0:
        cv2.imwrite(f'kmeans_segmentation_{j}.jpg', cv2.cvtColor(kmeans_output, cv2.COLOR_LAB2BGR))
    else:
        cv2.imwrite(f'kmeans_segmentation_{j}.jpg', cv2.cvtColor(kmeans_output, cv2.COLOR_HSV2BGR))



# model_type = "vit_l"
# #device = "cuda"
# sam = sam_model_registry[model_type](checkpoint="sam_hq_vit_l.pth")
# #sam.to(device=device)
# predictor = SamPredictor(sam)
# predictor.set_image(small_image)
# hq_token_only = True
# input_box = np.array([[0, 0, image_size, image_size]]) # Cover Full Image
# offset = image_size//32
# input_point = np.array([[image_size//2,image_size//2],[offset,offset],[offset,image_size-offset],[image_size-offset,offset],[image_size-offset,image_size-offset]])
# input_label = np.array([1,0,0,0,0])
# masks, scores, logits = predictor.predict(
#                     point_coords=input_point,
#                     point_labels=input_label,
#                     box = input_box,
#                     multimask_output=False,
#                     hq_token_only=hq_token_only, 
#                 )
# show_res(masks,scores,input_point, input_label, input_box, "SAM_segmentation",small_image)
            



# fill_size = image_size//32
# filled_mask = fill_large_holes(masks[0], kernel_size=fill_size)
# max_area, top_left, bottom_right = max_spanning_rectangle(filled_mask) # finds biggest rectange in the clothing mask
# rectangle_mask = create_mask_from_rectangle(image_size, image_size, top_left, bottom_right)
# eroded_mask = apply_erosion(rectangle_mask, kernel_size=5, iterations=5)


# --- Mean-Shift Segmentation ---
print("Mean-Shift Segmentation")

for j, image in enumerate(image_formats):# Convert to LAB and apply a Gaussian blur to smooth color variations
    #blurred_image = cv2.GaussianBlur(image, (1, 1), 0)
    channel_1, channel_2, channel_3 = cv2.split(image)
    extra_channel = np.full_like(channel_1, fill_value=150)
    if j == 0:
        image = np.dstack((extra_channel, channel_2, channel_3))
    elif j == 1:
        image = np.dstack((channel_1, channel_2, extra_channel))
        
    mean_shift_output = cv2.pyrMeanShiftFiltering(image, sp=10, sr=15)

    # Convert Mean-Shift output back to RGB for visualization
    if j % 2 == 0:
        mean_shift_output = cv2.cvtColor(mean_shift_output, cv2.COLOR_LAB2RGB)
    else:
        mean_shift_output = cv2.cvtColor(mean_shift_output, cv2.COLOR_HSV2RGB)

    #mean_shift_output = cv2.cvtColor(mean_shift_output, cv2.COLOR_LAB2RGB)
    mean_shift_output = cv2.resize(mean_shift_output, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Save Results
    cv2.imwrite(f'mean_shift_segmentation_{j}.jpg', cv2.cvtColor(mean_shift_output, cv2.COLOR_RGB2BGR))

print("Done!")

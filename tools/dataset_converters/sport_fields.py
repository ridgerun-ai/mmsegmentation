import argparse
import cv2
import numpy as np
import os
import shutil
import random

def convert_and_save_mask(mask_input_path, mask_output_path):
    mask = cv2.imread(mask_input_path, cv2.IMREAD_UNCHANGED)
    mask = np.uint8(mask/255)
    cv2.imwrite(mask_output_path, mask)

parser = argparse.ArgumentParser(description="Overlay masks")
parser.add_argument("--images", type=str, default="dataset/images",
                    help="Path to the folder of images")
parser.add_argument("--masks", type=str, default="dataset/masks",
                    help="Path to a folder of masks, "
                    "mask name should match the image name")    
parser.add_argument("--output", type=str, default="data/sport_fields/",
                    help="Path to save dataset structure")
args = parser.parse_args()

# Define the source dataset folder and target folders
source_dataset_images = args.images
source_dataset_masks = args.masks
output_dir = args.output

target_images_folder = os.path.join(output_dir, 'images')
target_images_val_folder = os.path.join(target_images_folder, 'val')
target_images_train_folder = os.path.join(target_images_folder, 'train')
target_ann_folder = os.path.join(output_dir, 'annotations')
target_ann_val_folder = os.path.join(target_ann_folder, 'val')
target_ann_train_folder = os.path.join(target_ann_folder, 'train')

# Create target directories if they don't exist
os.makedirs(target_images_train_folder, exist_ok=True)
os.makedirs(target_images_val_folder, exist_ok=True)
os.makedirs(target_ann_train_folder, exist_ok=True)
os.makedirs(target_ann_val_folder, exist_ok=True)

# List files in the source dataset folder
image_files = [f for f in os.listdir(source_dataset_images) if f.endswith(('.jpg', '.png'))]
mask_files = [f for f in os.listdir(source_dataset_masks) if f.endswith('.png')]

# Shuffle the list of files to ensure a random split
random.shuffle(image_files)

# Calculate the number of files for training (90%) and validation (10%)
num_train_files = int(len(image_files) * 0.9)
train_files = image_files[:num_train_files]
val_files = image_files[num_train_files:]

# Function to get the base name without extension
def get_base_name(filename):
    return os.path.splitext(filename)[0]

# Move the files
for filename in train_files:
    shutil.copy(os.path.join(source_dataset_images, filename),
                os.path.join(target_images_train_folder, filename))
    # Masks are always png
    mask_filename = get_base_name(filename) + '.png'
    convert_and_save_mask(os.path.join(source_dataset_masks, mask_filename),
                os.path.join(target_ann_train_folder, mask_filename))

for filename in val_files:
    shutil.copy(os.path.join(source_dataset_images, filename),
                os.path.join(target_images_val_folder, filename))
    # Masks are always png
    mask_filename = get_base_name(filename) + '.png'
    convert_and_save_mask(os.path.join(source_dataset_masks, mask_filename),
                os.path.join(target_ann_val_folder, mask_filename))

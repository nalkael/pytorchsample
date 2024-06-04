import os # handling directory operations
import shutil # handling moving files
import random

# source and destination directories
source_dir = '/home/rdluhu/Dokumente/preprecess/sample_images/images'

jpg_train_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/images/train'
jpg_val_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/images/val'

txt_train_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/labels/train'
txt_val_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/labels/val'

# ensure all corresponding directories exist
os.makedirs(jpg_train_dir, exist_ok=True)
os.makedirs(jpg_val_dir, exist_ok=True)
os.makedirs(txt_train_dir, exist_ok=True)
os.makedirs(txt_val_dir, exist_ok=True)

# shuffle the images and divide them into training dataset and validation dataset
image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith('.jpg')]
random.shuffle(image_files)

# the total number of the image files
total_images = len(image_files)

# the split ration (e.g., 80% of images are used as training data, 20% as validation data)
train_ration = 0.8
num_train = int(total_images * train_ration)
num_val = total_images - num_train

# move files 1 to 50 into corresponding folders
for i in range(1, 51):
    jpg_filename = f'img{i}.jpg'
    txt_filename = f'img{i}.txt'
    
    # get source path of images and labels
    jpg_source_path = os.path.join(source_dir, jpg_filename)
    txt_source_path = os.path.join(source_dir, txt_filename)

    if os.path.exists(jpg_source_path) and os.path.exists(txt_source_path):
        shutil.copy(jpg_source_path, jpg_train_dir)
        shutil.copy(txt_source_path, txt_train_dir)
    else:
        print(f'File {jpg_filename} or {txt_filename} does not exist in the source directory.')
    
# move files 51 to 63 to the val folder
for i in range(51, 64):
    jpg_filename = f'img{i}.jpg'
    txt_filename = f'img{i}.txt'
    
    # get source path of images and labels
    jpg_source_path = os.path.join(source_dir, jpg_filename)
    txt_source_path = os.path.join(source_dir, txt_filename)

    if os.path.exists(jpg_source_path) and os.path.exists(txt_source_path):
        shutil.copy(jpg_source_path, jpg_val_dir)
        shutil.copy(txt_source_path, txt_val_dir)
    else:
        print(f'File {jpg_filename} or {txt_filename} does not exist in the source directory.')
import os # handling directory operations
import shutil # handling moving files

# source and destination directories
source_dir = '/home/rdluhu/Dokumente/preprecess/sample_images/images'

jpg_train_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/images/train'
jpg_val_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/images/val'

txt_train_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/labels/train'
txt_val_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/labels/val'

os.makedirs(jpg_train_dir, exist_ok=True)
os.makedirs(jpg_val_dir, exist_ok=True)
os.makedirs(txt_train_dir, exist_ok=True)
os.makedirs(txt_val_dir, exist_ok=True)

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
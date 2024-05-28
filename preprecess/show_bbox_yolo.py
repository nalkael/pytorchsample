'''
 a simple program that displays an image with bounding boxes 
 based on coordinates stores in a YOLO format text file
 '''

import cv2

def draw_bbox(image, bbox_file):
    # load the image file
    img = cv2.imread(image)

    # load the txt file contains bounding box information
    with open(bbox_file, 'r') as file:
        pass
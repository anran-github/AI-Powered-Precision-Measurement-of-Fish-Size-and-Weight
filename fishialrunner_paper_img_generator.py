# -*- coding: utf-8 -*-
"""FishialRUNNER.ipynb

Generate results for papers.

"""

# Commented out IPython magic to ensure Python compatibility.
import os
import sys
import copy
import random
import json
import yaml
import glob
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from   zipfile import ZipFile
# %matplotlib inline
import pandas as pd
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)

def download_and_unzip(url, save_path, extract_dir):
    print("Downloading assets...")
    file = requests.get(url)

    open(save_path, "wb").write(file.content)
    print("Download completed.")

    try:
        if save_path.endswith(".zip"):
            with ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Extraction Done")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_image(url):
    filename = os.path.basename(url)

    response = requests.get(url)
    response.raise_for_status()

    with open(filename, 'wb') as file:
        file.write(response.content)

    return os.path.abspath(filename)

def get_basename(path):
  return os.path.basename(path)

def print_fish_data(fish_data):
    for idx, fish in enumerate(fish_data, start=1):
        print(f"ID: {idx}")
        print(f"Name: {fish['name']}")
        print(f"Species ID: {fish['species_id']}")
        # print(f"Distance: {fish['distance']:.3f}")
        print(f"Accuracy: {fish['accuracy']:.2%}")
        print("-" * 40)

# Links to models
MODEL_URLS = {
    'classification': 'https://storage.googleapis.com/fishial-ml-resources/classification_rectangle_v7-1.zip',
    'segmentation': 'https://storage.googleapis.com/fishial-ml-resources/segmentator_fpn_res18_416_1.zip',
    'detection': 'https://storage.googleapis.com/fishial-ml-resources/detector_v10_m3.zip',
    'face': 'https://storage.googleapis.com/fishial-ml-resources/face_yolo.zip'
}

# Model directories
MODEL_DIRS = {
    'classification': "models/classification",
    'segmentation': "models/segmentation",
    'detection': "models/detection",
    'face': "models/face_detector"
}

# Create directories and download models
for model_name, url in MODEL_URLS.items():
    model_dir = MODEL_DIRS[model_name]
    zip_path = os.path.join(os.getcwd(), get_basename(url))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
        download_and_unzip(url, zip_path, model_dir)  # Download and unzip the model

        # Remove the zip file after extraction
        try:
            os.remove(zip_path)
            logging.info(f"Removed zip file {zip_path}")
        except Exception as e:
            logging.error(f"Failed to remove zip file {zip_path}: {e}")



from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference
from models.face_detector.inference import YOLOInference as FaceInference


# Model initialization
classifier = EmbeddingClassifier(
    os.path.join(MODEL_DIRS['classification'], 'model.ts'),
    os.path.join(MODEL_DIRS['classification'], 'database.pt')
)

segmentator = Inference(
    model_path=os.path.join(MODEL_DIRS['segmentation'], 'model.ts'),
    image_size=416
)

detector = YOLOInference(
    os.path.join(MODEL_DIRS['detection'], 'model.ts'),
    imsz=(640, 640),
    conf_threshold=0.9,
    nms_threshold=0.3,
    yolo_ver='v10'
)

# face_detector = FaceInference(
#     os.path.join(MODEL_DIRS['face'], 'model.ts'),
#     imsz=(640, 640),
#     conf_threshold=0.69,
#     nms_threshold=0.5,
#     yolo_ver='v8'
# )

# You can change link below to your image with fish

fish_path = '/media/anranli/DATA/data/fish/D1 12_3_24'
fish_list = os.listdir(fish_path)

fish_list_higherror = ['/media/anranli/DATA/data/fish/Growth Study Day 3 [12-18-24]/27.JPG']

# for fish_i in fish_list:
#     fish_bgr_np = cv2.imread(os.path.join(fish_path,fish_i))
for fish_i in fish_list_higherror:
    fish_bgr_np = cv2.imread(fish_i)
    visulize_img_bgr = fish_bgr_np.copy()

    visulize_img_rgb = cv2.cvtColor(fish_bgr_np, cv2.COLOR_BGR2RGB)
    visulize_img = copy.deepcopy(visulize_img_rgb)


    # face_boxes = face_detector.predict(visulize_img_rgb)[0]

    # for box in face_boxes:
    #   box.draw_label(visulize_img, "Face")
    #   box.draw_box(visulize_img)
    # plt.imshow(visulize_img)
    # plt.show()

    boxes = detector.predict(visulize_img_rgb)[0]

    for box in boxes:
        cropped_fish_bgr = box.get_mask_BGR()
        cropped_fish_rgb = box.get_mask_RGB()
        cv2.imwrite('paper_image/cropped_fish_bgr.png',cropped_fish_rgb)

        segmented_polygons = segmentator.predict(cropped_fish_bgr)[0]

        croped_fish_mask = segmented_polygons.mask_polygon(cropped_fish_rgb)

        segmented_polygons.move_to(box.x1, box.y1)
        segmented_polygons.draw_polygon(visulize_img)

        classification_result = classifier.batch_inference([cropped_fish_bgr])[0]

        label = f"{classification_result[0]['name']} | {round(classification_result[0]['accuracy'], 3)}" if len(classification_result) else "Not Found"
        box.draw_label(visulize_img, label)
        box.draw_box(visulize_img)

        print(50 * "=")
        # plt.imsave(f'fish_{int(time.time())}.png',croped_fish_mask)
        croped_fish_mask = cv2.cvtColor(croped_fish_mask, cv2.COLOR_BGR2RGB)
        plt.imsave('paper_image/cropped_fish_mask.png',croped_fish_mask)
        plt.imshow(croped_fish_mask)
        plt.show()
        # class fish: 
        print_fish_data(classification_result)



        croped_fish_mask = np.zeros_like(cropped_fish_rgb)
        # Fill the object inside the boundary with white
        bw_background = np.zeros_like(visulize_img,np.uint8)
        cv2.fillPoly(bw_background, [np.array(segmented_polygons.points)], (255,255,255))  # 255 for white color
        cv2.imwrite('paper_image/fish_mask_ori_image.png',bw_background)
        # =================ROTATION FISH EXP===============================

        # Load the binary segmented mask
        mask = croped_fish_mask.copy()

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Ensure the mask is binary
        # _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming the object is represented by the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        
        # cv2.drawContours(mask, [contour], -1, (0, 255, 0), 2) 
        # cv2.imshow('contour find',mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Compute the minimum area rotated bounding rectangle
        rect = cv2.minAreaRect(contour)

        # Get the box points (corner points of the rotated rectangle)
        boxx = cv2.boxPoints(rect)

        # return location to origon image:
        boxx[:,0] += box.x1
        boxx[:,1] += box.y1
        boxx = np.intp(boxx)  # Convert to integer

        # ******Reculate width and height --> Directly applied as fish size *******
        fish_height = np.sqrt((boxx[0,0]-boxx[1,0])**2 + (boxx[0,1]-boxx[1,1])**2 )
        fish_width = np.sqrt((boxx[2,0]-boxx[1,0])**2 + (boxx[2,1]-boxx[1,1])**2 )
        if fish_height > fish_width:
            # swap height and width
            tmp = fish_height
            fish_height = fish_width
            fish_width = tmp


        # BBOX VERIFY:
        # Draw the bounding box on the ori image (for visualization)
        print(f'width: {fish_width}')

        # instead drawcountours, crop this region as a new image:

        cv2.drawContours(visulize_img, [boxx], 0, (0, 255, 0), 4)

        
        # Display the result
        visulize_img = cv2.cvtColor(visulize_img,cv2.COLOR_RGB2BGR)
        imS = cv2.resize(visulize_img, (visulize_img.shape[1]//4, visulize_img.shape[0]//4))                # Resize image
        cv2.imwrite('final_visual.png',imS)
        cv2.imshow("output", imS) 
        # cv2.imshow('Rotated Bounding Box', visulize_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # =================================================================






    # plt.imsave(f'fish_{int(time.time())}.png',visulize_img)
    plt.imshow(visulize_img)
    plt.show()
    plt.close()

    """# FU"""
    '''
    import time
    import cv2

    # Image Load
    fish_bgr_np = cv2.imread(fish_path)
    fish_rgb_np = cv2.cvtColor(fish_bgr_np, cv2.COLOR_BGR2RGB)

    times_array = []
    for _ in range(3):
    start_time_complex = time.time()
    print(20 * "=")
    #   start_time = time.time()
    #   face_boxes = face_detector.predict(face_bgr_np)[0]
    #   end_time = time.time()
    #   print(f"Face detection time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    boxes = detector.predict(fish_bgr_np)[0]
    end_time = time.time()
    print(f"Fish detection time: {end_time - start_time:.4f} seconds")

    # Обработка каждого объекта
    for box in boxes:
        cropped_fish_bgr = box.get_mask_BGR()

        # Segmentation
        start_time = time.time()
        segmented_polygons = segmentator.predict(cropped_fish_bgr)[0]
        end_time = time.time()
        print(f"Segmentation time: {end_time - start_time:.4f} seconds")

        # Classification
        #   start_time = time.time()
        #   classification_result = classifier.batch_inference([cropped_fish_bgr])[0]
        #   end_time = time.time()
        #   print(f"Classification time: {end_time - start_time:.4f} seconds")
    times_array.append(time.time() - start_time_complex)

    print(f"Average time: {sum(times_array)/len(times_array)}")
    '''
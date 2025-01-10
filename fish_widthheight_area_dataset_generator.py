# -*- coding: utf-8 -*-
"""
Read images and use current YOLO and segment NN:

genrate width, height, and area of detected fish. (UNIT: PIXEL SIZE)

All data is saved in dataset.json for weight prediction NN training.

"""

# Commented out IPython magic to ensure Python compatibility.
import os
import copy
import json
import cv2
import numpy as np
from tqdm import tqdm
import requests
from   zipfile import ZipFile
# %matplotlib inline
import logging

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
    conf_threshold=0.65,
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

# If you extend your dataset, add the extended path below:
fish_pathes = [
               '/media/anranli/DATA/data/fish/Growth Study Day 2 [12-11-24]',
               '/media/anranli/DATA/data/fish/Growth Study Day 3 [12-18-24]',
               '/media/anranli/DATA/data/fish/Growth Study Day 4 [12-30-24]',
               '/media/anranli/DATA/data/fish/Tk 4 - varied data',
               '/media/anranli/DATA/data/fish/Tk 5 - varied data']
save_path = 'bbox_area_dataset.json'
res = []
for fish_path in fish_pathes:
    fish_list = os.listdir(fish_path)

    img_higherror_list =  ["/media/anranli/DATA/data/fish/Growth Study Day 3 [12-18-24]/57.JPG"]
    
    # for fish_i in tqdm(fish_list):
    for fish_i in img_higherror_list:
        if not fish_i.endswith(('.JPG','.jpeg')):
            continue
        
        # fish_bgr_np = cv2.imread(os.path.join(fish_path,fish_i))
        fish_bgr_np = cv2.imread(fish_i)
        visulize_img_bgr = fish_bgr_np.copy()

        visulize_img_rgb = cv2.cvtColor(fish_bgr_np, cv2.COLOR_BGR2RGB)
        visulize_img = copy.deepcopy(visulize_img_rgb)


        boxes = detector.predict(visulize_img_rgb)[0]
        
        # only save data having bbox:
        for box in boxes:
            cropped_fish_bgr = box.get_mask_BGR()
            cropped_fish_rgb = box.get_mask_RGB()
            segmented_polygons = segmentator.predict(cropped_fish_bgr)[0]

            croped_fish_mask = segmented_polygons.mask_polygon(cropped_fish_rgb)

            # output_seg = cv2.cvtColor(croped_fish_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite('cropped_fish.png',croped_fish_mask)
            cv2.imshow('cropped fish mask',croped_fish_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            croped_fish_mask = np.zeros_like(cropped_fish_rgb)
            # Fill the object inside the boundary with white
            cv2.fillPoly(croped_fish_mask, [segmented_polygons.points], (255,255,255))  # 255 for white color


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




            # segmented_polygons.move_to(box.x1, box.y1)
            # segmented_polygons.draw_polygon(visulize_img)

            # classification_result = classifier.batch_inference([cropped_fish_bgr])[0]

            # label = f"{classification_result[0]['name']} | {round(classification_result[0]['accuracy'], 3)}" if len(classification_result) else "Not Found"
            # box.draw_label(visulize_img, label)
            # box.draw_box(visulize_img)
            res.append([os.path.join(fish_path,fish_i),fish_width,fish_height,segmented_polygons.to_dict()['area']])
            # res.append([box.width,box.height,segmented_polygons.to_dict()['area']])




# with open(save_path,'w') as f:
#     json.dump(res,f)
#     # f.write(';\n')

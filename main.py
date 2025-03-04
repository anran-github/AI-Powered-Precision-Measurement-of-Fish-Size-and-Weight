from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import copy
import json
import matplotlib.pyplot as plt
import requests
from   zipfile import ZipFile
# %matplotlib inline
import logging
import torch

from fish_weight_model import WeightNet
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference
from models.face_detector.inference import YOLOInference as FaceInference


app = FastAPI()

# calibration process

def crop_selected_img(image):
    '''
    Select image and crop via opencv roi toolbox
    '''
    image_copy = image.copy()
    ref_point = []
    cropping = False
    resize_idx = 4
    image_resize = cv2.resize(image,(image.shape[1]//resize_idx,image.shape[0]//resize_idx))


    while True:
        # Display the image and let the user select a region
        r = cv2.selectROI("Image", image_resize)
        x1, y1, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])

        # reproject to origional size:
        x2, y2 = x1 + w, y1 + h
        x1,y1,x2,y2 = int(x1*resize_idx),int(y1*resize_idx),int(x2*resize_idx),int(y2*resize_idx)

        if w > 0 and h > 0:
            
            roi = image[y1:y2, x1:x2]
            # cv2.imwrite('paper_image/cropped_dot.png',roi)
            # cv2.imshow("Processed ROI", roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            break


    
    return roi

def find_coin_diameter(image):

    # denoise image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # # Gaussian Blur
    # gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # # Median Blur
    # median_blur = cv2.medianBlur(image, 5)


    image = crop_selected_img(image)

    # Brighten the image
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)  # Increase alpha and beta for brightness
    # Non-Local Means Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ================PREVIOUS METHOD: Find edges using edge detection ===============
    edges = cv2.Canny(blurred, 20, 250)
    # Find the contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ================OTHER METHOD: FIND CIRCLE WITH BLUE COLOR======================
    # # Convert the image to HSV for better color segmentation
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # Define the blue color range in HSV
    # lower_blue = np.array([100, 150, 50])  # Adjust these values if necessary
    # upper_blue = np.array([140, 255, 255])
    # # Create a binary mask where blue is white and other colors are black
    # mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    # # Find contours in the mask
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    
    
    contours_cnt = [len(x) for x in contours]
    perimeter = cv2.arcLength(contours[np.argmax(contours_cnt)], True)
    # Draw contours
    output_image = image.copy()
    cv2.drawContours(output_image, [contours[np.argmax(contours_cnt)]], -1, (0, 255, 0), 2)  # Draw in green with thickness 2

    # Find the largest contour (which should be the coin)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the minimum enclosing circle of the largest contour
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    # Calculate the diameter of the coin
    diameter = 2 * radius

    return 1.27/diameter, output_image # Convert to cm/pixel


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_model = WeightNet().to(device)

weight_model.load_state_dict(torch.load('fish_saved_weights/model_epoch80_0.15009590983390808.pth'))
weight_model.eval()
print('model loaded')


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global variable to store calibration factor
calibration_factor = None

@app.post("/calibrate/")
async def calibrate(file: UploadFile = File(...)):
    global calibration_factor
    
    # Read the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = np.array(image)

    calibration_factor,out_img = find_coin_diameter(image)

    cv2.imshow("Calibration Circle", out_img)
    # Wait and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Fake calibration logic (Replace with actual calibration process)
    # reference_object_real_length = 10.0  # cm (example)
    # reference_object_pixel_length = image.shape[1] // 4  # Assume detected object width
    
    # calibration_factor = reference_object_real_length / reference_object_pixel_length
    
    return {"scale_factor": calibration_factor}


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Read the image
    res = []
    global calibration_factor

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = np.array(image)

    # Fake object detection logic (Replace with real detection)
    h, w, _ = image.shape

    fish_bgr_np = image.copy()

    visulize_img_rgb = cv2.cvtColor(fish_bgr_np, cv2.COLOR_BGR2RGB)
    visulize_img = copy.deepcopy(visulize_img_rgb)


    boxes = detector.predict(visulize_img_rgb)[0]
    
    # only save data having bbox:
    for box in boxes:
        cropped_fish_bgr = box.get_mask_BGR()
        cropped_fish_rgb = box.get_mask_RGB()
        segmented_polygons = segmentator.predict(cropped_fish_bgr)[0]

        croped_fish_mask = segmented_polygons.mask_polygon(cropped_fish_rgb)

        # # output_seg = cv2.cvtColor(croped_fish_mask, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('cropped_fish.png',croped_fish_mask)
        # cv2.imshow('cropped fish mask',croped_fish_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        croped_fish_mask = np.zeros_like(cropped_fish_rgb)
        # Fill the object inside the boundary with white
        cv2.fillPoly(croped_fish_mask, [segmented_polygons.points], (255,255,255))  # 255 for white color


        # =================ROTATION FISH EXP===============================

        # Load the binary segmented mask
        mask = croped_fish_mask.copy()

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

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
        
        # Display the result
        # visulize_img = cv2.cvtColor(visulize_img,cv2.COLOR_RGB2BGR)
        # imS = cv2.resize(visulize_img, (visulize_img.shape[1]//4, visulize_img.shape[0]//4))                # Resize image
        # # cv2.imwrite('final_visual.png',imS)
        # cv2.imshow("output", imS) 
        # # cv2.imshow('Rotated Bounding Box', visulize_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # =================================================================

        # get result: [width, height, area]
        # res.append([box.width,box.height,segmented_polygons.to_dict()['area']])


    # convert to real size:
    if len(boxes) != 0:
        fish_width = fish_width * calibration_factor
        fish_height = fish_height * calibration_factor
        fish_area = segmented_polygons.to_dict()['area'] * calibration_factor * calibration_factor

        # predict weight:
        with torch.no_grad():
            input_data = torch.tensor([fish_width, fish_height, fish_area]).to(device).float().unsqueeze(0)
            # input_data = torch.tensor([6.642561445470936, 1.5305083096582157, 6.966261551986847]).to(device).float().unsqueeze(0)
            fish_weight = weight_model(input_data).item()
        # print(h,w)
        # print(input_data.shape)
        # print(fish_width,fish_height,fish_area,fish_weight)


        # Convert to list of lists
        bbox = boxx.tolist()

        return {
            "bounding_box": bbox,  # List of 4 (x, y) points
            "fish_width": fish_width,
            "fish_height": fish_height,
            "fish_area": fish_area,  
            "fish_mass": fish_weight
        }

    else:
        return {
            "bounding_box": [[0,0],[0,0],[0,0],[0,0]],  # List of 4 (x, y) points
            "fish_width": 0,
            "fish_height": 0,
            "fish_area": 0, 
            "fish_mass": 0
        }
    

@app.post("/save_results/")
async def save_results(data: list[dict]):
    df = pd.DataFrame(data)

    # Save to Excel (append if file exists)
    file_name = "measurement_results.xlsx"
    try:
        existing_df = pd.read_excel(file_name)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass  # No existing file, create a new one

    df.to_excel(file_name, index=False)
    return {"message": "Results saved successfully"}
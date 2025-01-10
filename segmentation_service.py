from flask import Flask, request, jsonify, send_file
import numpy as np
import torch
from PIL import Image
import io
import os
import copy
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from   zipfile import ZipFile
# %matplotlib inline
import torch


from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference
from models.face_detector.inference import YOLOInference as FaceInference



app = Flask(__name__)



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


# Model directories
MODEL_DIRS = {
    'classification': "models/classification",
    'segmentation': "models/segmentation",
    'detection': "models/detection",
    'face': "models/face_detector"
}


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




def model_prediction(fish_bgr_np):
    visulize_img_bgr = fish_bgr_np.copy()

    visulize_img_rgb = cv2.cvtColor(fish_bgr_np, cv2.COLOR_BGR2RGB)
    visulize_img = copy.deepcopy(visulize_img_rgb)



    boxes = detector.predict(visulize_img_rgb)[0]

    # save result for each box
    res = {'box':[],'seg_poly':[]}
    for box in boxes:
        cropped_fish_bgr = box.get_mask_BGR()
        cropped_fish_rgb = box.get_mask_RGB()
        segmented_polygons = segmentator.predict(cropped_fish_bgr)[0]

        croped_fish_mask = segmented_polygons.mask_polygon(cropped_fish_rgb)

        segmented_polygons.move_to(box.x1, box.y1)
        segmented_polygons.draw_polygon(visulize_img)

        classification_result = classifier.batch_inference([cropped_fish_bgr])[0]

        label = f"{classification_result[0]['name']} | {round(classification_result[0]['accuracy'], 3)}" if len(classification_result) else "Not Found"
        box.draw_label(visulize_img, label)
        box.draw_box(visulize_img)

        # res['box'].extend(box.to_dict()['box'])
        res['box'].append([box.width,box.height])
        res['seg_poly'].append(segmented_polygons.to_dict()['area'])

    return res



# Preprocessing function
def preprocess_image(image: Image.Image):
    # Convert to tensor and normalize (update based on your model's requirements)
    transform = torch.nn.Sequential(
        torch.nn.functional.interpolate(size=(256, 256)),  # Resize image
        lambda x: x / 255.0  # Normalize to [0, 1]
    )
    image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1))  # HWC to CHW
    image_tensor = transform(image_tensor.unsqueeze(0))  # Add batch dimension
    return image_tensor

# Postprocessing function
def postprocess_mask(mask_tensor: torch.Tensor):
    # Convert to binary mask and PIL Image
    mask = (mask_tensor.squeeze(0).detach().numpy() > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)

@app.route('/segment', methods=['POST'])
def segment_fish():
    try:
        # Ensure an image file is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        # Read the image
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')

        res = model_prediction(np.array(image))

        # Return the bbox and the area of segment region.
        return  jsonify(res)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()

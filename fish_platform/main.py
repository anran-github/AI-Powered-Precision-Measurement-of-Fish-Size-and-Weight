from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = np.array(image)

    # Fake object detection logic (replace with real model)
    h, w, _ = image.shape
    x, y, box_w, box_h = w // 4, h // 4, w // 2, h // 2  # Fake bounding box
    accuracy = np.random.uniform(70, 99)  # Fake accuracy percentage

    return {
        "bounding_box": [x, y, box_w, box_h],
        "accuracy": accuracy
    }


import cv2
import numpy as np
import torch
import copy
from PIL import Image
import os
from time import time
from tqdm import tqdm
import pandas as pd

from fish_weight_model import WeightNet
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

# ---------------------------------------------------------
# USER PARAMETERS
# ---------------------------------------------------------
DETECTION_THRESHOLD = 0.7
RATIO_WH            = 4
AREA_THRESHOLD      = 400*100
PRINT_DEBUG = False

TILE_SIZE    = 640
TILE_OVERLAP = 0.3        # 30% overlap – required for small-object detection

# ---------------------------------------------------------
# Load your models
# ---------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_model = WeightNet().to(device)

weight_model.load_state_dict(torch.load('fish_saved_weights/model_epoch80_0.15009590983390808.pth'))
weight_model.eval()
print('Weight model loaded')

detector = YOLOInference(
    "models/detection/model.ts",
    imsz=(TILE_SIZE, TILE_SIZE),   # important: inference on tiles
    conf_threshold=0.5,
    nms_threshold=0.3,
    yolo_ver='v10'
)

segmentator = Inference(
    model_path="models/segmentation/model.ts",
    image_size=416
)


class ContourDetector:
    def __init__(self):
        self.image = None

    def obj_segmentation(self, image):
        """
        Detect contours from image
        Return: [[x,y,w,h], ...]
        """
        self.image = image
        if self.image is None:
            print("Error: Could not load image.")
            return None, None

        mm_per_px = self.compute_mm_per_px()
        if mm_per_px is None:
            return None, None

        img = self.image.copy()

        # ==========================================
        # 1. Convert to grayscale
        # ==========================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # 2. Blur slightly to suppress noise
        # ==========================================
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ==========================================
        # 3. Adaptive threshold (better for small objects)
        # Fish are darker than background → invert threshold
        # ==========================================
        binary = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,   # block size (must be odd)
            2
        )

        # ==========================================
        # 4. Morphological operations
        # Make small fish more solid
        # ==========================================
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.dilate(binary, kernel_big, iterations=1)

        # ==========================================
        # 5. Find contours
        # ==========================================
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_contours = []

        # Lower this a lot to ensure small fish are kept
        min_contour_area_mm2 = 7  # was 7, too large
        min_contour_area_px = min_contour_area_mm2 / (mm_per_px ** 2)

        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area_px:
                filtered_contours.append(contour)

        tiles = []
        coords = []

        # ==========================================
        # 6. Bounding boxes
        # ==========================================
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Expand bounding box slightly (important!)
            pad = 5
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, img.shape[1])
            y2 = min(y + h + pad, img.shape[0])

            cropped = img[y1:y2, x1:x2]

            tiles.append(cropped)
            coords.append([x1, y1])

        return tiles, coords

    def compute_mm_per_px(self):
        return 0.1



def tile_image(img, tile_size=640, overlap=0.3):

    H, W, _ = img.shape
    step = int(tile_size * (1 - overlap))

    tiles = []
    coords = []

    for y in range(0, H, step):
        for x in range(0, W, step):

            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)

            tile = img[y:y2, x:x2]
            tiles.append(tile)
            coords.append((x, y))

    return tiles, coords


# ---------------------------------------------------------
# Helper: convert local tile boxes → global-frame boxes
# ---------------------------------------------------------
def shift_boxes(global_dets, tile_x, tile_y):
    for b in global_dets:
        b.x1 += tile_x
        b.x2 += tile_x
        b.y1 += tile_y
        b.y2 += tile_y
    return global_dets



# ---------------------------------------------------------
# Helper: merge all tile detections, then apply NMS
# ---------------------------------------------------------
def merge_all_boxes(all_dets, iou_thresh=0.4):

    if len(all_dets) == 0:
        return []

    # Convert to format: [x1, y1, x2, y2, score]
    boxes = []
    for b in all_dets:
        boxes.append([b.x1, b.y1, b.x2, b.y2, b.score])

    boxes = np.array(boxes)

    if len(boxes) == 0:
        return []

    # apply NMS
    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes[:, :4].tolist(),
        scores=boxes[:, 4].tolist(),
        score_threshold=0.0,
        nms_threshold=iou_thresh
    )

    merged = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            merged.append(all_dets[i])

    return merged



# ---------------------------------------------------------
# Same: compute rotated bounding box from your segmentation
# ---------------------------------------------------------
def get_fish_size_and_box(full_img_rgb, box):

    cropped_fish_bgr = box.get_mask_BGR()
    cropped_fish_rgb = box.get_mask_RGB()

    segmented_polygons = segmentator.predict(cropped_fish_bgr)[0]

    crop_mask_rgb = np.zeros_like(cropped_fish_rgb)
    cv2.fillPoly(crop_mask_rgb, [segmented_polygons.points], (255,255,255))

    mask_gray = cv2.cvtColor(crop_mask_rgb, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None, None

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect)

    # shift back to global coordinates
    box_pts[:,0] += box.x1
    box_pts[:,1] += box.y1
    box_pts = np.intp(box_pts)

    # compute L, H
    h1 = np.linalg.norm(box_pts[0] - box_pts[1])
    w1 = np.linalg.norm(box_pts[1] - box_pts[2])
    length = max(h1, w1)
    height = min(h1, w1)

    return length, height, box_pts, segmented_polygons.to_dict()['area']



# calibration process
def find_coin_diameter_from_crop(crop_rgb,detected_save_path):
    """
    crop_rgb: numpy array in RGB (HxWx3). Returns (scale_cm_per_pixel, annotated_rgb_image).
    Uses your previous processing but WITHOUT calling selectROI or imshow.
    """
    # convert RGB->BGR for OpenCV ops (if crop is RGB)
    img = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    # image enhancement / denoising
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 20, 250)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the provided crop. Try a larger crop or better lighting.")

    # find largest contour and min enclosing circle
    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    diameter_px = 2.0 * radius
    if diameter_px <= 0.0:
        raise ValueError("Detected diameter is zero or negative.")

    # annotated image (draw contour + circle)
    annotated = img.copy()
    cv2.drawContours(annotated, [largest_contour], -1, (0, 255, 0), 2)
    center = (int(round(x)), int(round(y)))
    cv2.circle(annotated, center, int(round(radius)), (0, 0, 255), 2)  # circle in red

    # convert annotated to RGB for returning
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    cv2.imwrite(detected_save_path,annotated_rgb)

    # cv2.imshow('res',annotated_rgb)
    # cv2.waitKey(0)

    # coin real diameter in cm (your code used 1.27 cm)
    coin_cm = 1.27
    cm_per_pixel = coin_cm / diameter_px

    return cm_per_pixel, True



# ---------------------------------------------------------
# Main video processing with tiling inference
# ---------------------------------------------------------
def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    writer = None
    first_frame_written = False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1. Define your starting frame (0-indexed)
    start_frame = 0 
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # flag for blue-dot: real-size correction
    REAL_SIZE_FOUND = False 
    weight_model_input = []
    for i in tqdm(range(start_frame,total_frames), desc="Rendering", unit="fps"):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # create writer AFTER first frame
        # -----------------------------
        if output_path is not None and not first_frame_written:
            h, w = frame_bgr.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)

            # use MP4V or XVID for maximum compatibility
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            first_frame_written = True
        

        # Search For Real Size
        if not REAL_SIZE_FOUND:
            calibration_factor, REAL_SIZE_FOUND = find_coin_diameter_from_crop(frame_rgb,video_path.split('/')[-1].replace('.MOV','.png'))
        # -------------------------------------------
        # 1. Tile the large frame
        # -------------------------------------------
        # tiles, coords = ContourDetector().obj_segmentation(frame_bgr)
        tiles, coords = tile_image(frame_rgb, TILE_SIZE, TILE_OVERLAP)

        all_detects = []

        # -------------------------------------------
        # 2. Detect in each tile
        # -------------------------------------------
        dets = detector.predict(tiles)
        for det, (tx, ty) in zip(dets, coords):
            if len(det)==0:
                continue

            # shift tile detections back to global coordinates
            dets_shifted = shift_boxes(det, tx, ty)
            all_detects.extend(dets_shifted)

        # -------------------------------------------
        # 3. Merge tile detections → global NMS
        # -------------------------------------------
        detections = merge_all_boxes(all_detects, iou_thresh=0.4)

        # -------------------------------------------
        # 4. Process all remaining detections
        # -------------------------------------------
        for box in detections:
            
            # consider score large than threshold
            conf = box.score
            if conf < DETECTION_THRESHOLD:
                continue

            # segmentation + rotated bounding box
            L, H, box_pts, area_pixels = get_fish_size_and_box(frame_rgb, box)
            ratio_hw = L/H
            # get rid of unfair wh ratio and big areas.
            if box_pts is None or ratio_hw<RATIO_WH or L*H>AREA_THRESHOLD:
                continue

            weight_model_input.append(np.array([L* calibration_factor, H* calibration_factor, area_pixels* calibration_factor**2]))
            
            # draw rotated box
            if writer is not None:
                cv2.polylines(frame_bgr, [box_pts], True, (0,0,255), 2)

                x, y = box_pts[0]
                text = f"{L:.1f} x {H:.1f} (conf {conf:.2f})"
                cv2.putText(frame_bgr, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # -------------------------------------------
        # 5. Display & save
        # -------------------------------------------
        # cv2.imshow("Fish Detection", frame_bgr)
        if writer is not None:
            writer.write(frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -------------------------------------------
    # Weight Estimation
    # -------------------------------------------
    if REAL_SIZE_FOUND:
        input_data = np.stack(weight_model_input,axis=0)

        # predict weight:
        with torch.no_grad():
            input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
            # input_data = torch.tensor([6.642561445470936, 1.5305083096582157, 6.966261551986847]).to(device).float().unsqueeze(0)
            fish_weight = weight_model(input_data)
            weight_mean = fish_weight.mean().item()
            

    cap.release()
    if writer:
        writer.release()
    # cv2.destroyAllWindows()

    mean_width = input_data[:,0].mean().item()
    mean_height = input_data[:,1].mean().item()
    print('------------Summary----------------')
    print(f'Average Width: {mean_width}')
    print(f'Average Height: {mean_height}')
    print(f'Average Mass: {weight_mean}')
    
    return mean_width, mean_height, weight_mean, calibration_factor
     



# -------------------------------------------------------------------
# Run example
# -------------------------------------------------------------------
if __name__ == "__main__":

    # --- Configuration ---
    video_path = '/home/dnn/Downloads/OneDrive_1_2-23-2026/Week 4'
    save_file = 'results.csv'

    # Get list of videos (filtering for common video extensions)
    video_list = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi', '.MOV'))]

    # video_list = ['IMG_2570.MOV']

    # --- Processing Loop ---
    for each_video in video_list:
        full_path = os.path.join(video_path, each_video)
        
        try:
            # 1. Run your processing function
            mean_width, mean_height, weight_mean, calibration_factor = process_video(full_path)
            
            # 2. Create a temporary DataFrame for the current row
            df_row = pd.DataFrame([{
                'video_name': os.path.join(video_path.split('/')[-1],each_video),
                'mean_width': mean_width,
                'mean_height': mean_height,
                'weight_mean': weight_mean,
                'calibration_factor':calibration_factor
            }])
            
            # 3. Append to CSV
            # header=not os.path.exists(...) ensures header is only written once
            df_row.to_csv(save_file, 
                        mode='a', 
                        index=False, 
                        header=not os.path.exists(save_file))
            
            print(f"Successfully processed: {each_video}")

        except Exception as e:
            print(f"Error processing {each_video}: {e}")

    print(f"\nAll results saved to {save_file}")

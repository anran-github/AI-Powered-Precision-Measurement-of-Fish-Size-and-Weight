import cv2
import numpy as np
import torch
import copy
from PIL import Image

from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

# ---------------------------------------------------------
# USER PARAMETERS
# ---------------------------------------------------------
DETECTION_THRESHOLD = 0.7
PRINT_DEBUG = False

TILE_SIZE = 640
TILE_OVERLAP = 0.3        # 30% overlap – required for small-object detection

# ---------------------------------------------------------
# Load your models
# ---------------------------------------------------------
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



# ---------------------------------------------------------
# Helper: tile the frame into overlapping patches
# ---------------------------------------------------------
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

    return length, height, box_pts



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


    while True:
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
        

        # -------------------------------------------
        # 1. Tile the large frame
        # -------------------------------------------
        tiles, coords = tile_image(frame_rgb, TILE_SIZE, TILE_OVERLAP)

        all_detects = []

        # -------------------------------------------
        # 2. Detect in each tile
        # -------------------------------------------
        for tile, (tx, ty) in zip(tiles, coords):

            dets = detector.predict(tile)[0]   # detect in tile

            # shift tile detections back to global coordinates
            dets_shifted = shift_boxes(dets, tx, ty)
            all_detects.extend(dets_shifted)

        # -------------------------------------------
        # 3. Merge tile detections → global NMS
        # -------------------------------------------
        detections = merge_all_boxes(all_detects, iou_thresh=0.4)

        # -------------------------------------------
        # 4. Process all remaining detections
        # -------------------------------------------
        for box in detections:

            conf = box.score
            if conf < DETECTION_THRESHOLD:
                continue

            # segmentation + rotated bounding box
            L, H, box_pts = get_fish_size_and_box(frame_rgb, box)
            if box_pts is None:
                continue

            # draw rotated box
            cv2.polylines(frame_bgr, [box_pts], True, (0,0,255), 2)

            x, y = box_pts[0]
            text = f"{L:.1f} x {H:.1f} (conf {conf:.2f})"
            cv2.putText(frame_bgr, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # -------------------------------------------
        # 5. Display & save
        # -------------------------------------------
        cv2.imshow("Fish Detection", frame_bgr)
        if writer is not None:
            writer.write(frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()



# -------------------------------------------------------------------
# Run example
# -------------------------------------------------------------------
if __name__ == "__main__":
    video_path = "/home/dnn/Downloads/OneDrive_2026-02-18/Week 1/IMG_2342.MOV"
    output_path = "fish_detect_output.mp4"
    process_video(video_path, output_path)

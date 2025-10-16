#!/usr/bin/env python3


import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict

# ---------------------------
# Configuration - edit these
# ---------------------------
# Absolute path where outputs and report will be saved
OUTPUT_DIR = r"C:\Users\ghina\Desktop\Personal Code\IIUM Internship\Assignment\partA_q1\Q1_output"

# Relative to this script
DATA_FRONT = os.path.join(os.path.dirname(__file__), "Q1_data", "front")
DATA_REAR = os.path.join(os.path.dirname(__file__), "Q1_data", "rear")

# YOLO model path (file placed next to this script)
YOLO_MODEL = os.path.join(os.path.dirname(__file__), "LP-detection.pt")

# Detection tuning
MIN_PLATE_WIDTH = 50
MIN_PLATE_HEIGHT = 15
PLATE_ASPECT_MIN = 2.0
PLATE_ASPECT_MAX = 6.0
YOLO_CONF = 0.25
YOLO_IOU = 0.5

# Broken-character contour filters
BROKEN_AREA_MIN = 20
BROKEN_AREA_MAX = 500
BROKEN_W_MIN = 5
BROKEN_W_MAX = 50
BROKEN_H_MIN = 10
BROKEN_H_MAX = 60

# ---------------------------
# Utility functions
# ---------------------------

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_yolo_model(path: str) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YOLO model not found at: {path}")
    return YOLO(path)

def safe_imread(path: str):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img

def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale, equalize histogram and return a 3-channel image (for YOLO input).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def run_yolo_detect(model: YOLO, image: np.ndarray, conf=YOLO_CONF, iou=YOLO_IOU) -> List[Tuple[int,int,int,int]]:
    """
    Run YOLO model and return list of bounding boxes (x1,y1,x2,y2) that pass heuristic filters.
    """
    raw_results = model.predict(image, conf=conf, iou=iou, verbose=False)
    boxes = []
    for r in raw_results:
        # r.boxes.xyxy might be a tensor-like => iterate rows
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            ar = w / float(h)
            if PLATE_ASPECT_MIN <= ar <= PLATE_ASPECT_MAX and w >= MIN_PLATE_WIDTH and h >= MIN_PLATE_HEIGHT:
                boxes.append((x1, y1, x2, y2))
    return boxes

def crop_region(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    # clamp to image bounds
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return img[y1:y2, x1:x2]

def fallback_center_crop(img: np.ndarray, width_ratio=0.4, height_ratio=0.15) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    h, w = img.shape[:2]
    cw = int(w * width_ratio)
    ch = int(h * height_ratio)
    cx, cy = w // 2, int(h * 0.78)  # plate usually near bottom half
    x1 = max(0, cx - cw // 2)
    x2 = min(w, cx + cw // 2)
    y1 = max(0, cy - ch // 2)
    y2 = min(h, cy + ch // 2)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def detect_broken_characters(plate_img: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Return count of small contour fragments likely to be broken characters and annotated plate image.
    """
    if plate_img.size == 0:
        return 0, plate_img
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold => more robust to lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    annotated = plate_img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if BROKEN_AREA_MIN < area < BROKEN_AREA_MAX and BROKEN_W_MIN < w < BROKEN_W_MAX and BROKEN_H_MIN < h < BROKEN_H_MAX:
            count += 1
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,0,255), 2)
    return count, annotated

def stitch_side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Resize images to same height and stitch horizontally.
    """
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        return img1 if img2 is None else img2
    h = min(img1.shape[0], img2.shape[0])
    r1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    r2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    return cv2.hconcat([r1, r2])

# ---------------------------
# Main processing
# ---------------------------

def process_pairs(front_dir: str, rear_dir: str, output_dir: str, model: YOLO) -> List[Dict]:
    """
    Process paired images from front_dir and rear_dir (matching by sorted order).
    Returns a list of dicts for the report.
    """
    front_files = sorted([f for f in os.listdir(front_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    rear_files = sorted([f for f in os.listdir(rear_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    pairs = list(zip(front_files, rear_files))
    report_rows = []

    print(f"Processing {len(pairs)} pairs...")

    for front_name, rear_name in pairs:
        front_path = os.path.join(front_dir, front_name)
        rear_path = os.path.join(rear_dir, rear_name)
        try:
            front_img = safe_imread(front_path)
            rear_img = safe_imread(rear_path)
        except Exception as e:
            print(f"Skipping pair ({front_name}, {rear_name}) due to read error: {e}")
            continue

        # Preprocess for detection
        f_proc = preprocess_for_detection(front_img)
        r_proc = preprocess_for_detection(rear_img)

        f_boxes = run_yolo_detect(model, f_proc)
        r_boxes = run_yolo_detect(model, r_proc)

        broken_front_count = 0
        broken_rear_count = 0

        # Front plate handling
        if f_boxes:
            f_plate = crop_region(front_img, f_boxes[0])
            broken_front_count, f_annot = detect_broken_characters(f_plate)
            x1,y1,x2,y2 = f_boxes[0]
            cv2.rectangle(front_img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(front_img, f"Broken:{broken_front_count}", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            f_plate, fallback_bbox = fallback_center_crop(front_img)
            broken_front_count, f_annot = detect_broken_characters(f_plate)
            x1,y1,x2,y2 = fallback_bbox
            cv2.putText(front_img, f"Broken:{broken_front_count} (fb)", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Rear plate handling
        if r_boxes:
            r_plate = crop_region(rear_img, r_boxes[0])
            broken_rear_count, r_annot = detect_broken_characters(r_plate)
            x1,y1,x2,y2 = r_boxes[0]
            cv2.rectangle(rear_img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(rear_img, f"Broken:{broken_rear_count}", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            r_plate, fallback_bbox = fallback_center_crop(rear_img)
            broken_rear_count, r_annot = detect_broken_characters(r_plate)
            x1,y1,x2,y2 = fallback_bbox
            cv2.putText(rear_img, f"Broken:{broken_rear_count} (fb)", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # If any broken characters found, save stitched annotated image
        if broken_front_count > 0 or broken_rear_count > 0:
            stitched = stitch_side_by_side(front_img, rear_img)
            out_name = os.path.join(output_dir, f"broken_{front_name}")
            cv2.imwrite(out_name, stitched)
            print(f"Saved: {out_name} (Front broken={broken_front_count}, Rear broken={broken_rear_count})")
            report_rows.append({
                "pair_front": front_name,
                "pair_rear": rear_name,
                "broken_front_count": broken_front_count,
                "broken_rear_count": broken_rear_count,
                "stitched_image": os.path.basename(out_name)
            })
        else:
            # Optionally you can log non-broken cases too; here we skip saving to reduce clutter
            pass

    return report_rows

def save_report_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        # create empty CSV with header to indicate nothing found
        header = ["pair_front", "pair_rear", "broken_front_count", "broken_rear_count", "stitched_image"]
        with open(out_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"No broken plates found. Empty report created at {out_path}")
        return

    keys = ["pair_front", "pair_rear", "broken_front_count", "broken_rear_count", "stitched_image"]
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Report saved at {out_path}")

# ---------------------------
# Entry point
# ---------------------------

def main():
    ensure_output_dir(OUTPUT_DIR)
    model = load_yolo_model(YOLO_MODEL)

    if not os.path.isdir(DATA_FRONT) or not os.path.isdir(DATA_REAR):
        raise FileNotFoundError(f"Input folders not found. Expected front: {DATA_FRONT}, rear: {DATA_REAR}")

    rows = process_pairs(DATA_FRONT, DATA_REAR, OUTPUT_DIR, model)
    report_csv_path = os.path.join(OUTPUT_DIR, "broken_license_plate_report.csv")
    save_report_csv(rows, report_csv_path)
    print("Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3


import os, csv, cv2, numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict

# ---------------------------
# Config
# ---------------------------
OUTPUT_DIR = r"C:\Users\ghina\Desktop\Personal Code\IIUM Internship\Assignment\partA_q1\Q1_output"
DATA_FRONT = os.path.join(os.path.dirname(__file__), "Q1_data", "front")
DATA_REAR = os.path.join(os.path.dirname(__file__), "Q1_data", "rear")
YOLO_MODEL = os.path.join(os.path.dirname(__file__), "LP-detection.pt")

# YOLO detection tuning
MIN_PLATE_W, MIN_PLATE_H, ASPECT_MIN, ASPECT_MAX = 50, 15, 2.0, 6.0
YOLO_CONF, YOLO_IOU = 0.25, 0.5

# Broken character filters
BROKEN_AREA, BROKEN_W, BROKEN_H = (20,500), (5,50), (10,60)

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_yolo(path: str) -> YOLO:
    if not os.path.exists(path): raise FileNotFoundError(f"YOLO model not found: {path}")
    return YOLO(path)

def safe_read(path: str):
    img = cv2.imread(path)
    if img is None: raise IOError(f"Failed to read image: {path}")
    return img

def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)

def run_yolo(model: YOLO, img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    boxes = []
    for r in model.predict(img, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False):
        for box in r.boxes.xyxy:
            x1,y1,x2,y2 = map(int, box)
            w, h = x2-x1, y2-y1
            ar = w/h if h else 0
            if ASPECT_MIN<=ar<=ASPECT_MAX and w>=MIN_PLATE_W and h>=MIN_PLATE_H:
                boxes.append((x1,y1,x2,y2))
    return boxes

def crop(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    h,w = img.shape[:2]
    x1,y1,x2,y2 = max(0,bbox[0]), max(0,bbox[1]), min(w,bbox[2]), min(h,bbox[3])
    return img[y1:y2, x1:x2]

def fallback_crop(img: np.ndarray, w_ratio=0.4, h_ratio=0.15) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    h,w = img.shape[:2]
    cw,ch = int(w*w_ratio), int(h*h_ratio)
    cx,cy = w//2, int(h*0.78)
    x1,x2 = max(0,cx-cw//2), min(w,cx+cw//2)
    y1,y2 = max(0,cy-ch//2), min(h,cy+ch//2)
    return img[y1:y2, x1:x2], (x1,y1,x2,y2)

def detect_broken(plate: np.ndarray) -> Tuple[int, np.ndarray]:
    if plate.size==0: return 0, plate
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,11,2)
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    count, annotated = 0, plate.copy()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if BROKEN_AREA[0]<area<BROKEN_AREA[1] and BROKEN_W[0]<w<BROKEN_W[1] and BROKEN_H[0]<h<BROKEN_H[1]:
            count+=1
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,0,255),2)
    return count, annotated

def stitch(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    h = min(img1.shape[0], img2.shape[0])
    r1 = cv2.resize(img1,(int(img1.shape[1]*h/img1.shape[0]),h))
    r2 = cv2.resize(img2,(int(img2.shape[1]*h/img2.shape[0]),h))
    return cv2.hconcat([r1,r2])

def analyze_plate(img: np.ndarray, model: YOLO) -> Tuple[int, np.ndarray]:
    boxes = run_yolo(model, preprocess(img))
    if boxes: plate = crop(img, boxes[0]); fb=False; bbox=boxes[0]
    else: plate, bbox = fallback_crop(img); fb=True
    count, _ = detect_broken(plate)
    x1,y1,x2,y2 = bbox
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(img,f"Broken:{count}{' (fb)' if fb else ''}",(x1,max(0,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    return count, plate

# ---------------------------
# Main processing
# ---------------------------
def process_pairs(front_dir:str,rear_dir:str,output_dir:str,model:YOLO) -> List[Dict]:
    front_files = sorted(f for f in os.listdir(front_dir) if f.lower().endswith((".jpg",".jpeg",".png")))
    rear_files  = sorted(f for f in os.listdir(rear_dir)  if f.lower().endswith((".jpg",".jpeg",".png")))
    report_rows=[]
    for f_name,r_name in zip(front_files,rear_files):
        f_img,r_img = safe_read(os.path.join(front_dir,f_name)), safe_read(os.path.join(rear_dir,r_name))
        f_count,_ = analyze_plate(f_img,model)
        r_count,_ = analyze_plate(r_img,model)
        if f_count>0 or r_count>0:
            stitched = stitch(f_img,r_img)
            out_name = os.path.join(output_dir,f"broken_{f_name}")
            cv2.imwrite(out_name,stitched)
            report_rows.append({"pair_front":f_name,"pair_rear":r_name,
                                "broken_front_count":f_count,"broken_rear_count":r_count,
                                "stitched_image":os.path.basename(out_name)})
            print(f"Saved: {out_name} (Front={f_count}, Rear={r_count})")
    return report_rows

def save_csv(rows: List[Dict], path: str):
    keys = ["pair_front","pair_rear","broken_front_count","broken_rear_count","stitched_image"]
    with open(path,"w",newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Report saved at {path}")

# ---------------------------
# Entry
# ---------------------------
def main():
    ensure_dir(OUTPUT_DIR)
    model = load_yolo(YOLO_MODEL)
    rows = process_pairs(DATA_FRONT, DATA_REAR, OUTPUT_DIR, model)
    save_csv(rows, os.path.join(OUTPUT_DIR,"broken_license_plate_report.csv"))
    print("Done.")

if __name__=="__main__":
    main()

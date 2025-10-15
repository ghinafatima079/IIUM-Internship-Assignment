import cv2
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np

# -----------------------------
# Load pre-trained YOLOv8 license plate model
model_path = os.path.join(os.path.dirname(__file__), "LP-detection.pt")
model = YOLO(model_path)  # Use the Hugging Face downloaded model

# -----------------------------
# Output folder
output_folder = r"C:\Users\ghina\Desktop\Personal Code\IIUM Internship\Assignment\partA_q1\Q1_output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Initialize report
report = []

# -----------------------------
# Preprocess image for better detection
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    img_processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img_processed

# -----------------------------
# Detect license plates using YOLO
def detect_license_plate(img):
    results = model.predict(img, conf=0.25, iou=0.5, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = w / h
            if 2 <= aspect_ratio <= 6 and w > 50 and h > 15:
                boxes.append((x1, y1, x2, y2))
    return boxes

# -----------------------------
# Detect broken characters with adaptive threshold and filtering
def detect_broken_characters(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for better results under varying lighting
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    broken_count = 0
    annotated_plate = plate_img.copy()
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        # Only consider contours that resemble broken character fragments
        if 20 < area < 500 and 5 < w < 50 and 10 < h < 60:
            broken_count += 1
            cv2.rectangle(annotated_plate, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return broken_count, annotated_plate

# -----------------------------
# Crop plate image from full image
def crop_plate(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]

# -----------------------------
# Stitch front and rear images side by side
def stitch_images(front_img, rear_img):
    h = min(front_img.shape[0], rear_img.shape[0])
    front_resized = cv2.resize(front_img, (int(front_img.shape[1] * h / front_img.shape[0]), h))
    rear_resized = cv2.resize(rear_img, (int(rear_img.shape[1] * h / rear_img.shape[0]), h))
    stitched = cv2.hconcat([front_resized, rear_resized])
    return stitched

# -----------------------------
# Process paired images
front_folder = os.path.join(os.path.dirname(__file__), 'Q1_data', 'front')
rear_folder = os.path.join(os.path.dirname(__file__), 'Q1_data', 'rear')

front_files = sorted([f for f in os.listdir(front_folder) if f.endswith((".jpg", ".png"))])
rear_files = sorted([f for f in os.listdir(rear_folder) if f.endswith((".jpg", ".png"))])

for f_file, r_file in zip(front_files, rear_files):
    front_img = cv2.imread(os.path.join(front_folder, f_file))
    rear_img = cv2.imread(os.path.join(rear_folder, r_file))

    front_img_proc = preprocess_image(front_img)
    rear_img_proc = preprocess_image(rear_img)

    front_boxes = detect_license_plate(front_img_proc)
    rear_boxes = detect_license_plate(rear_img_proc)

    broken_front = 0
    broken_rear = 0

    # Check front plate
    if front_boxes:
        front_plate = crop_plate(front_img, front_boxes[0])
        broken_front, front_plate_annotated = detect_broken_characters(front_plate)
        x1, y1, x2, y2 = front_boxes[0]
        cv2.rectangle(front_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(front_img, f"Broken: {broken_front}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # Fallback: center region
        h, w, _ = front_img.shape
        x1, y1 = int(w * 0.3), int(h * 0.7)
        x2, y2 = int(w * 0.7), int(h * 0.85)
        front_plate = front_img[y1:y2, x1:x2]
        broken_front, front_plate_annotated = detect_broken_characters(front_plate)
        cv2.putText(front_img, f"Broken: {broken_front} (Fallback)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Check rear plate
    if rear_boxes:
        rear_plate = crop_plate(rear_img, rear_boxes[0])
        broken_rear, rear_plate_annotated = detect_broken_characters(rear_plate)
        x1, y1, x2, y2 = rear_boxes[0]
        cv2.rectangle(rear_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rear_img, f"Broken: {broken_rear}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # Fallback: center region
        h, w, _ = rear_img.shape
        x1, y1 = int(w * 0.3), int(h * 0.7)
        x2, y2 = int(w * 0.7), int(h * 0.85)
        rear_plate = rear_img[y1:y2, x1:x2]
        broken_rear, rear_plate_annotated = detect_broken_characters(rear_plate)
        cv2.putText(rear_img, f"Broken: {broken_rear} (Fallback)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save stitched image if any broken plate detected
    if broken_front > 0 or broken_rear > 0:
        stitched = stitch_images(front_img, rear_img)
        out_name = os.path.join(output_folder, f"broken_{f_file}")
        cv2.imwrite(out_name, stitched)
        report.append({
            "Car Image": f_file,
            "Broken Front": "Yes" if broken_front > 0 else "No",
            "Broken Rear": "Yes" if broken_rear > 0 else "No"
        })
        print(f"Saved stitched broken plate for {f_file}: Front={broken_front}, Rear={broken_rear}")

# Save report CSV
df = pd.DataFrame(report)
df.to_csv(os.path.join(output_folder, "broken_license_plate_report.csv"), index=False)
print("Processing complete. Report saved at:", os.path.join(output_folder, "broken_license_plate_report.csv"))

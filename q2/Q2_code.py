import cv2
import numpy as np
import os
from ultralytics import YOLO
import json

# ========== Utility Functions ==========
def get_dominant_color(roi):
    if roi.size == 0:
        return (0, 0, 0)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    data = roi_rgb.reshape((-1, 3)).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        data,
        1,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    return tuple(map(int, centers[0]))


def rgb_to_name(rgb):
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r > 150 and g < 100 and b < 100:
        return "red"
    elif b > 150 and r < 100 and g < 100:
        return "blue"
    elif g > 150 and r < 100 and b < 100:
        return "green"
    else:
        return f"rgb{rgb}"


def get_lane(x_center, img_width):
    return "Left" if x_center < img_width / 2 else "Right"


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def annotate_image(img, vehicles):
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        label = f"{v['type']} ({v['color']}, {v['lane']})"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return img


# ========== Main Class ==========
class VehicleSceneAnalyzer:
    def __init__(self):
        # use your trained model for car brands
        self.vehicle_model = YOLO("yolo11n.pt")
        self.vehicle_classes = ["car", "motorcycle", "bus", "truck"]

    def analyze(self, img):
        h, w = img.shape[:2]
        results = self.vehicle_model(img)[0]
        vehicles = []

        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = self.vehicle_model.names[cls_id]
            if cls_name not in self.vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            roi = img[y1:y2, x1:x2]

            dom_rgb = get_dominant_color(roi)
            color_name = rgb_to_name(dom_rgb)
            cx, cy = bbox_center((x1, y1, x2, y2))
            lane = get_lane(cx, w)

            vehicles.append(
                {
                    "type": cls_name,
                    "bbox": [x1, y1, x2, y2],
                    "color": color_name,
                    "lane": lane,
                    "make": None,
                    "logo_bbox": None,
                    "license_plate_present": False,
                    "license_plate_bbox": None,
                    "license_plate_color": None,
                }
            )

        incoming = any(v["lane"] == "Left" for v in vehicles)
        outgoing = any(v["lane"] == "Right" for v in vehicles)

        return {
            "incoming_traffic": incoming,
            "outgoing_traffic": outgoing,
            "vehicle_count": len(vehicles),
            "vehicles": vehicles,
        }


# ========== Batch Processing ==========
def analyze_folder(
    folder_path,
    save_json=False,
    json_folder="Q2_output",
    save_annotated=True,
    annotated_folder="Q2_annotated_images",
):
    analyzer = VehicleSceneAnalyzer()
    parent_folder = os.path.dirname(folder_path)  # Get parent folder (e.g., 'q2')

    json_path_full = os.path.join(parent_folder, json_folder)
    annotated_path_full = os.path.join(parent_folder, annotated_folder)

    os.makedirs(json_path_full, exist_ok=True)
    if save_annotated:
        os.makedirs(annotated_path_full, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            result = analyzer.analyze(img)

            print(f"Results for {filename}:")
            print(json.dumps(result, indent=4))
            print("-" * 40)

            if save_json:
                json_path = os.path.join(json_path_full, f"{os.path.splitext(filename)[0]}.json")
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=4)
                print(f"[INFO] JSON saved to: {json_path}")

            if save_annotated:
                annotated_img = annotate_image(img.copy(), result["vehicles"])
                save_path = os.path.join(annotated_path_full, filename)
                cv2.imwrite(save_path, annotated_img)
                print(f"[INFO] Annotated image saved to: {save_path}")


# ========== Run ==========
if __name__ == "__main__":
    folder = "q2/Q2_input_images"  # input image folder
    analyze_folder(folder, save_json=True, save_annotated=True)

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import json

# -------------------------
# Load ImageNet classes
# -------------------------
LABELS_PATH = os.path.join("q7", "Q7_data", "imagenet_classifications.txt")
with open(LABELS_PATH, "r") as f:
    idx_to_labels = [line.strip() for line in f.readlines()]

# -------------------------
# Define dog and cat classes
# -------------------------
dog_classes = [label.lower() for label in idx_to_labels if 
               any(x in label.lower() for x in ["dog", "retriever", "terrier", "spaniel", "sheepdog"])]

cat_classes = [label.lower() for label in idx_to_labels if "cat" in label.lower()]

# -------------------------
# Preprocessing & model
# -------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# -------------------------
# Classify a single image
# -------------------------
def classify_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top1_idx = torch.argmax(probs).item()
    top1_label = idx_to_labels[top1_idx].lower()

    # Determine category
    if top1_label in dog_classes:
        category = "dog"
    elif top1_label in cat_classes:
        category = "cat"
    else:
        category = "other"

    return category, top1_label

# -------------------------
# Test all images in a folder
# -------------------------
def test_images(folder_path):
    results = []

    # Save output in q7/Q7_output
    parent_folder = os.path.abspath(os.path.join(folder_path, "..", ".."))
    output_folder = os.path.join(parent_folder, "Q7_output")
    os.makedirs(output_folder, exist_ok=True)
    report_path = os.path.join(output_folder, "cat_dog_classification_report.json")

    # Iterate images
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)

        # Expected category based on filename
        expected_category = "cat" if "cat" in filename.lower() else "dog"

        predicted_category, predicted_label = classify_image(img_path)
        misclassified = predicted_category != expected_category

        results.append({
            "filename": filename,
            "predicted_category": predicted_category,
            "predicted_label": predicted_label,
            "expected_category": expected_category,
            "misclassified": misclassified
        })

        print(f"{filename}: predicted as {predicted_category} ({predicted_label})")

    total_tested = len(results)
    misclassified_count = sum(r["misclassified"] for r in results)

    report = {
        "total_tested": total_tested,
        "misclassified_count": misclassified_count,
        "results": results
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n✅ Report saved to: {report_path}")
    print(f"Total images tested: {total_tested}, Misclassified: {misclassified_count}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    folder = os.path.join("q7", "Q7_data", "Q7_input_images")
    if not os.path.exists(folder):
        print(f"⚠️ Folder '{folder}' not found. Please add images.")
    else:
        test_images(folder)

# test_Q6_code.py
import random
import csv
from pathlib import Path
import pytest
import sys
import os

# Add Q5 module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from q5 import Q5_code  # Make sure this points to your string_similarity function

# --- Configuration ---
NUM_PLATES = 1000
PLATE_LENGTH = 9
SIMILARITY_THRESHOLD = 70
RESULT_FOLDER = Path(__file__).parent / "Q6_output"
RESULT_FOLDER.mkdir(exist_ok=True)

# --- Helper: generate synthetic Indian license plates ---
def generate_plate():
    state_code = random.choice("MHDLRJGUPKNCH") + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(random.choices("0123456789", k=2))
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    number = "".join(random.choices("0123456789", k=3))
    return f"{state_code}{digits}{letters}{number}"[:PLATE_LENGTH]

# --- Helper: generate similar plate with few character changes ---
def generate_similar_plate(base, num_changes=1):
    plate = list(base)
    for _ in range(num_changes):
        idx = random.randint(0, len(plate)-1)
        plate[idx] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return "".join(plate)

# --- Generate plates with some intentional similarities ---
base_plates = ["MH12AB123", "DL34CD456", "KA56EF789", "TN78GH012", "RJ90IJ345"]
plates = []

# Add each base plate and a few similar variations
for base in base_plates:
    plates.append(base)
    for _ in range(5):  # 5 variations per base plate
        plates.append(generate_similar_plate(base, num_changes=1))

# Fill the rest with fully random plates
while len(plates) < NUM_PLATES:
    plates.append(generate_plate())

# --- Pytest test ---
def test_plate_similarity():
    total_matches = 0
    total_comparisons = 0
    output_file = RESULT_FOLDER / "license_plate_similarity.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Plate1", "Plate2", "Similarity(%)", "Matches", "Mismatches", "MatchAboveThreshold"])

        for i in range(len(plates)):
            for j in range(i + 1, len(plates)):
                plate1, plate2 = plates[i], plates[j]

                # Use Q5 function
                result = Q5_code.string_similarity(plate1, plate2)
                similarity = result["Similarity%"]
                match_pattern = result["MatchPattern"]
                matches = match_pattern.count("✔")
                mismatches = match_pattern.count("✘")

                match_flag = "YES" if similarity >= SIMILARITY_THRESHOLD else "NO"
                writer.writerow([plate1, plate2, f"{similarity:.2f}", matches, mismatches, match_flag])

                if match_flag == "YES":
                    total_matches += 1
                total_comparisons += 1

    avg_matches = total_matches / total_comparisons * 100
    print(f"\n✅ Total comparisons: {total_comparisons}")
    print(f"✅ Total matches above {SIMILARITY_THRESHOLD}%: {total_matches}")
    print(f"✅ Match percentage: {avg_matches:.2f}%")
    print(f"✅ Full CSV saved at: {output_file}")

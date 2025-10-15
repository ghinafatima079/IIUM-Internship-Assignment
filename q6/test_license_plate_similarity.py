"""
Q6. Automated License Plate Similarity Testing

- Generates 1000 synthetic license plates (valid and invalid variations).
- Compares them pairwise with the Q5 string similarity function.
- Summarizes matches above a threshold (e.g., 70% similarity).
- Saves results in CSV for analysis.
"""

import random
import csv
import os
import sys
from pathlib import Path

# Add Q5 module to path
sys.path.append(str(Path(__file__).parent.parent / "q5"))
from q5 import Q5_code


# --- Configuration ---
NUM_PLATES = 1000
PLATE_LENGTH = 9  # Indian plates typically ~9 chars
RESULT_FOLDER = str(Path(__file__).parent / "result")
SIMILARITY_THRESHOLD = 70  # Considered a "match"

# Ensure result folder exists
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Synthetic plate generation ---
def generate_plate():
    """
    Generates a synthetic Indian license plate-like string.
    Format example: "MH12AB1234" -> We'll use random letters & digits
    """
    state_code = random.choice("MHDLRJGUPKNCH") + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 2 letters
    digits = "".join(random.choices("0123456789", k=2))
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    number = "".join(random.choices("0123456789", k=3))
    return f"{state_code}{digits}{letters}{number}"[:PLATE_LENGTH]  # Ensure fixed length

# --- Generate plates ---
plates = [generate_plate() for _ in range(NUM_PLATES)]

# --- Compare plates pairwise and save results ---

# Helper to get a unique filename
def get_unique_filename(folder, base_name="license_plate_similarity", ext=".csv"):
    i = 1
    while True:
        fname = os.path.join(folder, f"{base_name}{'' if i == 1 else '_' + str(i)}{ext}")
        if not os.path.exists(fname):
            return fname
        i += 1

output_file = get_unique_filename(RESULT_FOLDER)

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Plate1", "Plate2", "Similarity(%)", "Matches", "Mismatches", "MatchAboveThreshold"])
    
    # Compare each plate with every other plate (could also sample to reduce time)
    for i in range(len(plates)):
        for j in range(i+1, len(plates)):
            plate1 = plates[i]
            plate2 = plates[j]
            similarity, matches, mismatches, _ = Q5_code.compare_strings(plate1, plate2)
            match_flag = "YES" if similarity >= SIMILARITY_THRESHOLD else "NO"
            writer.writerow([plate1, plate2, f"{similarity:.2f}", matches, mismatches, match_flag])

print(f"✅ Completed! Report saved to: {output_file}")

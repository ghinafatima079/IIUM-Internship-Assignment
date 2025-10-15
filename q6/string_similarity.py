import os

def string_similarity(str1: str, str2: str):
    """
    Compare two strings (6–10 chars).
    Returns similarity percentage and a match report.
    """

    # Alignment: pad shorter string with '-' so lengths match
    max_len = max(len(str1), len(str2))     # Find the longer string’s length
    str1 = str1.ljust(max_len, "-")         # Pad str1 with '-' (on the right) to match length
    str2 = str2.ljust(max_len, "-")         # Pad str2 with '-' (on the right) to match length

    matches = []        # Store ✔ or ✘ for each comparison
    total = max_len     # Total number of positions being compared
    match_count = 0     # Counter for matched characters

    # Compare characters one by one
    for i in range(max_len):
        if str1[i] == str2[i]:
            matches.append("✔")   # character match at position i
            match_count += 1      # increase match count
        else:
            matches.append("✘")   # mismatch at position i

    # Calculate similarity percentage
    similarity = (match_count / total) * 100

    # Prepare report as dictionary
    report = {
        "String1": str1,
        "String2": str2,
        "MatchPattern": "".join(matches),
        "Similarity%": round(similarity, 2)
    }

    return report


# -----------------------------
# Run as standalone program
# -----------------------------
if __name__ == "__main__":
    # Ask user for inputs
    s1 = input("Enter first string (6–10 chars): ").strip()
    s2 = input("Enter second string (6–10 chars): ").strip()

    # Validate input length
    if not (6 <= len(s1) <= 10 and 6 <= len(s2) <= 10):
        print("❌ Both strings must be between 6 and 10 characters.")
    else:
        # Call function and get results
        result = string_similarity(s1, s2)

        # Prepare formatted report text
        report_text = (
            "=== Match Report ===\n"
            f"String1:       {result['String1']}\n"
            f"String2:       {result['String2']}\n"
            f"Match Pattern: {result['MatchPattern']}\n"
            f"Similarity:    {result['Similarity%']}%\n"
        )

        # -----------------------------
        # Create folder relative to this script
        # -----------------------------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result_folder = os.path.join(script_dir, "result", "Q6")
        os.makedirs(result_folder, exist_ok=True)

        # Generate a unique file name (report_1.txt, report_2.txt, etc.)
        base_name = "string_similarity_report"
        ext = ".txt"
        counter = 1
        file_path = os.path.join(result_folder, f"{base_name}{ext}")
        while os.path.exists(file_path):
            file_path = os.path.join(result_folder, f"{base_name}_{counter}{ext}")
            counter += 1

        # Save report to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Display results
        print("\n" + report_text)
        print(f"✅ Report saved to: {file_path}")

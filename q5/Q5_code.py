import os

def string_similarity(str1: str, str2: str):
    """
    Compare two strings (6–10 chars).
    Returns similarity percentage and a match report.
    """
    max_len = max(len(str1), len(str2))
    str1 = str1.ljust(max_len, "-")
    str2 = str2.ljust(max_len, "-")

    matches = []
    match_count = 0

    for i in range(max_len):
        if str1[i] == str2[i]:
            matches.append("✔")
            match_count += 1
        else:
            matches.append("✘")

    similarity = (match_count / max_len) * 100

    return {
        "String1": str1,
        "String2": str2,
        "MatchPattern": "".join(matches),
        "Similarity%": round(similarity, 2)
    }


if __name__ == "__main__":
    s1 = input("Enter first string (6–10 chars): ").strip()
    s2 = input("Enter second string (6–10 chars): ").strip()

    if not (6 <= len(s1) <= 10 and 6 <= len(s2) <= 10):
        print("❌ Both strings must be between 6 and 10 characters.")
    else:
        result = string_similarity(s1, s2)

        report_text = (
            "=== Match Report ===\n"
            f"String1:       {result['String1']}\n"
            f"String2:       {result['String2']}\n"
            f"Match Pattern: {result['MatchPattern']}\n"
            f"Similarity:    {result['Similarity%']}%\n"
        )

        # Folder where report will be saved
        result_path = r"C:\Users\ghina\Desktop\Personal Code\IIUM Internship\Assignment\q5"
        os.makedirs(result_path, exist_ok=True)

        # ✅ Use ONE consistent file
        file_path = os.path.join(result_path, "Q5_output.txt")

        # Append to the same file
        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n" + "-"*30 + "\n")
            f.write(report_text)

        print("\n" + report_text)
        print(f"✅ Appended report to: {file_path}")

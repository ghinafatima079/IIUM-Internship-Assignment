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
        "String1": str1,                      # Padded string 1
        "String2": str2,                      # Padded string 2
        "MatchPattern": "".join(matches),     # Pattern of ✔ and ✘
        "Similarity%": round(similarity, 2)   # Percentage similarity (2 decimals)
    }

    return report   # Return the dictionary report


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

        # Display formatted match report
        print("\n=== Match Report ===")
        print("String1:      ", result["String1"])
        print("String2:      ", result["String2"])
        print("Match Pattern:", result["MatchPattern"])
        print("Similarity:   ", result["Similarity%"], "%")

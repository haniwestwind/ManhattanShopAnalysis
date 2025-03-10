import csv
import re

def parse_text_file(input_filename, output_filename):
    category_list = []

    with open(input_filename, "r", encoding="utf-8") as file:
        for line in file:
            # Match lines with parentheses
            match = re.match(r"^(.+?) \((\w+),", line)
            if match:
                category_name = match.group(1).strip()
                code_name = match.group(2).strip()
                category_list.append((category_name, code_name))

    # Write to CSV
    with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category Name", "Code Name"])  # Header row
        writer.writerows(category_list)

    print(f"Successfully saved {len(category_list)} categories to '{output_filename}'.")

# Example usage
parse_text_file("categories.txt", "categories.csv")

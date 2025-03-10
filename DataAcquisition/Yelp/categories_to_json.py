import csv
import json

def generate_json_from_csv(input_csv, output_json):
    categories = {}

    with open(input_csv, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Interested"].strip() == "1":  # Only add categories with '1' in Interested column
                category_name = row["SubCategory Name"].strip()
                code_name = row["Code Name"].strip()
                categories[category_name] = code_name

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(categories, json_file, indent=4)

    print(f"Successfully saved {len(categories)} interested categories to '{output_json}'.")

# Example usage
generate_json_from_csv("categories.csv", "categories.json")

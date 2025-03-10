import requests
import json
from config import API_KEY, db, offset_collection, business_collection

# Yelp API Setup
SEARCH_URL = "https://api.yelp.com/v3/businesses/search"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
LIMIT = 50  # Yelp API max per request
API_CALL_LIMIT = 1000 # Daily API call limit
MAX_OFFSET = 240  # Prevent offset exceeding safe limit

# File paths
CATEGORIES_FILE = "categories.json"
SEARCHED_FILE = "searched_categories.json"


def load_json(filename):
    """Load JSON data from a file."""
    # Return empty dict if file doesn't exist or is invalid
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}  


def save_json(filename, data):
    """Save JSON data to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_last_offset(category):
    """Retrieve last offset for a given category from MongoDB."""
    entry = offset_collection.find_one({"_id": category})
    return entry["offset"] if entry else 0


def update_offset(category, new_offset):
    """Update the last queried offset for a category in MongoDB."""
    offset_collection.update_one({"_id": category}, {"$set": {"offset": new_offset}}, upsert=True)


def fetch_and_store_businesses(category_name, category_code, location="Manhattan, NY"):
    """Fetch businesses from Yelp and store them in MongoDB."""
    offset = get_last_offset(category_name)
    search_count = LIMIT
    # Ensure we don't exceed the max offset
    if offset + LIMIT >= MAX_OFFSET:
        # count_left = MAX_OFFSET - offset
        # print("Max reached but still have ", count_left)
        # if count_left <= 0:
        #     search_count = count_left
        print(f"Skipping '{category_name}' (offset {offset} exceeds {MAX_OFFSET}).")
        return False  

    params = {
        "categories": category_code,
        "location": location,
        "limit": search_count,
        "offset": offset,
        "sort_by": "review_count"
    }

    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    data = response.json()

    if "businesses" in data and data["businesses"]:
        business_count = len(data["businesses"])  # Get actual number of businesses retrieved

        for business in data["businesses"]:
            print("Retrieved ", business)
            business["_id"] = business["id"]
            business_collection.update_one({"_id": business["_id"]}, {"$set": business}, upsert=True)

        # Update offset in MongoDB with actual count
        new_offset = offset + business_count
        update_offset(category_name, new_offset)

        # Update searched categories file
        searched_categories = load_json(SEARCHED_FILE)
        searched_categories[category_name] = new_offset
        save_json(SEARCHED_FILE, searched_categories)

        print(f"Stored {business_count} businesses for '{category_name}'. New offset: {new_offset}")

        # Stop querying if results are less than LIMIT (indicating no more results)
        if business_count < LIMIT:
            print(f" No more results for '{category_name}'. Marking as fully searched.")
            searched_categories[category_name] = "Completed"
            save_json(SEARCHED_FILE, searched_categories)
            return False

        return True  # Continue querying
    else:
        print(f" No results found for '{category_name}'. Marking as completed.")
        
        # Mark the category as fully searched in searched_categories.json
        searched_categories = load_json(SEARCHED_FILE)
        searched_categories[category_name] = "Completed"
        save_json(SEARCHED_FILE, searched_categories)

        return False  # Stop querying


def automated_query():
    """Automates the Yelp querying process within the API limit."""
    categories = load_json(CATEGORIES_FILE)
    searched_categories = load_json(SEARCHED_FILE)
    api_calls = 0

    for category_name, category_code in categories.items():
        # print(category_code)
        # continue
        if api_calls >= API_CALL_LIMIT:
            print(" API call limit reached. Stopping execution.")
            break

        if category_name in searched_categories and (searched_categories[category_name] == "Completed" or searched_categories[category_name] >= MAX_OFFSET):
            print(f"Skipping '{category_name}', already searched fully.")
            continue

        print(f"\n Querying '{category_name}' ({category_code})...")

        # Query until we hit a stopping condition
        while api_calls < API_CALL_LIMIT:
            continue_search = fetch_and_store_businesses(category_name, category_code)
            api_calls += 1

            # Stop if there are no more results or we exceed the offset limit
            if not continue_search:
                break

    print(f"\nAutomated querying completed! Used {api_calls} calls")


if __name__ == "__main__":
    automated_query()

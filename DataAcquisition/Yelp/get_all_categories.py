import pymongo

def get_all_category_aliases():
    """Connects to MongoDB and retrieves all unique category aliases."""

    client = pymongo.MongoClient("mongodb://localhost:27017/") 
    db = client["yelp_data"]  
    stores_collection = db["businesses"]

    # Aggregate query to get unique category aliases
    pipeline = [
        {"$unwind": "$categories"},  # Unwind the categories array
        {"$group": {"_id": "$categories.alias"}},  # Group by alias
    ]
    unique_aliases = stores_collection.aggregate(pipeline)

    # Extract aliases
    aliases = [doc["_id"] for doc in unique_aliases]
    client.close()
    return aliases

def save_aliases_to_file(aliases, filename="category_aliases.txt"):
    """Saves the list of aliases to a text file."""

    with open(filename, "w") as f:
        for alias in aliases:
            f.write(alias + "\n")

if __name__ == "__main__":
    aliases = get_all_category_aliases()
    save_aliases_to_file(aliases)
    print(f"Category aliases saved to 'category_aliases.txt'")
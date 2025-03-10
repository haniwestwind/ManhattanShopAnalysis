import pandas as pd
import pymongo
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from scipy.spatial import KDTree
import pickle
import os
import time
import numpy as np
import sys


"""
Connects to MongoDB, retrieves store data, finds the closest zip code for each store,
and updates the database with income data from an Excel file.
"""
#  Configuration 
mongodb_uri = "mongodb://localhost:27017/" 
  
database_name = "yelp_data" 
collection_name = "businesses" 

#  MongoDB Connection 
# Setting up client to communicate with MongoDB and retrieve data
client = pymongo.MongoClient(mongodb_uri)
db = client[database_name]
collection = db[collection_name]
    
def main():

    #  Update Store Data Based on ZIP Code 
    for store in collection.find():
        # MARK - Print and see what's in each store data
        print("Store information\n", store)

        # MARK - Get zip code
        store_zip = store['location']['zip_code']  # Get the store's ZIP code as a string
        if store_zip == "" or store_zip is None:
            print(f"Skipping store {store['_id']} due to missing ZIP code.")
        
        latitude = store["coordinates"]["latitude"]
        longitude = store["coordinates"]["longitude"]
        print("Latitude ", latitude)
        print("Longitude ", longitude)

        category = store["categories"][0]["alias"]
        print("Category ", category) 

        if store.get("price") is not None:
            price_level = store["price"]
            print("Price level ", price_level)

        store_name = store["name"]
        print("Store_name ", store_name)

        rating = store["rating"]
        print("Rating ", rating)

        review_count = store["review_count"]
        print("Review count ", review_count)

    client.close()
    print("Store data updated with ZIP-based income information.")


def check_store_data_completeness():
    """
    Queries MongoDB to check how many stores have ZIP codes and latitude/longitude data
    out of the total number of stores.
    """
    #  Configuration 
    mongodb_uri = "mongodb://localhost:27017/"  
    database_name = "yelp_data"  
    collection_name = "businesses"  

    #  MongoDB Connection 
    client = pymongo.MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Initialize counters
    total_stores = collection.count_documents({})
    stores_with_zip = 0
    stores_with_location = 0
    stores_with_both = 0

    #  Iterate Through Each Store 
    for store in collection.find():
        print("Store", store)
        store_zip = store.get('location', {}).get('zip_code', "").strip()
        latitude = store.get("coordinates", {}).get("latitude", None)
        longitude = store.get("coordinates", {}).get("longitude", None)

        # Check if ZIP code exists
        if store_zip:
            stores_with_zip += 1

        # Check if both latitude and longitude exist
        if latitude is not None and longitude is not None:
            stores_with_location += 1

        # Check if store has both ZIP and location data
        if store_zip and latitude is not None and longitude is not None:
            stores_with_both += 1

    # Print out results
    print("\nData Completeness Report:")
    print(f"Total stores: {total_stores}")
    print(f"Stores with ZIP codes: {stores_with_zip} ({(stores_with_zip / total_stores) * 100:.2f}%)")
    print(f"Stores with Latitude/Longitude: {stores_with_location} ({(stores_with_location / total_stores) * 100:.2f}%)")
    print(f"Stores with both ZIP & Location: {stores_with_both} ({(stores_with_both / total_stores) * 100:.2f}%)")

    client.close()
    print("Store data check completed.")


if __name__ == "__main__":
    # Run only the completeness check, not the full main function
    check_store_data_completeness()

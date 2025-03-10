import pandas as pd
import pymongo
from geopy.distance import geodesic
from scipy.spatial import KDTree
import pickle
import os
import time
import numpy as np
import math

def get_data_from_excel(excel_file):
    """
    Reads data from an Excel file and returns it as a DataFrame.

    Args:
        excel_file (str): Path to the Excel file.

    Returns:
        pandas.DataFrame: DataFrame containing the complaint data.
    """
    try:
        df = pd.read_csv(excel_file)
        return df
    except FileNotFoundError:
        print(f"Error: Excel file '{excel_file}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the Excel file: {e}")
        return None

def get_data_from_mongodb(mongodb_uri, database_name, collection_name):
    """
    Retrieves data from a MongoDB collection and returns it as a DataFrame.

    Args:
        mongodb_uri (str): MongoDB connection string.
        database_name (str): Name of the database.
        collection_name (str): Name of the collection.

    Returns:
        pandas.DataFrame: DataFrame containing the store data.
    """
    try:
        client = pymongo.MongoClient(mongodb_uri)
        db = client[database_name]
        collection = db[collection_name]
        data = list(collection.find())
        df = pd.DataFrame(data)
        client.close()
        return df
    except Exception as e:
        print(f"An unexpected error occurred while reading the MongoDB collection: {e}")
        return None

def build_kd_tree(complaints_df, kd_tree_file):
    """
    Builds or loads a KD-Tree from complaint coordinates.

    Args:
        complaints_df (pandas.DataFrame): DataFrame containing complaint data with latitude and longitude.
        kd_tree_file (str): Path to the KD-Tree file.

    Returns:
        scipy.spatial.KDTree: KD-Tree built from complaint coordinates.
    """
    # Remove rows with NaN or inf values in Latitude and Longitude
    complaints_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    complaints_df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    complaint_coords = list(zip(complaints_df['Latitude'].astype(float), complaints_df['Longitude'].astype(float)))

    complaint_coords = list(zip(complaints_df['Latitude'].astype(float), complaints_df['Longitude'].astype(float)))

    if not os.path.exists(kd_tree_file):
        start_time = time.time()
        kd_tree = KDTree(complaint_coords)
        end_time = time.time()
        print(f"Time taken to build KD-Tree: {end_time - start_time} seconds")
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(kd_tree, f)
    else:
        with open(kd_tree_file, 'rb') as f:
            kd_tree = pickle.load(f)
    return kd_tree

def count_complaints_in_radius(store_coords, complaints_df, kd_tree, radius_miles=0.5):
    """
    Counts the number of complaints within a given radius of store coordinates.

    Args:
        store_coords (tuple): Latitude and longitude coordinates of the store.
        complaints_df (pandas.DataFrame): DataFrame containing complaint data.
        kd_tree (scipy.spatial.KDTree): KD-Tree built from complaint coordinates.
        radius_miles (float): Radius in miles.

    Returns:
        int: Number of complaints within the radius.
    """
    if np.isnan(store_coords).any() or np.isinf(store_coords).any():
        return 0

    radius_degrees = radius_miles / 69.0  # Approximate conversion from miles to degrees

    indices = kd_tree.query_ball_point(store_coords, r=radius_degrees)
    return len(indices)

def update_mongodb_with_complaints(mongodb_uri, database_name, collection_name, complaints_df, kd_tree, radius_miles=0.5):
    """
    Updates the MongoDB collection with the number of complaints within the radius.

    Args:
        mongodb_uri (str): MongoDB connection string.
        database_name (str): Name of the database.
        collection_name (str): Name of the collection.
        complaints_df (pandas.DataFrame): DataFrame containing complaint data.
        kd_tree (scipy.spatial.KDTree): KD-Tree built from complaint coordinates.
        radius_miles (float): Radius in miles.
    """
    client = pymongo.MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]

    for store in collection.find():
        print(f"Processing store: {store.get('name', 'Unnamed')}")
        store_coords = (store['coordinates']['latitude'], store['coordinates']['longitude'])

        if not isinstance(store_coords[0], (int, float)) or not isinstance(store_coords[1], (int, float)) or math.isnan(store_coords[0]) or math.isnan(store_coords[1]) or math.isinf(store_coords[0]) or math.isinf(store_coords[1]):
            print("Skipping store due to invalid coordinates.")
            collection.update_one(
                {"_id": store["_id"]},
                {"$set": {
                    "complaints_within_radius": 0
                }}
            )
            continue

        complaint_count = count_complaints_in_radius(store_coords, complaints_df, kd_tree, radius_miles)

        collection.update_one(
            {"_id": store["_id"]},
            {"$set": {
                "complaints_within_radius": complaint_count
            }}
        )

    client.close()
    print("MongoDB collection updated with complaint counts.")

def main():
    """
    Main function to execute the complaint counting and MongoDB update process.
    """
    mongodb_uri = "mongodb://localhost:27017/"  
    database_name = "manhattan_yelp_data"  
    collection_name = "manhattan_businesses"  
    kd_tree_file = 'complaints_kd_tree.pkl'

    excel_file = 'NYPD_Complaint_Data_Current__Year_To_Date__20250217.csv'  

    #  Load Complaint Data 
    complaints_df = get_data_from_excel(excel_file)
    if complaints_df is None:
        return

    # Build or load KD-Tree
    kd_tree = build_kd_tree(complaints_df, kd_tree_file)

    # Update MongoDB with complaint counts
    update_mongodb_with_complaints(mongodb_uri, database_name, collection_name, complaints_df, kd_tree)

if __name__ == "__main__":
    main()
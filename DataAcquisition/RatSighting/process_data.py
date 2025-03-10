import pandas as pd
import pymongo
from geopy.geocoders import Nominatim
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
        pandas.DataFrame: DataFrame containing the income data.
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

def find_closest_zip_code(store_coords, zip_codes_df, kd_tree):
    """
    Finds the closest zip code to the given store coordinates using a KD-Tree.

    Args:
        store_coords (tuple): Latitude and longitude coordinates of the store.
        zip_codes_df (pandas.DataFrame): DataFrame containing zip code data with latitude and longitude.
        kd_tree (scipy.spatial.KDTree): KD-Tree built from zip code coordinates.

    Returns:
        dict: A dictionary containing the closest zip code and its distance.
    """

    # if np.isnan(store_coords).any() or np.isinf(store_coords).any():
    #     print("Error: store_location contains NaN or inf values.")
    #     # Handle the error (e.g., skip the query, use a default value)
    # else:
    #     distances, indices = kd_tree.query(store_coords, k=10)
    latitude = store_coords[0]  # Access latitude using index 0
    longitude = store_coords[1]  # Access longitude using index 1


    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)) or math.isnan(latitude) or math.isnan(longitude) or math.isinf(latitude) or math.isinf(longitude):
        print("Error: store_location contains NaN or inf values.")
        
        return None # pd.DataFrame(columns=["Park_Name", "Borough", "Zipcode", "Distance"])
        # Handle the error (e.g., skip the query, use a default value)
    
    distance, index = kd_tree.query(store_coords, k=1)  # Find the nearest neighbor
    print("Index ", index)
    if index >= len(zip_codes_df):
        print("Error: Index out of range.")
        return None
    # closest_zip = zip_codes_df.iloc[index]['ZIP code [1]']
    closest_zip_coords = (zip_codes_df.iloc[index]['Latitude'], zip_codes_df.iloc[index]['Longitude'])
    print("Closest Zip Coords ", closest_zip_coords)
    distance_miles = geodesic(store_coords, closest_zip_coords).miles
    return { 'distance_miles': distance_miles, 'index': index}

def main():
    """
    Connects to MongoDB, retrieves store data, finds the closest zip code for each store,
    and updates the database with income data from an Excel file.
    """
    #  Configuration 
    excel_file = 'rat_sightings_by_zipcode.csv'  
      
    mongodb_uri = "mongodb://localhost:27017/"  
    database_name = "manhattan_yelp_data"  
    collection_name = "manhattan_businesses"  

    #  Load Income Data 
    data_df = get_data_from_excel(excel_file)
    if data_df is None:
        return

    #  MongoDB Connection 
    client = pymongo.MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]

    #  Prepare Zip Code Data 
    geolocator = Nominatim(user_agent="zip_code_locator")
    # zip_codes = data_df['Zip Code'].astype(str).tolist()
    zip_code_coords = []
    # print(data_df['ZIP code [1]'])
    zip_code_coords = list(zip(data_df['Latitude'], data_df['Longitude']))
    # Some rows have nan values for latitude and longitude
    print(zip_code_coords)
    print(len(zip_code_coords))

    #  Save KD-Tree 
    kd_tree_file = 'rat_sighting_data_kd_tree.pkl'
    if not os.path.exists(kd_tree_file):
        #  KD-Tree Construction 
        # KD Tree file was not found. Reconstruct it. 
        start_time = time.time()
        kd_tree = KDTree(zip_code_coords)
        end_time = time.time()
        print(f"Time taken to build KD-Tree: {end_time - start_time} seconds")
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(kd_tree, f)
    else:
        with open(kd_tree_file, 'rb') as f:
            kd_tree = pickle.load(f)
    

    #  Update Store Data 
    for store in collection.find():
        print(f"Processing store: {store.get('name', 'Unnamed')}")
        store_coords = (store['coordinates']['latitude'], store['coordinates']['longitude'])
        closest_zip_data = find_closest_zip_code(store_coords, data_df, kd_tree)
        if closest_zip_data is None:
            print("Skipping store due to invalid zip code index.")
            continue
        print(closest_zip_data['index'])

        _data = data_df.iloc[int(closest_zip_data['index'])].to_dict()

        print(_data)
        # Update the store document with the closest zip code and income data
        collection.update_one(
            {"_id": store["_id"]},
            {"$set": {
                "closest_rat_sighting_count": _data["Sighting Count"],
                "closest_rat_sighting_distance": closest_zip_data["distance_miles"],
                "closest_rat_sighting_coordinates": (_data["Latitude"], _data["Longitude"])
            }}
        )

    client.close()
    print("Store data updated with closest zip code and income information.")

if __name__ == "__main__":
    main()
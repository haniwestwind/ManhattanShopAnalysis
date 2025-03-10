import pandas as pd
import numpy as np
import pymongo
from geopy.distance import geodesic
from scipy.spatial import KDTree

#  CONFIGURATION 

mongodb_uri = "mongodb://localhost:27017/"
database_name = "manhattan_yelp_data"
collection_name = "manhattan_businesses"

#  LOAD DATA 
subway_df = pd.read_csv("MTA_Subway_Stations.csv")  # Adjust path if needed

# MongoDB Connection
client = pymongo.MongoClient(mongodb_uri)
db = client[database_name]
collection = db[collection_name]

# Extract subway coordinates for KDTree
subway_coords = subway_df[['GTFS Latitude', 'GTFS Longitude']].dropna().values
subway_tree = KDTree(subway_coords)  # Fast spatial search

# Function to find the top k nearest subway stations (only if within 1 mile)
def find_top_k_nearest(lat, lon, k, max_distance=5.0):
    if np.isnan(lat) or np.isnan(lon):
        return []

    distances, indices = subway_tree.query((lat, lon), k=k)  # Get 3 closest stations

    nearest_stations = []
    for i, idx in enumerate(indices):
        station = subway_df.iloc[idx]
        # Wrong unit
        # distance_miles = distances[i] * 0.621371  # Convert km to miles
        store_coords = (lat, lon)
        closest_zip_coords = (station['GTFS Latitude'], station['GTFS Longitude'])
        distance_miles = geodesic(store_coords, closest_zip_coords).miles
        # distance_miles = geodesic(store_coords, closest_zip_coords).miles

        if distance_miles <= max_distance:  # Only keep stations within max_distance
            nearest_stations.append({
                "station_name": station['Stop Name'],
                "subway_distance_miles": round(distance_miles, 3),
                "subway_line": station['Line']
            })

    return nearest_stations  # Could be 0, 1, 2, or 3 stations

# Function to find all subway stations within a radius
# def find_stations_within_radius(lat, lon, radius=0.5):
#     if np.isnan(lat) or np.isnan(lon):
#         return []

#     # Use query_ball_point instead of query to directly get all points within radius
#     indices = subway_tree.query_ball_point((lat, lon), radius / 0.621371)  # Convert miles to km
#     return len(indices)

def find_stations_within_radius(lat, lon, radius=0.5):
    if np.isnan(lat) or np.isnan(lon):
        return []

    # Convert miles to degrees (approximate conversion)
    radius_degrees = radius / 69.0  # 1 degree of latitude is approximately 69 miles

    indices = subway_tree.query_ball_point((lat, lon), radius_degrees)
    return len(indices)
  
#  PROCESS & PRINT FIRST 10 STORES BEFORE UPDATING MONGODB 
preview_count = 0  # Track how many stores have been printed
processed_count = 0
for store in collection.find():
    store_id = store['_id']
    latitude = store.get("coordinates", {}).get("latitude", None)
    longitude = store.get("coordinates", {}).get("longitude", None)

    if latitude is None or longitude is None:
        print(f"Skipping store {store_id} due to missing location data.")
        continue  # Skip stores without coordinates

    # Get top 3 nearest subway stations (within 1 mile)
    nearest_subways = find_top_k_nearest(latitude, longitude, k = 10, max_distance=3.0)

    # Get all subway stations within 0.5 miles and 1 mile
    subway_density_0_3mi = find_stations_within_radius(latitude, longitude, radius=0.3)
    subway_density_0_5mi = find_stations_within_radius(latitude, longitude, radius=0.5)
    subway_density_1mi = find_stations_within_radius(latitude, longitude, radius=1.0)

    # Determine if the store has subway access
    has_subway_access = subway_density_0_3mi > 0
    # print("Nearest subways ", nearest_subways)

    # Update the database
    collection.update_one(
        {"_id": store["_id"]},
        {"$set": {"closest_subways": nearest_subways,
                  "subway_density_0_3mi": subway_density_0_3mi,
                  "subway_count_0_5mi": subway_density_0_5mi,
                "subway_count_1mi": subway_density_1mi,
                "has_subway_access": has_subway_access,
                  }}  # Add the closest_precincts to the restaurant document
    )
    print("Processed store ", store.get("name", "Unknown"), processed_count)
    processed_count += 1
    # # Print results for 10 stores before updating MongoDB
    # if preview_count < 10:
    #     print(f"\nStore {store_id} - {store['name']}")
    #     print(f"--Nearest Subways: {nearest_subways}")
    #     print(f"--Subway Density (0.5mi): {subway_density_0_5mi}")
    #     print(f"--Subway Density (1mi): {subway_density_1mi}")
    #     print(f"--Has Subway Access? {has_subway_access}")
    #     preview_count += 1

    # # Stop after printing 10 stores
    # if preview_count == 10:
    #     print("\nPreview Complete: Stopping before MongoDB update.")
    #     break  # Exit loop after showing 10 stores

# Close MongoDB connection
client.close()
print("MongoDB connection closed.")

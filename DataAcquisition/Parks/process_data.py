from shapely.wkt import loads
import pandas as pd
from geopy.distance import geodesic
import pymongo
from scipy.spatial import KDTree
import numpy as np
import time
import math

#  Calculate centroid of park 

# Load parks dataset
parks_file = "Parks_Properties_20250217.csv"
df_parks = pd.read_csv(parks_file)

# Extract longitude/latitude from MULTIPOLYGON (take centroid)
def extract_centroid(polygon_wkt):
    polygon = loads(polygon_wkt)  # Convert WKT to polygon
    centroid = polygon.centroid  # Compute centroid
    return centroid.x, centroid.y  # Return longitude, latitude

df_parks["Longitude"], df_parks["Latitude"] = zip(*df_parks["multipolygon"].apply(extract_centroid))

# Keep only relevant columns
df_parks = df_parks[["NAME311", "BOROUGH", "ZIPCODE", "Longitude", "Latitude", "ACRES"]]
df_parks.columns = ["Park_Name", "Borough", "Zipcode", "Longitude", "Latitude", "Acres"]

print("Check df parks ", df_parks.head())  # Debug output

#  Find top 10 closest parks for each store using KDTree 

# Load Store Data from MongoDB 
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")  
db = mongo_client["manhattan_yelp_data"]  
stores_collection = db["manhattan_businesses"]  

# Query store data
stores_cursor = stores_collection.find({}, {"_id": 1, "name": 1, "coordinates": 1})
stores_cursor = stores_collection.find()
df_stores = pd.DataFrame(list(stores_cursor))

# Debug: Check column names
print("Store Data Columns:", df_stores.columns.tolist())

# Ensure Longitude and Latitude exist
if "coordinates" not in df_stores.columns:
    print("Error: longitude and latitude columns not found in store data.")
    exit()

# Convert coordinates to float
# df_stores["longitude"] = df_stores["longitude"].astype(float)
# df_stores["latitude"] = df_stores["latitude"].astype(float)

#  KDTree Setup 
park_coords = df_parks[["Latitude", "Longitude"]].values
kd_tree = KDTree(park_coords)

# Function to find top 10 parks using KDTree
def find_top_parks_kdtree(store, kd_tree, parks, park_coords):
    # store_location = (store["coordinates"]["latitude"], store["coordinates"]["longitude"])
    store_location = np.array([store["coordinates"]["latitude"], store["coordinates"]["longitude"]]) #convert to numpy array
    latitude = store["coordinates"]["latitude"]
    longitude = store["coordinates"]["longitude"]

    # Query KDTree for 10 nearest neighbors
    # distances, indices = kd_tree.query(store_location, k=10)

    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)) or math.isnan(latitude) or math.isnan(longitude) or math.isinf(latitude) or math.isinf(longitude):
        print("Error: store_location contains NaN or inf values.")
        print(f"Store {store['name']} does not have the coordinate.")
        
        return pd.DataFrame(columns=["Park_Name", "Borough", "Zipcode", "Distance"])
        # Handle the error (e.g., skip the query, use a default value)
    else:
        distances, indices = kd_tree.query(store_location, k=10)
    
    nearest_parks = []
    for i in range(len(indices)):
        park_index = indices[i]
        park_name = parks.iloc[park_index]["Park_Name"]
        borough = parks.iloc[park_index]["Borough"]
        zipcode = parks.iloc[park_index]["Zipcode"]
        
        # Calculate precise distance using geodesic
        park_location = (park_coords[park_index][0], park_coords[park_index][1])
        distance = geodesic(store_location, park_location).miles
        
        nearest_parks.append((park_name, borough, zipcode, distance))
        
    return pd.DataFrame(nearest_parks, columns=["Park_Name", "Borough", "Zipcode", "Distance"])

# Apply function to each store
store_parks_mapping = {}

for _, store in df_stores.iterrows():
    top_parks = find_top_parks_kdtree(store, kd_tree, df_parks, park_coords)
    store_parks_mapping[store["name"]] = top_parks

    # Debug: Print the top parks for this store
    print(f"\nTop 10 parks near {store['name']}:")
    print(top_parks)
    top_parks_list = top_parks.to_dict(orient='records')
    print(top_parks_list)

    # Update the database
    stores_collection.update_one(
        {"_id": store["_id"]},
        {"$set": {"closest_parks": top_parks_list}}  # Add the closest_precincts to the restaurant document
    )
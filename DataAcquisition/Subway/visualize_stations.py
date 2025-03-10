import pandas as pd
import numpy as np
import pymongo
from geopy.distance import geodesic
from scipy.spatial import KDTree
import folium  # Import folium for map visualization

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
        distance_miles = distances[i] * 0.621371  # Convert km to miles

        if distance_miles <= max_distance:  # Only keep stations within max_distance
            nearest_stations.append({
                "station_name": station['Stop Name'],
                "subway_distance_miles": round(distance_miles, 3),
                "subway_line": station['Line']
            })

    return nearest_stations  # Could be 0, 1, 2, or 3 stations

# Function to find all subway stations within a radius
def find_stations_within_radius(lat, lon, radius=0.5):
    if np.isnan(lat) or np.isnan(lon):
        return []

    # Use query_ball_point instead of query to directly get all points within radius
    indices = subway_tree.query_ball_point((lat, lon), radius / 0.621371)  # Convert miles to km
    return len(indices)


#  VISUALIZATION 
map_nyc = folium.Map(location=[40.7128, -74.0060], zoom_start=12)  # Center on NYC


#  PROCESS & PRINT FIRST 10 STORES BEFORE UPDATING MONGODB 
# preview_count = 0  # Track how many stores have been printed
processed_count = 0
for store in collection.find():
    store_id = store['_id']
    latitude = store.get("coordinates", {}).get("latitude", None)
    longitude = store.get("coordinates", {}).get("longitude", None)

    if latitude is None or longitude is None:
        print(f"Skipping store {store_id} due to missing location data.")
        continue  # Skip stores without coordinates
    folium.Marker(
            location=[latitude, longitude],
            popup=store['name'] ,
            icon=folium.Icon(color='red', icon='store', prefix='fa')  # Subway icon
        ).add_to(map_nyc)
    processed_count+=1
    if processed_count==100:
        break


# Add subway stations to the map
for index, row in subway_df.iterrows():
    folium.Marker(
        location=[row['GTFS Latitude'], row['GTFS Longitude']],
        popup=row['Stop Name'] + " (" + row['Line'] + ")",
        icon=folium.Icon(color='green', icon='train', prefix='fa')  # Subway icon
    ).add_to(map_nyc)

# Save the map
map_nyc.save("nyc_subway_map.html")

# Close MongoDB connection
client.close()
print("MongoDB connection closed.")
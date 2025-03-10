import pandas as pd
import numpy as np
import pymongo
import folium
from scipy.spatial import KDTree
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler

""" Averaging the monthly traffic count per street  """
# Load the dataset
file_path = "Bi-Annual_Pedestrian_Counts_20250131.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Check the path and try again.")
    exit()

# Extract location-based and street information
df["Longitude"] = df["the_geom"].apply(lambda x: float(x.split("(")[-1].split()[0]))
df["Latitude"] = df["the_geom"].apply(lambda x: float(x.split(" ")[-1].strip(")")))

# Ensure "Borough" exists
if "Borough" not in df.columns:
    print("Warning: 'Borough' column not found in dataset. Check dataset headers!")
else:
    df["Borough"] = df["Borough"].astype(str).str.strip()  # Remove spaces

# Extract traffic count columns
traffic_cols = [col for col in df.columns if any(month in col for month in ["May", "Sept", "Oct", "June"])]
df_melted = df.melt(id_vars=["Borough", "Street_Nam", "From_Stree", "To_Street", "Longitude", "Latitude"], 
                     value_vars=traffic_cols, 
                     var_name="Month_Year", 
                     value_name="Traffic_Count")

# Extract month and year
df_melted["Month"] = df_melted["Month_Year"].str.extract(r'([A-Za-z]+)')
df_melted["Year"] = df_melted["Month_Year"].str.extract(r'(\d{2,4})')

# Convert traffic count to numeric
df_melted["Traffic_Count"] = pd.to_numeric(df_melted["Traffic_Count"], errors='coerce')

# Aggregate total traffic per street (averaging across all months)
df_street_avg = df_melted.groupby(["Borough", "Street_Nam", "From_Stree", "To_Street", "Longitude", "Latitude"]).agg(
    {"Traffic_Count": "mean"}).reset_index()

# Filter for Manhattan borough only
if "Borough" in df_street_avg.columns:
    df_manhattan = df_street_avg[df_street_avg["Borough"] == "Manhattan"]
else:
    print("Error: Borough column missing after grouping!")
    exit()

""" Visualizing the average traffic count per street  """

#  MongoDB Setup 
client = pymongo.MongoClient("mongodb://localhost:27017/")  
db = client["yelp_data"]  
stores_collection = db["businesses"]

#  KDTree Setup for Streets 
street_coords = df_manhattan[["Latitude", "Longitude"]].values
street_kd_tree = KDTree(street_coords)

#  Visualization 
# nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=13)  # Center on Manhattan


# Normalize Traffic Count
scaler = MinMaxScaler()
df_manhattan["Normalized_Traffic"] = scaler.fit_transform(df_manhattan[["Traffic_Count"]])

#  Visualization 
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=13)  # Center on Manhattan

# Add circles for each street
for index, row in df_manhattan.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=row["Normalized_Traffic"] * 45,  # Adjust multiplier for circle size
        color='blue',
        fill=True,
        fill_color='skyblue',
        fill_opacity=0.6,
        popup=f"Street: {row['Street_Nam']}<br>Avg Traffic: {row['Traffic_Count']:.2f}",
    ).add_to(nyc_map)

#  Save Map 
nyc_map.save("manhattan_traffic_map.html")
print("Map saved as 'manhattan_traffic_map.html'")
#  Load and Process Stores 
# stores_cursor = stores_collection.find({}, {"name": 1, "coordinates": 1, "_id": 1})
# df_stores = pd.DataFrame(list(stores_cursor))

# visualize_cnt = 0
# for _, store in df_stores.iterrows():
#     print("Processing store: ", store.get('name', 'Unnamed'))
#     try:
#         store_location = (store["coordinates"]["latitude"], store["coordinates"]["longitude"])

#         # Query KDTree for nearest streets
#         distances, indices = street_kd_tree.query(store_location, k=5)

#         # Add store marker
#         folium.Marker(
#             location=[store["coordinates"]["latitude"], store["coordinates"]["longitude"]],
#             popup=f"<b>{store['name']}</b>",
#             icon=folium.Icon(color='red', icon='home'),
#         ).add_to(nyc_map)

#         # Add street markers and lines to the map
#         for i, index in enumerate(indices):
#             street = df_manhattan.iloc[index]
#             street_location = (street["Latitude"], street["Longitude"])
#             distance = geodesic(store_location, street_location).miles

#             folium.Marker(
#                 location=street_location,
#                 popup=f"<b>{street['Street_Nam']}</b><br>Distance: {distance:.2f} miles<br>Avg Traffic: {street['Traffic_Count']:.2f}",
#                 icon=folium.Icon(color='blue', icon='road'),
#             ).add_to(nyc_map)

#             # Add line from store to street
#             folium.PolyLine(
#                 locations=[store_location, street_location],
#                 color='green',
#                 weight=2.5,
#                 opacity=0.5,
#             ).add_to(nyc_map)

#     except (TypeError, KeyError) as e:
#         print(f"Error processing store {store.get('name', 'unknown')}: {e}")
#     visualize_cnt += 1
#     if visualize_cnt > 10:
#         break

#  Save Map 
# nyc_map.save("store_streets_map.html")
# print("Map saved as 'store_streets_map.html'")

client.close()
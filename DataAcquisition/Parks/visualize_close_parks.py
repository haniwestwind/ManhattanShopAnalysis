import pymongo
import geopandas as gpd
import folium
from scipy.spatial import KDTree
from geopy.distance import geodesic
import pandas as pd
from shapely.wkt import loads

#  MongoDB Setup 
client = pymongo.MongoClient("mongodb://localhost:27017/")  
db = client["yelp_data"]  
stores_collection = db["businesses"]

#  Load Park Data 
parks_file = "Parks_Properties_20250217.csv"
df_parks = pd.read_csv(parks_file)

def extract_centroid(polygon_wkt):
    polygon = loads(polygon_wkt)
    centroid = polygon.centroid
    return centroid.x, centroid.y

df_parks["Longitude"], df_parks["Latitude"] = zip(*df_parks["multipolygon"].apply(extract_centroid))
df_parks = df_parks[["NAME311", "BOROUGH", "ZIPCODE", "Longitude", "Latitude", "ACRES"]]
df_parks.columns = ["Park_Name", "Borough", "Zipcode", "Longitude", "Latitude", "Acres"]

#  KDTree Setup for Parks 
park_coords = df_parks[["Latitude", "Longitude"]].values
kd_tree = KDTree(park_coords)

#  Visualization 
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)  # Center on NYC

#  Load and Process Stores 
stores_cursor = stores_collection.find({}, {"name": 1, "coordinates": 1, "_id": 1})
df_stores = pd.DataFrame(list(stores_cursor))

visualize_cnt = 0
for _, store in df_stores.iterrows():
    print("Processing store: ", store.get('name', 'Unnamed'))
    try:
        store_location = (store["coordinates"]["latitude"], store["coordinates"]["longitude"])

        # Query KDTree for nearest parks
        distances, indices = kd_tree.query(store_location, k=10)

        # Add store marker
        folium.Marker(
            location=[store["coordinates"]["latitude"], store["coordinates"]["longitude"]],
            popup=f"<b>{store['name']}</b>",
            icon=folium.Icon(color='red', icon='home'),
        ).add_to(nyc_map)

        # Add park markers and lines to the map
        for i, index in enumerate(indices):
            park = df_parks.iloc[index]
            park_location = (park["Latitude"], park["Longitude"])
            distance = geodesic(store_location, park_location).miles

            folium.Marker(
                location=park_location,
                popup=f"<b>{park['Park_Name']}</b><br>Distance: {distance:.2f} miles",
                icon=folium.Icon(color='green', icon='tree'),
            ).add_to(nyc_map)

            # Add line from store to park
            folium.PolyLine(
                locations=[store_location, park_location],
                color='blue',
                weight=2.5,
                opacity=0.5,
            ).add_to(nyc_map)

    except (TypeError, KeyError) as e:
        print(f"Error processing store {store.get('name', 'unknown')}: {e}")
    visualize_cnt += 1
    if visualize_cnt > 10:
        break

#  Save Map 
nyc_map.save("store_parks_map.html")
print("Map saved as 'store_parks_map.html'")

client.close()
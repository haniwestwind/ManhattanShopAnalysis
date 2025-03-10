import pymongo
import geopandas as gpd
from geopy.distance import geodesic
import folium
from scipy.spatial import KDTree

from geo_util import find_closest_precincts_using_kdtree, find_closest_precincts

USE_KDTREE = True

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")  
db = client["yelp_data"]  
# Get a list of database names
database_names = client.list_database_names()
print(database_names)

# List all collections (tables) in the database
collection_names = db.list_collection_names()
print(collection_names)
collection = db["businesses"]  



# Load police precinct GeoJSON
precincts = gpd.read_file('./NYC_Police_Precincts_-8413637686511439451.geojson')
precincts['centroid'] = precincts.geometry.centroid

#  KD-Tree Construction 
precinct_coords = [(precinct['centroid'].y, precinct['centroid'].x) for _, precinct in precincts.iterrows()]
kd_tree = KDTree(precinct_coords)  # Build the KD-tree *once*


#  Visualization Part 
# Create a base map centered on NYC 
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Add precinct centroids to the map 
for _, precinct in precincts.iterrows():
    folium.Marker(
        location=[precinct['centroid'].y, precinct['centroid'].x],
        popup=f"Precinct: {precinct['Precinct']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(nyc_map)

# Add restaurant markers with closest precinct info
for restaurant in collection.find():
    try:  # Handle cases where coordinates might be missing
        # for restaurant in collection.find():
        #     # print(restaurant)
        print("Processing restaurant: ", restaurant.get('name', 'Unnamed'))
        if USE_KDTREE:
            closest_precincts = find_closest_precincts_using_kdtree(restaurant, kd_tree, precincts, precinct_coords)    
        else:
            closest_precincts = find_closest_precincts(restaurant, precincts, precinct_coords)  
        print(closest_precincts)
        collection.update_one(
            {"_id": restaurant["_id"]},
            {"$set": {"closest_precincts": closest_precincts}}  # Add the closest_precincts to the restaurant document
        )
        restaurant_coords = (restaurant['coordinates']['latitude'], restaurant['coordinates']['longitude'])
        popup_text = f"<b>{restaurant['name']}</b><br>"
        print("Check! ",restaurant)
        if 'closest_precincts' in restaurant: #Check if the closest_precincts field exists.
            for precinct_name, distance, precinct_coords in restaurant['closest_precincts']:
                print(precinct_name, distance)
                popup_text += f"Closest Precinct: {precinct_name} ({distance:.2f} km)<br>"
                print("Check ", precinct['Precinct'] )
                print(precinct)
                
                print(precinct_coords)
                folium.Marker(
                    location=precinct_coords,
                    popup=f"Precinct: {precinct_name} ({distance:.2f} km)",
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(nyc_map)
        else:
            popup_text += "Closest Precincts data not available.<br>"
        folium.Marker(
            location=restaurant_coords,
            popup=popup_text,
            icon=folium.Icon(color='red', icon='home')  # Use a different icon for restaurants
        ).add_to(nyc_map)

    except (TypeError, KeyError): #Catch the error when the coordinates are missing.
        print(f"Coordinates missing for {restaurant.get('name', 'a restaurant')}") #Print the name of the restaurant without coordinates.
    break

# Save the map as an HTML file (same as before)
nyc_map.save("nyc_restaurants_map.html")
print("Map saved as 'nyc_restaurants_map.html'")

client.close()

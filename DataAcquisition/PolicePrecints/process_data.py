import pymongo
import geopandas as gpd
from geopy.distance import geodesic
import folium
import time
from scipy.spatial import KDTree
from geo_util import find_closest_precincts_using_kdtree, find_closest_precincts

USE_KDTREE = True

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")  
db = client["manhattan_yelp_data"]  
# Get a list of database names
database_names = client.list_database_names()
# print(database_names)

# List all collections (tables) in the database
collection_names = db.list_collection_names()
# print(collection_names)
collection = db["manhattan_businesses"]  



# Load police precinct GeoJSON 
precincts = gpd.read_file('./NYC_Police_Precincts_-8413637686511439451.geojson')
precincts['centroid'] = precincts.geometry.centroid

#  KD-Tree Construction 
precinct_coords = [(precinct['centroid'].y, precinct['centroid'].x) for _, precinct in precincts.iterrows()]
kd_tree = KDTree(precinct_coords)  # Build the KD-tree *once*

# # Function to calculate distances and find closest precincts
# def find_closest_precincts(restaurant):
#     restaurant_coords = (restaurant['coordinates']['latitude'], restaurant['coordinates']['longitude'])
#     distances = []
#     for _, precinct in precincts.iterrows():
#         precinct_coords = (precinct['centroid'].y, precinct['centroid'].x)
#         distance = geodesic(restaurant_coords, precinct_coords).km  # Calculate distance in kilometers
#         distances.append((precinct['Precinct'], distance)) #Store Precinct Name and distance
#         # print(precinct['Precinct'], distance)
#     # Sort by distance and get the 3 closest
#     closest_precincts = sorted(distances, key=lambda x: x[1])[:8]
#     print(closest_precincts)
#     # return closest_precincts
#     return [(name, float(dist)) for name, dist in closest_precincts] # Convert distance to standard float.


# #  Efficient Nearest Neighbor Search 
# def find_closest_precincts_using_kdtree(restaurant):
#     restaurant_coords = (restaurant['coordinates']['latitude'], restaurant['coordinates']['longitude'])
#     # Query the KD-tree for the 8 nearest neighbors
#     distances, indices = kd_tree.query(restaurant_coords, k=8)

#     closest_precincts = []
#     for i in range(len(indices)):
#         precinct_index = indices[i]
#         precinct_name = precincts.iloc[precinct_index]['Precinct'] #Get the precinct name.
#         distance = geodesic(restaurant_coords, precinct_coords[precinct_index]).km
        
#         closest_precincts.append((precinct_name, distance))
#     return [(name, float(dist)) for name, dist in closest_precincts] # Convert distance to standard float.

    # return closest_precincts

start_time = time.time()
cnt = 0
# # Iterate through restaurants in MongoDB and update the collection
for restaurant in collection.find():
    print(f"Processing restaurant {cnt}: {restaurant.get('name', 'Unnamed')}")
    cnt += 1
    if USE_KDTREE:
        closest_precincts = find_closest_precincts_using_kdtree(restaurant, kd_tree, precincts, precinct_coords)    
    else:
        closest_precincts = find_closest_precincts(restaurant, precincts, precinct_coords)  
    collection.update_one(
        {"_id": restaurant["_id"]},
        {"$set": {"closest_precincts": closest_precincts}}  # Add the closest_precincts to the restaurant document
    )
end_time = time.time()
print(f"Time taken to update the database: {end_time - start_time} seconds")

print("Restaurant data updated with closest precinct information.")
print(f"The number of precints searched from {len(precinct_coords)}")
client.close() #Close the connection to MongoDB
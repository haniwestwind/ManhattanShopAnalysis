
from geopy.distance import geodesic
from scipy.spatial import KDTree
import numpy
import time

total_compute_time = 0
compute_count = 0

#  Efficient Nearest Neighbor Search 
def find_closest_precincts_using_kdtree(restaurant, kd_tree, precincts, precinct_coords):
    # Initialize a global variable to store the total compute time and count
    
    global total_compute_time, compute_count
    compute_count += 1
    start_time = time.time()
    restaurant_coords = (restaurant['coordinates']['latitude'], restaurant['coordinates']['longitude'])
    
    # print(restaurant_coords)
    if all(coord is None for coord in restaurant_coords):
        print(f"Invalid coordinates for restaurant: {restaurant.get('name', 'Unnamed')}")
        with open('invalid_restaurants.txt', 'a') as f:
            f.write(f"None coordinates for restaurant: {restaurant}\n")
        return  # Or return a default value

    if not all( numpy.isfinite(coord) for coord in restaurant_coords):
        print(f"Invalid coordinates for restaurant: {restaurant.get('name', 'Unnamed')}")
        with open('invalid_restaurants.txt', 'a') as f:
            f.write(f"Infinite coordinates for restaurant: {restaurant}\n")
        return  # Or return a default value

    # Query the KD-tree for the 8 nearest neighbors
    distances, indices = kd_tree.query(restaurant_coords, k=8)
    # print(indices)
    closest_precincts = []
    for i in range(len(indices)):
        precinct_index = indices[i]
        precinct_name = precincts.iloc[precinct_index]['Precinct'] #Get the precinct name.
        # print(precinct_index, precinct_name)
        distance = geodesic(restaurant_coords, precinct_coords[precinct_index]).km
        # closest_precincts.append((precinct_name, distance))
        latitude, longitude = precinct_coords[precinct_index]

        # print(latitude, longitude)
        closest_precincts.append((precinct_name, distance, (latitude, longitude)))  # Store Precinct Name, distance, and coordinates
    end_time = time.time()
    print(f"Time taken to find coordinates: {end_time - start_time} seconds")
    total_compute_time += (end_time - start_time)
    average_compute_time = total_compute_time / compute_count
    print(f"Average compute time: {average_compute_time} seconds")
    # print(closest_precincts)
    return [(str(name), float(dist), coord) for name, dist, coord in closest_precincts] # Convert distance to standard float.

# Function to calculate distances and find closest precincts
def find_closest_precincts(restaurant, precincts, precinct_coords):
    global total_compute_time, compute_count
    compute_count += 1
    start_time = time.time()

    restaurant_coords = (restaurant['coordinates']['latitude'], restaurant['coordinates']['longitude'])
    distances = []
    for index, precinct in precincts.iterrows():
        coords = (precinct['centroid'].y, precinct['centroid'].x)
        distance = geodesic(restaurant_coords, precinct_coords[index]).km # Calculate distance in kilometers
        distances.append((precinct['Precinct'], distance, coords))  # Store Precinct Name, distance, and coordinates
        # print(precinct['Precinct'], distance)
    # Sort by distance and get the 3 closest
    
    closest_precincts = sorted(distances, key=lambda x: x[1])[:8]
    print(closest_precincts)
    # return closest_precincts
    end_time = time.time()
    print(f"Time taken to find coordinates: {end_time - start_time} seconds")
    total_compute_time += (end_time - start_time)
    average_compute_time = total_compute_time / compute_count
    print(f"Average compute time: {average_compute_time} seconds")
    return [(str(name), float(dist), coord) for name, dist, coord in closest_precincts] # Convert distance to standard float.


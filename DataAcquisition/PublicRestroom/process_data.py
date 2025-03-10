import pymongo
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from geopy.distance import great_circle
import folium


# Connect to MongoDB & Load Business Data

# client = pymongo.MongoClient("mongodb://localhost:27017/")  
client = pymongo.MongoClient("mongodb://localhost:27017/")  
db = client["manhattan_yelp_data"]  
collection = db["manhattan_businesses"]  

# Fetch only Manhattan businesses (filtered by ZIP code)
manhattan_zipcodes = [str(zip) for zip in range(10001, 10283)]  # NYC ZIP code range for Manhattan
business_data = list(collection.find({"location.zip_code": {"$in": manhattan_zipcodes}}, {"_id": 1, "coordinates": 1}))

# Convert to DataFrame
df_businesses = pd.DataFrame(business_data)

# Extract Longitude and Latitude for businesses
df_businesses["Longitude"] = df_businesses["coordinates"].apply(lambda x: x.get("longitude", None) if x else None)
df_businesses["Latitude"] = df_businesses["coordinates"].apply(lambda x: x.get("latitude", None) if x else None)

# Drop missing coordinate values
df_businesses.dropna(subset=["Longitude", "Latitude"], inplace=True)


# Load Public Restroom Data & Clean It

restrooms_file = "Public_Restrooms_20250217.csv"
df_restrooms = pd.read_csv(restrooms_file)

# Ensure column names match the dataset
df_restrooms.rename(columns={"Facility Name": "Restroom_Name", "Latitude": "Latitude", "Longitude": "Longitude"}, inplace=True)

# Drop missing latitude/longitude values
df_restrooms.dropna(subset=["Longitude", "Latitude"], inplace=True)


# Use KDTree for Fast Nearest Restroom Search

# Build KDTree for quick lookup
restroom_coords = np.array(df_restrooms[["Latitude", "Longitude"]])
restroom_tree = KDTree(restroom_coords)


def visualize_closest_restrooms(df_businesses, df_restrooms, num_stores=10):
    """
    Visualizes the closest restrooms to businesses on a map.

    Args:
        df_businesses: DataFrame containing business data with closest restrooms information.
        df_restrooms: DataFrame containing restroom data.
        num_stores: Number of stores to visualize.
    """

    # Create a map centered on Manhattan
    map_center = [40.758896, -73.985130]
    nyc_map = folium.Map(location=map_center, zoom_start=12)

    # Sample businesses for visualization
    sample_businesses = df_businesses.sample(n=num_stores, random_state=42)

    # Add business markers
    for _, business in sample_businesses.iterrows():
        folium.Marker(
            location=[business["Latitude"], business["Longitude"]],
            popup=f"<b>{business['_id']}</b>",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(nyc_map)

        # Add markers for closest restrooms
        for restroom in business["Closest_Restrooms"]:
            restroom_location = df_restrooms[df_restrooms["Restroom_Name"] == restroom["Restroom_Name"]][["Latitude", "Longitude"]].values[0]
            folium.Marker(
                location=restroom_location,
                popup=f"<b>{restroom['Restroom_Name']}</b><br>Distance: {restroom['Distance']:.2f} miles",
                icon=folium.Icon(color="green", icon="female"),
            ).add_to(nyc_map)

            # Draw a line between the business and the restroom
            folium.PolyLine(
                locations=[
                    [business["Latitude"], business["Longitude"]],
                    restroom_location
                ],
                color="red",
                weight=1.5,
                opacity=0.5
            ).add_to(nyc_map)

    # Save the map
    nyc_map.save("closest_restrooms_map.html")
    print("Map saved to closest_restrooms_map.html")

# Function to find closest restrooms
def find_closest_restrooms_kdtree(business, top_n=10):
    business_location = (business["Latitude"], business["Longitude"])
    
    if pd.isnull(business_location[0]) or pd.isnull(business_location[1]):
        return []
    
    # Query KDTree for nearest restrooms
    distances, indices = restroom_tree.query(business_location, k=min(top_n, len(df_restrooms)))
    
    # Convert to list if only one value
    if isinstance(indices, np.int64):
        indices = [indices]
        distances = [distances]

    # Store top N closest restrooms
    closest_restrooms = [
        {
            "Restroom_Name": df_restrooms.iloc[idx]["Restroom_Name"],
            "Distance": great_circle(business_location, (df_restrooms.iloc[idx]["Latitude"], df_restrooms.iloc[idx]["Longitude"])).miles
        }
        for idx in indices
    ]
    
    return closest_restrooms

# Apply function to all businesses
df_businesses["Closest_Restrooms"] = df_businesses.apply(lambda row: find_closest_restrooms_kdtree(row, top_n=10), axis=1)


# Print Sample Results Before Saving

print("\nSample of Closest Restrooms for 5 Businesses:\n")
print(df_businesses[["_id", "Closest_Restrooms"]].head())

# visualize_closest_restrooms(df_businesses, df_restrooms)


# Save Results to MongoDB

for _, row in df_businesses.iterrows():
   collection.update_one(
       {"_id": row["_id"]},
       {"$set": {"closest_restrooms": row["Closest_Restrooms"]}}
   )

print("\n Closest restrooms successfully saved to MongoDB!")

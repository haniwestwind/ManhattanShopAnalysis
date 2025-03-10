import pymongo
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from geopy.distance import great_circle
import folium



def visualize_stores_with_schools(df_businesses, df_schools, num_stores=10):
    """
    Visualizes stores and their closest schools on a map.

    Args:
        df_businesses: DataFrame containing business data with closest schools information.
        df_schools: DataFrame containing school data.
        num_stores: Number of stores to visualize.
    """

    # Sample 10 random stores
    sample_stores = df_businesses.sample(n=num_stores, random_state=42)  # Use random_state for reproducibility

    # Create a map centered on Manhattan
    map_center = [40.7589, -73.9851]  # Latitude and longitude for Manhattan
    m = folium.Map(location=map_center, zoom_start=12)

    # Add markers for schools
    for _, school in df_schools.iterrows():
        folium.Marker(
            location=[school["Latitude"], school["Longitude"]],
            popup=school["School_Name"],
            icon=folium.Icon(color="green", icon="book")
        ).add_to(m)

    # Add markers for stores and connect them to closest schools
    for _, store in sample_stores.iterrows():
        folium.Marker(
            location=[store["Latitude"], store["Longitude"]],
            popup=store["_id"],
            icon=folium.Icon(color="blue", icon="shopping-cart")
        ).add_to(m)

        for school_data in store["Closest_Schools"]:
            school_name = school_data["School_Name"]
            school_location = df_schools[df_schools["School_Name"] == school_name][["Latitude", "Longitude"]].values[0]
            folium.PolyLine(
                locations=[[store["Latitude"], store["Longitude"]], school_location],
                color="red",
                weight=1.5,
                opacity=0.7
            ).add_to(m)

    # Save the map
    m.save("schools_near_stores_map.html")
    print("\nMap saved as 'schools_near_stores_map.html'")

# Visualize the data


# Connect to MongoDB & Load Manhattan Business Data

# client = pymongo.MongoClient("mongodb://localhost:27017/") 
client = pymongo.MongoClient("mongodb://localhost:27017/") 
db = client["manhattan_yelp_data"]  
collection = db["manhattan_businesses"] 

# Fetch businesses, filtering for Manhattan ZIP codes (10001 - 10282)
manhattan_zips = [str(z) for z in range(10001, 10283)]
business_data = list(collection.find(
    {"location.zip_code": {"$in": manhattan_zips}}, 
    {"_id": 1, "coordinates": 1, "location.zip_code": 1}
))

# Convert to DataFrame
df_businesses = pd.DataFrame(business_data)

# Extract Longitude and Latitude
df_businesses["Longitude"] = df_businesses["coordinates"].apply(lambda x: x.get("longitude", None) if x else None)
df_businesses["Latitude"] = df_businesses["coordinates"].apply(lambda x: x.get("latitude", None) if x else None)

# Drop missing coordinate values
df_businesses.dropna(subset=["Longitude", "Latitude"], inplace=True)

print(f"Manhattan businesses loaded: {len(df_businesses)}")



# Load & Clean Schools Data

schools_file = "2019_-_2020_School_Locations_20250217.csv"  
df_schools = pd.read_csv(schools_file)

# Ensure column names match dataset
df_schools.rename(columns={"LONGITUDE": "Longitude", "LATITUDE": "Latitude", "location_name": "School_Name"}, inplace=True)

# Drop missing latitude/longitude values
df_schools.dropna(subset=["Longitude", "Latitude"], inplace=True)

print(f"Schools loaded: {len(df_schools)}")



# Use KDTree for Fast Nearest School Search

# Build KDTree for quick lookup
school_coords = np.array(df_schools[["Latitude", "Longitude"]])
school_tree = KDTree(school_coords)

# Function to find closest schools
def find_closest_schools_kdtree(business, top_n=10):
    business_location = (business["Latitude"], business["Longitude"])
    
    if pd.isnull(business_location[0]) or pd.isnull(business_location[1]):
        return []
    
    # Query KDTree for nearest schools
    distances, indices = school_tree.query(business_location, k=min(top_n, len(df_schools)))
    
    # Convert to list if only one value
    if isinstance(indices, np.int64):
        indices = [indices]
        distances = [distances]

    # Store top N closest schools
    closest_schools = [
        {
            "School_Name": df_schools.iloc[idx]["School_Name"],
            "Distance": round(great_circle(business_location, (df_schools.iloc[idx]["Latitude"], df_schools.iloc[idx]["Longitude"])).miles, 2)
        }
        for idx in indices
    ]
    
    return closest_schools

# Apply function to all businesses
df_businesses["Closest_Schools"] = df_businesses.apply(lambda row: find_closest_schools_kdtree(row, top_n=10), axis=1)


# Print Sample Results Before Saving

print("\nSample of Closest Schools for 5 Manhattan Businesses:\n")
print(df_businesses[["_id", "Closest_Schools"]].head())

# visualize_stores_with_schools(df_businesses, df_schools)




# Save Results to MongoDB

# Update MongoDB collection by adding the closest schools
for _, row in df_businesses.iterrows():
   collection.update_one(
       {"_id": row["_id"]},
       {"$set": {"closest_schools": row["Closest_Schools"]}}
   )

print("\nClosest schools successfully saved to MongoDB!")

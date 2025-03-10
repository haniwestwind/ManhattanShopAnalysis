import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import folium

"""K-means clustering with the same 5 variables as v1, filtered for only Manhattan businesses"""


# 1. Load Data from MongoDB


# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")  
db = client["yelp_data"]  
collection = db["businesses"]  

# Fetch relevant store data
store_data = list(collection.find({}, {
    "_id": 1, "coordinates": 1, "rating": 1, "review_count": 1, 
    "average_income_data": 1, "closest_precincts": 1, 
    "closest_parks": 1, "closest_subways": 1, "has_subway_access": 1, 
    "closest_rat_sighting_count": 1, "closest_rat_sighting_distance": 1
}))

# Convert to DataFrame
df_stores = pd.DataFrame(store_data)


# 2. Filter for Manhattan-Only Businesses


# Manhattan's rough latitude and longitude bounds
MANHATTAN_LAT_MIN, MANHATTAN_LAT_MAX = 40.68, 40.88
MANHATTAN_LON_MIN, MANHATTAN_LON_MAX = -74.03, -73.91

# Extract latitude and longitude
df_stores["Latitude"] = df_stores["coordinates"].apply(lambda x: x.get("latitude", None))
df_stores["Longitude"] = df_stores["coordinates"].apply(lambda x: x.get("longitude", None))

# Filter only Manhattan stores
df_manhattan = df_stores[
    (df_stores["Latitude"].between(MANHATTAN_LAT_MIN, MANHATTAN_LAT_MAX)) &
    (df_stores["Longitude"].between(MANHATTAN_LON_MIN, MANHATTAN_LON_MAX))
]


# 3. Extract Relevant Features for Clustering


df_manhattan["Income"] = df_manhattan["average_income_data"]  
df_manhattan["closest_precinct_distance"] = df_manhattan["closest_precincts"].apply(
    lambda x: x[0][1] if isinstance(x, list) and len(x) > 0 else None)

df_manhattan["closest_park_distance"] = df_manhattan["closest_parks"].apply(
    lambda x: x[0]["Distance"] if isinstance(x, list) and len(x) > 0 else None)

df_manhattan["closest_subway_distance"] = df_manhattan["closest_subways"].apply(
    lambda x: x[0]["subway_distance_miles"] if isinstance(x, list) and len(x) > 0 else None)

df_manhattan["has_subway_access"] = df_manhattan["has_subway_access"].astype(int)

df_manhattan["closest_rat_sighting_count"] = df_manhattan["closest_rat_sighting_count"]
df_manhattan["closest_rat_sighting_distance"] = df_manhattan["closest_rat_sighting_distance"]

# Drop missing values
df_manhattan.dropna(subset=[
    "Income", "closest_precinct_distance", "closest_park_distance", 
    "closest_subway_distance", "has_subway_access", 
    "closest_rat_sighting_count", "closest_rat_sighting_distance"
], inplace=True)


# 4. Perform K-Means Clustering


# Select clustering features
clustering_features = [
    "Income", "closest_precinct_distance", "closest_park_distance",
    "closest_subway_distance", "has_subway_access",
    "closest_rat_sighting_count", "closest_rat_sighting_distance"
]

X = df_manhattan[clustering_features]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_manhattan["Cluster"] = kmeans.fit_predict(X_scaled)


# 5. Analyze Store Success Across Clusters


# Define store success (1 if rating >= 4.0, else 0)
df_manhattan["Success"] = (df_manhattan["rating"] >= 4.0).astype(int)

# Group by cluster
cluster_summary = df_manhattan.groupby("Cluster").agg({
    "Latitude": "mean",
    "Longitude": "mean",
    "Income": "mean",
    "closest_precinct_distance": "mean",
    "closest_park_distance": "mean",
    "closest_subway_distance": "mean",
    "closest_rat_sighting_count": "mean",
    "Success": ["count", "mean"],  # Count = number of stores, mean = success rate
    "rating": "mean",
    "review_count": "mean"
}).reset_index()

# Rename columns
cluster_summary.columns = [
    "Cluster", "Latitude", "Longitude", "Income", "Avg_Precinct_Dist",
    "Avg_Park_Dist", "Avg_Subway_Dist", "Avg_Rat_Count",
    "Num_Stores", "Success_Rate", "Avg_Rating", "Avg_Review_Count"
]

# Print cluster summary
print("\nCluster Summary:\n", cluster_summary)


# 6. Visualizing Clusters on a Scatter Plot


# Create a directory to save figures if it doesn’t exist
save_dir = "figures"
os.makedirs(save_dir, exist_ok=True)

#  1. Scatter plot of clusters based on latitude and longitude 
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_manhattan["Longitude"], y=df_manhattan["Latitude"], hue=df_manhattan["Cluster"], palette="viridis", alpha=0.6)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Distribution of Store Clusters in Manhattan")
plt.legend(title="Cluster")
plt.savefig(os.path.join(save_dir, "clusters_scatter_plot.png"))
plt.close()
print(f"Scatter plot saved.")

#  2. Manhattan cluster map 
# Set Manhattan center coordinates
manhattan_center = [40.758896, -73.985130]  # Approximate center

# Create a Manhattan map
map_manhattan = folium.Map(location=manhattan_center, zoom_start=12)

# Define cluster colors
cluster_colors = {0: "red", 1: "blue", 2: "green", 3: "purple"}

# Plot clusters on the Manhattan map
for _, row in df_manhattan.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        color=cluster_colors.get(row["Cluster"], "black"),  # Default to black if cluster missing
        fill=True,
        fill_color=cluster_colors.get(row["Cluster"], "black"),
        fill_opacity=0.7,
        popup=f"Cluster: {row['Cluster']}, Rating: {row['rating']:.1f}",
    ).add_to(map_manhattan)

map_path = os.path.join(save_dir, "manhattan_clusters_map.html")
map_manhattan.save(map_path)
print(f"Interactive Manhattan cluster map saved: {map_path}")

#  3. Box plots for key variables 
features_to_plot = ["Income", "closest_precinct_distance", "closest_park_distance", 
                    "closest_subway_distance", "closest_rat_sighting_distance", "closest_rat_sighting_count"]

for feature in features_to_plot:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df_manhattan["Cluster"], y=df_manhattan[feature], palette="coolwarm")
    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.title(f"Distribution of {feature} across Clusters")
    plt.savefig(os.path.join(save_dir, "clusters_box_plot.png"))
    plt.close()
print(f"Box plot saved.")

#  4. Bar plot for success rate per cluster 
plt.figure(figsize=(8, 5))
sns.barplot(x=df_manhattan["Cluster"], y=df_manhattan["Success"], estimator=lambda x: sum(x)/len(x), palette="Blues")
plt.xlabel("Cluster")
plt.ylabel("Success Rate")
plt.title("Success Rate per Cluster")
plt.savefig(os.path.join(save_dir, "clusters_bar_plot.png"))
plt.close()
print(f"Bar plot saved.")
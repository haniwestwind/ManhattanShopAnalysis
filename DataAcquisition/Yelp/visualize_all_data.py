import pymongo
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
import os
import folium
from sklearn.preprocessing import MinMaxScaler

from shapely.geometry import Point
from shapely.ops import transform
import pyproj
from functools import partial

def calculate_success_score(rating, review_count):
    """Calculates a success score based on rating and review count."""
    return rating


def insert_manhattan_data_in_chunks(df_manhattan, manhattan_businesses_collection, chunk_size=10):
    manhattan_data = df_manhattan.drop(columns=["Latitude", "Longitude"]).to_dict(orient="records")
    total_documents = len(manhattan_data)
    print(f"Total documents to insert: {total_documents}")

    if not manhattan_data:
        print("No Manhattan businesses found.")
        return

    for i in range(0, total_documents, chunk_size):
        chunk = manhattan_data[i:i + chunk_size]
        try:
            for doc in chunk:
                try:
                    result = manhattan_businesses_collection.update_many(
                        {"_id": doc["_id"]},
                        {"$set": doc},
                        upsert=True
                    )
                    #print(f"Updated/Inserted document with _id: {doc['_id']}")
                except pymongo.errors.PyMongoError as e:
                    print(f"Error updating/inserting document with _id: {doc['_id']}: {e}")

            # result = manhattan_businesses_collection.insert_many(chunk, ordered=False) #ordered=False is important
            # result = manhattan_businesses_collection.update_many(
            #         {"_id": doc["_id"]},  # Match documents with the same _id
            #         {"$set": doc},  # Update fields with the new values
            #         upsert=True #if the document does not exist, insert it.
            #     )
            # print(f"Inserted {len(result.upserted_id)} documents from {i} to {i + len(chunk)}")
        except pymongo.errors.BulkWriteError as e:
            print(f"BulkWriteError in chunk {i} to {i + chunk_size}:")
            # print(e.details)
            # You can log the error details to a file if needed
            # with open(f"bulk_write_error_chunk_{i}.log", "w") as f:
            #     f.write(str(e.details))

    print("Manhattan data insertion completed.")

def filter_and_save_manhattan_data():
    """Filters Manhattan businesses, saves to a new MongoDB collection, and visualizes."""

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["manhattan_yelp_data"]
    businesses_collection = db["manhattan_businesses"]

    # Fetch all store data
    # store_data = list(businesses_collection.find({}, {"_id": 1, "coordinates": 1, "rating": 1, "review_count": 1, "name": 1}))
    store_data = list(businesses_collection.find())
    df_stores = pd.DataFrame(store_data)

    # Extract latitude and longitude
    df_stores["Latitude"] = df_stores["coordinates"].apply(lambda x: x.get("latitude", None))
    df_stores["Longitude"] = df_stores["coordinates"].apply(lambda x: x.get("longitude", None))

    # Filter only Manhattan stores using the provided shapefile
    # df_manhattan = filter_manhattan_businesses_with_shapefile(df_stores)

    # # Create a new database and collection for Manhattan data
    # manhattan_db = client["manhattan_yelp_data"]
    # manhattan_businesses_collection = manhattan_db["manhattan_businesses"]

    # # Convert DataFrame back to list of dictionaries for MongoDB insertion
    # manhattan_data = df_manhattan.drop(columns=["Latitude", "Longitude"]).to_dict(orient="records")
    # print(len(manhattan_data))
    # insert_manhattan_data_in_chunks(df_manhattan, manhattan_businesses_collection, chunk_size=10)
    # Insert Manhattan data into the new collection
    # if manhattan_data:
    #     manhattan_businesses_collection.insert_many(manhattan_data)
    #     print("Manhattan data saved to 'manhattan_yelp_data.manhattan_businesses'")
    # else:
    #     print("No Manhattan businesses found.")

    # Visualize Manhattan stores
    visualize_manhattan_stores(df_stores)

    client.close()

def filter_manhattan_businesses_with_shapefile(df_stores):
    """Filters Manhattan businesses using the provided shapefile."""

    # Load Manhattan boundary from the provided shapefile
    # gdf = gpd.read_file("taxi_zones.shp")
    gdf = gpd.read_file("taxi_zones.shp")
    gdf = gdf[gdf.borough == 'Manhattan']
   # Convert the coordinate system to WGS 84 (latitude/longitude)
    gdf = gdf.to_crs(epsg=4326)  # EPSG:4326 is the code for WGS 84

    # Now you can proceed with creating the polygon and filtering
    manhattan_polygon = gdf.unary_union

    # Filter only Manhattan stores (using the converted coordinates)
    manhattan_mask = df_stores.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]).within(manhattan_polygon),
        axis=1
    )
    df_manhattan = df_stores[manhattan_mask]
    print(f"Number of Manhattan businesses: {df_manhattan.shape[0]}")
    return df_manhattan

def visualize_manhattan_stores(df_manhattan):
    """Visualizes Manhattan stores on a map."""

    if df_manhattan.empty:
        print("No Manhattan stores to visualize.")
        return

    # Calculate success scores
    df_manhattan.loc[:, "success_score"] = df_manhattan.apply(
        lambda row: calculate_success_score(row["rating"], row["review_count"]), axis=1
    )

    # Normalize success scores and review counts
    scaler = MinMaxScaler()
    df_manhattan.loc[:, "normalized_score"] = scaler.fit_transform(df_manhattan[["success_score"]])
    df_manhattan.loc[:, "normalized_review_count"] = scaler.fit_transform(df_manhattan[["review_count"]])

    # Visualization
    nyc_map = folium.Map(location=[40.758896, -73.985130], zoom_start=12)  # Center on Manhattan
    count_coord_none = 0
    for _, store in df_manhattan.iterrows():
        if store["coordinates"] is None or store["coordinates"]["latitude"] is None or store["coordinates"]["longitude"] is None:
            count_coord_none += 1
            continue
        folium.CircleMarker(
            location=[store["coordinates"]["latitude"], store["coordinates"]["longitude"]],
            radius=store["normalized_score"] * 20,  
            color="red",
            fill=True,
            fill_color="lightcoral",
            fill_opacity=0.6,
            popup=f"<b>{store['name']}</b><br>Rating: {store['rating']}<br>Reviews: {store['review_count']}<br>Success Score: {store['success_score']:.2f}",
        ).add_to(nyc_map)

        folium.CircleMarker(
            location=[store["coordinates"]["latitude"], store["coordinates"]["longitude"]],
            radius=store["normalized_review_count"] * 40,  
            color="blue",
            fill=True,
            fill_color="lightblue",
            fill_opacity=0.6,
            popup=f"<b>{store['name']}</b><br>Rating: {store['rating']}<br>Reviews: {store['review_count']}<br>Success Score: {store['success_score']:.2f}",
        ).add_to(nyc_map)
    print(f"Number of stores with missing coordinates: {count_coord_none}")
    # Save map
    # save_dir = "manhattan_maps"
    save_dir = "./"
    os.makedirs(save_dir, exist_ok=True)
    map_filename = os.path.join(save_dir, "all_stores_map.html")
    nyc_map.save(map_filename)
    print(f"Manhattan stores map saved as '{map_filename}'")

if __name__ == "__main__":
    filter_and_save_manhattan_data()
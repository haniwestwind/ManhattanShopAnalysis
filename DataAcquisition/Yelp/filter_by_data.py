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


def filter_invalid_data():
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

    # Filter for valid data before filtering by location
    df_stores_with_valid_data = df_stores.dropna(subset=["coordinates", "rating", "review_count"])

    # Combine filters: valid data 
    valid_manhattan_mask = df_stores_with_valid_data.index.isin(df_stores.index)

    invalid_data_mask = ~valid_manhattan_mask

    # Delete invalid businesses from the database
    invalid_ids = df_stores[invalid_data_mask]["_id"].tolist()
    print(f"Deleting {len(invalid_ids)} invalid businesses from the database.")
    if invalid_ids:
        result = businesses_collection.delete_many({"_id": {"$in": invalid_ids}})
        print(f"Deleted {result.deleted_count} invalid businesses from the database.")
    # Print the remaining number of data
    remaining_store_data = list(businesses_collection.find())
    print(f"Remaining number of businesses in the database: {len(remaining_store_data)}")

    client.close()


if __name__ == "__main__":
    filter_invalid_data()
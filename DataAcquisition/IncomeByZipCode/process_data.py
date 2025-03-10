import pandas as pd
import pymongo
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from scipy.spatial import KDTree
import pickle
import os
import time
import numpy as np
import sys

# Reading excel/csv data 
def get_income_data_from_excel(excel_file):
    """
    Reads income data from an Excel file and returns it as a DataFrame.

    Args:
        excel_file (str): Path to the Excel file.

    Returns:
        pandas.DataFrame: DataFrame containing the income data.
    """
    try:
        df = pd.read_excel(excel_file)
        return df
    except FileNotFoundError:
        print(f"Error: Excel file '{excel_file}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the Excel file: {e}")
        return None

# Define a function to get the midpoint of an income range
def get_income_midpoint(income_bracket):
    if pd.isna(income_bracket):  # Handle NaN values
        return 0
    income_bracket = str(income_bracket)  # Ensure it's a string
    if "under" in income_bracket:
        low, high = map(int, income_bracket.replace("$", "").replace(",", "").split(" under "))
        return (low + high) / 2
    elif "or more" in income_bracket:
        low = int(income_bracket.replace("$", "").replace(",", "").split(" or more")[0])
        return low * 1.5  # Assuming 1.5x of lower bound for open-ended bracket
    return 0  # If it doesn't match, return NaN


def main():
    """
    Connects to MongoDB, retrieves store data, finds the closest zip code for each store,
    and updates the database with income data from an Excel file.
    """
    #  Configuration 
    excel_file = 'income_data.xlsx'  
    mongodb_uri = "mongodb://localhost:27017/"  
    database_name = "manhattan_yelp_data"  
    collection_name = "manhattan_businesses"  

    #  Load Income Data 
    income_data_df = get_income_data_from_excel(excel_file)
    if income_data_df is None:
        return

    #  MongoDB Connection 
    # Setting up client to communicate with MongoDB and retrieve data
    client = pymongo.MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]

    #  Prepare Zip Code Data 
    # Using Nominatim, convert ZIP code to longitude/latitude
    geolocator = Nominatim(user_agent="zip_code_locator")

    print(income_data_df['ZIP code [1]'])

    # Apply the function to create a new column with midpoints
    income_data_df["income_midpoint"] = income_data_df["Size of adjusted gross income"].apply(get_income_midpoint)

    # Calculate weighted average income for each ZIP code
    weighted_income = (
        income_data_df.groupby("ZIP code [1]")
        .apply(lambda x: np.average(x["income_midpoint"], weights=x["Number of individuals [3]"]))
        .reset_index()
    )

    # Rename columns
    weighted_income.columns = ["ZIP code [1]", "weighted_avg_income"]
    print(weighted_income)
    zip_codes = weighted_income['ZIP code [1]'].astype(str).tolist()

    
    weighted_income["ZIP code [1]"] = weighted_income["ZIP code [1]"].astype(int).astype(str).str.zfill(5)
    print("weighted_income[ZIP code]", weighted_income["ZIP code [1]"])

    #  Update Store Data Based on ZIP Code 
    for store in collection.find():
        store_zip = store['location']['zip_code']  # Get the store's ZIP code as a string
        
        # Find matching income data for the ZIP code
        # Ensure ZIP codes are properly formatted as 5-digit strings

        # Get ZIP code, ensuring it is not empty or None
        store_zip = store['location'].get('zip_code', '').strip()

        # If ZIP code is empty, skip the update
        if not store_zip or not store_zip.isdigit():
            print(f"Skipping store {store['_id']} due to missing ZIP code.")
            continue  # Skip this store and move to the next one

        # Convert ZIP to a 5-digit string
        store_zip = store_zip.zfill(5)

        # Find matching income data
        income_info = weighted_income[weighted_income["ZIP code [1]"] == store_zip]

        if not income_info.empty:
            avg_income = income_info["weighted_avg_income"].values[0]  # Extract income value
        else:
            avg_income = None  # No matching income found

        # Update the MongoDB document with income data
        collection.update_one(
            {"_id": store["_id"]},
            {"$set": {
                "average_income_data": avg_income
            }}
        )

        print(f"Updated store {store['_id']} with income data for ZIP {store_zip}: {avg_income}")

    client.close()
    print("Store data updated with ZIP-based income information.")

if __name__ == "__main__":
    main()
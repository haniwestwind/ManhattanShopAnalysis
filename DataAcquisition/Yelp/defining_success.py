import pymongo
import pandas as pd
import numpy as np

 
# Connect to MongoDB
 
client = pymongo.MongoClient("mongodb://localhost:27017/")  
db = client["manhattan_yelp_data"] 
collection = db["manhattan_businesses"] 

 
# Define Manhattan ZIP Codes
 
manhattan_zipcodes = [str(z) for z in range(10001, 10283)]  # Covers all Manhattan ZIPs

 
# Fetch Manhattan Businesses by ZIP Code
 
query = {
    "location.zip_code": {"$in": manhattan_zipcodes}  # Filter by ZIP codes
}
projection = {
    "_id": 1,
    "rating": 1,
    "review_count": 1,
    "location.zip_code": 1  # Keep for debugging
}

# business_data = list(collection.find(query, projection))
business_data = list(collection.find({}, projection))

# Convert to DataFrame
df_stores = pd.DataFrame(business_data)

# Check if data is empty
if df_stores.empty:
    print("No Manhattan businesses found in MongoDB.")
else:
    print(f"Loaded {len(df_stores)} businesses from Manhattan based on ZIP code.")

# Drop rows with missing values in rating or review_count
df_stores.dropna(subset=["rating", "review_count"], inplace=True)

 
# Calculate Success Scores
 
# Global stats
C = df_stores["rating"].mean()  # Mean rating across all stores
m = 50  # Minimum reviews for reliability (tuneable hyperparameter)

# Bayesian Weighted Average
df_stores["Bayesian_Score"] = (df_stores["rating"] * df_stores["review_count"] + C * m) / (df_stores["review_count"] + m)

# IMDB style weighted rating
V_max = df_stores["review_count"].max()
df_stores["IMDB_Score"] = df_stores["rating"] * (np.log(1 + df_stores["review_count"]) / np.log(1 + V_max))

 
# Print & Save Results
 
print("\nSample of Manhattan Store Success Scores:\n", df_stores.head())

 
# Save to MongoDB
 
for index, row in df_stores.iterrows():
    collection.update_one({"_id": row["_id"]}, {"$set": {"bayesian_score": row["Bayesian_Score"], "imdb_score": row["IMDB_Score"]}})
print("Updated MongoDB with success scores.")
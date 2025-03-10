import pymongo


# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["manhattan_yelp_data"]
business_collection = db["manhattan_businesses"]


# Define the fields we are interested in
fields = [
    "_id", "rating", "review_count", "coordinates", 
    "average_income_data", "closest_precincts", 
    "closest_parks", "closest_subways", "has_subway_access", 
    "closest_rat_sighting_distance", "closest_rat_sighting_count", 
    "subway_count_0_5mi",  "subway_density_0_3mi", "complaints_within_radius",
    "closest_schools", "closest_restrooms", "bayesian_score", "imdb_score", "categories" 

]

# categories

# Fetch store data with additional variables
store_data = list(business_collection.find({}, {field: 1 for field in fields}))
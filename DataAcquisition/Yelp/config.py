from pymongo import MongoClient

api_key_file = "API_KEY_YELP"
with open("API_KEY_YELP", "r") as file:
    API_KEY = file.read().strip()
# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
# db = client["yelp_data"]
db = client["manhattan_yelp_data"]
offset_collection = db["query_offsets_again"]
business_collection = db["manhattan_businesses"]

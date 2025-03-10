import pymongo
import pandas as pd
import folium
from sklearn.preprocessing import MinMaxScaler
import argparse

def calculate_success_score(rating, review_count):
    """Calculates a success score based on rating and review count."""
    # A simple example: weighted average. Adjust weights as needed.
    return rating # + (review_count / 100 * 0.3)  # Example weights

def visualize_stores_by_category(category_name):
    """Queries MongoDB, calculates success scores, and visualizes stores."""

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["yelp_data"]  
    stores_collection = db["businesses"]

    # Query stores with the specified category
    query = {"categories.alias": category_name}
    stores_cursor = stores_collection.find(query)
    df_stores = pd.DataFrame(list(stores_cursor))

    if df_stores.empty:
        print(f"No stores found with category: {category_name}")
        client.close()
        return

    # Calculate success scores
    df_stores["success_score"] = df_stores.apply(
        lambda row: calculate_success_score(row["rating"], row["review_count"]), axis=1
    )

    # Normalize success scores
    scaler = MinMaxScaler()
    df_stores["normalized_score"] = scaler.fit_transform(
        df_stores[["success_score"]]
    )

    df_stores["normalized_review_count"] = scaler.fit_transform(
        df_stores[["review_count"]]
    )
    # Visualization
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)  # Center on NYC

    for _, store in df_stores.iterrows():
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
            fill_color="lightcoral",
            fill_opacity=0.6,
            popup=f"<b>{store['name']}</b><br>Rating: {store['rating']}<br>Reviews: {store['review_count']}<br>Success Score: {store['success_score']:.2f}",
        ).add_to(nyc_map)

    # Save map
    map_filename = f"./VisualizationHtml/{category_name}_stores_map.html"
    nyc_map.save(map_filename)
    print(f"Map saved as '{map_filename}'")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize stores by category.")
    parser.add_argument("category", type=str, help="The category to search for.")
    args = parser.parse_args()

    visualize_stores_by_category(args.category)
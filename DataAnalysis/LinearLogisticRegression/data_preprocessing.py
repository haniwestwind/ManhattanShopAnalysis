from data_reader import store_data, fields
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import pandas as pd

df_manhattan = pd.DataFrame(store_data)
# print("Columns in df_manhattan:", df_manhattan.columns.tolist())  # Debugging

# Extract category aliases
df_manhattan['category_aliases'] = df_manhattan['categories'].apply(lambda x: [item['alias'] for item in x] if isinstance(x, list) else [])

# One-Hot Encoding using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
encoded_categories = pd.DataFrame(mlb.fit_transform(df_manhattan['category_aliases']), columns=mlb.classes_)
# print(encoded_categories.head())
# Concatenate the encoded categories with the original DataFrame
df_manhattan = pd.concat([df_manhattan, encoded_categories], axis=1)

# Standardization
scaler = StandardScaler()
encoded_category_columns = encoded_categories.columns #save the column names.
# df[encoded_category_columns] = scaler.fit_transform(df[encoded_category_columns])

# print("Check the category columns", encoded_category_columns)
df_manhattan[encoded_category_columns] = scaler.fit_transform(df_manhattan[encoded_category_columns])
# print(df_manhattan[encoded_category_columns].head())

# Concatenate encoded categories with the original DataFrame
df_manhattan = pd.concat([df_manhattan, encoded_categories], axis=1)

# Drop the original 'categories' and 'category_aliases' columns if you don't need them
df_manhattan = df_manhattan.drop(['categories', 'category_aliases'], axis=1)

# Convert necessary fields
df_manhattan["closest_precinct_distance"] = df_manhattan["closest_precincts"].apply(
    lambda x: x[0][1] if isinstance(x, list) and len(x) > 0 else None)
df_manhattan["closest_park_distance"] = df_manhattan["closest_parks"].apply(
    lambda x: x[0]["Distance"] if isinstance(x, list) and len(x) > 0 else None)
df_manhattan["closest_subway_distance"] = df_manhattan["closest_subways"].apply(
    lambda x: x[0]["subway_distance_miles"] if isinstance(x, list) and len(x) > 0 else None)

# df_manhattan["closest_rat_sighting_count"] = df_manhattan["closest_rat_sighting_count"]
# df_manhattan["closest_rat_sighting_distance"] = df_manhattan["closest_rat_sighting_distance"]
# Extract closest school distance
df_manhattan["closest_school_distance"] = df_manhattan["closest_schools"].apply(
    lambda x: x[0]["Distance"] if isinstance(x, list) and len(x) > 0 else None)

# Extract closest restroom distance
df_manhattan["closest_restroom_distance"] = df_manhattan["closest_restrooms"].apply(
    lambda x: x[0]["Distance"] if isinstance(x, list) and len(x) > 0 else None)

df_manhattan["normalized_average_income_data"] = df_manhattan["average_income_data"] / (df_manhattan["average_income_data"].max() + 1)

df_manhattan["has_subway_access"] = df_manhattan["has_subway_access"].fillna(0).astype(int)

# Normalize complaints_within_radius with the maximum value
df_manhattan["normalized_complaints_within_radius"] = df_manhattan["complaints_within_radius"] / (df_manhattan["complaints_within_radius"].max() + 1)

# Normalize distances with the maximum value in each column
df_manhattan["normalized_precinct_distance"] = df_manhattan["closest_precinct_distance"] / (df_manhattan["closest_precinct_distance"].max() + 1)
df_manhattan["normalized_park_distance"] = df_manhattan["closest_park_distance"] / (df_manhattan["closest_park_distance"].max() + 1)
df_manhattan["normalized_subway_distance"] = df_manhattan["closest_subway_distance"] / (df_manhattan["closest_subway_distance"].max() + 1)
df_manhattan["normalized_rat_sighting_distance"] = df_manhattan["closest_rat_sighting_distance"] / (df_manhattan["closest_rat_sighting_distance"].max() + 1)
df_manhattan["normalized_closest_rat_sighting_count"] = df_manhattan["closest_rat_sighting_count"] / (df_manhattan["closest_rat_sighting_count"].max() + 1)


# Drop missing values
df_manhattan.dropna(subset=[
    "normalized_average_income_data", "has_subway_access", "normalized_complaints_within_radius", "normalized_precinct_distance",
    "normalized_park_distance", "normalized_subway_distance", "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"
], inplace=True)


df_manhattan["success"] = (df_manhattan["bayesian_score"] >= df_manhattan["bayesian_score"].mean()).astype(int)

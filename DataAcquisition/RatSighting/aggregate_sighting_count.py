import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

def get_zipcode(address, city="New York City", state="NY"):
    """
    Retrieves the zipcode for a given address using geocoding.

    Args:
        address (str): The street address.
        city (str): The city.
        state (str): The state.

    Returns:
        str: The zipcode, or None if not found.
    """
    geolocator = Nominatim(user_agent="zipcode_finder")
    location = None
    try:
        location = geolocator.geocode(f"{address}, {city}, {state}")
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Geocoding error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    if location:
        if hasattr(location, 'raw') and 'address' in location.raw and 'postcode' in location.raw['address']:
            print("Found the location")
            return location.raw['address']['postcode']
        else:
            print(f"Could not extract postcode from geocoding result for address: {address}")
            return None
    else:
        print(f"Location not found for address: {address}")
        return None

def process_rat_sightings(input_file_path, output_file_path):
    """
    Processes rat sighting data, aggregates by zipcode, and saves the results to a CSV.

    Args:
        input_file_path (str): The path to the input CSV file.
        output_file_path (str): The path to the output CSV file.
    """
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file at {input_file_path} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the CSV file at {input_file_path}. Please ensure it is valid.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV file: {e}")
        return

    zipcode_counts = {}
    nan_count = 0
    for index, row in df.iterrows():
        # print(row['Incident Zip'])

        if pd.isna(row['Incident Zip']):
            nan_count += 1
            continue
        zipcode = str(int(row['Incident Zip']))
        

        # address = row['Incident Address']
        # if pd.isna(address):
        #     zipcode = row['Incident Zip']
        #     if pd.isna(zipcode):
        #         continue
        #     else:
        #         zipcode = str(int(zipcode))
        # else:
        #     zipcode = get_zipcode(address)
        #     if zipcode is None:
        #         continue

        if zipcode in zipcode_counts:
            zipcode_counts[zipcode] += 1
        else:
            zipcode_counts[zipcode] = 1

    # Create a DataFrame from the zipcode_counts dictionary
    result_df = pd.DataFrame(list(zipcode_counts.items()), columns=['Zip Code', 'Sighting Count'])
    result_df['Latitude'] = None
    result_df['Longitude'] = None
    geolocator = Nominatim(user_agent="zip_code_locator")

    for index, row in result_df.iterrows():
        zip_code = row['Zip Code']
        try:
            print("Processing ZIP code:", zip_code)
            location = geolocator.geocode(f"{zip_code}, USA")
            if location:
                result_df.at[index, 'Latitude'] = location.latitude
                result_df.at[index, 'Longitude'] = location.longitude
            else:
                print(f"Location not found for zip code: {zip_code}")
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Geocoding error for zip code {zip_code}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for zip code {zip_code}: {e}")

    # Save the DataFrame to a CSV file
    try:
        result_df.to_csv(output_file_path, index=False)
        print(f"Results saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
    print(f"Nan count {nan_count}")
# Example usage:
input_file_path = 'Rat_Sightings.csv' 
output_file_path = 'rat_sightings_by_zipcode.csv'  
process_rat_sightings(input_file_path, output_file_path)
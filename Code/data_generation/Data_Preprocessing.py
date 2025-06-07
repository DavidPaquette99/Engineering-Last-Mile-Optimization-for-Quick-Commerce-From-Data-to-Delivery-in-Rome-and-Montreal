##### LIBRARIES #####

import pandas as pd
import re
import unicodedata
import googlemaps
import time
import folium
from geopy.geocoders import Nominatim

##### DATASET #####

# Restaurants
italy_data = pd.read_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Rome/glovo-foodi-ml-dataset.csv', index_col = 0)
canada_data = pd.read_csv('/Users/davidpaquette/Documents/Thesis/Project/data/Montreal/cleaned_full_data.csv', index_col = 0)

# Weather
rome_weather = pd.read_csv('/Users/davidpaquette/Documents/Thesis/Project/data/Rome/export.csv')
mtl_weather = pd.read_csv('/Users/davidpaquette/Documents/Thesis/Project/data/Montreal/weatherstats_montreal_daily.csv')

# Traffic
# Might use APIs instead

##### API #####

# Initialize Google Maps API
gmaps = googlemaps.Client(key="AIzaSyCVCIC2uaYM3cwEq7nLmJ4-B4gMGbTsde0")

##### PREPROCESSING - RESTAURANTS #####

### ITALY - ROME ###

# Filter for Italy
italy = italy_data[italy_data['country_code'] == 'IT']

# Filter for Rome
rome = italy[italy['city_code'] == 'ROM'].reset_index(drop = True)

# Dropping columns, and filtering out useless info
rome = rome[rome['HIER'] == True]

# Only need the stores, not the products, descriptions, etc.
rome = rome.drop(columns = ['product_name', 'collection_section', 'product_description' ,'subset', 'hash','aux_store', 'HIER', 's3_path'])

# Shape = (12775, 3)
rome.shape

# Check for nulls - None
rome_null = rome.isnull().sum()

# Check for duplicate restos - keep original names in seperate column for coordinate search
rome['original_store_name'] = rome['store_name']

# Normalize store names
def normalize_name(name):

    # Remove accents
    nfkd_form = unicodedata.normalize('NFKD', name)
    without_accents = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Remove special symbols (keep alphanumeric and spaces)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", without_accents)

    # Lowercase
    return cleaned.lower().strip()

# Apply normalization
rome['store_name'] = rome['store_name'].apply(normalize_name)

# Drop duplicates based on normalized names
rome = rome.drop_duplicates(subset = 'store_name', keep = 'first').reset_index(drop = True)

# Drop store_name, and change original_store_name back to store_name
rome = rome.drop(columns = 'store_name')
rome = rome.rename(columns = {'original_store_name' : 'store_name'})

# Shape = (447, 3)
rome.shape

# Geocode and collect results
coords = []
for name in rome['store_name']:
    try:
        query = f"{name}, Rome, Italy"
        result = gmaps.geocode(query)
        if result:
            lat = result[0]['geometry']['location']['lat']
            lon = result[0]['geometry']['location']['lng']
        else:
            lat, lon = None, None
        coords.append([lat, lon])
        time.sleep(0.1)  # Respect API rate limits

    except Exception as e:
        print(f"Error geocoding {name}: {e}")
        coords.append([None, None])
        time.sleep(1)

# Add coordinates to DataFrame
rome[['latitude', 'longitude']] = pd.DataFrame(coords, index = rome.index)

# Save final result
rome.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Rome/rome_geocoded.csv', index = False)

# Reimport CSV
rome_geocoded = pd.read_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Rome/rome_geocoded.csv')

# Only historical center

# Define bounding box limits
min_lat, max_lat = 41.890, 41.905
min_lon, max_lon = 12.460, 12.515

# Filter the restaurants inside the bounding box
rome_historical_center = rome_geocoded[
    (rome_geocoded['latitude'] >= min_lat) & (rome_geocoded['latitude'] <= max_lat) &
    (rome_geocoded['longitude'] >= min_lon) & (rome_geocoded['longitude'] <= max_lon)
]

# Find areas to better filter
geolocator = Nominatim(user_agent="rome_restaurants_locator")

def get_neighborhood(lat, lon):
    try:
        time.sleep(1)  # Still be polite to API
        location = geolocator.reverse((lat, lon), exactly_one=True, addressdetails=True)
        if location:
            address = location.raw['address']
            # Check multiple possible fields
            for field in ['neighbourhood', 'quarter', 'suburb', 'city_district', 'residential', 'municipality']:
                if field in address:
                    return address[field]
            # Fallback: use part of the display name
            return location.raw.get('display_name', '').split(',')[0]
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Geocode with progress bar
rome_historical_center['neighborhood'] = rome_historical_center.progress_apply(
    lambda row: get_neighborhood(row['latitude'], row['longitude']),
    axis=1
)

# Save CSV
rome_historical_center.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Rome/rome_historical_center.csv')

# List of desired neighborhoods
target_neighborhoods = ['Monti', 'Trevi', 'Campo Marzio', 'Trastevere', "Sant'Eustachio", 'Parione', 'Ponte', 'Colonna']

# Filter restaurants to only those in the target neighborhoods
rome_target = rome_historical_center[rome_historical_center['neighborhood'].isin(target_neighborhoods)].copy().reset_index(drop = True)

# Save CSV
rome_target.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Rome/rome_target.csv')

### CANADA - MONTREAL ###

# Filter for Montreal
montreal = canada_data[canada_data['city'] == 'Montreal'].reset_index(drop = True)

# Only need the stores, not the products, descriptions, etc.
montreal = montreal.drop(columns = ['url', 'distance', 'star' ,'num_reviews', 'price_range', 'category_1', 'category_2'])

# Shape = (568, 2)
montreal.shape

# Check for duplicate restos - keep original names in seperate column for coordinate search
montreal['original_restaurant_name'] = montreal['restaurant']

# Apply normalization
montreal['restaurant'] = montreal['restaurant'].apply(normalize_name)

# Drop duplicates based on normalized names
montreal = montreal.drop_duplicates(subset = 'restaurant', keep = 'first').reset_index(drop = True)

# Drop restaurant, and change original_restaurant_name back to restaurant_name
montreal = montreal.drop(columns = 'restaurant')
montreal = montreal.rename(columns = {'original_restaurant_name' : 'restaurant'})

# Geocode and collect results
coords = []
for name in montreal['restaurant']:
    try:
        query = f"{name}, Montreal, Canada"
        result = gmaps.geocode(query)
        if result:
            lat = result[0]['geometry']['location']['lat']
            lon = result[0]['geometry']['location']['lng']
        else:
            lat, lon = None, None
        coords.append([lat, lon])
        time.sleep(0.1)  # Respect API rate limits

    except Exception as e:
        print(f"Error geocoding {name}: {e}")
        coords.append([None, None])
        time.sleep(1)

# Add coordinates to DataFrame
montreal[['latitude', 'longitude']] = pd.DataFrame(coords, index = montreal.index)

# Save final result
montreal.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Montreal/montreal_geocoded.csv', index = False)

# Reimport CSV
montreal_geocoded = pd.read_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Montreal/montreal_geocoded.csv')

# Add neighborhoods to filter boroughs
geolocator = Nominatim(user_agent="montreal_restaurants_locator")

def get_montreal_neighborhood(lat, lon):
    try:
        time.sleep(1)  # Respect API limits
        location = geolocator.reverse((lat, lon), exactly_one=True, addressdetails=True)
        if location:
            address = location.raw['address']
            for field in ['neighbourhood', 'suburb', 'city_district', 'municipality']:
                if field in address:
                    return address[field]
            return location.raw.get('display_name', '').split(',')[0]
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

montreal_geocoded['neighborhood'] = montreal_geocoded.progress_apply(
    lambda row: get_montreal_neighborhood(row['latitude'], row['longitude']),
    axis=1
)

# Define target Montreal neighborhoods
target_neighborhoods_mtl = ['Quartier des Spectacles', 'Ville-Marie', 'Le Plateau-Mont-Royal', 'TOD quartier Berri UQAM',
                            'Saint-Henri', 'Mile-End', 'Quartier Chinois', 'Vieux-Montréal', 'Saint-Laurent', 'Le Sud-Ouest']

# Filter
montreal_target = montreal_geocoded[
    montreal_geocoded['neighborhood'].isin(target_neighborhoods_mtl)
].copy()

# Save CSV
montreal_target.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Montreal/montreal_target.csv', index = False)

# Randomly select 40 restaurants (473 is too many)
montreal_sampled = montreal_target.sample(n = 40, random_state = 42)  # random_state ensures reproducibility

# Save dataset
montreal_sampled.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Montreal/montreal_sampled.csv', index=False)

##### PREPROCESSING - WEATHER #####

### ROME ###

#Fill missing snow values with 0
rome_weather['snow'] = rome_weather['snow'].fillna(0)

# Drop unnecessary columns
rome_weather = rome_weather.drop(columns=['wdir', 'wspd','wpgt', 'pres','tsun'])

# Classify weather condition based on precipitation and snow
def classify_weather(row):
    if row['snow'] > 0:
        return 'snow'
    elif row['prcp'] > 0:
        return 'rain'
    else:
        return 'clear'

rome_weather['weather_condition'] = rome_weather.apply(classify_weather, axis = 1)

# Create binary flags for rain and snow
rome_weather['is_raining'] = (rome_weather['weather_condition'] == 'rain').astype(int)
rome_weather['is_snowing'] = (rome_weather['weather_condition'] == 'snow').astype(int)

# Parse the date column properly
rome_weather['date'] = pd.to_datetime(rome_weather['date'])

# Save cleaned dataset
rome_weather.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Rome/rome_weather_cleaned.csv', index=False)

### Montreal ###

# Select only columns needed
mtl_weather = mtl_weather[['date', 'avg_temperature', 'precipitation', 'snow']]

# Fill missing values
mtl_weather['precipitation'] = mtl_weather['precipitation'].fillna(0)
mtl_weather['snow'] = mtl_weather['snow'].fillna(0)

# Create weather condition
def classify_weather_mtl(row):
    if row['snow'] > 0:
        return 'snow'
    elif row['precipitation'] > 0:
        return 'rain'
    else:
        return 'clear'

# Apply same classifier as rome
mtl_weather['weather_condition'] = mtl_weather.apply(classify_weather_mtl, axis=1)

# Create binary flags
mtl_weather['is_raining'] = (mtl_weather['weather_condition'] == 'rain').astype(int)
mtl_weather['is_snowing'] = (mtl_weather['weather_condition'] == 'snow').astype(int)

# Parse dates properly
mtl_weather['date'] = pd.to_datetime(mtl_weather['date'])

# Save cleaned version
mtl_weather.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Montreal/montreal_weather_cleaned.csv', index=False)

##### PREPROCESSING - STANDARDIZE DATASET #####

### Restaurants

# Create ID variable for better referencing
rome_target['store_id'] = ['ROM' + str(i) for i in range(len(rome_target))]
montreal_sampled['store_id'] = ['MTL' + str(i) for i in range(len(montreal_sampled))]

# Add city column just in case
rome_target['city'] = 'Rome'
montreal_sampled['city'] = 'Montreal'

# Standardize Columns
final_columns = ['store_id', 'store_name', 'latitude', 'longitude', 'city', 'neighborhood']

montreal_sampled = montreal_sampled.rename(columns = {'restaurant' : 'store_name'})

rome_target = rome_target[final_columns]
montreal_sampled = montreal_sampled[final_columns].reset_index(drop = True)

# Save CSVs
rome_target.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Rome/rome_target.csv', index=False)
montreal_sampled.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Montreal/montreal_sampled.csv', index=False)

### Weather

# Drop cols not in MTL dataset
rome_weather = rome_weather.drop(columns=['tmin', 'tmax'])

# Rename cols
rome_weather = rome_weather.rename(columns={
    'tavg': 'temperature_celsius',
    'prcp': 'precipitation_mm',
    'snow': 'snowfall_cm'
})

mtl_weather = mtl_weather.rename(columns={
    'avg_temperature': 'temperature_celsius',
    'precipitation': 'precipitation_mm',
    'snow': 'snowfall_cm'
})

# Save Final CSVs
rome_weather.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Rome/rome_weather_cleaned.csv', index=False)
mtl_weather.to_csv('/Users/davidpaquette/Documents/Thesis/Project/Data/Montreal/montreal_weather_cleaned.csv', index=False)

##### PLOTTING #####

### ROME ###

# Center the map on Rome
rome_center = [41.9028, 12.4964]  # Approx center of Rome

# Create a base map
m = folium.Map(location = rome_center, zoom_start = 13)

# Add restaurant markers
for _, row in rome_geocoded.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=row['store_name']
    ).add_to(m)

# Save to HTML or display
m.save("rome_restaurants_map.html")

m = folium.Map(location = rome_center, zoom_start = 13)

# Add restaurant markers
for _, row in rome_target.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=row['store_name']
    ).add_to(m)

m.save("rome_target.html")

### MONTREAL

# Center the map on Montreal
montreal_center = [45.5019, -73.5674]  # Approx center of Montreal (Ville-Marie)

# Create base map
m = folium.Map(location = montreal_center, zoom_start = 13)

for _, row in montreal_geocoded.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=row['restaurant']
    ).add_to(m)

# Map with all restaurants
m.save("montreal_restaurants_map.html")

# Target Areas
m = folium.Map(location = montreal_center, zoom_start = 13)

for _, row in montreal_target.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=row['restaurant']
    ).add_to(m)

# Filtered map
m.save("montreal_target.html")

# Randomly Sampled Areas
m = folium.Map(location = montreal_center, zoom_start = 13)

for _, row in montreal_sampled.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=row['restaurant']
    ).add_to(m)

# Filtered map
m.save("montreal_sampled.html")

##### LIBRARIES #####

import math
import requests
import time
from itertools import cycle

##### DISTANCE FUNCTIONS #####

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points (haversine formula).
    Returns distance in kilometers.
    """
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

##### TRAVEL TIME FUNCTIONS #####

VEHICLE_SPEEDS = {'bike': 15, 'scooter': 30, 'car': 40}  # km/h

def travel_time_haversine(lat1, lon1, lat2, lon2, vehicle_type):
    dist_km = haversine(lat1, lon1, lat2, lon2)
    speed_kmh = VEHICLE_SPEEDS.get(vehicle_type, 15)
    return (dist_km / speed_kmh) * 60, dist_km  # minutes, km

def get_google_maps_travel_time(pickup_lat, pickup_lon, delivery_lat, delivery_lon, api_key, mode='driving'):
    """
    Query Google Maps Distance Matrix API for realistic travel time.
    Returns travel time in minutes.
    """
    origin = f"{pickup_lat},{pickup_lon}"
    destination = f"{delivery_lat},{delivery_lon}"
    url = (
        f"https://maps.googleapis.com/maps/api/distancematrix/json"
        f"?origins={origin}&destinations={destination}&mode={mode}&key={api_key}"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        element = data['rows'][0]['elements'][0]
        travel_time_sec = element['duration']['value']
        distance_m = element['distance']['value']

        travel_time_min = int(math.ceil(travel_time_sec / 60))
        distance_km = round(distance_m / 1000, 2)

        return travel_time_min, distance_km
    
    except Exception as e:
        print(f"⚠️ Google API failed: {e}")
        return None

route_cache = {}

def get_realistic_travel_time(pickup_lat, pickup_lon, delivery_lat, delivery_lon, mode, api_key):
    """
    Get realistic travel time using Google Maps API.
    Falls back to haversine if API fails.
    Caches routes to avoid redundant API calls.
    """

    # Create a unique key for this route
    key = (pickup_lat, pickup_lon, delivery_lat, delivery_lon, mode)

    # Check if result already in cache
    if key in route_cache:
        return route_cache[key]

    # Try Google Maps API up to 3 times
    tries = 3
    for attempt in range(tries):
        result = get_google_maps_travel_time(pickup_lat, pickup_lon, delivery_lat, delivery_lon, api_key, mode=mode)

        if result is not None:
            travel_time_min, distance_km = result
            route_cache[key] = (travel_time_min, distance_km)
            return travel_time_min, distance_km
        else:
            if attempt < tries - 1:
                print("Google API failed. Retrying...")
                time.sleep(2)
            else:
                print("Google API failed after retries. Using fallback.")

    # Fallback to haversine if all retries failed
    distance_km = haversine(pickup_lat, pickup_lon, delivery_lat, delivery_lon)

    if mode == 'bike':
        speed_kmh = 15
    elif mode == 'scooter':
        speed_kmh = 30
    else:
        speed_kmh = 40

    travel_time_min = (distance_km / speed_kmh) * 60

    route_cache[key] = (travel_time_min, distance_km)
    return travel_time_min, distance_km

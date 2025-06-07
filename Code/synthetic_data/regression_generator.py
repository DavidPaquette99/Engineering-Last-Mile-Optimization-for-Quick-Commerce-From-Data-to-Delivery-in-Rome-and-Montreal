##### LIBRARIES #####

import pandas as pd
import numpy as np
import random
import os
import time
from utils.Distance_Calculator import haversine

##### SETTINGS #####
SAVE_PATH = "/Users/davidpaquette/Documents/Thesis/Project/Data/ANFIS"
CITIES = ["Montreal", "Rome"]
SAMPLES_PER_CITY = 10000
random.seed(42)

##### CONFIG #####
city_bounds = {
    "Montreal": {"lat": (45.48, 45.55), "lon": (-73.6, -73.54)},
    "Rome": {"lat": (41.88, 41.91), "lon": (12.46, 12.52)}
}
vehicle_map = {"bike": 1, "scooter": 2, "car": 3}
vehicle_speeds = {"bike": 15, "scooter": 25, "car": 40}
weather_map = {"clear": 0, "rain": 1, "snow": 2}
period_map = {(6, 11): 0, (11, 14): 1, (17, 21): 2}

def get_period(hour):
    for (start, end), label in period_map.items():
        if start <= hour < end:
            return label
    return 3

def estimated_time_minutes(lat1, lon1, lat2, lon2, vehicle_type):
    distance_km = haversine(lat1, lon1, lat2, lon2)
    speed = vehicle_speeds.get(vehicle_type, 15)
    return (distance_km / speed) * 60

# NEW: Synthetic demand feature generators
def synthetic_local_demand():
    # Simulate plausible local demand in next 60 min within 1 km
    return random.randint(0, 5)

def synthetic_global_demand():
    # Simulate plausible global demand in next 60 min for whole city
    return random.randint(10, 50)

def generate_sample(city):
    lat_bounds = city_bounds[city]["lat"]
    lon_bounds = city_bounds[city]["lon"]

    order = {
        "pickup_lat": random.uniform(*lat_bounds),
        "pickup_lon": random.uniform(*lon_bounds),
        "delivery_lat": random.uniform(*lat_bounds),
        "delivery_lon": random.uniform(*lon_bounds),
        "prep_time": random.randint(5, 15),
        "size": random.randint(1, 3),
        "hour": random.randint(10, 22),
        "weather": random.choice(list(weather_map.keys()))
    }

    period = get_period(order["hour"])
    weather_code = weather_map[order["weather"]]

    courier_pool = []
    for _ in range(10):
        vehicle_type = random.choice(list(vehicle_map.keys()))
        vehicle_code = vehicle_map[vehicle_type]
        active_minutes = random.randint(0, 300)
        capacity = random.randint(1, 4)
        lat = random.uniform(*lat_bounds)
        lon = random.uniform(*lon_bounds)

        try:
            to_pickup = estimated_time_minutes(lat, lon, order["pickup_lat"], order["pickup_lon"], vehicle_type)
            to_delivery = estimated_time_minutes(order["pickup_lat"], order["pickup_lon"], order["delivery_lat"], order["delivery_lon"], vehicle_type)
        except Exception:
            return None

        # Add synthetic demand features
        local_demand = synthetic_local_demand()
        global_demand = synthetic_global_demand()

        features = [
            to_pickup, to_delivery, active_minutes, vehicle_code, weather_code,
            order["prep_time"], period, capacity, order["size"],
            local_demand, global_demand
        ]
        courier_pool.append((features, total_time := max(order["prep_time"], to_pickup) + to_delivery))

    if len(courier_pool) != 10:
        return None

    features_flat = np.array([feat for feat, _ in courier_pool]).flatten().tolist()
    target = min([t for _, t in courier_pool])
    return features_flat, target

# MAIN GENERATION
for city in CITIES:
    print(f"Generating {SAMPLES_PER_CITY} samples for {city}...")
    X_all, y_all = [], []
    attempts = 0

    while len(X_all) < SAMPLES_PER_CITY and attempts < SAMPLES_PER_CITY * 2:
        sample = generate_sample(city)
        if sample:
            X_all.append(sample[0])
            y_all.append(sample[1])
        attempts += 1
        time.sleep(0.001)

    # Now 11 features per courier × 10 couriers = 110 features
    df = pd.DataFrame(X_all, columns=[f"f{i+1}" for i in range(110)])
    df["target_time_min"] = y_all
    file_path = os.path.join(SAVE_PATH, f"{city}_regression_training_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} samples for {city} to {file_path}")

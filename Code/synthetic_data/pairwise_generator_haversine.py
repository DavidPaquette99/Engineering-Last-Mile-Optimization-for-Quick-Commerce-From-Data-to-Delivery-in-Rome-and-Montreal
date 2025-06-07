##### LIBRARIES #####

import pandas as pd
import numpy as np
import random
import os
from utils.Distance_Calculator import haversine

##### SETTINGS #####
SAVE_PATH = "/Users/davidpaquette/Documents/Thesis/Project/Data/ANFIS"
CITIES = ["Montreal", "Rome"]
SAMPLES_PER_CITY = 10000
random.seed(42)

##### CONFIG #####
city_bounds = {
    "Montreal": {"lat": (45.49, 45.54), "lon": (-73.59, -73.55), "center": (45.51, -73.57)},
    "Rome": {"lat": (41.890, 41.905), "lon": (12.460, 12.515), "center": (41.8975, 12.4875)}
}
vehicle_map = {"bike": 1, "scooter": 2, "car": 3}
vehicle_speeds = {"bike": 15, "scooter": 30, "car": 40}
weather_map = {"clear": 0, "rain": 1, "snow": 2}
weather_conditions = list(weather_map.keys())

def get_period(hour):
    if 6 <= hour < 11:
        return 0
    elif 11 <= hour < 14:
        return 1
    elif 17 <= hour < 21:
        return 2
    else:
        return 3

def estimated_time_minutes(km, vehicle_type):
    speed = vehicle_speeds.get(vehicle_type, 15)
    return (km / speed) * 60

def synthetic_local_demand():
    return random.randint(0, 5)

def synthetic_global_demand():
    return random.randint(10, 50)

def noisy(val, std_frac=0.05):
    std = std_frac * abs(val) if val != 0 else 0.01
    return float(val) + np.random.normal(0, std)

def generate_pairwise_haversine_samples(n_samples=10000, city="Montreal"):
    bounds = city_bounds[city]
    lat_range = bounds["lat"]
    lon_range = bounds["lon"]
    center = bounds["center"]

    X_all, y_all = [], []

    for _ in range(n_samples):
        hour = random.randint(8, 22)
        period = get_period(hour)
        weather = random.choice(weather_conditions)
        weather_code = weather_map[weather]
        prep_time = random.randint(5, 20)
        order_size = random.randint(1, 2)

        # Order locations
        pickup_lat = random.uniform(*lat_range)
        pickup_lon = random.uniform(*lon_range)
        delivery_lat = random.uniform(*lat_range)
        delivery_lon = random.uniform(*lon_range)

        # Courier info
        vehicle_type = random.choice(list(vehicle_map.keys()))
        vehicle_code = vehicle_map[vehicle_type]
        active_minutes = random.randint(0, 300)

        courier_lat = random.uniform(*lat_range)
        courier_lon = random.uniform(*lon_range)

        # Distance/time to new order pickup & delivery
        to_pickup_km = haversine(courier_lat, courier_lon, pickup_lat, pickup_lon)
        to_pickup_min = estimated_time_minutes(to_pickup_km, vehicle_type)
        to_delivery_km = haversine(pickup_lat, pickup_lon, delivery_lat, delivery_lon)
        to_delivery_min = estimated_time_minutes(to_delivery_km, vehicle_type)

        wait_time = max(0, prep_time - to_pickup_min)
        total_time = to_pickup_min + wait_time + to_delivery_min

        dist_to_center = haversine(courier_lat, courier_lon, *center)

        # Add noise to features
        to_pickup_min = noisy(to_pickup_min, 0.05)
        to_delivery_min = noisy(to_delivery_min, 0.05)
        active_minutes = noisy(active_minutes, 0.04)
        prep_time = noisy(prep_time, 0.04)

        local_demand = synthetic_local_demand()
        global_demand = synthetic_global_demand()

        # Features
        features = [
            to_pickup_min,
            to_delivery_min,
            active_minutes,
            vehicle_code,
            weather_code,
            prep_time,
            period,
            order_size,
            dist_to_center,
            local_demand,
            global_demand
        ]
        X_all.append(features)
        y_all.append(total_time)

    columns = [
        "to_pickup_min", "to_delivery_min", "active_minutes", "vehicle_code", "weather_code",
        "prep_time", "period", "order_size", "dist_to_center", "local_demand", "global_demand"
    ]
    df = pd.DataFrame(X_all, columns=columns)
    df["target_time_min"] = y_all
    return df

##### MAIN #####
for city in CITIES:
    print(f"Generating {SAMPLES_PER_CITY} haversine pairwise samples for {city} (no batching)...")
    df = generate_pairwise_haversine_samples(n_samples=SAMPLES_PER_CITY, city=city)
    file_path = os.path.join(SAVE_PATH, f"{city}_pairwise_training_data_haversine_nobatch.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} samples for {city} to {file_path}")

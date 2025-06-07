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

##### NOISE FUNCTION #####
def noisy(val, std_frac=0.05):
    std = std_frac * abs(val) if val != 0 else 0.01
    return float(val) + np.random.normal(0, std)

def get_period(hour):
    for (start, end), label in period_map.items():
        if start <= hour < end:
            return label
    return 3

def estimated_time_minutes(lat1, lon1, lat2, lon2, vehicle_type):
    distance_km = haversine(lat1, lon1, lat2, lon2)
    speed = vehicle_speeds.get(vehicle_type, 15)
    return (distance_km / speed) * 60, distance_km

def synthetic_local_demand():
    return random.randint(0, 5)

def synthetic_global_demand():
    return random.randint(10, 50)

def generate_sample(city):
    lat_bounds = city_bounds[city]["lat"]
    lon_bounds = city_bounds[city]["lon"]

    # Candidate order to assign
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

    vehicle_type = random.choice(list(vehicle_map.keys()))
    vehicle_code = vehicle_map[vehicle_type]
    active_minutes = random.randint(0, 300)
    courier_lat = random.uniform(*lat_bounds)
    courier_lon = random.uniform(*lon_bounds)

    # Compute haversine times/distances with noise
    try:
        to_pickup_time, to_pickup_dist = estimated_time_minutes(
            courier_lat, courier_lon, order["pickup_lat"], order["pickup_lon"], vehicle_type)
        to_delivery_time, to_delivery_dist = estimated_time_minutes(
            order["pickup_lat"], order["pickup_lon"], order["delivery_lat"], order["delivery_lon"], vehicle_type)
    except Exception:
        return None

    # Add noise to features
    to_pickup_time = noisy(to_pickup_time, 0.05)
    to_delivery_time = noisy(to_delivery_time, 0.05)
    active_minutes = noisy(active_minutes, 0.04)
    prep_time = noisy(order["prep_time"], 0.04)
    local_demand = synthetic_local_demand()
    global_demand = synthetic_global_demand()

    # Feature vector 
    features = [
        to_pickup_time,      # 1
        to_delivery_time,    # 2
        active_minutes,      # 3
        vehicle_code,        # 4
        weather_code,        # 5
        prep_time,           # 6
        period,              # 7
        order["size"],       # 8
        local_demand,        # 9
        global_demand        # 10
    ]

    total_time = max(prep_time, to_pickup_time) + to_delivery_time
    target = noisy(total_time, 0.07)  # Slightly more noise on the label

    return features, target

# MAIN GENERATION
columns = [
    "to_pickup_time", "to_delivery_time", "active_minutes", "vehicle_code", "weather_code",
    "prep_time", "period", "order_size", "local_demand", "global_demand"
]
for city in CITIES:
    print(f"Generating {SAMPLES_PER_CITY} samples for {city} (haversine, no batching, with noise)...")
    X_all, y_all = [], []
    attempts = 0

    while len(X_all) < SAMPLES_PER_CITY and attempts < SAMPLES_PER_CITY * 2:
        sample = generate_sample(city)
        if sample:
            X_all.append(sample[0])
            y_all.append(sample[1])
        attempts += 1
        time.sleep(0.0005)

    df = pd.DataFrame(X_all, columns=columns)
    df["target_time_min"] = y_all
    file_path = os.path.join(SAVE_PATH, f"{city}_regression_training_data_haversine_nobatch.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} samples for {city} to {file_path}")


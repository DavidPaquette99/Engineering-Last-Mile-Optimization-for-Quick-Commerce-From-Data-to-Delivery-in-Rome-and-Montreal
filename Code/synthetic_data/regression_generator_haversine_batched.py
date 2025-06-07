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
    return (distance_km / speed) * 60, distance_km

def synthetic_local_demand():
    return random.randint(0, 5)

def synthetic_global_demand():
    return random.randint(10, 50)

def generate_batch_context(capacity, lat_bounds, lon_bounds, vehicle_type, pickup_lat, pickup_lon):
    """Randomly generate a batch of up to (capacity-1) orders already assigned to courier."""
    batch_size = random.randint(0, capacity - 1) if capacity > 1 else 0
    batch_orders = []
    total_batch_distance = 0.0
    total_batch_time = 0.0
    last_lat, last_lon = None, None
    for _ in range(batch_size):
        order_pickup_lat = random.uniform(*lat_bounds)
        order_pickup_lon = random.uniform(*lon_bounds)
        order_delivery_lat = random.uniform(*lat_bounds)
        order_delivery_lon = random.uniform(*lon_bounds)
        _, to_pickup_dist = estimated_time_minutes(
            pickup_lat if last_lat is None else last_lat,
            pickup_lon if last_lon is None else last_lon,
            order_pickup_lat, order_pickup_lon, vehicle_type)
        time_to_pickup, _ = estimated_time_minutes(
            pickup_lat if last_lat is None else last_lat,
            pickup_lon if last_lon is None else last_lon,
            order_pickup_lat, order_pickup_lon, vehicle_type)
        time_to_delivery, _ = estimated_time_minutes(
            order_pickup_lat, order_pickup_lon, order_delivery_lat, order_delivery_lon, vehicle_type)
        batch_orders.append({
            "pickup_lat": order_pickup_lat,
            "pickup_lon": order_pickup_lon,
            "delivery_lat": order_delivery_lat,
            "delivery_lon": order_delivery_lon,
            "size": random.randint(1, 3),
            "prep_time": random.randint(5, 15)
        })
        total_batch_distance += to_pickup_dist
        total_batch_time += time_to_pickup + time_to_delivery
        last_lat, last_lon = order_delivery_lat, order_delivery_lon
    return batch_orders, batch_size, total_batch_distance, total_batch_time

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
    capacity = random.randint(1, 4)
    available_capacity = capacity - order["size"]
    courier_lat = random.uniform(*lat_bounds)
    courier_lon = random.uniform(*lon_bounds)

    # Create batch context (simulate already having up to capacity-1 orders)
    batch_orders, batch_size, batch_dist, batch_time = generate_batch_context(
        capacity, lat_bounds, lon_bounds, vehicle_type, order["pickup_lat"], order["pickup_lon"]
    )

    # Time/distance from courier's location to this new order's pickup
    try:
        to_pickup_time, to_pickup_dist = estimated_time_minutes(
            courier_lat, courier_lon, order["pickup_lat"], order["pickup_lon"], vehicle_type)
        to_delivery_time, to_delivery_dist = estimated_time_minutes(
            order["pickup_lat"], order["pickup_lon"], order["delivery_lat"], order["delivery_lon"], vehicle_type)
    except Exception:
        return None

    # Demand features
    local_demand = synthetic_local_demand()
    global_demand = synthetic_global_demand()

    # --- Batch-aware features ---
    features = [
        to_pickup_time,  
        to_delivery_time,  
        active_minutes,
        vehicle_code,
        weather_code,
        order["prep_time"],
        period,
        capacity,
        order["size"],
        available_capacity,
        batch_size,
        batch_time,     
        batch_dist,     
        local_demand,
        global_demand
    ]
    
    candidate_total_time = max(
        batch_time, max(order["prep_time"], to_pickup_time) + to_delivery_time
    )

    noise = np.random.normal(loc=0, scale=0.5)  # mean=0, std=0.5min 
    target = candidate_total_time + noise


    return features, target

# MAIN GENERATION
feature_names = [
    "to_pickup_time", "to_delivery_time", "active_minutes", "vehicle_code", "weather_code",
    "prep_time", "period", "capacity", "order_size", "available_capacity",
    "batch_size", "batch_time", "batch_distance", "local_demand", "global_demand"
]
for city in CITIES:
    print(f"Generating {SAMPLES_PER_CITY} samples for {city} (batch-aware)...")
    X_all, y_all = [], []
    attempts = 0

    while len(X_all) < SAMPLES_PER_CITY and attempts < SAMPLES_PER_CITY * 2:
        sample = generate_sample(city)
        if sample:
            X_all.append(sample[0])
            y_all.append(sample[1])
        attempts += 1
        time.sleep(0.001)

    df = pd.DataFrame(X_all, columns=feature_names)
    df["target_time_min"] = y_all
    file_path = os.path.join(SAVE_PATH, f"{city}_regression_training_data_haversine_batch.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} samples for {city} to {file_path}")

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

def generate_batch_context(capacity, lat_range, lon_range, vehicle_type, courier_lat, courier_lon):
    """
    Randomly generate a batch of up to (capacity-1) orders already assigned to courier.
    """
    batch_size = random.randint(0, capacity - 1) if capacity > 1 else 0
    batch_orders = []
    total_batch_distance = 0.0
    total_batch_time = 0.0
    last_lat, last_lon = courier_lat, courier_lon
    for _ in range(batch_size):
        order_pickup_lat = random.uniform(*lat_range)
        order_pickup_lon = random.uniform(*lon_range)
        order_delivery_lat = random.uniform(*lat_range)
        order_delivery_lon = random.uniform(*lon_range)
        # From last point to next pickup, then to delivery
        to_pickup_km = haversine(last_lat, last_lon, order_pickup_lat, order_pickup_lon)
        to_pickup_min = estimated_time_minutes(to_pickup_km, vehicle_type)
        to_delivery_km = haversine(order_pickup_lat, order_pickup_lon, order_delivery_lat, order_delivery_lon)
        to_delivery_min = estimated_time_minutes(to_delivery_km, vehicle_type)
        batch_orders.append({
            "pickup_lat": order_pickup_lat,
            "pickup_lon": order_pickup_lon,
            "delivery_lat": order_delivery_lat,
            "delivery_lon": order_delivery_lon,
            "size": random.randint(1, 2),
            "prep_time": random.randint(5, 20)
        })
        total_batch_distance += to_pickup_km + to_delivery_km
        total_batch_time += to_pickup_min + to_delivery_min
        last_lat, last_lon = order_delivery_lat, order_delivery_lon
    return batch_orders, batch_size, total_batch_distance, total_batch_time

def generate_pairwise_samples(n_samples=10000, city="Montreal"):

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
        idle_minutes = random.randint(0, 60)
        capacity = random.randint(1, 3)

        if order_size > capacity:
            continue  # skip and regenerate sample (start next loop iteration)
        available_capacity = random.randint(order_size, capacity)

        courier_lat = random.uniform(*lat_range)
        courier_lon = random.uniform(*lon_range)

        # Create batch context (simulate already having up to capacity-1 orders)
        batch_orders, batch_size, batch_dist, batch_time = generate_batch_context(
            capacity, lat_range, lon_range, vehicle_type, courier_lat, courier_lon
        )

        # Distance/time to new order pickup & delivery
        to_pickup_km = haversine(courier_lat, courier_lon, pickup_lat, pickup_lon)
        to_pickup_min = estimated_time_minutes(to_pickup_km, vehicle_type)
        to_delivery_km = haversine(pickup_lat, pickup_lon, delivery_lat, delivery_lon)
        to_delivery_min = estimated_time_minutes(to_delivery_km, vehicle_type)

        wait_time = max(0, prep_time - to_pickup_min)
        noise = np.random.normal(0, 1.0)
        total_time = max(batch_time, to_pickup_min + wait_time + to_delivery_min) + noise

        # Synthetic demand forecast features
        local_demand = synthetic_local_demand()
        global_demand = synthetic_global_demand()

        features = [
            to_pickup_min,         
            to_delivery_min,       
            active_minutes,        
            vehicle_code,          
            weather_code,          
            prep_time,             
            period,                
            capacity,              
            available_capacity,    
            order_size,            
            batch_size,            
            batch_time,            
            batch_dist,            
            local_demand,          
            global_demand          
        ]


        X_all.append(features)
        y_all.append(total_time)

    columns = [
        "to_pickup_min", "to_delivery_min", "active_minutes", "vehicle_code", "weather_code",
        "prep_time", "period", "capacity", "available_capacity", "order_size",
        "batch_size", "batch_time", "batch_dist", "local_demand", "global_demand"
    ]
    df = pd.DataFrame(X_all, columns=columns)
    df["target_time_min"] = y_all

    os.makedirs(SAVE_PATH, exist_ok=True)
    df.to_csv(f"{SAVE_PATH}/{city}_pairwise_training_data_haversine_batch.csv", index=False)
    print(f"✅ Saved {n_samples} batch-aware pairwise samples for {city} to {SAVE_PATH}/{city}_pairwise_training_data_haversine_batch.csv")

    return df

##### MAIN #####

if __name__ == "__main__":
    for city in CITIES:
        generate_pairwise_samples(n_samples=SAMPLES_PER_CITY, city=city)

##### LIBRARIES #####

import numpy as np
import pickle
import os
import joblib
import uuid
from utils.Distance_Calculator import haversine, travel_time_haversine

##### CONFIG #####

base_path = "/Users/davidpaquette/Documents/Thesis/Project/Data/ANFIS"

def load_anfis_regression_model(city, base_path):
    model_path = os.path.join(base_path, f"{city}_anfis_model_haversine_batched.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler_X = joblib.load(os.path.join(base_path, f"{city}_scaler_X_haversine_batched.pkl"))
    scaler_y = joblib.load(os.path.join(base_path, f"{city}_scaler_y_haversine_batched.pkl"))
    return model, scaler_X, scaler_y

##### DEMAND FORECAST FEATURE HELPERS #####

def forecast_local_demand(orders, courier, current_minute, window=180, radius_km=0.2):
    count = 0
    for o in orders:
        if current_minute - 30 <= o.order_time <= current_minute + window:
            d = haversine(courier.latitude, courier.longitude, o.pickup_lat, o.pickup_lon)
            if d <= radius_km:
                count += 1
    return count

def forecast_global_demand(orders, current_minute, window=180):
    return sum(1 for o in orders if current_minute - 30 <= o.order_time <= current_minute + window)

##### FEATURES #####

def extract_features(order, courier, current_minute, weather_code, period_of_day, all_orders):
    tp, _ = travel_time_haversine(
        courier.latitude, courier.longitude,
        order.pickup_lat, order.pickup_lon,
        courier.vehicle_type
    )
    td, _ = travel_time_haversine(
        order.pickup_lat, order.pickup_lon,
        order.delivery_lat, order.delivery_lon,
        courier.vehicle_type
    )
    vehicle_map = {'bike': 1, 'scooter': 2, 'car': 3}
    vehicle_code = vehicle_map.get(courier.vehicle_type, 1)

    available_capacity = getattr(courier, "available_capacity", courier.capacity)
    batch_size = getattr(courier, "batch_size", 0)
    batch_time = getattr(courier, "batch_time", 0.0)
    batch_distance = getattr(courier, "batch_dist", 0.0)
    local_demand = forecast_local_demand(all_orders, courier, current_minute)
    global_demand = forecast_global_demand(all_orders, current_minute)

    return [
        tp,              
        td,              
        courier.active_minutes,
        vehicle_code,
        weather_code,
        order.prep_wait_time,
        period_of_day,
        courier.capacity,
        order.order_size,
        available_capacity,
        batch_size,
        batch_time,
        batch_distance,
        local_demand,
        global_demand
    ]


def get_period_of_day(current_minute):
    if  6 * 60 <= current_minute < 11 * 60: return 0
    if 11 * 60 <= current_minute < 14 * 60: return 1
    if 17 * 60 <= current_minute < 21 * 60: return 2
    return 3

def get_weather_code(weather_condition):
    return {'clear': 0, 'rain': 1, 'snow': 2}.get(weather_condition.lower(), 0)

##### ANFIS ASSIGNMENT WITH BATCH TRACKING AND BONUS #####
def anfis_assignment(
        available_couriers, new_orders, current_minute, weather_condition, city,
        model, scaler_X, scaler_y, all_orders, batching_enabled=True,
        batching_bonus=0.90  # 10% reduction in latency for batching
    ):
    """
    Assign orders to couriers based on ANFIS regression predictions of delivery latency,
    with batching and explicit tracking of batched orders (with batch_id).
    Batching is *encouraged* via a latency bonus for batchable couriers.
    """

    weather_code = get_weather_code(weather_condition)
    period_of_day = get_period_of_day(current_minute)
    completed_orders = []

    # Ensure all couriers have batch tracking attributes
    for c in available_couriers:
        if not hasattr(c, 'batched_orders'):
            c.batched_orders = []
        if not hasattr(c, 'current_batch_id'):
            c.current_batch_id = None
        if not hasattr(c, 'last_pickup'):
            c.last_pickup = None

    for order in list(new_orders):
        if order.assigned or order.order_time > current_minute:
            continue

        # --------- Prioritize batchable couriers FIRST ------------
        batchable_couriers = [
            c for c in available_couriers
            if c.available_capacity >= order.order_size and
               haversine(c.latitude, c.longitude, order.pickup_lat, order.pickup_lon) < 0.2
        ]
        idle_couriers = [
            c for c in available_couriers
            if c.status == 'idle' and c.available_at <= current_minute and c.available_capacity >= order.order_size
        ]
        eligible = batchable_couriers if batchable_couriers else idle_couriers
        if not eligible:
            continue

        preds = []
        for c in eligible:
            feats = np.array(
                extract_features(order, c, current_minute, weather_code, period_of_day, all_orders)
            ).reshape(1, -1)
            X_scaled = scaler_X.transform(feats)
            y_s = model.predict(X_scaled)
            latency = scaler_y.inverse_transform(y_s.reshape(-1, 1)).flatten()[0]
            # Apply batching bonus if at pickup and already batching
            at_pickup = haversine(c.latitude, c.longitude, order.pickup_lat, order.pickup_lon) < 0.2
            if at_pickup and len(c.batched_orders) > 0:
                latency *= batching_bonus
            preds.append((c, latency))

        # Sort by latency (bonus applied for batching)
        preds.sort(key=lambda x: x[1])
        selected, _ = preds[0]

        at_pickup = haversine(selected.latitude, selected.longitude, order.pickup_lat, order.pickup_lon) < 0.2

        if at_pickup:
            # Start a new batch if new pickup location
            if selected.last_pickup != (order.pickup_lat, order.pickup_lon):
                selected.current_batch_id = str(uuid.uuid4())
                selected.last_pickup = (order.pickup_lat, order.pickup_lon)
                selected.batched_orders = []
            order.batched = True
            order.batch_id = selected.current_batch_id
            selected.batched_orders.append(order)
        else:
            order.batched = False
            order.batch_id = None

        # Compute actual times/distances
        if at_pickup:
            tp, d1 = 0.0, 0.0  # already at pickup, zero time/distance
        else:
            tp, d1 = travel_time_haversine(
                selected.latitude, selected.longitude,
                order.pickup_lat, order.pickup_lon,
                selected.vehicle_type
            )
        td, d2 = travel_time_haversine(
            order.pickup_lat, order.pickup_lon,
            order.delivery_lat, order.delivery_lon,
            selected.vehicle_type
        )
        total_time = tp + max(0, order.prep_wait_time - tp) + td
        total_km = d1 + d2

        # Update courier (batch attributes if batching)
        selected.latitude = order.delivery_lat
        selected.longitude = order.delivery_lon
        selected.total_deliveries += 1
        selected.total_distance_km += total_km
        selected.active_minutes += total_time
        selected.available_at = current_minute + total_time
        selected.available_capacity -= order.order_size
        if at_pickup:
            selected.batch_size = getattr(selected, 'batch_size', 0) + 1
            selected.batch_time = getattr(selected, 'batch_time', 0.0) + total_time
            selected.batch_dist = getattr(selected, 'batch_dist', 0.0) + total_km
        else:
            selected.batch_size = 1
            selected.batch_time = total_time
            selected.batch_dist = total_km
        if not hasattr(selected, 'assigned_orders'):
            selected.assigned_orders = []
        selected.assigned_orders.append(order)

        # Standardized wait time calculation (realistic!)
        courier_arrival_time = current_minute + tp
        order_ready_time = order.order_time + order.prep_wait_time
        wait_time = max(0, courier_arrival_time - order_ready_time)
        order.wait_time = wait_time

        # Mark order as completed
        order.assigned = True
        order.assigned_courier_id = selected.courier_id
        completed_orders.append(order)

        # Only remove courier if not batching at pickup
        if not at_pickup:
            available_couriers.remove(selected)
        # If batching, courier remains available at that pickup for further batch assignments

    return completed_orders, available_couriers

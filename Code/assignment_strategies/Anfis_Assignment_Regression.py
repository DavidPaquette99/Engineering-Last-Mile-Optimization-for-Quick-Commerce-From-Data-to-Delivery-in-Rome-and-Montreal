##### LIBRARIES #####

import numpy as np
import pickle
import os
import joblib
from utils.Distance_Calculator import haversine, get_realistic_travel_time

##### CONFIG #####

base_path = "/Users/davidpaquette/Documents/Thesis/Project/Data/ANFIS"

def load_anfis_regression_model(city, base_path):
    model_path = os.path.join(base_path, f"{city}_anfis_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler_X = joblib.load(os.path.join(base_path, f"{city}_scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(base_path, f"{city}_scaler_y.pkl"))
    pca = joblib.load(os.path.join(base_path, f"{city}_pca.pkl"))
    return model, scaler_X, scaler_y, pca

##### DEMAND FORECAST FEATURE HELPERS #####

def forecast_local_demand(orders, courier, current_minute, window=60, radius_km=1.0):
    count = 0
    for o in orders:
        if current_minute < o.order_time <= current_minute + window:
            d = haversine(courier.latitude, courier.longitude, o.pickup_lat, o.pickup_lon)
            if d <= radius_km:
                count += 1
    return count

def forecast_global_demand(orders, current_minute, window=60):
    return sum(1 for o in orders if current_minute < o.order_time <= current_minute + window)

##### FEATURES #####

def extract_features(order, courier, current_minute, weather_code, period_of_day, api_key, all_orders):
    tp, _ = get_realistic_travel_time(
        courier.latitude, courier.longitude,
        order.pickup_lat, order.pickup_lon,
        mode=courier.vehicle_type,
        api_key=api_key
    )
    td, _ = get_realistic_travel_time(
        order.pickup_lat, order.pickup_lon,
        order.delivery_lat, order.delivery_lon,
        mode=courier.vehicle_type,
        api_key=api_key
    )
    vehicle_map = {'bike': 1, 'scooter': 2, 'car': 3}
    vehicle_code = vehicle_map.get(courier.vehicle_type, 1)

    local_demand = forecast_local_demand(all_orders, courier, current_minute, window=60, radius_km=1.0)
    global_demand = forecast_global_demand(all_orders, current_minute, window=60)

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

##### ANFIS ASSIGNMENT #####

def anfis_assignment(
        available_couriers, new_orders, current_minute, weather_condition, city, api_key,
        model, scaler_X, scaler_y, pca, all_orders):
    """
    Assign orders to couriers based on ANFIS regression predictions of delivery latency.
    """

    weather_code = get_weather_code(weather_condition)
    period_of_day = get_period_of_day(current_minute)
    completed_orders = []

    for order in list(new_orders):
        if order.assigned or order.order_time > current_minute:
            continue

        wait = current_minute - order.order_time

        eligible = [
            c for c in available_couriers
            if c.available_at <= current_minute and c.available_capacity >= order.order_size
        ]
        if not eligible:
            continue

        # predict latency for each eligible courier
        preds = []
        for c in eligible:
            feats = np.array(
                extract_features(order, c, current_minute, weather_code, period_of_day, api_key, all_orders)
            ).reshape(1, -1)
            feats10 = np.tile(feats, (1, 10))
            X_scaled = scaler_X.transform(feats10)
            X_pca = pca.transform(X_scaled)
            y_s = model.predict(X_pca)
            latency = scaler_y.inverse_transform(y_s.reshape(-1, 1)).flatten()[0]
            preds.append((c, latency))

        preds.sort(key=lambda x: x[1])
        selected, _ = preds[0]

        # actual times/distances
        tp, d1 = get_realistic_travel_time(
            selected.latitude, selected.longitude,
            order.pickup_lat, order.pickup_lon,
            selected.vehicle_type, api_key
        )
        td, d2 = get_realistic_travel_time(
            order.pickup_lat, order.pickup_lon,
            order.delivery_lat, order.delivery_lon,
            selected.vehicle_type, api_key
        )
        wait_time = max(0, order.prep_wait_time - tp)
        total_time = tp + wait_time + td
        total_km = d1 + d2

        # update courier
        selected.latitude = order.delivery_lat
        selected.longitude = order.delivery_lon
        selected.total_deliveries += 1
        selected.total_distance_km += total_km
        selected.active_minutes += total_time
        selected.available_at = current_minute + total_time
        selected.available_capacity -= order.order_size
        selected.assigned_orders.append(order)

        # mark order
        order.assigned = True
        order.assigned_courier_id = selected.courier_id
        order.wait_time = wait
        completed_orders.append(order)

        available_couriers.remove(selected)

    return completed_orders, available_couriers

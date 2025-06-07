##### LIBRARIES #####
import numpy as np
import pickle
import os
import joblib
import uuid
from datetime import datetime
from utils.Distance_Calculator import haversine, travel_time_haversine
from scipy.optimize import linear_sum_assignment

##### CONFIG #####
base_path = "/Users/davidpaquette/Documents/Thesis/Project/Data/ANFIS"

##### LOAD MODEL + TRANSFORMERS #####
def load_pairwise_model(city, base_path):
    model_path = os.path.join(base_path, f"{city}_pairwise_model_haversine_batched.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler_X = joblib.load(os.path.join(base_path, f"{city}_pairwise_scaler_X_haversine_batched.pkl"))
    scaler_y = joblib.load(os.path.join(base_path, f"{city}_pairwise_scaler_y_haversine_batched.pkl"))
    return model, scaler_X, scaler_y

##### ADAPTIVE ALPHA #####
def get_adaptive_alpha(weather_condition, day_of_week, hour):
    alpha = 0.6
    if weather_condition in ['rain', 'snow']:
        alpha += 0.1
    if 18 <= hour <= 22:
        alpha += 0.1
    if day_of_week in ['Friday', 'Saturday', 'Sunday']:
        alpha += 0.1
    return alpha

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

##### FEATURE EXTRACTION #####
def extract_pairwise_features(order, courier, weather_code, period, city_center, all_orders, current_minute):
    vcode = {"bike":1, "scooter":2, "car":3}[courier.vehicle_type]
    speed = {1:15, 2:30, 3:40}[vcode]
    km1 = haversine(courier.latitude, courier.longitude, order.pickup_lat, order.pickup_lon)
    km2 = haversine(order.pickup_lat, order.pickup_lon, order.delivery_lat, order.delivery_lon)
    to_pickup = (km1 / speed) * 60
    to_delivery = (km2 / speed) * 60
    active = courier.active_minutes
    vehicle_code = vcode
    cap = courier.capacity
    avail = courier.available_capacity
    order_size = order.order_size
    batch_size = getattr(courier, 'batch_size', 0)
    batch_time = getattr(courier, 'batch_time', 0.0)
    batch_dist = getattr(courier, 'batch_dist', 0.0)
    local_demand = forecast_local_demand(all_orders, courier, current_minute)
    global_demand = forecast_global_demand(all_orders, current_minute)

    return [
        to_pickup,         # 1
        to_delivery,       # 2
        active,            # 3
        vehicle_code,      # 4
        weather_code,      # 5
        order.prep_wait_time, # 6
        period,            # 7
        cap,               # 8
        order_size,        # 9
        avail,             # 10
        batch_size,        # 11
        batch_time,        # 12
        batch_dist,        # 13
        local_demand,      # 14
        global_demand      # 15
    ]

##### MAIN ASSIGNMENT #####
def pairwise_assignment(
    couriers, orders, current_minute, weather_condition, city,
    model, scaler_X, scaler_y, city_center, simulation_date_str, all_orders, batching_enabled=True,
    batching_bonus=0.90   # 10% latency reduction for batching at pickup
):
    completed_orders = []
    used_ids = set()
    weather_map = {"clear": 0, "rain": 1, "snow": 2}
    wcode = weather_map.get(weather_condition.lower(), 0)

    dt = datetime.strptime(simulation_date_str, "%Y-%m-%d") if simulation_date_str else datetime.now()
    day_of_week = dt.strftime("%A")
    hour = current_minute // 60

    # Ensure batch tracking attributes
    for c in couriers:
        if not hasattr(c, 'batched_orders'):
            c.batched_orders = []
        if not hasattr(c, 'current_batch_id'):
            c.current_batch_id = None
        if not hasattr(c, 'last_pickup'):
            c.last_pickup = None

    ready_orders = [o for o in orders if not o.assigned and o.order_time <= current_minute]
    if not ready_orders:
        return completed_orders, couriers

    # Build cost matrix (orders x couriers)
    eligible_couriers = [c for c in couriers if c.available_at <= current_minute and c.available_capacity > 0 and c.courier_id not in used_ids]
    if not eligible_couriers:
        return completed_orders, couriers

    cost_matrix = np.full((len(ready_orders), len(eligible_couriers)), np.inf)
    for i, order in enumerate(ready_orders):
        for j, courier in enumerate(eligible_couriers):
            if courier.available_capacity < order.order_size:
                continue  # infeasible

            at_pickup = batching_enabled and haversine(courier.latitude, courier.longitude, order.pickup_lat, order.pickup_lon) < 0.2
            period = get_period(order.order_time)
            feats = extract_pairwise_features(
                order, courier, wcode, period, city_center, all_orders, current_minute
            )
            X = np.array(feats).reshape(1, -1)
            X_scaled = scaler_X.transform(X)
            y_scaled = model.predict(X_scaled)
            latency = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()[0]
            # Batching bonus: apply reduction if at pickup and already batching
            if at_pickup and len(courier.batched_orders) > 0:
                latency *= batching_bonus

            # Cost calculation as before
            if at_pickup:
                distance_km = 0.0
            else:
                _, d1_cand = travel_time_haversine(
                    courier.latitude, courier.longitude,
                    order.pickup_lat, order.pickup_lon,
                    courier.vehicle_type
                )
                _, d2_cand = travel_time_haversine(
                    order.pickup_lat, order.pickup_lon,
                    order.delivery_lat, order.delivery_lon,
                    courier.vehicle_type
                )
                distance_km = d1_cand + d2_cand

            if city.lower() == 'rome':
                score = latency
            else:
                cost_per_min = (16.10 * 1.2) / 60
                cost_per_km = 0.22
                bonus = 1.10 if weather_condition in ['rain', 'snow'] else 1.0
                alpha = get_adaptive_alpha(weather_condition, day_of_week, hour)
                predicted_cost = bonus * (distance_km * cost_per_km + latency * cost_per_min)
                score = alpha * latency + (1 - alpha) * predicted_cost

            cost_matrix[i, j] = score

    order_idxs, courier_idxs = linear_sum_assignment(cost_matrix)

    for oi, cj in zip(order_idxs, courier_idxs):
        if cost_matrix[oi, cj] == np.inf:
            continue  # Skip infeasible matches

        order = ready_orders[oi]
        courier = eligible_couriers[cj]
        at_pickup = batching_enabled and haversine(courier.latitude, courier.longitude, order.pickup_lat, order.pickup_lon) < 0.2

        # Batching logic and tracking
        if at_pickup:
            if courier.last_pickup != (order.pickup_lat, order.pickup_lon):
                courier.current_batch_id = str(uuid.uuid4())
                courier.last_pickup = (order.pickup_lat, order.pickup_lon)
                courier.batched_orders = []
            order.batched = True
            order.batch_id = courier.current_batch_id
            courier.batched_orders.append(order)
        else:
            order.batched = False
            order.batch_id = None

        if at_pickup:
            tp, d1 = 0.0, 0.0
        else:
            tp, d1 = travel_time_haversine(
                courier.latitude, courier.longitude,
                order.pickup_lat, order.pickup_lon,
                courier.vehicle_type
            )
        td, d2 = travel_time_haversine(
            order.pickup_lat, order.pickup_lon,
            order.delivery_lat, order.delivery_lon,
            courier.vehicle_type
        )

        total_time = tp + max(0, order.prep_wait_time - tp) + td
        total_km = d1 + d2

        courier.latitude = order.delivery_lat
        courier.longitude = order.delivery_lon
        courier.total_deliveries += 1
        courier.total_distance_km += total_km
        courier.active_minutes += total_time
        courier.available_at = current_minute + total_time
        courier.available_capacity -= order.order_size
        if not hasattr(courier, 'assigned_orders'):
            courier.assigned_orders = []
        courier.assigned_orders.append(order)
        if at_pickup:
            courier.batch_size = getattr(courier, 'batch_size', 0) + 1
            courier.batch_time = getattr(courier, 'batch_time', 0.0) + total_time
            courier.batch_dist = getattr(courier, 'batch_dist', 0.0) + total_km
        else:
            courier.batch_size = 1
            courier.batch_time = total_time
            courier.batch_dist = total_km

        # Standardized wait time calculation (realistic!)
        courier_arrival_time = current_minute + tp
        order_ready_time = order.order_time + order.prep_wait_time
        wait_time = max(0, courier_arrival_time - order_ready_time)
        order.wait_time = wait_time

        order.assigned = True
        order.assigned_courier_id = courier.courier_id
        completed_orders.append(order)
        used_ids.add(courier.courier_id)

    return completed_orders, couriers

# Helper for period
def get_period(minute):
    hour = minute // 60
    if 6 <= hour < 11:
        return 0
    if 11 <= hour < 14:
        return 1
    if 17 <= hour < 21:
        return 2
    return 3

##### LIBRARIES #####

import pandas as pd
import random
from datetime import datetime, timedelta
import osmnx as ox
import os

##### LOAD GRAPHS #####

rome_graph = ox.graph_from_bbox(
    north=41.905, south=41.890,
    east=12.515, west=12.460,
    network_type='drive'
)

montreal_graph = ox.graph_from_bbox(
    north=45.54, south=45.49,
    east=-73.55, west=-73.59,
    network_type='drive'
)

##### ORDER CLASS #####

class Order:
    def __init__(self, order_id, pickup_lat, pickup_lon, delivery_lat, delivery_lon,
                 order_time, period, prep_wait_time, order_size_label, order_size):
        self.order_id = order_id
        self.pickup_lat = pickup_lat
        self.pickup_lon = pickup_lon
        self.delivery_lat = delivery_lat
        self.delivery_lon = delivery_lon
        self.order_time = order_time
        self.period = period
        self.prep_wait_time = prep_wait_time
        self.order_size_label = order_size_label
        self.order_size = order_size
        self.wait_time = 0
        self.wait_time_minutes = 0
        self.dropped = False
        self.assigned = False
        self.assigned_courier_id = None

##### FUNCTIONS #####

def generate_random_delivery_location(lat_bounds, lon_bounds, graph):
    delivery_lat = random.uniform(*lat_bounds)
    delivery_lon = random.uniform(*lon_bounds)
    nearest_node = ox.distance.nearest_nodes(graph, delivery_lon, delivery_lat)
    node_data = graph.nodes[nearest_node]
    return node_data['y'], node_data['x']

def generate_order_size():
    label = random.choices(['small', 'medium', 'large'], weights=[0.5, 0.35, 0.15])[0]
    size_map = {'small': 0.25, 'medium': 0.5, 'large': 0.8}
    return label, size_map[label]

def apply_demand_multipliers(base_orders, day_of_week, weather, hour, city):
    orders = base_orders
    if weather == 'rain':
        orders *= 1.6 if city.lower() == 'montreal' else 1.4
    elif weather == 'snow':
        orders *= 1.8 if city.lower() == 'montreal' else 1.5
    if day_of_week == 'Saturday':
        orders *= 1.3
    elif day_of_week in ['Friday', 'Sunday']:
        orders *= 1.2
    if 19 <= hour <= 22:
        orders *= 1.6 if day_of_week in ['Friday', 'Saturday', 'Sunday'] else 1.4
    return int(round(orders))

def get_weather_condition(weather_df, date_str):
    row = weather_df.loc[weather_df['date'] == date_str]
    if row.empty:
        return 'clear'
    if row.iloc[0]['is_snowing'] == 1:
        return 'snow'
    elif row.iloc[0]['is_raining'] == 1:
        return 'rain'
    return 'clear'

def get_period_from_hour(hour):
    if 6 <= hour < 11:
        return "morning"
    elif 11 <= hour < 14:
        return "lunch"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "offpeak"

def generate_orders(base_orders, city, restaurants_df, lat_bounds, lon_bounds,
                    start_time, end_time, day_of_week, weather, order_id_start, graph):
    orders = []
    order_id_counter = order_id_start

    adjusted_total_orders = apply_demand_multipliers(base_orders, day_of_week, weather, 20, city)

    time_buckets = [
        {"start": "10:00", "end": "12:00", "pct": 0.2},
        {"start": "12:00", "end": "15:00", "pct": 0.3},
        {"start": "18:00", "end": "22:00", "pct": 0.5}
    ]

    for bucket in time_buckets:
        start = datetime.strptime(bucket["start"], "%H:%M")
        end = datetime.strptime(bucket["end"], "%H:%M")
        duration_sec = int((end - start).total_seconds())
        bucket_orders = int(adjusted_total_orders * bucket["pct"])

        for _ in range(bucket_orders):
            order_seconds = random.randint(0, duration_sec)
            order_time = start + timedelta(seconds=order_seconds)
            period = get_period_from_hour(order_time.hour)

            pickup = restaurants_df.sample(1).iloc[0]
            pickup_lat = pickup['latitude']
            pickup_lon = pickup['longitude']

            delivery_lat, delivery_lon = generate_random_delivery_location(lat_bounds, lon_bounds, graph)
            prep_time_min = random.randint(5, 20)
            order_size_label, order_size = generate_order_size()

            order = Order(
                order_id=f"{city.lower()}_order_{order_id_counter}",
                pickup_lat=pickup_lat,
                pickup_lon=pickup_lon,
                delivery_lat=delivery_lat,
                delivery_lon=delivery_lon,
                order_time=int(order_time.hour * 60 + order_time.minute),
                period=period,
                prep_wait_time=prep_time_min,
                order_size_label=order_size_label,
                order_size=order_size
            )

            orders.append(order)
            order_id_counter += 1

    return orders

def generate_orders_for_day(city, date_str, weather_condition, restaurants_df, lat_bounds, lon_bounds, num_orders=100):
    
    graph = rome_graph if city.lower() == "rome" else montreal_graph

    orders = generate_orders(
        base_orders=num_orders,
        city=city,
        restaurants_df=restaurants_df,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        start_time=datetime.strptime("10:00", "%H:%M"),
        end_time=datetime.strptime("22:00", "%H:%M"),
        day_of_week=datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
        weather=weather_condition,
        order_id_start=0,
        graph=graph
    )

    save_dir = f"/Users/davidpaquette/Documents/Thesis/Project/Data/{city}/Order Generation"
    os.makedirs(save_dir, exist_ok=True)
    day_number = len([f for f in os.listdir(save_dir) if f.startswith(f"{city.lower()}_orders_day_")]) + 1

    df = pd.DataFrame([{
        'order_id': o.order_id,
        'pickup_lat': o.pickup_lat,
        'pickup_lon': o.pickup_lon,
        'delivery_lat': o.delivery_lat,
        'delivery_lon': o.delivery_lon,
        'order_time': f"{o.order_time//60:02}:{o.order_time%60:02}:00",
        'period': o.period,
        'prep_time_min': o.prep_wait_time,
        'order_size_label': o.order_size_label,
        'order_size': o.order_size
    } for o in orders])

    df.to_csv(f"{save_dir}/{city.lower()}_orders_day_{day_number}.csv", index=False)

    return orders

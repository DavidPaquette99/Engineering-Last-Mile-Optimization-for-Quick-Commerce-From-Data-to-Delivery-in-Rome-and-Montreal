##### LIBRARIES #####

import random
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os
import osmnx as ox

##### IMPORT GRAPHS ######

# Rome street graph
rome_graph = ox.graph_from_bbox(
    north=41.905, south=41.890,
    east=12.515, west=12.460,
    network_type='drive'
)

# Montreal street graph
montreal_graph = ox.graph_from_bbox(
    north=45.54, south=45.49,
    east=-73.55, west=-73.59,
    network_type='drive'
)

##### COURIER CLASS #####

# Define Courier class
class Courier:
    def __init__(self, courier_id, city, start_lat, start_lon, vehicle_type):
        self.courier_id = courier_id
        self.city = city
        self.latitude = start_lat
        self.longitude = start_lon
        self.vehicle_type = vehicle_type
        self.capacity = self.assign_capacity()
        self.available_capacity = self.capacity
        self.status = 'idle'
        self.available_at = 0
        self.current_order = None
        self.assigned_orders = []  # list of Order objects currently onboard
        self.active_minutes = 0
        self.idle_minutes = 0  # initialize idle time tracker
        self.total_distance_km = 0
        self.earnings = 0
        self.total_deliveries = 0

    def assign_capacity(self):
        if self.vehicle_type in ['bike', 'scooter']:
            return 1
        elif self.vehicle_type == 'car':
            return 2
        else:
            return 1

    def to_dict(self):
        return {
            'courier_id': self.courier_id,
            'city': self.city,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'vehicle_type': self.vehicle_type,
            'capacity': self.capacity,
            'available_capacity': self.available_capacity,
            'status': self.status,
            'available_at': self.available_at,
            'current_order': self.current_order,
            'active_minutes': self.active_minutes,
            'idle_minutes' : self.idle_minutes,
            'total_distance_km': self.total_distance_km,
            'earnings': self.earnings,
            'total_deliveries': self.total_deliveries
        }

# Function to generate couriers
def generate_couriers(num_couriers, city, graph, downtown_bounds=None):
    couriers = []
    nodes = list(graph.nodes(data=True))

    if city.lower() == 'rome':
        vehicle_types = ['bike', 'scooter', 'car']
        weights = [0.5, 0.4, 0.1]
    elif city.lower() == 'montreal':
        vehicle_types = ['bike', 'car']
        weights = [0.2, 0.8]
    else:
        vehicle_types = ['bike', 'scooter', 'car']
        weights = [0.33, 0.33, 0.34]

    vehicle_pool = random.choices(vehicle_types, weights=weights, k=num_couriers)

    for i in range(num_couriers):
        courier_id = f"{city.lower()}_courier_{i}"
        vehicle_type = vehicle_pool[i]

        if city.lower() == 'montreal' and vehicle_type == 'bike' and downtown_bounds is not None:
            lat_bounds, lon_bounds = downtown_bounds
            downtown_nodes = [
                node for node in nodes
                if (lat_bounds[0] <= node[1]['y'] <= lat_bounds[1]) and
                   (lon_bounds[0] <= node[1]['x'] <= lon_bounds[1])
            ]
            node = random.choice(downtown_nodes)
        else:
            node = random.choice(nodes)

        start_lat = node[1]['y']
        start_lon = node[1]['x']

        courier = Courier(courier_id, city, start_lat, start_lon, vehicle_type)
        couriers.append(courier)

    return couriers


# Convert list of couriers into DataFrame
def couriers_to_dataframe(courier_list):
    return pd.DataFrame([c.to_dict() for c in courier_list])

def generate_couriers_for_day(city, date_str, num_couriers=50):
    """
    Generate couriers for a specific day and save as CSV.
    
    Args:
        city (str): City name (Montreal or Rome).
        date_str (str): Date in format YYYY-MM-DD.
        num_couriers (int): Number of couriers to generate.
    
    Returns:
        list of Courier objects
    """

    # Select graph
    if city.lower() == "rome":
        graph = rome_graph
    elif city.lower() == "montreal":
        graph = montreal_graph
        downtown_bounds = ((45.5, 45.52), (-73.58, -73.55))  # Downtown lat/lon bounds for bikes
    else:
        raise ValueError("Unsupported city")

    # Generate couriers
    couriers = generate_couriers(
        num_couriers=num_couriers,
        city=city,
        graph=graph,
        downtown_bounds=downtown_bounds if city.lower() == "montreal" else None
    )

    # Save
    save_dir = f"/Users/davidpaquette/Documents/Thesis/Project/Data/{city}/Courier Generation"
    os.makedirs(save_dir, exist_ok=True)
    
    day_number = len([f for f in os.listdir(save_dir) if f.startswith(f"{city.lower()}_couriers_day_")]) + 1
    df = couriers_to_dataframe(couriers)
    df.to_csv(f"{save_dir}/{city.lower()}_couriers_day_{day_number}.csv", index=False)

    return couriers


##### PLOTTING #####

def plot_couriers(df_couriers, city_center, zoom_start=13, map_title='courier_map.html', polygon_bounds=None, polygon_name=None):
    color_map = {
        'bike': 'blue',
        'scooter': 'green',
        'car': 'red'
    }

    m = folium.Map(location=city_center, zoom_start=zoom_start)

    if polygon_bounds:
        folium.Polygon(
            locations=polygon_bounds,
            color='purple',
            fill=True,
            fill_opacity=0.1,
            tooltip=polygon_name if polygon_name else "Boundary"
        ).add_to(m)

    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df_couriers.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"ID: {row['courier_id']}<br>Vehicle: {row['vehicle_type']}",
            icon=folium.Icon(color=color_map.get(row['vehicle_type'], 'gray'))
        ).add_to(marker_cluster)

    m.save(map_title)
    print(f"Map saved to {map_title}")



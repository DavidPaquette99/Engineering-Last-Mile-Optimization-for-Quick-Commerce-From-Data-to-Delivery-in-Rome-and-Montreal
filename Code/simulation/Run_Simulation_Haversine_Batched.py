##### LIBRARIES #####
import os
import pandas as pd
from datetime import datetime, timedelta
from data_generation.Order_Generation import generate_orders_for_day, Order, get_weather_condition
from data_generation.Courier_Generation import generate_couriers_for_day, Courier
from Simulation_Engine_Haversine_Batched import simulate_day, calculate_payment
import random
import copy
import re

##### CONFIGURATION #####
NUM_COURIERS = 10
NUM_ORDERS = 100
cities = ["Montreal", "Rome"]
assignment_modes = ["anfis", "pairwise"]
random.seed(42)

##### USER INPUT #####
city = input("Choose city (Montreal or Rome): ").strip()
while city not in cities:
    city = input("Invalid. Choose city (Montreal or Rome): ").strip()

##### DATE SELECTION #####
date_range = pd.date_range(start="2021-03-16", end="2025-04-23")
print(f"Available dates: {date_range[0].strftime('%Y-%m-%d')} to {date_range[-1].strftime('%Y-%m-%d')}")

start_date_str = input("Choose start date (YYYY-MM-DD): ").strip()
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
while start_date not in date_range:
    start_date_str = input("Invalid date. Choose again (YYYY-MM-DD): ").strip()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

num_days = int(input(f"How many days to simulate (max {len(date_range)}): ").strip())

##### FILE PATHS #####
base_data_path = f"/Users/davidpaquette/Documents/Thesis/Project/Data/{city}"
output_dir = f"{base_data_path}/Results"
os.makedirs(output_dir, exist_ok=True)

# Load historical weather data
weather_df = pd.read_csv(f"{base_data_path}/{city.lower()}_weather_cleaned.csv")

##### SIMULATION LOOP #####
for day_offset in range(num_days):
    simulated_date = start_date + timedelta(days=day_offset)
    weather_today = get_weather_condition(weather_df, simulated_date.strftime("%Y-%m-%d"))

    if city.lower() == "rome":
        restaurants_df = pd.read_csv(f"{base_data_path}/rome_target.csv")
        lat_bounds = (41.890, 41.905)
        lon_bounds = (12.460, 12.515)
    else:
        restaurants_df = pd.read_csv(f"{base_data_path}/montreal_sampled.csv")
        lat_bounds = (45.49, 45.54)
        lon_bounds = (-73.59, -73.55)

    orders = generate_orders_for_day(
        city=city,
        date_str=simulated_date.strftime("%Y-%m-%d"),
        weather_condition=weather_today,
        restaurants_df=restaurants_df,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        num_orders=NUM_ORDERS
    )

    couriers = generate_couriers_for_day(
        city=city,
        date_str=simulated_date.strftime("%Y-%m-%d"),
        num_couriers=NUM_COURIERS
    )

    couriers_base = copy.deepcopy(couriers)
    orders_base = copy.deepcopy(orders)

    daily_summary_rows = []

    for assignment_mode in assignment_modes:
        print(f"Simulating {city} - {simulated_date.date()} - {assignment_mode}")

        couriers = copy.deepcopy(couriers_base)
        orders = copy.deepcopy(orders_base)

        completed_orders, updated_couriers = simulate_day(
            couriers=couriers,
            orders=orders,
            assignment_mode=assignment_mode,
            weather_condition=weather_today,
            city=city,
            simulation_date_str=simulated_date.strftime("%Y-%m-%d"),
        )

        # Save completed orders, including batch details if present
        orders_data = [{
            "order_id": o.order_id,
            "pickup_lat": o.pickup_lat,
            "pickup_lon": o.pickup_lon,
            "delivery_lat": o.delivery_lat,
            "delivery_lon": o.delivery_lon,
            "assigned": o.assigned,
            "assigned_courier_id": o.assigned_courier_id,
            "wait_time": o.wait_time,
            "batched": getattr(o, "batched", False),
            "batch_id": getattr(o, "batch_id", None)
        } for o in completed_orders]

        pd.DataFrame(orders_data).to_csv(
            f"{output_dir}/{city.lower()}_orders_{simulated_date.date()}_{assignment_mode}_batched_final.csv",
            index=False
        )

        # Save courier stats
        couriers_data = [{
            "courier_id": c.courier_id,
            "total_distance_km": c.total_distance_km,
            "active_minutes": c.active_minutes,
            "idle_minutes": c.idle_minutes,
            "total_deliveries": c.total_deliveries,
            "earnings": calculate_payment(c, city, weather_today),
            "total_batches": len(getattr(c, "batched_orders", [])),
        } for c in updated_couriers]

        pd.DataFrame(couriers_data).to_csv(
            f"{output_dir}/{city.lower()}_couriers_{simulated_date.date()}_{assignment_mode}_batched_final.csv",
            index=False
        )

        # Summary row
        total_deliveries = len(completed_orders)
        total_distance = sum(c["total_distance_km"] for c in couriers_data)
        total_cost = sum(c["earnings"] for c in couriers_data)

        summary_row = {
            "date": simulated_date.strftime("%Y-%m-%d"),
            "assignment_mode": assignment_mode,
            "total_deliveries": total_deliveries,
            "total_distance_km": round(total_distance, 2),
            "total_cost": round(total_cost, 2),
            "avg_delivery_distance_km": round(total_distance / total_deliveries, 2) if total_deliveries else 0,
            "avg_delivery_time_min": round(sum(c["active_minutes"] for c in couriers_data) / total_deliveries, 2) if total_deliveries else 0,
            "avg_wait_time_min": round(sum(o["wait_time"] for o in orders_data) / total_deliveries, 2) if total_deliveries else 0,
            "avg_cost": round(total_cost / total_deliveries, 2) if total_deliveries else 0,
            "total_batches": sum(c["total_batches"] for c in couriers_data)
        }

        daily_summary_rows.append(summary_row)

    # Save daily summary
    pd.DataFrame(daily_summary_rows).to_csv(
        f"{output_dir}/{city.lower()}_haversine_batched_summary_{simulated_date.date()}_final.csv",
        index=False
    )
    print(f"Finished simulations for {simulated_date.date()} — summary saved.")

##### COMPILE GLOBAL SUMMARY #####
summary_files = [
    f for f in os.listdir(output_dir)
    if re.match(f"{city.lower()}_haversine_batched_summary_\\d{{4}}-\\d{{2}}-\\d{{2}}\\_final.csv", f)
]

summary_dfs = [pd.read_csv(os.path.join(output_dir, f)) for f in summary_files]
full_summary = pd.concat(summary_dfs, ignore_index=True)
full_summary.to_csv(
    os.path.join(output_dir, f"{city.lower()}_haversine_batched_full_summary_final.csv"),
    index=False
)
print(f"Full summary saved to {output_dir}/{city.lower()}_haversine_batched_full_summary_final.csv")

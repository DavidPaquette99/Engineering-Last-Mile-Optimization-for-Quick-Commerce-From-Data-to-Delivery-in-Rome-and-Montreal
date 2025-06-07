##### LIBRARIES #####

import copy
import pandas as pd
from datetime import datetime
from data_generation.Order_Generation import generate_orders_for_day, get_weather_condition
from data_generation.Courier_Generation import generate_couriers_for_day
from Simulation_Engine import simulate_day, calculate_payment
import assignment_strategies.Pairwise_Assignment as PA

##### CONFIGURATION ######

CITIES = ["Montreal", "Rome"]
NUM_COURIERS = 5
NUM_ORDERS = 50
DATE = "2025-04-04"    # pick any valid date
OUTPUT_CSV = "alpha_tuning_results.csv"
API_KEY = "AIzaSyCVCIC2uaYM3cwEq7nLmJ4-B4gMGbTsde0"

for city in CITIES:

    ##### LOAD WEATHER #####
    weather_df = pd.read_csv(f"/Users/davidpaquette/Documents/Thesis/Project/Data/{city}/{city.lower()}_weather_cleaned.csv")
    weather = get_weather_condition(weather_df, DATE)

    ##### GENERATE ORDERS & COURIERS #####
    
    if city.lower() == "rome":

        restaurants_df = pd.read_csv(f"/Users/davidpaquette/Documents/Thesis/Project/Data/{city}/rome_target.csv")
        lat_bounds = (41.890, 41.905)
        lon_bounds = (12.460, 12.515)

    else:
        restaurants_df = pd.read_csv(f"/Users/davidpaquette/Documents/Thesis/Project/Data/{city}/montreal_sampled.csv")
        lat_bounds = (45.49, 45.54)
        lon_bounds = (-73.59, -73.55)

    orders_base = generate_orders_for_day(
        city=city,
        date_str=DATE,
        weather_condition=weather,
        restaurants_df=restaurants_df,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        num_orders=NUM_ORDERS
    )

    couriers_base = generate_couriers_for_day(
        city=city,
        date_str=DATE,
        num_couriers=NUM_COURIERS
    )

    ##### ALPHA LOOP #####
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for alpha in alphas:
        print(f"Running pairwise with alpha = {alpha}")
        # override the module constant
        PA.ALPHA = alpha

        # deep-copy to reset state each run
        orders = copy.deepcopy(orders_base)
        couriers = copy.deepcopy(couriers_base)

        # run one simulated day
        completed_orders, updated_couriers = simulate_day(
            couriers=couriers,
            orders=orders,
            assignment_mode="pairwise",
            api_key=API_KEY,
            weather_condition=weather,
            city=city
        )

        # compute total cost via calculate_payment
        total_cost = sum(
            calculate_payment(c, city, weather)
            for c in updated_couriers
        )

        # average delivery time = total active minutes / deliveries
        total_active = sum(c.active_minutes for c in updated_couriers)
        total_deliv = len(completed_orders)
        avg_time = total_active / total_deliv if total_deliv else float('nan')

        results.append({
            "alpha": alpha,
            "total_cost": round(total_cost, 2),
            "avg_delivery_time": round(avg_time, 2)
        })

    ##### SAVE RESULTS #####
    pd.DataFrame(results).to_csv(f"/Users/davidpaquette/Documents/Thesis/Project/Data/{city}/{city.lower()}_{OUTPUT_CSV}", index=False)
    print(f"Done. Tuning results saved to {OUTPUT_CSV}")
    print(pd.DataFrame(results))
    

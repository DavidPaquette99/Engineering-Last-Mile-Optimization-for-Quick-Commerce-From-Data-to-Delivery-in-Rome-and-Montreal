##### LIBRARIES #####

from assignment_strategies.Naive_Assignment import naive_assignment
from assignment_strategies.Hungarian_Assignment import hungarian_assignment
from assignment_strategies.Anfis_Assignment_Regression import load_anfis_regression_model, anfis_assignment
import assignment_strategies.Pairwise_Assignment as PA

##### PAYMENT FUNCTION #####
def calculate_payment(courier, city, weather_condition):
    if city.lower() == 'rome':
        base_pay_per_minute = 10 / 60  # €10/hour
    elif city.lower() == 'montreal':
        base_pay_per_minute = (16.10 * 1.2) / 60  # CA$19.32/hour
    else:
        base_pay_per_minute = 10 / 60

    base_earnings = courier.active_minutes * base_pay_per_minute
    bonus_multiplier = 1.0
    if weather_condition in ['rain', 'snow']:
        bonus_multiplier += 0.10

    total_earnings = base_earnings * bonus_multiplier
    if city.lower() == 'montreal':
        total_earnings += courier.total_distance_km * 0.22

    return round(total_earnings, 2)

##### SIMULATE DAY FUNCTION #####
def simulate_day(couriers, orders, assignment_mode, api_key, weather_condition, city, simulation_date_str = None, 
                 model_base_path="/Users/davidpaquette/Documents/Thesis/Project/Data/ANFIS"):

    # Load models if needed
    if assignment_mode in ["anfis", "pairwise"]:
        loader = {
            "anfis": load_anfis_regression_model,
            "pairwise": PA.load_pairwise_model
        }[assignment_mode]
        model, scaler_X, scaler_y, pca = loader(city, model_base_path)

    # City centers for pairwise feature
    city_centers = {
        "Montreal": (45.51, -73.57),
        "Rome":     (41.8975, 12.4875)
    }
    city_center = city_centers[city]

    current_minute     = 0
    active_orders      = []
    available_couriers = couriers.copy()

    while current_minute < 24 * 60:
        # add new orders arriving this minute
        new_orders = [o for o in orders if o.order_time == current_minute]
        active_orders.extend(new_orders)

        # free up couriers whose tasks are done
        busy_ids = {c.courier_id for c in available_couriers}
        for c in couriers:
            if c.available_at <= current_minute and c.courier_id not in busy_ids:
                c.status = "idle"
                c.available_capacity = c.capacity
                c.assigned_orders.clear()
                available_couriers.append(c)

        # drop only assigned orders from active pool
        active_orders = [o for o in active_orders if not o.assigned]

        if active_orders:
            if assignment_mode == "naive":
                naive_assignment(available_couriers, active_orders, current_minute, api_key)

            elif assignment_mode == "hungarian":
                hungarian_assignment(available_couriers, active_orders, current_minute, api_key)

            elif assignment_mode == "anfis":
                completed, available_couriers = anfis_assignment(
                    available_couriers=available_couriers,
                    new_orders=active_orders,
                    current_minute=current_minute,
                    api_key=api_key,
                    weather_condition=weather_condition,
                    city=city,
                    model=model,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y,
                    pca=pca
                )
                # mark those completed as assigned
                for o in completed:
                    o.assigned = True
                active_orders = [o for o in active_orders if not o.assigned]

            elif assignment_mode == "pairwise":
                # run pairwise on the idle pool
                completed, updated_available = PA.pairwise_assignment(
                    couriers=available_couriers,
                    orders=active_orders,
                    current_minute=current_minute,
                    weather_condition=weather_condition,
                    city=city,
                    api_key=api_key,
                    model=model,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y,
                    pca=pca,
                    city_center=city_center,
                    simulation_date_str=simulation_date_str,
                    all_orders=orders  # <-- add this!
                )

                # Fix 1: merge updated_available back into full couriers list
                for upd in updated_available:
                    for idx, orig in enumerate(couriers):
                        if orig.courier_id == upd.courier_id:
                            couriers[idx] = upd
                            break

                # Fix 2: recompute idle pool for the next minute
                available_couriers = [c for c in couriers if c.available_at <= current_minute]

                # mark orders assigned this minute
                for o in completed:
                    o.assigned = True
                active_orders = [o for o in active_orders if not o.assigned]

            else:
                raise ValueError(f"Unknown assignment mode: {assignment_mode}")

        current_minute += 1

    # collect all that got assigned over the day
    completed_orders = [o for o in orders if o.assigned]

    return completed_orders, couriers

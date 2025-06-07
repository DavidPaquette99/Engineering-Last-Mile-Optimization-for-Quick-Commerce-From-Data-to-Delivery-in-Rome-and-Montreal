##### LIBRARIES #####
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.Distance_Calculator import get_realistic_travel_time

##### HUNGARIAN ASSIGNMENT #####

def hungarian_assignment(available_couriers, new_orders, current_minute, api_key):
    """
    Assigns orders to couriers by minimizing pickup + wait time using the Hungarian algorithm.
    Couriers are removed from the available pool so they can be re-added when they finish.
    """
    # Guard against no assignable entities
    if not available_couriers or not new_orders:
        return

    num_couriers = len(available_couriers)
    num_orders = len(new_orders)
    # Initialize cost matrix with large default cost
    cost_matrix = np.full((num_couriers, num_orders), 1e6)

    # Build cost matrix for feasible matches
    for i, courier in enumerate(available_couriers):
        for j, order in enumerate(new_orders):
            if courier.available_capacity < order.order_size:
                continue
            pickup_time, _ = get_realistic_travel_time(
                courier.latitude, courier.longitude,
                order.pickup_lat, order.pickup_lon,
                mode=courier.vehicle_type, api_key=api_key
            )
            wait_time = max(0, order.prep_wait_time - pickup_time)
            cost_matrix[i, j] = pickup_time + wait_time

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Collect valid matches
    matched = []  # list of (courier_obj, order_obj, wait_time, total_service_time, total_distance)
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] >= 1e6:
            continue
        courier = available_couriers[i]
        order = new_orders[j]
        if order.assigned:
            continue
        # Compute full service time and distance
        pickup_time, pickup_dist = get_realistic_travel_time(
            courier.latitude, courier.longitude,
            order.pickup_lat, order.pickup_lon,
            mode=courier.vehicle_type, api_key=api_key
        )
        delivery_time, delivery_dist = get_realistic_travel_time(
            order.pickup_lat, order.pickup_lon,
            order.delivery_lat, order.delivery_lon,
            mode=courier.vehicle_type, api_key=api_key
        )
        wait_time = max(0, order.prep_wait_time - pickup_time)
        total_time = pickup_time + wait_time + delivery_time
        total_dist = pickup_dist + delivery_dist
        matched.append((courier, order, wait_time, total_time, total_dist))

    # Apply matches and remove couriers
    for courier, order, wait_time, service_time, dist in matched:
        # Update courier state
        courier.total_distance_km += dist
        courier.active_minutes += service_time
        courier.idle_minutes += wait_time
        courier.available_at = current_minute + service_time
        courier.total_deliveries += 1
        courier.latitude = order.delivery_lat
        courier.longitude = order.delivery_lon
        courier.available_capacity -= order.order_size

        # Update order state
        order.assigned = True
        order.assigned_courier_id = courier.courier_id
        order.wait_time = wait_time

        # Remove courier from pool so simulate_day can re-add later
        available_couriers.remove(courier)


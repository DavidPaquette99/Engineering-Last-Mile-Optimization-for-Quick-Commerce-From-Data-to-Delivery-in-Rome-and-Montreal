##### LIBRARIES #####
from utils.Distance_Calculator import haversine, travel_time_haversine

##### NAIVE ASSIGNMENT #####

def naive_assignment(available_couriers, new_orders, current_minute):
    for order in new_orders:
        if order.assigned:
            continue

        # Filter: courier must be idle, on time, and have enough capacity
        eligible = [
            c for c in available_couriers
            if c.status == 'idle' and c.available_at <= current_minute and c.available_capacity >= order.order_size
        ]
        if not eligible:
            continue

        # Pick closest courier based on haversine distance to pickup
        closest = min(
            eligible,
            key=lambda c: haversine(c.latitude, c.longitude, order.pickup_lat, order.pickup_lon)
        )

        # Estimate time and distance using Haversine
        travel_time, travel_km = travel_time_haversine(
            closest.latitude, closest.longitude,
            order.pickup_lat, order.pickup_lon,
            closest.vehicle_type
        )
        delivery_time, delivery_km = travel_time_haversine(
            order.pickup_lat, order.pickup_lon,
            order.delivery_lat, order.delivery_lon,
            closest.vehicle_type
        )

        total_time = travel_time + max(0, order.prep_wait_time - travel_time) + delivery_time
        total_km = travel_km + delivery_km

        # Update courier
        closest.active_minutes += total_time
        closest.total_distance_km += total_km
        closest.available_at = current_minute + total_time
        closest.latitude = order.delivery_lat
        closest.longitude = order.delivery_lon
        closest.available_capacity -= order.order_size
        closest.total_deliveries += 1

        # Standardized wait time calculation
        courier_arrival_time = current_minute + travel_time
        order_ready_time = order.order_time + order.prep_wait_time
        wait_time = max(0, courier_arrival_time - order_ready_time)
        order.wait_time = wait_time

        # Update order
        order.assigned = True
        order.assigned_courier_id = closest.courier_id

        # Remove courier for this minute
        available_couriers.remove(closest)


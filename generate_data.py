"""Generate a realistic SYNTHETIC delivery dataset (no real customer data).

Creates 5 linked tables in ./data:
    warehouses.csv, carriers.csv, orders.csv, shipments.csv, delivery_events.csv

Delay patterns are deliberately planted so the analysis has real signal to find:
  * Carrier "SwiftLine" is the least reliable; "BlueArrow" the most.
  * Warehouse WH-MUM is congested on Fridays and in festive season (Oct-Dec).
  * Remote / long-distance routes and bulky categories (Furniture, Appliances) run late more.
  * Monsoon months (Jun-Sep) add delay on the East and South-East routes.
  * A few unusual spike days (e.g. a strike, a flood) hit specific warehouses.

Run:  python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 15000
OUT = Path(__file__).parent / "data"


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ dims
    warehouses = pd.DataFrame(
        {
            "warehouse_id": ["WH-BLR", "WH-MUM", "WH-DEL", "WH-HYD", "WH-KOL"],
            "warehouse_name": ["Bengaluru Hub", "Mumbai Hub", "Delhi Hub", "Hyderabad Hub", "Kolkata Hub"],
            "region": ["South", "West", "North", "South", "East"],
            "daily_capacity": [900, 1100, 1000, 700, 600],
        }
    )
    carriers = pd.DataFrame(
        {
            "carrier_id": ["CR-1", "CR-2", "CR-3", "CR-4"],
            "carrier_name": ["SwiftLine", "BlueArrow", "RapidRoute", "EcoExpress"],
            "service_type": ["Standard", "Premium", "Standard", "Economy"],
        }
    )

    # ---------------------------------------------------------------- orders
    start, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
    n_days = (end - start).days + 1
    # festive season gets more orders
    day_weights = np.ones(n_days)
    dates_all = pd.date_range(start, end)
    day_weights[dates_all.month >= 10] *= 1.6
    day_weights /= day_weights.sum()
    order_dates = dates_all[rng.choice(n_days, size=N_ORDERS, p=day_weights)]

    categories = ["Electronics", "Fashion", "Grocery", "Furniture", "Appliances", "Books", "Beauty"]
    cat_p = [0.20, 0.25, 0.15, 0.08, 0.10, 0.12, 0.10]
    category = rng.choice(categories, size=N_ORDERS, p=cat_p)

    wh_ids = rng.choice(warehouses["warehouse_id"], size=N_ORDERS, p=[0.24, 0.26, 0.22, 0.15, 0.13])
    dest_zones = ["Metro", "Tier-2", "Tier-3", "Remote"]
    dest_zone = rng.choice(dest_zones, size=N_ORDERS, p=[0.40, 0.30, 0.20, 0.10])

    orders = pd.DataFrame(
        {
            "order_id": [f"ORD{100000 + i}" for i in range(N_ORDERS)],
            "order_date": order_dates,
            "product_category": category,
            "order_value": np.round(rng.lognormal(7.2, 0.8, N_ORDERS), 2),
            "warehouse_id": wh_ids,
            "destination_zone": dest_zone,
        }
    )

    # ------------------------------------------------------------- shipments
    carrier_ids = rng.choice(carriers["carrier_id"], size=N_ORDERS, p=[0.30, 0.25, 0.25, 0.20])
    dist_base = {"Metro": 250, "Tier-2": 600, "Tier-3": 950, "Remote": 1500}
    distance = np.array([max(30, rng.normal(dist_base[z], dist_base[z] * 0.3)) for z in dest_zone]).round(0)
    weight = np.round(
        np.where(np.isin(category, ["Furniture", "Appliances"]), rng.uniform(8, 40, N_ORDERS), rng.uniform(0.2, 5, N_ORDERS)),
        1,
    )

    # promised days based on service type & distance (what the customer was told)
    svc = carriers.set_index("carrier_id")["service_type"].reindex(carrier_ids).to_numpy()
    promised_days = np.where(distance < 400, 2, np.where(distance < 900, 4, np.where(distance < 1300, 5, 7)))
    promised_days = promised_days + np.where(svc == "Economy", 1, 0) - np.where(svc == "Premium", 1, 0)
    promised_days = np.clip(promised_days, 1, 9)

    # processing delay at warehouse (days between order and ship)
    ship_lag = rng.choice([0, 1, 2], size=N_ORDERS, p=[0.35, 0.50, 0.15]).astype(float)

    # --------------------------------------------------- planted delay drivers
    delay_effect = np.zeros(N_ORDERS)
    od = pd.DatetimeIndex(orders["order_date"])
    dow = od.dayofweek.to_numpy()
    month = od.month.to_numpy()

    carrier_effect = {"CR-1": 0.9, "CR-2": -0.3, "CR-3": 0.2, "CR-4": 0.35}
    delay_effect += np.array([carrier_effect[c] for c in carrier_ids])

    is_mum = wh_ids == "WH-MUM"
    delay_effect += is_mum * ((dow == 4) * 1.1 + (month >= 10) * 0.8)       # Friday + festive congestion
    delay_effect += (wh_ids == "WH-KOL") * 0.4
    delay_effect += (dest_zone == "Remote") * 1.3 + (dest_zone == "Tier-3") * 0.5
    delay_effect += np.isin(category, ["Furniture", "Appliances"]) * 0.7
    delay_effect += (month >= 11) * 0.5                                      # festive volume
    monsoon = (month >= 6) & (month <= 9)
    delay_effect += monsoon * np.isin(wh_ids, ["WH-KOL", "WH-HYD"]) * 0.9
    delay_effect += (distance / 1000) * 0.4
    delay_effect += (weight > 25) * 0.3
    delay_effect += ship_lag * 0.35

    # unusual spikes (one-off incidents)
    spike = np.zeros(N_ORDERS)
    spike += ((od >= "2025-03-10") & (od <= "2025-03-13") & (wh_ids == "WH-DEL")) * 2.5   # transport strike
    spike += ((od >= "2025-08-18") & (od <= "2025-08-22") & (wh_ids == "WH-KOL")) * 3.0   # flooding
    spike += ((od >= "2025-11-25") & (od <= "2025-11-29")) * 1.5                           # sale week surge
    delay_effect += spike

    noise = rng.normal(0, 1.1, N_ORDERS)
    realised_delay = np.round(delay_effect + noise - 2.0)   # shift so ~70-75% are on time
    realised_delay = np.clip(realised_delay, -1, 14)

    transit_days = np.maximum(1, promised_days + realised_delay).astype(int)
    ship_date = od + pd.to_timedelta(ship_lag, unit="D")
    promised_date = od + pd.to_timedelta(promised_days + 1, unit="D")   # promise made at order time
    actual_date = ship_date + pd.to_timedelta(transit_days, unit="D")

    shipments = pd.DataFrame(
        {
            "shipment_id": [f"SHP{500000 + i}" for i in range(N_ORDERS)],
            "order_id": orders["order_id"],
            "carrier_id": carrier_ids,
            "ship_date": ship_date,
            "promised_delivery_date": promised_date,
            "actual_delivery_date": actual_date,
            "distance_km": distance,
            "weight_kg": weight,
        }
    )
    # route = origin warehouse city -> destination zone
    city = warehouses.set_index("warehouse_id")["warehouse_name"].str.replace(" Hub", "", regex=False)
    shipments["route"] = city.reindex(wh_ids).to_numpy() + " -> " + dest_zone

    # --------------------------------------------------------- delivery events
    ev_rows = []
    ship_ids = shipments["shipment_id"].to_numpy()
    for sid, od_, sd, ad, td in zip(ship_ids, orders["order_date"], ship_date, actual_date, transit_days):
        ev_rows.append((sid, "Order Packed", od_ + pd.Timedelta(hours=10)))
        ev_rows.append((sid, "Dispatched", sd + pd.Timedelta(hours=14)))
        ev_rows.append((sid, "In Transit", sd + pd.Timedelta(days=max(1, td // 2), hours=9)))
        ev_rows.append((sid, "Out for Delivery", ad + pd.Timedelta(hours=8)))
        ev_rows.append((sid, "Delivered", ad + pd.Timedelta(hours=15)))
    events = pd.DataFrame(ev_rows, columns=["shipment_id", "event_type", "event_time"])
    events.insert(0, "event_id", [f"EVT{i:07d}" for i in range(len(events))])

    # extra exception events for late shipments (the "why" signal)
    late_mask = np.asarray(actual_date > promised_date)
    reasons = ["Weather Hold", "Address Issue", "Hub Congestion", "Vehicle Breakdown", "Customer Unavailable"]
    late_ids = ship_ids[late_mask]
    pick = rng.random(len(late_ids)) < 0.45
    exc = pd.DataFrame(
        {
            "shipment_id": late_ids[pick],
            "event_type": rng.choice(reasons, size=pick.sum()),
            "event_time": pd.to_datetime(shipments.loc[late_mask, "ship_date"].to_numpy()[pick]) + pd.Timedelta(days=1, hours=11),
        }
    )
    exc.insert(0, "event_id", [f"EXC{i:07d}" for i in range(len(exc))])
    events = pd.concat([events, exc], ignore_index=True).sort_values(["shipment_id", "event_time"])

    # ------------------------------------------------------------------ save
    warehouses.to_csv(OUT / "warehouses.csv", index=False)
    carriers.to_csv(OUT / "carriers.csv", index=False)
    orders.to_csv(OUT / "orders.csv", index=False)
    shipments.to_csv(OUT / "shipments.csv", index=False)
    events.to_csv(OUT / "delivery_events.csv", index=False)

    on_time = (actual_date <= promised_date).mean() * 100
    print(f"Saved 5 tables to {OUT}")
    print(f"Orders: {len(orders):,} | Events: {len(events):,} | On-time rate: {on_time:.1f}%")


if __name__ == "__main__":
    main()

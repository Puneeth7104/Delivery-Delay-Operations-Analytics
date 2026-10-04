"""Data loading, merging and delay analytics helpers."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def season_of(month: int) -> str:
    if month in (10, 11, 12):
        return "Festive (Oct-Dec)"
    if month in (6, 7, 8, 9):
        return "Monsoon (Jun-Sep)"
    if month in (3, 4, 5):
        return "Summer (Mar-May)"
    return "Winter (Jan-Feb)"


def load_data() -> pd.DataFrame:
    """Combine orders + shipments + warehouses + carriers + event summary into one table."""
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    shipments = pd.read_csv(
        DATA / "shipments.csv", parse_dates=["ship_date", "promised_delivery_date", "actual_delivery_date"]
    )
    warehouses = pd.read_csv(DATA / "warehouses.csv")
    carriers = pd.read_csv(DATA / "carriers.csv")
    events = pd.read_csv(DATA / "delivery_events.csv", parse_dates=["event_time"])

    df = (
        orders.merge(shipments, on="order_id", how="left")
        .merge(warehouses, on="warehouse_id", how="left")
        .merge(carriers, on="carrier_id", how="left")
    )

    # exception reason from delivery events (first exception per shipment, if any)
    normal = {"Order Packed", "Dispatched", "In Transit", "Out for Delivery", "Delivered"}
    exc = events[~events["event_type"].isin(normal)].drop_duplicates("shipment_id")[["shipment_id", "event_type"]]
    exc = exc.rename(columns={"event_type": "exception_reason"})
    df = df.merge(exc, on="shipment_id", how="left")
    df["exception_reason"] = df["exception_reason"].fillna("None")

    # core metrics
    df["delay_days"] = (df["actual_delivery_date"] - df["promised_delivery_date"]).dt.days
    df["is_late"] = (df["delay_days"] > 0).astype(int)
    df["days_late"] = df["delay_days"].clip(lower=0)
    df["processing_days"] = (df["ship_date"] - df["order_date"]).dt.days
    df["day_of_week"] = pd.Categorical(df["order_date"].dt.dayofweek.map(dict(enumerate(DOW))), DOW, ordered=True)
    df["month"] = df["order_date"].dt.month
    df["month_start"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["season"] = df["month"].map(season_of)
    return df


def kpis(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return dict(orders=0, on_time_pct=0.0, avg_delay=0.0, avg_late_delay=0.0, severe_pct=0.0)
    late = df[df["is_late"] == 1]
    return dict(
        orders=len(df),
        on_time_pct=100 * (1 - df["is_late"].mean()),
        avg_delay=df["days_late"].mean(),
        avg_late_delay=late["days_late"].mean() if len(late) else 0.0,
        severe_pct=100 * (df["delay_days"] >= 3).mean(),
    )


def segment(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = (
        df.groupby(col, observed=True)
        .agg(orders=("order_id", "count"), late_orders=("is_late", "sum"), avg_days_late=("days_late", "mean"))
        .reset_index()
    )
    g["late_pct"] = 100 * g["late_orders"] / g["orders"]
    g["on_time_pct"] = 100 - g["late_pct"]
    return g.round(2)


def hotspots(df: pd.DataFrame, min_orders: int = 40, top: int = 15) -> pd.DataFrame:
    """High-delay combinations of warehouse x carrier x product category."""
    g = (
        df.groupby(["warehouse_name", "carrier_name", "product_category"], observed=True)
        .agg(orders=("order_id", "count"), late_orders=("is_late", "sum"), avg_days_late=("days_late", "mean"))
        .reset_index()
    )
    g = g[g["orders"] >= min_orders].copy()
    g["late_pct"] = 100 * g["late_orders"] / g["orders"]
    g["lost_orders"] = g["late_orders"]  # volume impact
    return g.sort_values("late_pct", ascending=False).head(top).round(2)


def daily_spikes(df: pd.DataFrame, z: float = 2.5, min_orders: int = 8) -> pd.DataFrame:
    """Warehouse-days where the late-order rate is unusually high (z-score vs that warehouse's own daily mean)."""
    d = (
        df.groupby(["warehouse_name", "order_date"])
        .agg(orders=("order_id", "count"), late_pct=("is_late", "mean"))
        .reset_index()
    )
    d["late_pct"] *= 100
    d = d[d["orders"] >= min_orders].copy()
    grp = d.groupby("warehouse_name")["late_pct"]
    d["z_score"] = (d["late_pct"] - grp.transform("mean")) / grp.transform("std")
    out = d[d["z_score"] >= z].sort_values(["z_score"], ascending=False)
    out["z_score"] = out["z_score"].round(2)
    out["late_pct"] = out["late_pct"].round(1)
    out["order_date"] = out["order_date"].dt.date
    return out


def recommendations(df: pd.DataFrame) -> list[dict]:
    """Auto-generate prioritized recommendations from the data (impact = extra late orders vs average)."""
    base = df["is_late"].mean()
    recs = []

    def excess(sub):  # late orders above what the average rate would predict
        return float(sub["is_late"].sum() - base * len(sub))

    for col, label in [
        ("carrier_name", "Carrier"),
        ("warehouse_name", "Warehouse"),
        ("destination_zone", "Destination zone"),
        ("product_category", "Product category"),
        ("day_of_week", "Order day"),
        ("season", "Season"),
    ]:
        for key, sub in df.groupby(col, observed=True):
            if len(sub) < 200:
                continue
            ex = excess(sub)
            rate = 100 * sub["is_late"].mean()
            if ex > 0 and rate > 100 * base * 1.15:
                recs.append(
                    dict(
                        area=f"{label}: {key}",
                        late_pct=round(rate, 1),
                        vs_avg=round(rate - 100 * base, 1),
                        excess_late_orders=int(ex),
                    )
                )
    out = pd.DataFrame(recs)
    if out.empty:
        return []
    out = out.sort_values("excess_late_orders", ascending=False).head(8).reset_index(drop=True)
    actions = {
        "Carrier": "Review SLA / penalties, shift volume to better carriers, run a monthly scorecard.",
        "Warehouse": "Add shift capacity or cut-off times; rebalance order allocation to nearby hubs.",
        "Destination zone": "Use regional carriers or buffer the promised date for these zones.",
        "Product category": "Use specialised handling / freight for bulky items and quote longer promises.",
        "Order day": "Smooth dispatch on this weekday; add weekend sorting shifts.",
        "Season": "Pre-book capacity and extra staff before this season starts.",
    }
    out["recommended_action"] = out["area"].map(lambda a: actions[a.split(":")[0]])
    out.insert(0, "priority", range(1, len(out) + 1))
    return out.to_dict("records")

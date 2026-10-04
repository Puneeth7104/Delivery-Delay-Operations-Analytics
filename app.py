"""Delivery Delay & Operations Analytics - Streamlit dashboard."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import analysis as A
from model import CAT, predict_risk, train_models

st.set_page_config(page_title="Delivery Delay & Operations Analytics", page_icon="🚚", layout="wide")

ROOT = Path(__file__).parent
if not (ROOT / "data" / "orders.csv").exists():  # first run on a fresh clone / cloud
    subprocess.run([sys.executable, str(ROOT / "generate_data.py")], check=True)


@st.cache_data
def get_data() -> pd.DataFrame:
    return A.load_data()


@st.cache_resource
def get_model(_df: pd.DataFrame):
    return train_models(_df)


df_all = get_data()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Filters")
date_min, date_max = df_all["order_date"].min().date(), df_all["order_date"].max().date()
date_range = st.sidebar.date_input("Order date range", (date_min, date_max), min_value=date_min, max_value=date_max)


def multi(label, col):
    opts = sorted(df_all[col].astype(str).unique())
    return st.sidebar.multiselect(label, opts, default=opts)


wh_sel = multi("Warehouse", "warehouse_name")
car_sel = multi("Carrier", "carrier_name")
cat_sel = multi("Product category", "product_category")
zone_sel = multi("Destination zone", "destination_zone")

df = df_all.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    df = df[(df["order_date"].dt.date >= date_range[0]) & (df["order_date"].dt.date <= date_range[1])]
df = df[
    df["warehouse_name"].isin(wh_sel)
    & df["carrier_name"].isin(car_sel)
    & df["product_category"].isin(cat_sel)
    & df["destination_zone"].isin(zone_sel)
]

st.title("🚚 Delivery Delay & Operations Analytics")
st.caption("Synthetic data · find why orders arrive late and where the operational bottlenecks are.")

if df.empty:
    st.warning("No orders match the current filters.")
    st.stop()

tab_over, tab_seg, tab_hot, tab_model, tab_rec = st.tabs(
    ["📊 Overview", "🔍 Segments", "🔥 Hotspots & Spikes", "🤖 Delay-Risk Model", "✅ Recommendations"]
)

# ----------------------------------------------------------------- overview
with tab_over:
    k = A.kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total orders", f"{k['orders']:,}")
    c2.metric("On-time delivery", f"{k['on_time_pct']:.1f}%")
    c3.metric("Avg days late (all orders)", f"{k['avg_delay']:.2f}")
    c4.metric("Avg days late (late orders)", f"{k['avg_late_delay']:.2f}")
    c5.metric("Severely late (3+ days)", f"{k['severe_pct']:.1f}%")

    m = df.groupby("month_start").agg(orders=("order_id", "count"), on_time=("is_late", lambda s: 100 * (1 - s.mean())),
                                      avg_days_late=("days_late", "mean")).reset_index()
    l, r = st.columns(2)
    with l:
        fig = px.line(m, x="month_start", y="on_time", markers=True, title="On-time % by month")
        fig.update_yaxes(range=[0, 100], title="On-time %")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, width="stretch")
    with r:
        fig = px.bar(m, x="month_start", y="avg_days_late", title="Average days late by month")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Days")
        st.plotly_chart(fig, width="stretch")

    dist = df["delay_days"].clip(-2, 8).value_counts().sort_index().reset_index()
    dist.columns = ["delay_days", "orders"]
    fig = px.bar(dist, x="delay_days", y="orders", title="Delay distribution (days vs promise; negative = early; capped at -2/+8)")
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------- segments
with tab_seg:
    st.subheader("Late-order rate by segment")
    pairs = [
        ("warehouse_name", "Warehouse"),
        ("carrier_name", "Carrier"),
        ("route", "Route"),
        ("product_category", "Product type"),
        ("day_of_week", "Day of week (order placed)"),
        ("season", "Season"),
    ]
    choice = st.selectbox("Segment by", [p[1] for p in pairs])
    col = dict((v, k) for k, v in pairs)[choice]
    seg = A.segment(df, col)
    if col == "route":
        seg = seg.sort_values("late_pct", ascending=False)
    fig = px.bar(seg, x=col, y="late_pct", color="late_pct", color_continuous_scale="Reds",
                 hover_data=["orders", "avg_days_late"], title=f"Late % by {choice.lower()}")
    fig.update_yaxes(title="Late %")
    fig.update_xaxes(title="", tickangle=-40 if col == "route" else 0)
    fig.add_hline(y=100 * df["is_late"].mean(), line_dash="dash", annotation_text="overall average")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(seg.sort_values("late_pct", ascending=False), width="stretch", hide_index=True)

    st.subheader("Warehouse × Day of week heatmap (late %)")
    hm = df.pivot_table(index="warehouse_name", columns="day_of_week", values="is_late", aggfunc="mean", observed=True) * 100
    st.plotly_chart(px.imshow(hm.round(1), text_auto=True, color_continuous_scale="Reds", aspect="auto"), width="stretch")

    st.subheader("Carrier × Destination zone heatmap (late %)")
    hm2 = df.pivot_table(index="carrier_name", columns="destination_zone", values="is_late", aggfunc="mean") * 100
    st.plotly_chart(px.imshow(hm2.round(1), text_auto=True, color_continuous_scale="Reds", aspect="auto"), width="stretch")

# ------------------------------------------------------------------ hotspots
with tab_hot:
    st.subheader("Top high-delay combinations (warehouse × carrier × product)")
    min_o = st.slider("Minimum orders per combination", 10, 150, 40, step=10)
    st.dataframe(A.hotspots(df, min_orders=min_o), width="stretch", hide_index=True)

    st.subheader("Unusual spikes (warehouse-days with abnormally high late %)")
    st.caption("Flagged when a warehouse's daily late % is 2.5+ standard deviations above its own average.")
    sp = A.daily_spikes(df)
    if sp.empty:
        st.info("No statistically unusual spikes in the current selection.")
    else:
        st.dataframe(sp.head(30), width="stretch", hide_index=True)
    weekly = df.assign(week=df["order_date"].dt.to_period("W").dt.start_time)
    weekly = weekly.groupby(["week", "warehouse_name"]).agg(late_pct=("is_late", "mean")).reset_index()
    weekly["late_pct"] *= 100
    fig = px.line(weekly, x="week", y="late_pct", color="warehouse_name", title="Weekly late % by warehouse (spikes stand out)")
    fig.update_yaxes(title="Late %")
    fig.update_xaxes(title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Exception reasons logged in delivery events (late orders)")
    ex = df[(df["is_late"] == 1) & (df["exception_reason"] != "None")]["exception_reason"].value_counts().reset_index()
    ex.columns = ["reason", "orders"]
    if not ex.empty:
        st.plotly_chart(px.bar(ex, x="reason", y="orders"), width="stretch")

# -------------------------------------------------------------------- model
with tab_model:
    st.subheader("Delay-risk model")
    st.write("Models trained on the full dataset (75/25 train-test split) to predict whether an order will be late.")
    model, best_name, results, drivers, rules = get_model(df_all)
    st.dataframe(results, width="stretch", hide_index=True)
    st.caption(f"Best model by ROC-AUC: **{best_name}** (used for the risk estimator below).")

    a, b = st.columns(2)
    with a:
        st.markdown("**Top delay drivers** (logistic regression coefficients; positive = more late)")
        st.plotly_chart(px.bar(drivers.sort_values("coef"), x="coef", y="feature", orientation="h"), width="stretch")
    with b:
        st.markdown("**Decision-tree rules** (top levels)")
        st.code(rules, language="text")

    st.subheader("Try it: estimate delay risk for an order")
    d = df_all
    c1, c2, c3 = st.columns(3)
    wh = c1.selectbox("Warehouse", sorted(d["warehouse_name"].unique()), key="p_wh")
    ca = c2.selectbox("Carrier", sorted(d["carrier_name"].unique()), key="p_ca")
    pc = c3.selectbox("Product category", sorted(d["product_category"].unique()), key="p_pc")
    c4, c5, c6 = st.columns(3)
    zn = c4.selectbox("Destination zone", ["Metro", "Tier-2", "Tier-3", "Remote"], key="p_zn")
    dw = c5.selectbox("Order day", A.DOW, index=4, key="p_dw")
    se = c6.selectbox("Season", sorted(d["season"].unique()), key="p_se")
    c7, c8, c9 = st.columns(3)
    dist_km = c7.number_input("Distance (km)", 30, 2500, 600, step=50)
    wt = c8.number_input("Weight (kg)", 0.1, 60.0, 2.0, step=0.5)
    pdays = c9.selectbox("Warehouse processing days", [0, 1, 2], index=1)
    risk = predict_risk(
        model,
        dict(warehouse_name=wh, carrier_name=ca, product_category=pc, destination_zone=zn, day_of_week=dw, season=se,
             distance_km=dist_km, weight_kg=wt, order_value=float(d["order_value"].median()), processing_days=pdays),
    )
    st.metric("Estimated probability of late delivery", f"{risk * 100:.1f}%")
    st.progress(min(max(risk, 0.0), 1.0))

# ---------------------------------------------------------- recommendations
with tab_rec:
    st.subheader("Prioritised recommendations")
    st.write("Ranked by **excess late orders** = late orders above what the overall average rate would predict. Updates with the filters.")
    recs = A.recommendations(df)
    if not recs:
        st.info("No segment is meaningfully worse than average in the current selection.")
    for r in recs:
        with st.container(border=True):
            st.markdown(f"**#{r['priority']} · {r['area']}**")
            st.write(f"Late rate **{r['late_pct']}%** ({r['vs_avg']:+} pts vs average) · about **{r['excess_late_orders']}** extra late orders")
            st.write(f"➡️ {r['recommended_action']}")

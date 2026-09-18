"""Interactive Streamlit dashboard for the location prediction prototype."""
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from predict import build_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "destination_regressor.joblib"
st.set_page_config(page_title="Destination Prediction", page_icon="📍", layout="wide")
st.title("Destination Location Prediction")
st.caption("Portfolio prototype using origin coordinates and trip-start time")
if not MODEL.exists():
    st.warning("Run `python train.py` before opening the dashboard.")
    st.stop()
artifact = joblib.load(MODEL)
metrics = artifact["metrics"]
for warning in metrics["warnings"]:
    st.warning(warning)
a, b = st.columns(2)
with a:
    lat = st.number_input("Starting latitude", value=47.409291, format="%.6f")
with b:
    lon = st.number_input("Starting longitude", value=8.546942, format="%.6f")
start_time = st.datetime_input("Trip start time", value=datetime(2017, 5, 24, 12, 21))
if st.button("Predict destination", type="primary"):
    destination = artifact["model"].predict(build_features(lat, lon, start_time.isoformat()))[0]
    x, y, z = st.columns(3)
    x.metric("Latitude", f"{destination[0]:.6f}")
    y.metric("Longitude", f"{destination[1]:.6f}")
    z.metric("Median validation error", f"{metrics['evaluation']['median_distance_error_km']:.2f} km")
    st.map(pd.DataFrame({"lat": [lat, destination[0]], "lon": [lon, destination[1]]}), zoom=11)
st.subheader("Dataset and validation")
summary = metrics["dataset_summary"]
one, two, three, four = st.columns(4)
one.metric("Rows", summary["rows"])
two.metric("Unique routes", summary["unique_routes"])
three.metric("Largest route share", f"{summary['largest_route_share']:.1%}")
four.metric("Mean error", f"{metrics['evaluation']['mean_distance_error_km']:.2f} km")
plot = ROOT / "reports" / "evaluation_plots.png"
if plot.exists():
    st.image(str(plot), caption="Group-aware held-out evaluation")

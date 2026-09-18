"""Predict a destination coordinate from known trip-start information."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL = Path(__file__).resolve().parent / "models" / "destination_regressor.joblib"


def build_features(lat, lon, start_time):
    timestamp = pd.Timestamp(start_time)
    return pd.DataFrame([{"latStart": lat, "lonStart": lon, "start_hour": timestamp.hour, "start_dayofweek": timestamp.dayofweek, "start_month": timestamp.month}])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--start-time", required=True, help="Example: 2017-05-24T12:21:00")
    args = parser.parse_args()
    if not MODEL.exists():
        raise FileNotFoundError("Run `python train.py` first.")
    destination = joblib.load(MODEL)["model"].predict(build_features(args.lat, args.lon, args.start_time))[0]
    print(f"Predicted destination: latitude={destination[0]:.6f}, longitude={destination[1]:.6f}")

"""Train a leakage-free destination coordinate prediction prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "location_dataset.csv"
FEATURES = ["latStart", "lonStart", "start_hour", "start_dayofweek", "start_month"]
TARGETS = ["latEnd", "lonEnd"]


def haversine_km(actual, predicted):
    lat1, lon1 = np.radians(actual[:, 0]), np.radians(actual[:, 1])
    lat2, lon2 = np.radians(predicted[:, 0]), np.radians(predicted[:, 1])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def load_data(path):
    df = pd.read_csv(path)
    required = {"eventTimeStart", "latStart", "lonStart", "latEnd", "lonEnd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["eventTimeStart"] = pd.to_datetime(df["eventTimeStart"], errors="coerce")
    df[["latStart", "lonStart", "latEnd", "lonEnd"]] = df[["latStart", "lonStart", "latEnd", "lonEnd"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required).copy()
    df["start_hour"] = df.eventTimeStart.dt.hour
    df["start_dayofweek"] = df.eventTimeStart.dt.dayofweek
    df["start_month"] = df.eventTimeStart.dt.month
    return df


def distance_scorer(estimator, x, y):
    return -float(haversine_km(y.to_numpy(), estimator.predict(x)).mean())


def train(data_path=DATA):
    df = load_data(data_path)
    x, y = df[FEATURES], df[TARGETS]
    groups = df.latStart.astype(str) + "," + df.lonStart.astype(str)
    routes = df.groupby(["latStart", "lonStart", "latEnd", "lonEnd"]).size()
    summary = {
        "rows": len(df), "unique_origins": int(groups.nunique()), "unique_destinations": int(df[TARGETS].drop_duplicates().shape[0]),
        "unique_routes": len(routes), "largest_route_share": round(float(routes.max() / len(df)), 4),
    }
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_i, test_i = next(split.split(x, y, groups))
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1),
    }
    folds = min(5, groups.nunique())
    cv = GroupKFold(n_splits=folds)
    comparison = {}
    for name, model in models.items():
        scores = cross_val_score(model, x, y, groups=groups, cv=cv, scoring=distance_scorer, n_jobs=1)
        comparison[name] = {"mean_distance_km": round(float(-scores.mean()), 3), "std_distance_km": round(float(scores.std()), 3)}
    best_name = min(comparison, key=lambda name: comparison[name]["mean_distance_km"])
    model = models[best_name].fit(x.iloc[train_i], y.iloc[train_i])
    predicted = model.predict(x.iloc[test_i])
    distances = haversine_km(y.iloc[test_i].to_numpy(), predicted)
    model.fit(x, y)
    metrics = {
        "best_model": best_name, "feature_columns": FEATURES, "dataset_summary": summary, "cross_validation": comparison,
        "evaluation": {"test_rows": len(test_i), "mean_distance_error_km": round(float(distances.mean()), 3), "median_distance_error_km": round(float(np.median(distances)), 3), "p95_distance_error_km": round(float(np.percentile(distances, 95)), 3)},
        "warnings": ["The dataset is route-imbalanced. Treat results as a prototype, not production accuracy.", "There are too few unique routes for a robust generalization claim."],
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    actual = y.iloc[test_i].to_numpy()
    axes[0].scatter(actual[:, 1], actual[:, 0], label="Actual")
    axes[0].scatter(predicted[:, 1], predicted[:, 0], marker="x", label="Predicted")
    axes[0].set(xlabel="Longitude", ylabel="Latitude", title="Held-out destinations")
    axes[0].legend()
    axes[1].hist(distances, bins=min(20, max(5, len(distances) // 2)), color="#2563eb")
    axes[1].set(xlabel="Prediction error (km)", ylabel="Trips", title="Held-out distance error")
    fig.tight_layout()
    fig.savefig(reports / "evaluation_plots.png", dpi=160)
    joblib.dump({"model": model, "metrics": metrics}, ROOT / "models" / "destination_regressor.joblib")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    print(json.dumps(train(parser.parse_args().data), indent=2))

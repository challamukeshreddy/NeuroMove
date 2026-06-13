"""Train the NeuroMove motor-intent classifier.

Run:
    python train.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.neuro_utils import (
    LABELS,
    METRICS_PATH,
    MODEL_PATH,
    clean_emg_data,
    compute_reduction,
    ensure_project_dirs,
    load_emg_dataset,
    spike_encode,
)


def train_model() -> dict:
    """Load data, preprocess it, train the model, and save artifacts."""

    ensure_project_dirs()
    raw_df = load_emg_dataset()
    df, label_column, feature_columns = clean_emg_data(raw_df)

    X = df[feature_columns].to_numpy()
    y = df[label_column].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)

    spikes = spike_encode(X_test, threshold=0.08)
    reduction = compute_reduction(X_test, spikes)

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "labels": LABELS,
        "spike_threshold": 0.08,
    }
    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
        "raw_samples": reduction["raw_samples"],
        "spike_events": reduction["spike_events"],
        "reduction_percent": reduction["reduction_percent"],
        "dataset_rows": int(len(df)),
        "feature_count": int(len(feature_columns)),
    }

    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return metrics


if __name__ == "__main__":
    training_metrics = train_model()
    print("NeuroMove training complete.")
    print(json.dumps(training_metrics, indent=2))

"""Shared utilities for NeuroMove training and dashboard code.

The functions in this file are intentionally small and beginner friendly so
that hackathon judges and teammates can quickly understand the full pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = DATA_DIR / "emg_gesture_data.csv"
MODEL_PATH = MODELS_DIR / "neuromove_rf_model.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.joblib"

LABELS = ["Open Hand", "Close Hand", "Rest"]
DEVICE_ICONS = {
    "Open Hand": "🖐️",
    "Close Hand": "✊",
    "Rest": "🤚",
}


def ensure_project_dirs() -> None:
    """Create project folders used by the prototype."""

    for folder in [DATA_DIR, MODELS_DIR, PROJECT_ROOT / "screenshots", PROJECT_ROOT / "notebooks"]:
        folder.mkdir(parents=True, exist_ok=True)


def generate_demo_emg_dataset(
    output_path: Path = DATASET_PATH,
    samples_per_class: int = 700,
    channels: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate an EMG-like dataset for offline demos.

    The shapes mimic common public EMG hand gesture datasets: multiple channels
    sampled over time, with higher activity during movement and low-amplitude
    activity during rest. Replace this CSV with a real public dataset for a
    formal experiment.
    """

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []
    channel_names = [f"emg_ch{i + 1}" for i in range(channels)]

    gesture_settings = {
        "Open Hand": {"base": 0.45, "frequency": 2.0, "phase": 0.0},
        "Close Hand": {"base": 0.85, "frequency": 3.4, "phase": 0.7},
        "Rest": {"base": 0.08, "frequency": 0.8, "phase": 1.4},
    }

    for label, settings in gesture_settings.items():
        for index in range(samples_per_class):
            t = index / samples_per_class
            row: dict[str, float | str] = {"label": label}

            for channel_index, channel in enumerate(channel_names):
                channel_gain = 1.0 + channel_index * 0.08
                wave = np.sin(2 * np.pi * settings["frequency"] * t + settings["phase"])
                burst = max(0.0, wave) * settings["base"] * channel_gain
                noise = rng.normal(0.0, 0.035)
                rest_jitter = rng.normal(0.0, 0.018) if label == "Rest" else 0.0
                row[channel] = float(np.clip(settings["base"] + burst + noise + rest_jitter, 0.0, 2.0))

            rows.append(row)

    dataset = pd.DataFrame(rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def find_label_column(columns: Iterable[str]) -> str:
    """Find a likely class label column in a public EMG CSV."""

    candidates = ["label", "gesture", "class", "target", "movement", "intent"]
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError(
        "Could not find a label column. Use one of: label, gesture, class, target, movement, intent."
    )


def load_emg_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load EMG data, generating the demo dataset when no CSV is present."""

    ensure_project_dirs()
    if not path.exists():
        return generate_demo_emg_dataset(path)
    return pd.read_csv(path)


def clean_emg_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """Clean a raw EMG dataframe and return numeric features plus labels."""

    df = df.copy()
    label_column = find_label_column(df.columns)

    # Keep numeric EMG columns and the label. Missing numeric values are filled
    # with the column median to keep the demo robust with real-world CSV files.
    feature_columns = [column for column in df.columns if column != label_column]
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric EMG feature columns found in the dataset.")

    df = df[numeric_features + [label_column]].dropna(subset=[label_column])
    for column in numeric_features:
        df[column] = df[column].fillna(df[column].median())

    # Normalize label text into the three target classes used by the prototype.
    df[label_column] = df[label_column].astype(str).str.strip()
    label_map = {
        "open": "Open Hand",
        "open hand": "Open Hand",
        "1": "Open Hand",
        "close": "Close Hand",
        "close hand": "Close Hand",
        "closed hand": "Close Hand",
        "fist": "Close Hand",
        "2": "Close Hand",
        "rest": "Rest",
        "idle": "Rest",
        "0": "Rest",
    }
    df[label_column] = df[label_column].str.lower().map(label_map).fillna(df[label_column])
    df = df[df[label_column].isin(LABELS)].reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset does not contain Open Hand, Close Hand, or Rest labels.")

    return df, label_column, numeric_features


def spike_encode(signal: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """Convert EMG samples into event spikes using signal-change thresholding."""

    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    changes = np.abs(np.diff(signal, axis=0, prepend=signal[[0], :]))
    return (changes > threshold).astype(int)


def compute_reduction(raw_signal: np.ndarray, spikes: np.ndarray) -> dict[str, float]:
    """Estimate event-driven computational reduction from spike sparsity."""

    raw_samples = int(np.asarray(raw_signal).size)
    spike_events = int(np.asarray(spikes).sum())
    reduction = 0.0 if raw_samples == 0 else (1.0 - spike_events / raw_samples) * 100.0
    return {
        "raw_samples": raw_samples,
        "spike_events": spike_events,
        "reduction_percent": round(max(0.0, reduction), 2),
    }


def load_model_bundle(path: Path = MODEL_PATH) -> dict:
    """Load the trained model bundle created by train.py."""

    return joblib.load(path)

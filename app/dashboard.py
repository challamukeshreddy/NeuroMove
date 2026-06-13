"""Streamlit dashboard for the NeuroMove prototype."""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.neuro_utils import (
    DEVICE_ICONS,
    LABELS,
    METRICS_PATH,
    MODEL_PATH,
    clean_emg_data,
    compute_reduction,
    load_emg_dataset,
    load_model_bundle,
    spike_encode,
)
from train import train_model


st.set_page_config(
    page_title="NeuroMove",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def cached_dataset() -> tuple[pd.DataFrame, str, list[str]]:
    """Load and clean EMG data once per Streamlit session."""

    raw_df = load_emg_dataset()
    return clean_emg_data(raw_df)


@st.cache_resource
def cached_model() -> dict:
    """Load the trained model, training automatically for first-time users."""

    if not MODEL_PATH.exists():
        train_model()
    return load_model_bundle()


def load_metrics() -> dict:
    """Load saved metrics, creating them when needed."""

    if not METRICS_PATH.exists():
        train_model()
    return joblib.load(METRICS_PATH)


def plot_signal(values: np.ndarray, title: str, ylabel: str) -> plt.Figure:
    """Create a compact line plot for raw EMG or spike data."""

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("Sample Window")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def predict_intent(bundle: dict, sample: pd.DataFrame) -> tuple[str, float, dict[str, float]]:
    """Predict motor intent and return class probabilities."""

    scaler = bundle["scaler"]
    model = bundle["model"]
    scaled_sample = scaler.transform(sample)
    prediction = model.predict(scaled_sample)[0]
    probabilities = model.predict_proba(scaled_sample)[0]
    confidence_by_label = dict(zip(model.classes_, probabilities))
    confidence = float(confidence_by_label[prediction])
    return prediction, confidence, confidence_by_label


def main() -> None:
    """Render the NeuroMove dashboard."""

    st.title("NeuroMove")
    st.caption("Neuromorphic EMG motor-intent detection for stroke rehabilitation")

    df, label_column, feature_columns = cached_dataset()
    bundle = cached_model()
    metrics = load_metrics()

    page = st.sidebar.radio(
        "Dashboard",
        ["Live Demo", "Model Performance", "Dataset"],
    )
    threshold = st.sidebar.slider(
        "Spike threshold",
        min_value=0.01,
        max_value=0.30,
        value=float(bundle.get("spike_threshold", 0.08)),
        step=0.01,
    )
    window_size = st.sidebar.slider("Signal window", 20, 180, 80, 10)

    if page == "Live Demo":
        st.subheader("Motor Intent Demo")
        row_index = st.slider("Select EMG sample", 0, len(df) - 1, 0)
        start_index = max(0, row_index - window_size // 2)
        end_index = min(len(df), start_index + window_size)
        signal_window = df.iloc[start_index:end_index][feature_columns].to_numpy()
        sample = df.iloc[[row_index]][feature_columns]

        spikes = spike_encode(signal_window, threshold=threshold)
        reduction = compute_reduction(signal_window, spikes)
        prediction, confidence, confidence_by_label = predict_intent(bundle, sample)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Predicted Intent", prediction)
        metric_cols[1].metric("Confidence", f"{confidence * 100:.1f}%")
        metric_cols[2].metric("Device Status", DEVICE_ICONS.get(prediction, "🤚"))
        metric_cols[3].metric("Compute Reduction", f"{reduction['reduction_percent']:.1f}%")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.pyplot(plot_signal(signal_window[:, 0], "Raw EMG Signal", "Amplitude"))
        with chart_cols[1]:
            st.pyplot(plot_signal(spikes[:, 0], "Spike-Encoded Signal", "Spike"))

        st.progress(confidence)
        st.write("Confidence by class")
        st.bar_chart(pd.Series(confidence_by_label).reindex(LABELS).fillna(0.0))

        st.write("Event-driven processing comparison")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Metric": "Raw samples processed", "Value": reduction["raw_samples"]},
                    {"Metric": "Spike events processed", "Value": reduction["spike_events"]},
                    {"Metric": "Estimated reduction (%)", "Value": reduction["reduction_percent"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    elif page == "Model Performance":
        st.subheader("Random Forest Performance")
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        st.metric("Training Dataset Rows", metrics["dataset_rows"])

        matrix = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=25)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(column, row, matrix[row, column], ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)

        st.write("Saved training metrics")
        st.json(metrics)

    else:
        st.subheader("Dataset Preview")
        st.write(f"Detected label column: `{label_column}`")
        st.write(f"Numeric EMG features: {len(feature_columns)}")
        st.dataframe(df.head(50), use_container_width=True)
        st.bar_chart(df[label_column].value_counts().reindex(LABELS).fillna(0))


if __name__ == "__main__":
    main()

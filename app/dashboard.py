"""Streamlit dashboard for the NeuroMove prototype."""

from __future__ import annotations

import html

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.neuro_utils import (
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
    page_icon=":brain:",
    layout="wide",
)


def apply_design_system() -> None:
    """Apply the NeuroMove healthcare AI visual system."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: Inter, Segoe UI, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 8%, rgba(34, 211, 238, .16), transparent 30%),
                radial-gradient(circle at 82% 0%, rgba(45, 212, 191, .12), transparent 26%),
                linear-gradient(135deg, #020617 0%, #07111f 50%, #0f172a 100%);
            color: #e5f6ff;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 47, 73, .94), rgba(2, 6, 23, .98));
            border-right: 1px solid rgba(34, 211, 238, .16);
            box-shadow: 12px 0 42px rgba(0, 0, 0, .24);
        }

        [data-testid="stSidebar"] * {
            color: #dbeafe;
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 10px;
        }

        [data-testid="stSidebar"] label {
            border: 1px solid rgba(34, 211, 238, .10);
            border-radius: 14px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: rgba(15, 23, 42, .36);
            transition: transform .18s ease, border-color .18s ease, background .18s ease;
        }

        [data-testid="stSidebar"] label:hover {
            transform: translateX(3px);
            border-color: rgba(34, 211, 238, .38);
            background: rgba(14, 116, 144, .24);
        }

        [data-testid="stSidebar"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(8, 145, 178, .36), rgba(15, 23, 42, .74));
            border-color: rgba(103, 232, 249, .72);
            box-shadow: 0 12px 32px rgba(8, 145, 178, .16);
        }

        .sidebar-brand {
            background: rgba(15, 23, 42, .58);
            border: 1px solid rgba(34, 211, 238, .24);
            border-radius: 18px;
            padding: 18px;
            margin: 4px 0 22px 0;
            box-shadow: 0 16px 38px rgba(0, 0, 0, .22);
        }

        .sidebar-logo {
            align-items: center;
            display: flex;
            gap: 10px;
            font-size: 1.25rem;
            font-weight: 800;
        }

        .sidebar-logo-icon {
            align-items: center;
            background: rgba(34, 211, 238, .14);
            border: 1px solid rgba(103, 232, 249, .38);
            border-radius: 14px;
            display: inline-flex;
            height: 38px;
            justify-content: center;
            width: 38px;
        }

        .sidebar-tagline {
            color: #a5f3fc;
            font-size: .82rem;
            line-height: 1.35;
            margin-top: 10px;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }

        h1, h2, h3 {
            color: #f8fafc;
            letter-spacing: 0;
        }

        p, li, label, span {
            letter-spacing: 0;
        }

        .hero {
            backdrop-filter: blur(20px);
            background:
                linear-gradient(135deg, rgba(8, 47, 73, .76), rgba(15, 23, 42, .72)),
                radial-gradient(circle at 88% 18%, rgba(34, 211, 238, .18), transparent 28%);
            border: 1px solid rgba(125, 211, 252, .26);
            border-radius: 26px;
            padding: 36px;
            margin-bottom: 24px;
            box-shadow: 0 26px 80px rgba(0, 0, 0, .34);
            overflow: hidden;
            position: relative;
        }

        .hero::after {
            background: linear-gradient(90deg, transparent, rgba(34, 211, 238, .26), transparent);
            content: "";
            height: 1px;
            left: 30px;
            position: absolute;
            right: 30px;
            top: 0;
        }

        .hero-title {
            align-items: center;
            display: flex;
            gap: 16px;
        }

        .brain-pulse {
            align-items: center;
            background: rgba(34, 211, 238, .12);
            border: 1px solid rgba(103, 232, 249, .48);
            border-radius: 999px;
            box-shadow: 0 0 0 rgba(34, 211, 238, .32);
            display: inline-flex;
            font-size: clamp(2rem, 4vw, 3.5rem);
            height: clamp(64px, 8vw, 86px);
            justify-content: center;
            min-width: clamp(64px, 8vw, 86px);
            animation: pulseGlow 2.4s infinite;
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(34, 211, 238, .34); }
            70% { box-shadow: 0 0 0 18px rgba(34, 211, 238, 0); }
            100% { box-shadow: 0 0 0 0 rgba(34, 211, 238, 0); }
        }

        .hero h1 {
            font-size: clamp(2.4rem, 5vw, 4.4rem);
            line-height: 1.02;
            margin: 0;
            font-weight: 800;
        }

        .hero .subtitle {
            color: #67e8f9;
            font-size: 1.24rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .hero .description {
            color: #cbd5e1;
            font-size: 1.04rem;
            max-width: 760px;
            margin-bottom: 0;
        }

        .section-title {
            align-items: center;
            color: #f8fafc;
            display: flex;
            gap: 10px;
            font-size: 1.35rem;
            font-weight: 800;
            margin: 22px 0 12px 0;
        }

        .section-title::after {
            background: linear-gradient(90deg, rgba(34, 211, 238, .45), transparent);
            content: "";
            flex: 1;
            height: 1px;
            margin-left: 6px;
        }

        .card {
            background: rgba(15, 23, 42, .86);
            border: 1px solid rgba(34, 211, 238, .22);
            border-radius: 18px;
            padding: 22px;
            min-height: 138px;
            box-shadow: 0 18px 48px rgba(0, 0, 0, .24);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        .card:hover, .comparison-card:hover, .workflow-card:hover, .chart-card:hover {
            border-color: rgba(103, 232, 249, .42);
            box-shadow: 0 24px 62px rgba(0, 0, 0, .34);
            transform: translateY(-2px);
        }

        .metric-icon {
            color: #22d3ee;
            font-size: 1.8rem;
            line-height: 1;
            margin-bottom: 12px;
        }

        .metric-label {
            color: #94a3b8;
            font-size: .9rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .metric-help {
            color: #a7f3d0;
            font-size: .9rem;
            margin-top: 8px;
        }

        .workflow-card {
            background: rgba(8, 47, 73, .82);
            border: 1px solid rgba(125, 211, 252, .24);
            border-radius: 16px;
            padding: 18px 14px;
            min-height: 112px;
            text-align: center;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        .workflow-icon {
            color: #67e8f9;
            font-size: 1.65rem;
            margin-bottom: 8px;
        }

        .workflow-title {
            color: #f8fafc;
            font-weight: 800;
            font-size: .98rem;
        }

        .arrow {
            color: #22d3ee;
            font-size: 2.1rem;
            font-weight: 800;
            text-align: center;
            padding-top: 32px;
            text-shadow: 0 0 18px rgba(34, 211, 238, .45);
        }

        .comparison-card {
            background: rgba(2, 6, 23, .74);
            border: 1px solid rgba(34, 211, 238, .18);
            border-radius: 18px;
            padding: 24px;
            min-height: 190px;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        .comparison-card h3 {
            margin-top: 0;
            margin-bottom: 10px;
        }

        .comparison-card p {
            color: #cbd5e1;
            font-size: 1rem;
            margin: 0;
        }

        .footer {
            border-top: 1px solid rgba(34, 211, 238, .18);
            color: #94a3b8;
            margin-top: 34px;
            padding-top: 18px;
            text-align: center;
        }

        .chart-card {
            background: rgba(15, 23, 42, .72);
            border: 1px solid rgba(34, 211, 238, .18);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 18px 48px rgba(0, 0, 0, .20);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        .impact-grid {
            display: grid;
            gap: 14px;
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .impact-card {
            background: rgba(8, 47, 73, .58);
            border: 1px solid rgba(34, 211, 238, .18);
            border-radius: 16px;
            padding: 18px;
        }

        .impact-card h3 {
            color: #67e8f9;
            font-size: .92rem;
            margin: 0 0 8px 0;
            text-transform: uppercase;
        }

        .impact-card p, .impact-card li {
            color: #cbd5e1;
            font-size: .96rem;
            margin: 0;
        }

        .impact-card ul {
            margin: 0;
            padding-left: 18px;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, .86);
            border: 1px solid rgba(34, 211, 238, .20);
            border-radius: 16px;
            padding: 18px;
        }

        .stDataFrame, .stPlotlyChart, [data-testid="stExpander"] {
            border-radius: 16px;
        }

        @media (max-width: 900px) {
            .hero {
                padding: 26px;
            }

            .hero-title {
                align-items: flex-start;
                flex-direction: column;
            }

            .impact-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def safe_text(value: object) -> str:
    """Escape user or data-derived values before rendering custom HTML."""

    return html.escape(str(value))


def metric_card(title: str, value: str, icon: str, help_text: str = "") -> str:
    """Return a reusable dashboard metric card."""

    help_markup = f"<div class='metric-help'>{safe_text(help_text)}</div>" if help_text else ""
    return f"""
    <div class="card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{safe_text(title)}</div>
        <div class="metric-value">{safe_text(value)}</div>
        {help_markup}
    </div>
    """


def section_title(icon: str, title: str) -> None:
    """Render a consistent section heading with a compact icon."""

    st.markdown(f"<div class='section-title'>{icon} {safe_text(title)}</div>", unsafe_allow_html=True)


def chart_card_start() -> None:
    """Start a lightweight visual card around a chart."""

    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)


def chart_card_end() -> None:
    """Close a lightweight visual card around a chart."""

    st.markdown("</div>", unsafe_allow_html=True)


def render_hero() -> None:
    """Render the healthcare product hero section."""

    st.markdown(
        """
        <section class="hero">
            <div class="hero-title">
                <span class="brain-pulse">&#129504;</span>
                <h1>NeuroMove</h1>
            </div>
            <div class="subtitle">Neuromorphic EMG Motor Intent Detection for Stroke Rehabilitation</div>
            <p class="description">
                Detecting patient movement intentions using low-power event-driven AI inspired by the human brain.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(prediction: str, confidence: float, reduction: dict[str, float]) -> None:
    """Render large hackathon-ready prediction cards."""

    intent_icon = {
        "Open Hand": "&#9995;",
        "Close Hand": "&#9994;",
        "Rest": "&#128564;",
    }.get(prediction, "&#129306;")
    device_icon = {
        "Open Hand": "&#9995;",
        "Close Hand": "&#9994;",
        "Rest": "&#129306;",
    }.get(prediction, "&#129306;")
    device_label = {
        "Open Hand": "Open Assist",
        "Close Hand": "Close Assist",
        "Rest": "Rest Mode",
    }.get(prediction, prediction)

    cards = st.columns(4)
    cards[0].markdown(
        metric_card("Predicted Intent", prediction, intent_icon, "Current motor intent"),
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        metric_card("Confidence", f"{confidence * 100:.1f}%", "&#128200;", "Model certainty"),
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        metric_card("Device Action", device_label, device_icon, "Assistive response"),
        unsafe_allow_html=True,
    )
    cards[3].markdown(
        metric_card("Compute Reduction", f"{reduction['reduction_percent']:.1f}%", "&#9889;", "Event-driven savings"),
        unsafe_allow_html=True,
    )


def render_patient_journey() -> None:
    """Render the end-to-end rehabilitation workflow."""

    section_title("&#128694;", "Patient Journey")
    steps = [
        ("&#129489;", "Patient Attempts Movement"),
        ("&#128246;", "EMG Sensor Captures Signals"),
        ("&#9889;", "Spike Encoder"),
        ("&#129504;", "AI Intent Prediction"),
        ("&#129470;", "Assistive Device Action"),
    ]

    columns = st.columns([1, 0.18, 1, 0.18, 1, 0.18, 1, 0.18, 1])
    for index, (icon, title) in enumerate(steps):
        columns[index * 2].markdown(
            f"""
            <div class="workflow-card">
                <div class="workflow-icon">{icon}</div>
                <div class="workflow-title">{safe_text(title)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if index < len(steps) - 1:
            columns[index * 2 + 1].markdown("<div class='arrow'>&#8594;</div>", unsafe_allow_html=True)


def render_neuromorphic_innovation(reduction: dict[str, float]) -> None:
    """Render the traditional vs neuromorphic processing comparison."""

    section_title("&#9889;", "Neuromorphic Innovation")
    left, right = st.columns(2)
    left.markdown(
        """
        <div class="comparison-card">
            <h3>Traditional AI</h3>
            <p>Processes every signal sample continuously, even when the patient signal is quiet or unchanged.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    right.markdown(
        """
        <div class="comparison-card">
            <h3>Neuromorphic AI</h3>
            <p>Processes only important spike events caused by meaningful signal changes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_col, spike_col, reduction_col = st.columns(3)
    raw_col.markdown(
        metric_card("Raw Samples", f"{reduction['raw_samples']:,}", "&#128202;"),
        unsafe_allow_html=True,
    )
    spike_col.markdown(
        metric_card("Spike Events", f"{reduction['spike_events']:,}", "&#9889;"),
        unsafe_allow_html=True,
    )
    reduction_col.markdown(
        metric_card("Reduction", f"{reduction['reduction_percent']:.1f}%", "&#127919;"),
        unsafe_allow_html=True,
    )


def render_architecture() -> None:
    """Render the system architecture page."""

    render_hero()
    section_title("&#128736;", "System Architecture")
    stages = [
        ("&#128202;", "EMG Signals", "Multi-channel muscle activity captured from the patient."),
        ("&#9889;", "Spike Encoding", "Signal changes above threshold become sparse events."),
        ("&#128295;", "Feature Extraction", "Clean normalized EMG features are prepared for inference."),
        ("&#127795;", "Random Forest Model", "A lightweight classifier predicts rehabilitation intent."),
        ("&#129504;", "Motor Intent Detection", "Open Hand, Close Hand, or Rest is selected."),
        ("&#129470;", "Assistive Device Control", "The virtual rehabilitation device responds instantly."),
    ]

    for index, (icon, title, detail) in enumerate(stages):
        st.markdown(
            f"""
            <div class="card" style="min-height:auto;margin-bottom:12px;">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="font-size:1.35rem;">{safe_text(title)}</div>
                <div class="metric-help">{safe_text(detail)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if index < len(stages) - 1:
            st.markdown("<div class='arrow' style='padding:2px 0 10px 0;'>&#8595;</div>", unsafe_allow_html=True)


def render_future_scope() -> None:
    """Render the future scope page."""

    render_hero()
    section_title("&#128640;", "Future Scope")
    items = [
        ("&#128246;", "Real-time EMG Sensors"),
        ("&#129504;", "EEG Integration"),
        ("&#9889;", "Spiking Neural Networks"),
        ("&#128421;", "Neuromorphic Hardware"),
        ("&#129470;", "Robotic Hand Control"),
        ("&#127973;", "Clinical Deployment"),
    ]

    for row in range(0, len(items), 3):
        columns = st.columns(3)
        for column, (icon, item) in zip(columns, items[row : row + 3]):
            column.markdown(
                f"""
                <div class="card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value" style="font-size:1.25rem;">{safe_text(item)}</div>
                    <div class="metric-help">Next milestone for clinical-grade rehabilitation impact.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_project_impact() -> None:
    """Render a compact hackathon impact summary near the bottom."""

    section_title("&#127919;", "Project Impact")
    st.markdown(
        """
        <div class="impact-grid">
            <div class="impact-card">
                <h3>Problem</h3>
                <p>Stroke patients may struggle to perform intended movements.</p>
            </div>
            <div class="impact-card">
                <h3>Solution</h3>
                <p>
                    NeuroMove detects motor intent from EMG signals and triggers assistive actions
                    using neuromorphic-inspired event-driven processing.
                </p>
            </div>
            <div class="impact-card">
                <h3>Benefits</h3>
                <ul>
                    <li>Lower compute requirements</li>
                    <li>Faster response time</li>
                    <li>Rehabilitation assistance</li>
                    <li>Scalable future deployment</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the hackathon demo footer."""

    st.markdown(
        """
        <div class="footer">
            <strong>NeuroMove</strong><br>
            Neuromorphic Stroke Rehabilitation Assistant<br>
            Built for Hackathon Prototype Demonstration
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render premium sidebar branding and return the selected page."""

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">
                <span class="sidebar-logo-icon">&#129504;</span>
                <span>NeuroMove</span>
            </div>
            <div class="sidebar-tagline">Neuromorphic Stroke Rehabilitation Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_options = {
        "Live Demo": "\U0001f4c8 Live Demo",
        "Architecture": "\U0001f6e0 Architecture",
        "Dataset": "\U0001f4ca Dataset",
        "Model Performance": "\U0001f3af Model Performance",
        "Future Scope": "\U0001f680 Future Scope",
    }
    selected_label = st.sidebar.radio("Navigation", list(page_options.values()))
    return next(page for page, label in page_options.items() if label == selected_label)


def plot_signal(values: np.ndarray, title: str, ylabel: str) -> plt.Figure:
    """Create a polished line plot for raw EMG or spike data."""

    fig, ax = plt.subplots(figsize=(10, 3.3), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(values, color="#22d3ee", linewidth=2.0)
    ax.set_title(title, color="#f8fafc", fontweight="bold")
    ax.set_xlabel("Sample Window", color="#cbd5e1")
    ax.set_ylabel(ylabel, color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, color="#334155", alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix: np.ndarray) -> plt.Figure:
    """Create a healthcare-themed confusion matrix."""

    fig, ax = plt.subplots(figsize=(5.8, 4.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=20)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted", color="#cbd5e1")
    ax.set_ylabel("Actual", color="#cbd5e1")
    ax.set_title("Confusion Matrix", color="#f8fafc", fontweight="bold")
    ax.tick_params(colors="#cbd5e1")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, matrix[row, column], ha="center", va="center", color="#0f172a")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.tick_params(colors="#cbd5e1")
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


def render_live_demo(df: pd.DataFrame, feature_columns: list[str], bundle: dict) -> None:
    """Render the primary motor-intent demo page."""

    render_hero()

    threshold = st.sidebar.slider(
        "Spike threshold",
        min_value=0.01,
        max_value=0.30,
        value=float(bundle.get("spike_threshold", 0.08)),
        step=0.01,
    )
    window_size = st.sidebar.slider("Signal window", 20, 180, 80, 10)
    row_index = st.slider("Select patient EMG sample", 0, len(df) - 1, 0)

    start_index = max(0, row_index - window_size // 2)
    end_index = min(len(df), start_index + window_size)
    signal_window = df.iloc[start_index:end_index][feature_columns].to_numpy()
    sample = df.iloc[[row_index]][feature_columns]

    spikes = spike_encode(signal_window, threshold=threshold)
    reduction = compute_reduction(signal_window, spikes)
    prediction, confidence, confidence_by_label = predict_intent(bundle, sample)

    render_metrics(prediction, confidence, reduction)
    render_patient_journey()

    chart_cols = st.columns(2)
    with chart_cols[0]:
        section_title("&#128202;", "Raw EMG Signal")
        chart_card_start()
        st.pyplot(plot_signal(signal_window[:, 0], "Raw EMG Signal", "Amplitude"), use_container_width=True)
        chart_card_end()
    with chart_cols[1]:
        section_title("&#9889;", "Spike-Encoded Signal")
        chart_card_start()
        st.pyplot(plot_signal(spikes[:, 0], "Spike-Encoded Signal", "Spike"), use_container_width=True)
        chart_card_end()

    confidence_col, distribution_col = st.columns([1, 1.2])
    with confidence_col:
        section_title("&#129504;", "Prediction Confidence")
        chart_card_start()
        st.progress(confidence)
        st.caption(f"The model is {confidence * 100:.1f}% confident that the intended movement is {prediction}.")
        chart_card_end()
    with distribution_col:
        section_title("&#128200;", "Confidence by Class")
        chart_card_start()
        st.bar_chart(pd.Series(confidence_by_label).reindex(LABELS).fillna(0.0))
        chart_card_end()

    render_neuromorphic_innovation(reduction)


def render_model_performance(metrics: dict) -> None:
    """Render the redesigned model performance page."""

    render_hero()
    section_title("&#127919;", "Model Performance")

    summary_cols = st.columns(3)
    summary_cols[0].markdown(
        metric_card("Accuracy", f"{metrics['accuracy'] * 100:.2f}%", "&#127919;", "Test dataset score"),
        unsafe_allow_html=True,
    )
    summary_cols[1].markdown(
        metric_card("Dataset Rows", f"{metrics['dataset_rows']:,}", "&#128196;", "Training dataset size"),
        unsafe_allow_html=True,
    )
    summary_cols[2].markdown(
        metric_card("EMG Channels", f"{metrics['feature_count']}", "&#128246;", "Numeric signal channels"),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="comparison-card" style="margin-top:16px;">
            <h3>Performance Summary</h3>
            <p>{metrics['accuracy'] * 100:.2f}% classification accuracy achieved on the test dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    matrix = np.array(metrics["confusion_matrix"])
    left, right = st.columns([1, 1])
    with left:
        section_title("&#128290;", "Confusion Matrix")
        chart_card_start()
        st.pyplot(plot_confusion_matrix(matrix), use_container_width=True)
        chart_card_end()
    with right:
        section_title("&#128161;", "Why It Matters")
        st.markdown(
            """
            <div class="comparison-card">
                <h3>Why it matters</h3>
                <p>
                    A lightweight Random Forest provides fast inference for a rehabilitation demo,
                    while the spike layer reduces event processing compared with raw continuous sampling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dataset_page(df: pd.DataFrame, label_column: str, feature_columns: list[str]) -> None:
    """Render a polished dataset summary page."""

    render_hero()
    section_title("&#128202;", "Dataset Summary")

    class_counts = df[label_column].value_counts().reindex(LABELS).fillna(0).astype(int)
    summary_cols = st.columns(4)
    summary_cols[0].markdown(metric_card("Total Samples", f"{len(df):,}", "&#128202;"), unsafe_allow_html=True)
    summary_cols[1].markdown(metric_card("EMG Channels", f"{len(feature_columns)}", "&#128246;"), unsafe_allow_html=True)
    summary_cols[2].markdown(metric_card("Classes", f"{len(LABELS)}", "&#129504;"), unsafe_allow_html=True)
    summary_cols[3].markdown(metric_card("Label Column", label_column, "&#127991;"), unsafe_allow_html=True)

    chart_cols = st.columns([1.2, 1])
    with chart_cols[0]:
        section_title("&#128200;", "Class Distribution")
        chart_card_start()
        st.bar_chart(class_counts)
        chart_card_end()
    with chart_cols[1]:
        section_title("&#9899;", "Class Balance")
        chart_card_start()
        fig, ax = plt.subplots(figsize=(4.8, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.pie(
            class_counts,
            labels=class_counts.index,
            autopct="%1.0f%%",
            colors=["#22d3ee", "#2dd4bf", "#60a5fa"],
            textprops={"color": "#f8fafc"},
        )
        ax.set_title("Class Balance", color="#f8fafc", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        chart_card_end()

    with st.expander("View raw dataset"):
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)


def main() -> None:
    """Render the NeuroMove dashboard."""

    apply_design_system()

    df, label_column, feature_columns = cached_dataset()
    bundle = cached_model()
    metrics = load_metrics()

    page = render_sidebar()

    if page == "Live Demo":
        render_live_demo(df, feature_columns, bundle)
    elif page == "Architecture":
        render_architecture()
    elif page == "Dataset":
        render_dataset_page(df, label_column, feature_columns)
    elif page == "Model Performance":
        render_model_performance(metrics)
    else:
        render_future_scope()

    render_project_impact()
    render_footer()


if __name__ == "__main__":
    main()
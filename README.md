# 🧠 NeuroMove

<div align="center">

## Neuromorphic EMG Motor Intent Detection System for Stroke Rehabilitation

**Detecting patient movement intentions using low-power, event-driven AI inspired by the human brain.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Pipeline-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Hackathon](https://img.shields.io/badge/Hackathon-Prototype-00D4FF?style=for-the-badge)

**Built for Hackathon Prototype Demonstration**

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Why This Matters](#-why-this-matters)
- [Solution Overview](#-solution-overview)
- [Neuromorphic Innovation](#-neuromorphic-innovation)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Dataset Information](#-dataset-information)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Model Performance](#-model-performance)
- [Results](#-results)
- [Future Scope](#-future-scope)
- [Installation Guide](#-installation-guide)
- [Running Instructions](#-running-instructions)
- [Project Structure](#-project-structure)
- [Team / Author](#-team--author)
- [License](#-license)

---

## 🎯 Problem Statement

Stroke patients often experience reduced motor control, making it difficult to perform intended hand movements. Rehabilitation devices can help, but they need fast and reliable detection of patient intent to respond naturally.

Traditional signal-processing systems may process every EMG sample continuously, which can increase computational load, latency, and power requirements.

**NeuroMove** addresses this challenge by designing a neuromorphic-inspired EMG system that detects motor intent and simulates assistive device activation using sparse event-driven processing.

---

## ❤️ Why This Matters

Stroke rehabilitation depends on repeated, meaningful movement practice. If an assistive device can detect when a patient is trying to move, it can support therapy at the exact moment of intent.

NeuroMove demonstrates how low-power AI can support:

- Faster response during rehabilitation exercises
- Reduced computational workload
- More energy-efficient assistive devices
- Scalable future integration with robotic hands or wearable rehabilitation systems

For judges unfamiliar with neuromorphic computing:  
**Neuromorphic AI is inspired by the brain. Instead of processing every signal all the time, it reacts mainly to important changes, or “events.”**

---

## 🧩 Solution Overview

NeuroMove is a Streamlit-based hackathon prototype that:

1. Loads EMG signal data.
2. Cleans and normalizes numeric EMG features.
3. Applies spike encoding to simulate neuromorphic event-driven processing.
4. Trains a Random Forest classifier.
5. Predicts motor intent:
   - Open Hand
   - Close Hand
   - Rest
6. Simulates assistive device action based on detected intent.
7. Compares raw sample processing against spike-event processing.

---

## ⚡ Neuromorphic Innovation

Traditional AI systems often process every EMG sample:

```text
sample_1 → process
sample_2 → process
sample_3 → process
...
```

NeuroMove uses a simple event-driven spike encoder:

```text
if abs(current_signal - previous_signal) > threshold:
    spike = 1
else:
    spike = 0
```

This means the system focuses on meaningful signal changes rather than every raw sample.

### Traditional AI vs Neuromorphic AI

| Approach                 | Processing Style                 | Impact                                      |
| ------------------------ | -------------------------------- | ------------------------------------------- |
| Traditional AI           | Processes every signal sample    | Higher compute load                         |
| Neuromorphic-inspired AI | Processes important spike events | Lower compute and faster response potential |

---

## 🏗 System Architecture

```text
Patient Movement Attempt
        ↓
EMG Signal Acquisition
        ↓
Spike Encoding
        ↓
Feature Extraction
        ↓
Random Forest Classifier
        ↓
Motor Intent Detection
        ↓
Assistive Device Action
```

### Architecture Explanation

- **Patient Movement Attempt:** Patient tries to open or close the hand.
- **EMG Signal Acquisition:** Muscle activity is captured as EMG signals.
- **Spike Encoding:** Only meaningful signal changes are converted into spike events.
- **Feature Extraction:** Numeric EMG features are prepared for inference.
- **Random Forest Classifier:** Predicts the movement intent.
- **Motor Intent Detection:** System classifies Open Hand, Close Hand, or Rest.
- **Assistive Device Action:** A virtual rehabilitation device responds to the prediction.

---

## 🛠 Technology Stack

| Category             | Tools                        |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Dashboard            | Streamlit                    |
| Machine Learning     | Scikit-learn                 |
| Data Processing      | Pandas, NumPy                |
| Visualization        | Matplotlib, Streamlit Charts |
| Model Storage        | Joblib                       |

---

## 📊 Dataset Information

NeuroMove is designed to work with a public EMG hand gesture dataset.

Expected dataset format:

- Numeric EMG channel columns
- One label column named one of:

```text
label, gesture, class, target, movement, intent
```

Supported classes:

```text
Open Hand
Close Hand
Rest
```

For hackathon reliability, if no dataset is found at `data/emg_gesture_data.csv`, the project can generate an EMG-like demo dataset so the prototype runs offline.

---

## ✨ Features

- 🧠 Neuromorphic-inspired spike encoding
- 💪 EMG motor intent detection
- 🌲 Random Forest classifier
- 📈 Accuracy and confusion matrix display
- ⚡ Raw samples vs spike events comparison
- 🖐 Virtual assistive device simulation
- 🏥 Healthcare AI themed dashboard
- 📊 Dataset summary and class distribution
- 🧭 Architecture and future scope pages
- 🧪 Hackathon-ready offline demo flow

---

## 📈 Model Performance

The current prototype uses a Random Forest classifier trained on EMG features.

Example performance from the demo run:

| Metric       |                       Value |
| ------------ | --------------------------: |
| Accuracy     |                      99.29% |
| Classes      | Open Hand, Close Hand, Rest |
| Model        |    Random Forest Classifier |
| Feature Type |        Numeric EMG channels |

The dashboard includes:

- Accuracy card
- Confusion matrix
- Confidence score
- Class-wise prediction confidence

---

## ✅ Results

NeuroMove demonstrates:

- Successful classification of EMG-based motor intent
- Clear separation between Open Hand, Close Hand, and Rest states
- Event-driven spike processing inspired by neuromorphic systems
- Reduced computational load by processing spike events instead of every raw sample
- A polished healthcare AI dashboard suitable for hackathon judging

Example compute comparison:

| Metric       | Description                                         |
| ------------ | --------------------------------------------------- |
| Raw Samples  | Total continuous EMG values processed traditionally |
| Spike Events | Signal changes that crossed the spike threshold     |
| Reduction %  | Estimated event-driven compute reduction            |

---

## 🚀 Future Scope

- ✅ Real-time EMG sensor integration
- ✅ EEG integration for hybrid neuro-rehabilitation
- ✅ Spiking Neural Networks
- ✅ Neuromorphic hardware deployment
- ✅ Robotic hand control
- ✅ Clinical validation and deployment

---

## ⚙️ Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/NeuroMove.git
cd NeuroMove
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## ▶️ Running Instructions

### Train the Model

```powershell
python train.py
```

This will:

- Load or generate the EMG dataset
- Preprocess the data
- Train the Random Forest model
- Save model artifacts into `models/`
- Print accuracy, confusion matrix, and spike-processing metrics

### Run the Streamlit App

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
NeuroMove/
├── app/
│   ├── __init__.py
│   ├── dashboard.py
│   └── neuro_utils.py
├── data/
│   ├── README.md
│   └── emg_gesture_data.csv
├── models/
│   ├── neuromove_rf_model.joblib
│   └── training_metrics.joblib
├── notebooks/
│   └── README.md
├── screenshots/
│   └── README.md
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

<div align="center">

## NeuroMove

**Neuromorphic Stroke Rehabilitation Assistant**

Built to explore faster, lower-power, patient-centered rehabilitation AI.

</div>

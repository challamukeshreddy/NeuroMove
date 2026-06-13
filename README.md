# NeuroMove

NeuroMove is a hackathon-ready prototype for neuromorphic EMG motor-intent
detection in stroke rehabilitation.

It detects:

- Open Hand
- Close Hand
- Rest

The system uses threshold-based spike encoding to simulate event-driven
neuromorphic processing and compares the number of raw samples against spike
events to estimate computational reduction.

## Project Structure

```text
NeuroMove/
├── data/
├── notebooks/
├── models/
├── app/
│   ├── __init__.py
│   ├── dashboard.py
│   └── neuro_utils.py
├── screenshots/
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## Dataset

For a real experiment, place a public EMG hand gesture CSV at:

```text
data/emg_gesture_data.csv
```

The CSV should contain numeric EMG columns and one label column named one of:

```text
label, gesture, class, target, movement, intent
```

Supported labels are normalized into:

```text
Open Hand, Close Hand, Rest
```

For hackathon demos, if no CSV exists, `train.py` automatically creates an
EMG-like demo dataset so the prototype runs offline on Windows.

## Setup on Windows with VS Code

1. Open the `NeuroMove` folder in VS Code.
2. Open a PowerShell terminal.
3. Create a virtual environment:

```powershell
python -m venv venv
```

4. Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

5. Install dependencies:

```powershell
pip install -r requirements.txt
```

6. Train the model:

```powershell
python train.py
```

7. Run the dashboard:

```powershell
streamlit run app.py
```

## Neuromorphic Spike Encoding

The spike encoder converts EMG changes into sparse events:

```text
if abs(current_signal - previous_signal) > threshold:
    spike = 1
else:
    spike = 0
```

This allows the dashboard to compare:

- Total raw samples processed
- Total spikes processed
- Estimated computational reduction percentage

## Model

The training pipeline:

1. Loads the EMG dataset.
2. Cleans missing values.
3. Normalizes numeric EMG features.
4. Splits data into train and test sets.
5. Trains a Random Forest classifier.
6. Saves model artifacts into `models/`.
7. Reports accuracy, confusion matrix, and spike reduction.

## Assistive Device Simulation

The dashboard maps predicted intent to a virtual assistive device:

```text
OPEN HAND  -> 🖐️
CLOSE HAND -> ✊
REST       -> 🤚
```

## Suggested Git Commit Messages

```text
feat: create NeuroMove project structure
feat: add EMG preprocessing and spike encoding pipeline
feat: train random forest motor intent classifier
feat: add Streamlit rehabilitation dashboard
docs: add setup instructions and hackathon demo guide
```

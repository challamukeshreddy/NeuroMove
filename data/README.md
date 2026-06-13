# Data Folder

Place a public EMG hand gesture CSV here as:

```text
data/emg_gesture_data.csv
```

The prototype expects numeric EMG columns and a label column named one of:

```text
label, gesture, class, target, movement, intent
```

If this file is missing, `python train.py` creates a small EMG-like demo dataset
for offline hackathon use.

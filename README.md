# batAI — Bioacoustic Bat Behavior Monitor

A CNN-LSTM deep learning system that classifies fruit bat vocalizations and predicts next-day food consumption — deployed live at the **Oregon Zoo**.

---

## Overview

Oregon Zoo zookeepers must order food for their Rodrigues flying foxes and straw-colored fruit bats one day in advance, without knowing how much the bats ate overnight. This leads to food waste or shortages.

**batAI** solves this by recording overnight vocalizations, classifying each call by behavior type, and automatically emailing zookeepers a food consumption prediction each morning.

The system found a moderate positive correlation (**r = 0.5**) between straw bat "want food" vocalizations and next-day food consumption.

---

## Results

| Metric               | Goal   | Achieved   |
|-----------------------|--------|------------|
| Test Accuracy         | ≥ 98%  | **98%** ✅ |
| Macro F1 Score        | ≥ 0.90 | **0.97** ✅ |
| Embedded System Cost  | ≤ $60  | **$35** ✅ |

---

## Model

A hybrid CNN-LSTM architecture built with Keras/TensorFlow. Audio is converted to mel-spectrograms (79 × 120), passed through two convolutional blocks (64 and 128 filters), an LSTM layer for temporal modeling, and dense layers for behavior classification.

**Behavior classes** include vocalizations such as rod fighting, straw fighting, straw talking, and "want food" calls, labeled from overnight recordings.

---

## Repository Structure

```
batAI/
├── AI Model/            # Trained model files and inference code
├── Embedded System/      # Raspberry Pi firmware / deployment scripts
├── Preprocessing/        # Audio segmentation, labeling, spectrogram generation
├── Training/              # Model training scripts and experiments
├── behavior_timeline_gui.py   # Tkinter GUI for reviewing/classifying recordings
└── README.md
```

---

## Components

**GUI** (`behavior_timeline_gui.py`) — A Tkinter app that loads an overnight recording, segments and classifies vocalizations (confidence threshold: 0.7), plots call frequency over time, and exports data to Google Sheets. A companion email script sends a daily prediction report.

**Embedded System** — A self-contained Raspberry Pi Zero 2W unit installed in the bat exhibit. Records at scheduled times, runs inference locally, and emails a full overnight report to zookeepers.

Hardware: INMP441 microphone, DS3231 real-time clock, TFT LCD + EC11 rotary encoder, 3D-printed case (Onshape / Prusa MK4S).

---

## Getting Started

### Requirements

```bash
pip install tensorflow keras numpy librosa
pip install tflite-runtime  # for Raspberry Pi deployment
```

### Usage

There are three main ways to use this repo, depending on what you're trying to do.

#### 1. Retrain the AI model

Use this if you want to reproduce training, fine-tune on new data, or retrain from scratch.

1. Go to the **`Training/`** folder — it contains the labeled audio clips used for training, along with the training script(s).
2. Confirm the clips are organized by behavior class (folder structure or filename prefix, matching the labels used during training).
3. Run the training script from inside `Training/`:
   ```bash
   cd Training
   python train.py
   ```
4. The script will preprocess the clips into mel-spectrograms, train the CNN-LSTM model, and output:
   - A trained model file (e.g. `model.h5` / `model.keras`)
   - A label encoder file (e.g. `label_encoder.pkl`) mapping class indices to behavior names
5. Check the specific script (`train.py` or equivalent) inside `Training/` for exact hyperparameters, file paths, and output locations — these are documented in-line in that folder.

#### 2. Deploy straight to the embedded system

Use this if you just want to run the existing trained model on hardware, without retraining.

1. Go to the **`AI Model/`** folder and copy out:
   - The trained model file (`.h5` / `.keras` / `.tflite`)
   - The label encoder file
2. Go to the **`Embedded System/`** folder, which contains the Raspberry Pi deployment code.
3. Place the model and label encoder files into the location expected by the embedded script (check the paths referenced at the top of the main script in `Embedded System/`).
4. Set up the hardware:
   - Raspberry Pi Zero 2W
   - INMP441 I2S microphone
   - DS3231 real-time clock module
   - TFT LCD + EC11 rotary encoder for on-device controls
   - 3D-printed enclosure
5. Install dependencies on the Pi:
   ```bash
   pip install tflite-runtime numpy librosa
   ```
6. Configure recording schedule, email credentials, and any exhibit-specific settings in the config section of the main script in `Embedded System/`.
7. Run the main script (check the folder for the exact entry-point filename, e.g. `main.py`) — it will record on schedule, classify overnight, and email zookeepers a report each morning.

#### 3. GUI, preprocessing

- **Preprocessing** (`Preprocessing/`) — this is how the labeled clips in `Training/` were originally created: segmenting raw overnight recordings into individual clips and labeling them by behavior. You generally don't need to re-run this unless you're adding brand-new raw recordings to expand the dataset.
- **GUI** (`behavior_timeline_gui.py`) — a standalone Tkinter app for manually reviewing a recording, classifying vocalizations (confidence threshold: 0.7), plotting call frequency over time, and exporting results to Google Sheets:
  ```bash
  python behavior_timeline_gui.py
  ```
#### 4. Analyze data

- **Spreadsheet** - all of the data collected for calculating the correlation over the course of 2 weeks.
- **AppsScript** - The .gs code used to automatically analyze the data (only works on Google Sheets)
---

## Awards

- 🥇 WSSEF — 1st Place, Animals & Behaviors
- 🥇 SWWSEF — 1st Place, Animals

---

## Authors

**Daniel Liu**
**Emily Liu**

---

## Acknowledgments

Thanks to **Hannah Molony** (bat keeper, Oregon Zoo) for providing recordings, behavioral labels, and exhibit access, and to **Danica Person** for adult sponsorship throughout the project.

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

**Deployed model** (`Embedded System/bat_model.tflite`, retrained October 2026): an
average of five small 1D CNNs over mel-spectrogram features (16 kHz audio, 451 frames ×
120 mel bands plus their deltas). It is trained with balanced class weights so the
rare calls are learned, and it ignores overall loudness (recording level/distance). A small
per-class count correction (`bat_model.calibration.json`) keeps nightly call counts in line
with reality. Its honest results — tested on recording nights it never trained on — are in
`AI Model/bat_model_report.md` and `AI Model/bat_model_calibration_report.md`.

The original CNN-LSTM (`AI Model/12_29_both_species.keras`) is kept for reference.

**Behavior classes:** Rods fighting, Straws fighting, Straws talking, and Straws "want food",
labeled from overnight recordings.

---

## Repository Structure

```
batAI/
├── AI Model/            # Trained models (.keras) and their evaluation reports
├── Embedded System/     # Everything the Raspberry Pi needs (code, model, installer, README)
├── Preprocessing/       # Audio segmentation, labeling, spectrogram generation
├── Results/             # Spreadsheet with data and AppsScript for data processing
├── Training/            # Labeled clips, labels, and the training scripts
├── behavior_timeline_gui.py   # Tkinter GUI for reviewing/classifying recordings
└── README.md
```

---

## Components

**GUI** (`behavior_timeline_gui.py`) — A Tkinter app that loads an overnight recording, segments and classifies vocalizations (confidence threshold: 0.7), plots call frequency over time, and exports data to Google Sheets. A companion email script sends a daily prediction report.

**Embedded System** — A self-contained Raspberry Pi Zero 2W unit installed in the bat exhibit. Records at scheduled times, runs inference locally, and emails a full overnight report to zookeepers. Only sounds louder than an adjustable threshold are classified. Settings can be changed on the device's screen and knob, or in a terminal control panel (`batmon`) that opens when you SSH in.

Hardware: INMP441 microphone, DS3231 real-time clock, TFT LCD + EC11 rotary encoder, 3D-printed case (Onshape / Prusa MK4S).

---

## Getting Started

### Requirements

- **Raspberry Pi:** `Embedded System/install.sh` installs everything (see below).
- **Training (desktop, NVIDIA GPU recommended):** `pip install -r Training/requirements-gpu.txt`
  (or `Training/requirements.txt` without a GPU).

### Usage

There are three main ways to use this repo, depending on what you're trying to do.

#### 1. Retrain the AI model

Use this if you want to reproduce training, fine-tune on new data, or retrain from scratch.

1. The labeled clips are in `Training/Audio files/`, and the labels in `Training/Labels`.
2. Train (on a desktop; about 1–3 hours with an NVIDIA GPU):
   ```bash
   cd Training
   python train_bat_model.py --data .
   ```
3. Every candidate model is tested on recording nights it never trained on. The best one is
   exported for the Pi with its count correction. Outputs in `Training/output/`:
   - `bat_model.tflite`, `bat_model.calibration.json`, `label_encoder.pkl` — copy to the Pi
   - `report.md` — per-call-type results compared with the current model
4. Full guide, options and tips: [`Training/README.md`](Training/README.md).
   The original training script (`Training/train_audio_model.py`) is kept for reference.

#### 2. Deploy straight to the embedded system

Use this if you just want to run the trained model on hardware. `Embedded System/` already
contains the trained model, so there is nothing to copy in.

1. Set up the hardware:
   - Raspberry Pi Zero 2W
   - INMP441 I2S microphone
   - DS3231 real-time clock module
   - TFT LCD + EC11 rotary encoder for on-device controls
   - 3D-printed enclosure (`Embedded System/Part Studio 1 - Box.3mf`)
2. On the Pi, get the code and run the installer (it installs the dependencies, enables I2C/SPI/I2S, and sets up the service), then reboot:
   ```bash
   git clone https://github.com/Gandalf3244/batAI.git
   cd "batAI/Embedded System"
   sudo bash install.sh
   sudo reboot
   ```
3. Set the schedule, email account and recipients either on the device (turn = move, click = select, hold = back) or over SSH — the `batmon` control panel opens automatically when you log in.
4. Start the service (it records on schedule, classifies overnight, and emails zookeepers a report each morning):
   ```bash
   sudo systemctl enable --now bat-monitor
   ```

Details, wiring and troubleshooting: [`Embedded System/README.md`](Embedded%20System/README.md).

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

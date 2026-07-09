# 🦇 batAI — Bioacoustic Bat Behavior Monitor

A CNN-LSTM deep learning system that classifies fruit bat vocalizations and predicts next-day food consumption — deployed live at the **Oregon Zoo**.

---

## Overview

Oregon Zoo zookeepers must order food for their Rodrigues flying foxes and straw-colored fruit bats one day in advance, without knowing how much the bats ate overnight. This leads to food waste or shortages.

**batAI** solves this by recording overnight vocalizations, classifying each call by behavior type, and automatically emailing zookeepers a food consumption prediction each morning.

The system found a moderate positive correlation (**r = 0.5**) between straw bat “want food” vocalizations and next-day food consumption.

---

## Results

| Metric | Goal | Achieved |
|---|---|---|
| Test Accuracy | ≥ 98% | **98%** ✅ |
| Macro F1 Score | ≥ 0.90 | **0.97** ✅ |
| Embedded System Cost | ≤ $60 | **$35** ✅ |

---

## Model

A hybrid CNN-LSTM architecture built with Keras/TensorFlow. Audio is converted to mel-spectrograms (79 × 120), passed through two convolutional blocks (64 and 128 filters), an LSTM layer for temporal modeling, and dense layers for behavior classification.

---

## Components

**GUI** — A Tkinter app that loads an overnight recording, segments and classifies vocalizations (confidence threshold: 0.7), plots frequency over time, and exports data to Google Sheets. A companion email script sends a daily prediction report.

**Embedded System** — A self-contained Raspberry Pi Zero 2W unit installed in the bat exhibit. Records at scheduled times, runs inference locally, and emails a full overnight report to zookeepers.

Hardware: INMP441 microphone, DS3231 real-time clock, TFT LCD + EC11 rotary encoder, 3D-printed case (Onshape / Prusa MK4S).

---

## Getting Started

```bash
pip install tensorflow keras numpy librosa
pip install tflite-runtime  # for Raspberry Pi deployment
```

```bash
python gui.py             # run the GUI
python train.py           # train the model
python convert_tflite.py  # convert for embedded deployment
```

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

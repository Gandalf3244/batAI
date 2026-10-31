# Training a new bat-call model (on your desktop)

## Why retrain

On 997 labeled clips, tested on recording nights it never trained on:

| | Rods fighting | Straws fighting | Straws talking | Straws want food |
|---|---|---|---|---|
| Original model* | 17% | 100% | 8% | **1%** |
| Retrained, Oct 2026 (`bat_model.tflite`, now installed) | 78% | 89% | 0% | **53%** |

\* The original model's scores are even on clips it was trained on. It labels almost
everything "Straws fighting", and it gets worse when the recording volume changes.
The food prediction depends on "want food" calls, so it was essentially always
predicting the base amount.

Why the old model's 97% score was wrong: the rare classes were copied up to ~180
times **before** splitting into training and validation. Validation was mostly
testing on copies of training clips. Real data: 896 Straws-fighting clips, but
only 46 Rods-fighting, 38 want-food and 17 talking.

This script fixes the training:

- **Honest testing.** It trains on 5 recording nights and tests on the 6th,
  for every night. That's the question that matters: does it work on a new night?
- **Balanced classes** through class weights, not copying clips.
- **Volume robustness.** It trains and tests at -12 dB, 0 dB and +6 dB, and the
  model ignores overall loudness.
- **Same features as the Pi.** It imports the Pi's own
  `feature_extraction.py`, so there's nothing to keep in sync.
- **Drop-in export.** The output TFLite file has the same input as before
  (1 × 451 × 360), so no Pi code changes are needed.

## 1. Get the code and clips onto the desktop

**From the batAI GitHub repo (easiest):** the clips and labels are already in
`Training/` (`Labels` + `Audio files/`).

```bash
git clone https://github.com/Gandalf3244/batAI.git ~/batAI
cd ~/batAI/Training
```

Use `--data .` in the commands below.

**Or copy the project folder:** copy the project folder and the `Fresh Start`
folder to the desktop. `Fresh Start` holds `Labeled_clips_consolidated.xlsx` and the
`Clustered_vocalizations-…` folder; keep them together. Then
`cd ~/"Bat Translator System On Chip/training"` and use `--data ~/"Fresh Start"`.

Either way the script finds the clips itself. Windows-style paths in the
spreadsheet (`folder\file.wav`) and moved folders are handled; clips are matched
by filename if needed.

## 2. Install (Ubuntu with an NVIDIA GPU)

First check the NVIDIA driver works. `nvidia-smi` should list your card. If it
doesn't, install the driver with `sudo ubuntu-drivers install` and reboot.

```bash
nvidia-smi
sudo apt install python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` installs TensorFlow with its own CUDA libraries, so you
don't need to install CUDA separately. Check TensorFlow sees the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

It should print one `PhysicalDevice(... device_type='GPU')`. If it prints `[]`,
the training still works, just slower on the CPU. (No GPU at all: use
`pip install -r requirements.txt` instead.)

## 3. Train

Check the setup first (takes a few minutes, makes a throwaway model):

```bash
python train_bat_model.py --data . --quick --out quick_test
```

Then the real run:

```bash
nohup python train_bat_model.py --data . > train.log 2>&1 &
tail -f train.log          # watch progress; Ctrl+C stops watching, not training
```

With a GPU it uses the `high` budget by default. It tries about 48 model
settings across 4 model types (simple, small neural net, 1D CNN, 2D CNN), each
scored on every recording night it didn't train on. It then re-runs the best 4
with 4 random seeds each and exports an average of 5 models as one file. Expect
roughly **1–3 hours** depending on the card. The log prints a time estimate per
model type.

| budget | what it does | time with a GPU |
|---|---|---|
| `--budget normal` | 12 settings, 3 seeds, 3-model ensemble, no 2D CNN | ~20–40 min |
| `--budget high` (default with GPU) | 48 settings, 4 seeds, 5-model ensemble | ~1–3 h |
| `--budget max` | 120 settings, 6 seeds, 7-model ensemble, longer training | ~4–10 h |

**It resumes.** Every finished test is saved in `output/search_cache.jsonl`, so if
the run stops (crash, reboot, Ctrl+C), start the same command again and it
picks up where it left off. The 3 GB feature cache is also reused.

**It stays Pi-sized.** The exported ensemble is timed. If it would take more
than ~1.5 s per detected sound on a Pi Zero 2 W, it drops models from the
ensemble until it fits.

Other options:

| option | what it does |
|---|---|
| `--families cnn1d,cnn2d` | only try these model types (`linear`, `mlp`, `cnn1d`, `cnn2d`) |
| `--ensemble 3` | number of models averaged in the exported file |
| `--out somewhere` | output folder (default `training/output`) |

More compute helps less than more data. With 17 talking and 38 want-food clips,
expect a solid but limited improvement. See "Getting better results" below.

## 4. Read `output/report.md`

It shows how often the new model and the current one identify each call type
correctly **on nights they never heard**. It also shows a confusion matrix
(which calls get mistaken for which), the best settings found, and the
estimated speed on the Pi.

Look at the "want food" column: that's what drives the food prediction.
If it's still low, the fix is more labeled want-food clips (see below), not
more training.

## 5. Install on the Pi

The Oct 2026 model is already in the project folder (`bat_model.tflite`, with
its report in `bat_model_report.md`). Copying the updated project to the Pi
installs it, and the Pi switches to it automatically on the next start.
For **future** retrains, copy the new files over like this:

From the desktop's `training` folder, back up the Pi's current label encoder,
then copy the two new files over. The old model file stays in place, so you
can switch back.

```bash
ssh pi@raspberrypi.local 'cp ~/bat-monitor/label_encoder.pkl ~/bat-monitor/label_encoder_old.pkl'
scp output/bat_model.tflite output/bat_model.calibration.json output/label_encoder.pkl pi@raspberrypi.local:~/bat-monitor/
```

Then point the Pi's config at the new model and restart:

```bash
ssh pi@raspberrypi.local
cd ~/bat-monitor
sed -i 's/"model_path": *"[^"]*"/"model_path": "bat_model.tflite"/' config.json
sudo systemctl restart bat-monitor
```

To switch back:

```bash
cd ~/bat-monitor
cp label_encoder_old.pkl label_encoder.pkl
sed -i 's/"model_path": *"[^"]*"/"model_path": "12_29_both_species.tflite"/' config.json
sudo systemctl restart bat-monitor
```

(Use your Pi's actual username/address if it isn't `pi@raspberrypi.local`.)

## Count correction

The training balances the call types so the rare ones get learned at all.
The side effect is that in real audio the model counts some call types too
often or too rarely. Each training run therefore also writes
`bat_model.calibration.json`: small offsets, tested on unseen nights, that
make the Pi's counts match reality. The Pi applies it automatically to the
model it was made for (checked by fingerprint). Delete the file to turn it off.

For the Oct 2026 model the correction is already installed. With the Pi's 60%
confidence cutoff, want-food calls were counted at 0.47x their true number
before and 0.92x after (details in `bat_model_calibration_report.md`).
To redo it for a model trained before this was added:

```bash
python calibrate_counts.py --data . --cache output/search_cache.jsonl --model output/bat_model.tflite
```

## Getting better results

The model can only be as good as the examples of the rare calls:

- **Talking (17 clips) and want food (38 clips) need more examples**, ideally
  100+ each, from several different nights. Clips from the same night sound alike,
  so 20 clips from 5 nights help more than 100 from one.
- Add new clips as rows in `Labeled_clips_consolidated.xlsx` (Filename,
  Label, Full_Path) and re-run. The night is read from the 14-digit timestamp
  in the filename, so keep that in new filenames.
- Delete `output/feature_cache.npz` if you change existing clips. New or
  removed rows are detected automatically.
- Consider an **"Other / noise"** class (keepers talking, doors, fans). Right
  now any loud non-bat sound is forced into one of the four bat classes. To
  add one, label some clips `Other`, add `"Other"` to `CLASSES` in the script,
  and the Pi will ignore sounds the model labels `Other` instead of counting them as calls.

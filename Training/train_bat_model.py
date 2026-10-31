#!/usr/bin/env python3
"""
Train a new bat-call model for the Pi - run this on a desktop (ideally with an
NVIDIA GPU), not the Pi.

    python train_bat_model.py --data ~/"Fresh Start"     (project folder layout)
    python train_bat_model.py --data .                   (in the batAI repo's Training/ folder)

--data is the folder that holds the labels spreadsheet (Labeled_clips_consolidated.xlsx
or Labels) and the clips. Clips are found by the spreadsheet's Full_Path column,
or by Filename anywhere under --data.

What it does
  1. Computes features for every labeled clip with the Pi's own code
     (../feature_extraction.py), so the trained model sees exactly what the
     Pi will feed it. Each clip is also used at -12 dB and +6 dB, because the
     Pi's microphone won't record at the same level as the original recorder.
  2. Searches many model types and settings. Every candidate is scored
     honestly: train on all recording nights but one, test on the night left
     out, repeat for every night. (The old 97% score came from testing on
     copies of training clips.)
  3. Re-runs the best settings with several random seeds so a lucky run can't
     win, and scores the average of those runs (an ensemble).
  4. Trains the winning ensemble on all clips and exports ONE drop-in TFLite
     file for the Pi, checks its speed, and writes a report.

The search is cached in --out, so an interrupted run resumes where it stopped.

Output (in --out, default ./output):
  bat_model.tflite   copy to the Pi
  label_encoder.pkl  copy to the Pi
  report.md          per-class results on unseen nights, vs. the current model
  search_results.csv, model_config.json, bat_model.keras, caches
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# The Pi code (feature_extraction.py, the current model): the project folder itself,
# or "Embedded System/" in the batAI repo
PI_DIR = next((p for p in (HERE.parent / "Embedded System", HERE.parent)
               if (p / "feature_extraction.py").exists()), HERE.parent)
sys.path.insert(0, str(PI_DIR))
from feature_extraction import SpectrogramExtractor  # noqa: E402  (same code as on the Pi)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's info chatter
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402
import logging  # noqa: E402

for _name in ("absl", "tensorflow", "feature_extraction"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Must match the Pi (config.json "model" section and main.py)
SAMPLE_RATE = 16000
N_MELS, N_FFT, HOP, FRAMES = 120, 2048, 512, 451
FEATURES = N_MELS * 3
GAINS_DB = (0, -12, 6)
CLASSES = ["Rods_Fighting", "Straws_Fighting", "Straws_Talking", "Straws_Want_Food"]

# How much searching each budget does. "search" = settings tried per model type,
# "seeds" = repeat runs of the best settings, "ensemble" = models averaged in the
# exported file.
BUDGETS = {
    "quick":  dict(search=1, confirm_top=1, seeds=2, ensemble=2, epochs_scale=0.05,
                   families=["linear", "mlp", "cnn1d", "cnn2d"]),
    "normal": dict(search=4, confirm_top=2, seeds=3, ensemble=3, epochs_scale=1.0,
                   families=["linear", "mlp", "cnn1d"]),
    "high":   dict(search=12, confirm_top=4, seeds=4, ensemble=5, epochs_scale=1.0,
                   families=["linear", "mlp", "cnn1d", "cnn2d"]),
    "max":    dict(search=30, confirm_top=6, seeds=6, ensemble=7, epochs_scale=1.5,
                   families=["linear", "mlp", "cnn1d", "cnn2d"]),
}

# Search spaces. The first value of each list is the default tried first.
SPACES = {
    "linear": {"dropout": [0.2, 0.0, 0.1, 0.4], "l2": [1e-3, 1e-4, 1e-2], "lr": [1e-3, 3e-4, 3e-3],
               "epochs": [80, 40, 150]},
    "mlp": {"hidden": [64, 32, 128, 256], "dropout": [0.3, 0.2, 0.5], "l2": [1e-3, 1e-4, 1e-2],
            "lr": [1e-3, 3e-4], "epochs": [80, 50, 150]},
    "cnn1d": {"filters": [64, 32, 48, 96], "layers": [3, 2, 4], "kernel": [5, 3, 7], "dropout": [0.2, 0.1, 0.3],
              "l2": [1e-3, 1e-4], "lr": [1e-3, 5e-4, 2e-3], "epochs": [30, 20, 45],
              "specaug": [True, False], "noise": [0.02, 0.0, 0.05]},
    "cnn2d": {"base": [16, 8, 12, 24], "dropout": [0.3, 0.2, 0.4], "l2": [1e-3, 1e-4],
              "lr": [1e-3, 5e-4, 2e-3], "epochs": [30, 20, 45], "specaug": [True, False], "noise": [0.02, 0.0]},
}

# The Pi Zero 2 W is very roughly this many times slower than one desktop core
PI_SLOWDOWN = 15
PI_BUDGET_S = 1.5   # max time per classified sound on the Pi
PI_THRESHOLD = 0.6  # the Pi's default confidence threshold (config "model.confidence_threshold")


def log(msg: str):
    print(time.strftime("%H:%M:%S "), msg, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_by_name: dict = {}


def resolve_clip(full_path: str, data_dir: Path) -> str:
    """Find a clip from the spreadsheet's Full_Path on any OS.

    The spreadsheet was written on Windows (backslashes, maybe C:\\ paths). If the
    path doesn't exist as given, look the file up by name anywhere under data_dir."""
    rel = full_path.strip().replace("\\", "/")
    candidate = Path(rel) if Path(rel).is_absolute() else data_dir / rel
    if candidate.exists():
        return str(candidate)
    if not _by_name:
        for root, _dirs, files in os.walk(data_dir):
            for f in files:
                _by_name.setdefault(f, []).append(os.path.join(root, f))
    matches = _by_name.get(rel.rsplit("/", 1)[-1], [])
    # Same filename in two folders would be ambiguous: only use a unique match
    return matches[0] if len(matches) == 1 else str(candidate)


SHEET_NAMES = ("Labeled_clips_consolidated.xlsx", "Labels", "Labels.xlsx")


def find_sheet(data_dir: Path, labels: Optional[str] = None) -> Path:
    """The labels spreadsheet: --labels if given, else a known name inside data_dir."""
    if labels:
        p = Path(labels) if Path(labels).is_absolute() else data_dir / labels
        if not p.exists():
            raise SystemExit(f"Labels spreadsheet not found: {p}")
        return p
    for name in SHEET_NAMES:
        if (data_dir / name).exists():
            return data_dir / name
    raise SystemExit(f"No labels spreadsheet ({' or '.join(SHEET_NAMES)}) in {data_dir} - use --labels")


def load_clips(data_dir: Path, quick: bool, labels: Optional[str] = None):
    sheet = find_sheet(data_dir, labels)
    df = pd.read_excel(sheet, engine="openpyxl")    # "Labels" has no .xlsx extension
    df = df[df.Label.isin(CLASSES)].reset_index(drop=True)
    # The recording night is the 14-digit timestamp in the filename; prefixes such
    # as "Rod_fighting_" are just renamed copies of the same recording.
    df["night"] = df.Filename.astype(str).str.extract(r"(\d{14})")[0]
    if df.night.isna().any():
        bad = df[df.night.isna()].Filename.tolist()[:5]
        raise SystemExit(f"Can't find a recording timestamp in some filenames, e.g. {bad}")
    paths = df.Full_Path if "Full_Path" in df.columns else df.Filename
    df["path"] = [resolve_clip(p, data_dir) for p in paths.astype(str)]
    missing = [p for p in df.path if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} clips not found, e.g. {missing[0]}\n"
                         "Check --data points at the folder containing the spreadsheet.")
    if quick:
        df = df.groupby("Label", group_keys=False).apply(lambda g: g.sample(min(len(g), 25), random_state=0))
        df = df.reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame, cache: Path):
    """Pi features for every clip at every gain: (gains, N, 451, 360) float16."""
    key = "|".join(df.path) + f"|{GAINS_DB}"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            log(f"Using cached features from {cache.name}")
            return z["X"], key
    import soundfile as sf
    import librosa
    fx = SpectrogramExtractor(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP,
                              target_length=FRAMES, model_feature_dim=FEATURES)

    def load(p):
        a, sr = sf.read(p, dtype="float32")
        a = a.mean(axis=1) if a.ndim > 1 else a   # same as the original training script
        if sr != SAMPLE_RATE:
            a = librosa.resample(a, orig_sr=sr, target_sr=SAMPLE_RATE)
        return a

    log(f"Loading {len(df)} clips...")
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        audio = list(ex.map(load, df.path))
    X = np.zeros((len(GAINS_DB), len(df), FRAMES, FEATURES), np.float16)
    for gi, g in enumerate(GAINS_DB):
        k = np.float32(10 ** (g / 20))
        log(f"Computing features at {g:+d} dB...")
        with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
            for i, f in enumerate(ex.map(lambda a: fx.extract_features(np.clip(a * k, -1, 1))[0], audio)):
                X[gi, i] = f
    np.savez(cache, X=X, key=np.array(key))
    return X, key


# ---------------------------------------------------------------------------
# Model layers (all TFLite built-in ops, so the Pi needs no extra code)
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable("bat")
class LevelNorm(keras.layers.Layer):
    """Zero-padding mask + removes the overall loudness from the log-mel part,
    so the model doesn't care how close the bat was or how the mic gain is set."""

    def call(self, x):
        mask = tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=-1), x.dtype)[:, :, None]   # (B,T,1)
        n = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)                                       # (B,1)
        mel = x[:, :, :N_MELS]
        level = tf.reduce_sum(tf.reduce_mean(mel, axis=2, keepdims=True) * mask, axis=1) / n    # (B,1)
        mel = mel - level[:, :, None]
        return tf.concat([mel, x[:, :, N_MELS:]], axis=2) * mask, mask


@keras.saving.register_keras_serializable("bat")
class Augment(keras.layers.Layer):
    """Training-only SpecAugment (hide random frequency bands and time spans) plus
    a little noise. Does nothing when the model is used for predictions."""

    def __init__(self, specaug=True, noise=0.0, **kw):
        super().__init__(**kw)
        self.specaug, self.noise = bool(specaug), float(noise)

    def get_config(self):
        return {**super().get_config(), "specaug": self.specaug, "noise": self.noise}

    def call(self, inputs, training=None):
        x, mask = inputs
        if not training:
            return x
        b = tf.shape(x)[0]
        if self.specaug:
            f = tf.range(N_MELS, dtype=x.dtype)[None, :]
            keep_f = tf.ones((b, N_MELS), x.dtype)
            for _ in range(2):
                w = tf.random.uniform((b, 1), 0, 12, dtype=x.dtype)
                f0 = tf.random.uniform((b, 1), 0, N_MELS, dtype=x.dtype)
                keep_f *= 1.0 - tf.cast((f >= f0) & (f < f0 + w), x.dtype)
            x = x * tf.tile(keep_f, [1, 3])[:, None, :]          # same bands in mel/delta/delta2
            n = tf.reduce_sum(mask, axis=1)                        # (B,1) real frames
            t = tf.range(FRAMES, dtype=x.dtype)[None, :]
            keep_t = tf.ones((b, FRAMES), x.dtype)
            for _ in range(2):
                w = tf.random.uniform((b, 1), 0, 1, dtype=x.dtype) * tf.minimum(30.0, 0.15 * n)
                t0 = tf.random.uniform((b, 1), 0, 1, dtype=x.dtype) * n
                keep_t *= 1.0 - tf.cast((t >= t0) & (t < t0 + w), x.dtype)
            x = x * keep_t[:, :, None]
        if self.noise:
            x = x + tf.random.normal(tf.shape(x), stddev=self.noise, dtype=x.dtype)
        return x * mask


@keras.saving.register_keras_serializable("bat")
class ClipStats(keras.layers.Layer):
    """Summarises a clip of any length (padded to 451 frames) as one vector:
    mel mean/std, delta spread, how the spectrum changes over 4 time slices, and length."""

    def call(self, inputs):
        x, mask = inputs
        n = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)                           # (B,1)
        mel, d1, d2 = x[:, :, :N_MELS], x[:, :, N_MELS:2 * N_MELS], x[:, :, 2 * N_MELS:]

        def mean(v):
            return tf.reduce_sum(v * mask, axis=1) / n

        def std(v):
            mu = mean(v)
            return tf.sqrt(tf.reduce_sum(tf.square(v - mu[:, None, :]) * mask, axis=1) / n + 1e-6)

        parts = [mean(mel), std(mel), std(d1), std(d2)]
        pos = tf.cumsum(mask[:, :, 0], axis=1) - 1.0                               # frame index among real frames
        part = tf.floor(4.0 * pos / n)
        for k in range(4):
            mk = mask[:, :, 0] * tf.cast(tf.equal(part, float(k)), x.dtype)
            nk = tf.maximum(tf.reduce_sum(mk, axis=1, keepdims=True), 1.0)
            parts.append(tf.reduce_sum(mel * mk[:, :, None], axis=1) / nk)
        parts.append(tf.math.log(n) / 6.0)
        return tf.concat(parts, axis=1)


@keras.saving.register_keras_serializable("bat")
class MaskedPool(keras.layers.Layer):
    """Mean and max over real (non-padded) frames only."""

    def call(self, inputs):
        h, mask = inputs
        n = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
        mean = tf.reduce_sum(h * mask, axis=1) / n
        mx = tf.reduce_max(h + (mask - 1.0) * 1e4, axis=1)
        return tf.concat([mean, mx], axis=1)


@keras.saving.register_keras_serializable("bat")
class ToImage(keras.layers.Layer):
    """(B, T, 360) -> (B, T, 120, 3): mel, delta, delta2 as image channels."""

    def call(self, x):
        return tf.stack([x[:, :, :N_MELS], x[:, :, N_MELS:2 * N_MELS], x[:, :, 2 * N_MELS:]], axis=-1)


@keras.saving.register_keras_serializable("bat")
class FreqMean(keras.layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=2)


# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------

def stats_extractor():
    inp = keras.Input((FRAMES, FEATURES))
    x, mask = LevelNorm()(inp)
    return keras.Model(inp, ClipStats()([x, mask]))


_STATS_EXT = None


def pooled_features(X: np.ndarray) -> np.ndarray:
    """Run the (parameter-free) stats layers over float16 features in batches."""
    global _STATS_EXT
    if _STATS_EXT is None:
        _STATS_EXT = stats_extractor()
    out = [_STATS_EXT(np.asarray(X[i:i + 128], np.float32), training=False).numpy() for i in range(0, len(X), 128)]
    return np.concatenate(out)


def build_head(kind, cfg, train_stats):
    norm = keras.layers.Normalization()
    norm.adapt(train_stats)
    reg = keras.regularizers.l2(cfg["l2"])
    inp = keras.Input((train_stats.shape[1],))
    x = keras.layers.Dropout(cfg["dropout"])(norm(inp))
    if kind == "mlp":
        x = keras.layers.Dense(cfg["hidden"], activation="relu", kernel_regularizer=reg)(x)
        x = keras.layers.Dropout(cfg["dropout"])(x)
    out = keras.layers.Dense(len(CLASSES), activation="softmax", kernel_regularizer=reg)(x)
    return keras.Model(inp, out)


def build_cnn(kind, cfg):
    reg = keras.regularizers.l2(cfg["l2"])
    inp = keras.Input((FRAMES, FEATURES))
    x, mask = LevelNorm()(inp)
    x = Augment(cfg["specaug"], cfg["noise"])([x, mask])
    if kind == "cnn1d":
        h = x
        for i in range(cfg["layers"]):
            h = keras.layers.Conv1D(cfg["filters"], cfg["kernel"], padding="same", dilation_rate=2 ** i,
                                    activation="relu", kernel_regularizer=reg)(h)
            h = keras.layers.Dropout(cfg["dropout"])(h)
        m = mask
    else:  # cnn2d: treat the spectrogram as a 3-channel image
        h = ToImage()(x)
        m = mask
        for mult in (1, 2, 4):
            h = keras.layers.Conv2D(cfg["base"] * mult, 3, padding="same", activation="relu",
                                    kernel_regularizer=reg)(h)
            h = keras.layers.BatchNormalization()(h)
            h = keras.layers.MaxPooling2D(2, padding="same")(h)
            h = keras.layers.Dropout(cfg["dropout"] / 2)(h)
            m = keras.layers.MaxPooling1D(2, padding="same")(m)
        h = FreqMean()(h)
    h = MaskedPool()([h, m])
    h = keras.layers.Dropout(cfg["dropout"])(h)
    out = keras.layers.Dense(len(CLASSES), activation="softmax", kernel_regularizer=reg)(h)
    return keras.Model(inp, out)


def class_weights(y):
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = len(y) / (len(CLASSES) * np.maximum(counts, 1))
    return {i: float(w[i]) for i in range(len(CLASSES))}


class Batches(keras.utils.PyDataset):
    """Feeds (clip, gain) pairs in shuffled batches, converting float16 -> float32 on the fly."""

    def __init__(self, X, idx, y, cw, batch, seed):
        super().__init__(workers=4, use_multiprocessing=False, max_queue_size=8)
        self.X, self.y, self.cw, self.batch = X, y, cw, batch
        self.items = np.array([(g, i) for g in range(X.shape[0]) for i in idx])
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.items) / self.batch)

    def on_epoch_end(self):
        self.rng.shuffle(self.items)

    def __getitem__(self, b):
        sel = self.items[b * self.batch:(b + 1) * self.batch]
        xb = self.X[sel[:, 0], sel[:, 1]].astype(np.float32)
        yb = self.y[sel[:, 1]]
        return xb, yb, np.array([self.cw[c] for c in yb], np.float32)


class Member:
    """One trained model of a given family/settings, usable on raw (N,451,360) features."""

    def __init__(self, kind, cfg, epochs_scale):
        self.kind, self.cfg = kind, cfg
        self.epochs = max(1, round(cfg["epochs"] * epochs_scale))

    def fit(self, X, stats, y, idx, seed):
        keras.utils.set_random_seed(seed)
        cw = class_weights(y[idx])
        opt = keras.optimizers.Adam(self.cfg["lr"])
        if self.kind in ("linear", "mlp"):
            S = np.concatenate([s[idx] for s in stats])
            yy = np.tile(y[idx], len(stats))
            self.model = build_head(self.kind, self.cfg, S)
            self.model.compile(opt, "sparse_categorical_crossentropy")
            self.model.fit(S, yy, epochs=self.epochs, batch_size=32, class_weight=cw, verbose=0)
        else:
            self.model = build_cnn(self.kind, self.cfg)
            self.model.compile(opt, "sparse_categorical_crossentropy", jit_compile="auto")
            self.model.fit(Batches(X, idx, y, cw, 32, seed), epochs=self.epochs, verbose=0)
        return self

    def predict(self, X, stats=None):
        if self.kind in ("linear", "mlp"):
            S = stats if stats is not None else pooled_features(X)
            return self.model.predict(S, batch_size=256, verbose=0)
        return self.model.predict(np.asarray(X, np.float32), batch_size=64, verbose=0)

    def full_model(self) -> keras.Model:
        """(B, 451, 360) -> probabilities."""
        if self.kind in ("linear", "mlp"):
            inp = keras.Input((FRAMES, FEATURES))
            x, mask = LevelNorm()(inp)
            return keras.Model(inp, self.model(ClipStats()([x, mask])))
        return self.model


def sample_configs(kind, n, rng):
    """First config = all defaults; the rest random, without repeats."""
    space = SPACES[kind]
    configs = [{k: v[0] for k, v in space.items()}]
    seen = {json.dumps(configs[0], sort_keys=True)}
    tries = 0
    while len(configs) < n and tries < 1000:
        tries += 1
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}
        c = {k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(c)
    return configs


# ---------------------------------------------------------------------------
# Honest evaluation: leave one recording night out
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, X, stats, y, nights, data_key, epochs_scale, cache_path: Path):
        self.X, self.stats, self.y, self.nights = X, stats, y, nights
        self.data_key, self.epochs_scale = data_key, epochs_scale
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = np.array(rec["probs"], np.float32)
                except (json.JSONDecodeError, KeyError):
                    pass
        self.fit_times = {}

    def lono(self, kind, cfg, seed):
        """Probabilities (gains, N, classes); each clip predicted by a model that never
        heard its recording night. Cached, so interrupted runs resume."""
        key = hashlib.sha1(json.dumps([self.data_key, kind, cfg, seed, self.epochs_scale],
                                      sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        probs = np.zeros((len(GAINS_DB), len(self.y), len(CLASSES)), np.float32)
        for night in np.unique(self.nights):
            tr, te = np.where(self.nights != night)[0], np.where(self.nights == night)[0]
            t0 = time.time()
            m = Member(kind, cfg, self.epochs_scale).fit(self.X, self.stats, self.y, tr, seed)
            for g in range(len(GAINS_DB)):
                probs[g, te] = m.predict(self.X[g][te], self.stats[g][te] if kind in ("linear", "mlp") else None)
            self.fit_times.setdefault(kind, []).append(time.time() - t0)
            keras.backend.clear_session()
        self.cache[key] = probs
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"key": key, "kind": kind, "cfg": cfg, "seed": seed,
                                "probs": np.round(probs, 5).tolist()}) + "\n")
        return probs

    def eta(self, kind, fits_left):
        t = self.fit_times.get(kind)
        return None if not t else np.mean(t) * fits_left


def scores(y, probs):
    """Scores averaged over the test volumes."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
    rows = []
    for g in range(probs.shape[0]):
        pred = probs[g].argmax(1)
        rec = recall_score(y, pred, average=None, labels=range(len(CLASSES)), zero_division=0)
        rows.append({"balanced_accuracy": balanced_accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
                     **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}})
    return pd.DataFrame(rows).mean().to_dict()


def baseline_probs(tflite_path: Path, X):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    if tuple(i["shape"][1:]) != (FRAMES, FEATURES) or o["shape"][-1] != len(CLASSES):
        return None
    probs = np.zeros((len(GAINS_DB), X.shape[1], len(CLASSES)), np.float32)
    for g in range(len(GAINS_DB)):
        for n in range(X.shape[1]):
            interp.set_tensor(i["index"], np.asarray(X[g, n:n + 1], np.float32))
            interp.invoke()
            probs[g, n] = interp.get_tensor(o["index"])[0]
    return probs


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def ensemble_model(members, batch_size=1) -> keras.Model:
    inp = keras.Input((FRAMES, FEATURES), batch_size=batch_size)
    outs = [m.full_model()(inp) for m in members]
    out = outs[0] if len(outs) == 1 else keras.layers.Average()(outs)
    return keras.Model(inp, out)


def to_tflite(model: keras.Model) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return conv.convert()


def tflite_latency(blob: bytes, X0) -> float:
    """Seconds per prediction on ONE desktop CPU thread."""
    interp = tf.lite.Interpreter(model_content=blob, num_threads=1)
    interp.allocate_tensors()
    i = interp.get_input_details()[0]
    xb = np.asarray(X0[:1], np.float32)
    interp.set_tensor(i["index"], xb)
    interp.invoke()
    t0 = time.perf_counter()
    for _ in range(10):
        interp.set_tensor(i["index"], xb)
        interp.invoke()
    return (time.perf_counter() - t0) / 10


def export(members, X0, out: Path):
    """Export the biggest ensemble that stays fast enough for the Pi."""
    while True:
        model = ensemble_model(members)
        blob = to_tflite(model)
        latency = tflite_latency(blob, X0)
        pi_est = latency * PI_SLOWDOWN
        if pi_est <= PI_BUDGET_S or len(members) == 1:
            break
        log(f"   {len(members)} models ≈ {pi_est:.1f}s per sound on the Pi - too slow, dropping one")
        members = members[:-1]
    model.save(out / "bat_model.keras")
    (out / "bat_model.tflite").write_bytes(blob)

    # The TFLite file must give the same answers as the Keras model
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    worst = 0.0
    for n in range(min(40, len(X0))):
        xb = np.asarray(X0[n:n + 1], np.float32)
        interp.set_tensor(i["index"], xb)
        interp.invoke()
        worst = max(worst, float(np.abs(interp.get_tensor(o["index"]) - model(xb, training=False).numpy()).max()))
    if worst > 1e-3:
        raise SystemExit(f"TFLite output differs from Keras by {worst} - export failed")
    log(f"TFLite export verified (max difference {worst:.1e}), input {[int(d) for d in i['shape']]}, "
        f"{len(blob) / 1e6:.1f} MB, {len(members)} model(s), ≈{pi_est:.2f}s per sound on the Pi")

    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder().fit(CLASSES)
    assert list(enc.classes_) == CLASSES
    with open(out / "label_encoder.pkl", "wb") as f:
        pickle.dump(enc, f)
    return len(members), latency, pi_est, len(blob)


def silence_check(out: Path):
    """What the new model says about silence and hiss (informational)."""
    fx = SpectrogramExtractor(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP,
                              target_length=FRAMES, model_feature_dim=FEATURES)
    interp = tf.lite.Interpreter(model_path=str(out / "bat_model.tflite"))
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    rows = []
    rng = np.random.default_rng(0)
    for name, a in [("silence", np.zeros(SAMPLE_RATE * 3, np.float32)),
                    ("quiet hiss", (rng.standard_normal(SAMPLE_RATE * 3) * 1e-3).astype(np.float32))]:
        interp.set_tensor(i["index"], fx.extract_features(a).astype(np.float32))
        interp.invoke()
        p = interp.get_tensor(o["index"])[0]
        rows.append(f"| {name} | {CLASSES[int(p.argmax())]} | {p.max():.0%} |")
    return rows


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def pct(v):
    return f"{v:.0%}"


def markdown_table(table: pd.DataFrame) -> str:
    lines = ["| | " + " | ".join(str(c) for c in table.columns) + " |", "|---" * (len(table.columns) + 1) + "|"]
    for idx, row in table.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def confusion_text(y, pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    short = [c.replace("Straws_", "S_").replace("Rods_", "R_") for c in CLASSES]
    lines = ["| true \\ predicted | " + " | ".join(short) + " |", "|---" * (len(CLASSES) + 1) + "|"]
    for c, row in zip(short, cm):
        lines.append(f"| **{c}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def score_row(name, s):
    return (f"| {name} | {pct(s['balanced_accuracy'])} | {s['macro_f1']:.2f} | "
            + " | ".join(pct(s[f'recall_{c}']) for c in CLASSES) + " |")


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    return gpus


# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # some consoles can't print every character
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder with the labels spreadsheet and the clips")
    ap.add_argument("--labels", default=None, help="Labels spreadsheet (default: Labeled_clips_consolidated.xlsx "
                                                   "or Labels inside --data)")
    ap.add_argument("--out", default=str(HERE / "output"), help="Output folder (default: output next to this script)")
    ap.add_argument("--budget", choices=list(BUDGETS), default=None,
                    help="How much to search: normal (~20 min), high (~1-2 h on a GPU, default with a GPU), "
                         "max (several hours)")
    ap.add_argument("--families", default=None, help="Model types to try, e.g. cnn1d,cnn2d")
    ap.add_argument("--ensemble", type=int, default=None, help="Models averaged in the exported file")
    ap.add_argument("--baseline", default=str(next((p for p in (PI_DIR / "bat_model.tflite",
                                                                PI_DIR / "12_29_both_species.tflite")
                                                     if p.exists()), PI_DIR / "bat_model.tflite")),
                    help="Current model to compare against")
    ap.add_argument("--quick", action="store_true", help="Tiny smoke test (25 clips per class, few epochs)")
    args = ap.parse_args()

    gpus = setup_gpu()
    budget_name = "quick" if args.quick else (args.budget or ("high" if gpus else "normal"))
    budget = dict(BUDGETS[budget_name])
    if args.families:
        budget["families"] = [f.strip() for f in args.families.split(",") if f.strip()]
        for f in budget["families"]:
            if f not in SPACES:
                raise SystemExit(f"Unknown model type '{f}' (choose from {', '.join(SPACES)})")
    if args.ensemble:
        budget["ensemble"] = args.ensemble
    if gpus:
        log(f"GPU: {', '.join(g.name for g in gpus)}")
    else:
        log("No GPU found - training on the CPU. (On Ubuntu with an NVIDIA card: check `nvidia-smi` "
            "works and that you installed requirements-gpu.txt.)")
    log(f"Budget '{budget_name}': {budget['search']} settings per model type "
        f"({', '.join(budget['families'])}), best {budget['confirm_top']} re-run with "
        f"{budget['seeds']} seeds, final ensemble of {budget['ensemble']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_clips(Path(args.data), args.quick, args.labels)
    y = df.Label.map(CLASSES.index).values
    nights = df.night.values
    n_nights = len(np.unique(nights))
    log(f"{len(df)} clips from {n_nights} recording nights")
    print(df.groupby(["Label", "night"]).size().unstack(fill_value=0).to_string(), flush=True)

    X, data_key = compute_features(df, out / ("feature_cache_quick.npz" if args.quick else "feature_cache.npz"))
    stats = [pooled_features(X[g]) for g in range(len(GAINS_DB))]
    ev = Evaluator(X, stats, y, nights, data_key, budget["epochs_scale"],
                   out / ("search_cache_quick.jsonl" if args.quick else "search_cache.jsonl"))

    # ---- Stage 1: search ---------------------------------------------------
    rng = np.random.default_rng(0)
    candidates = [(k, c) for k in budget["families"] for c in sample_configs(k, budget["search"], rng)]
    log(f"Stage 1/3: trying {len(candidates)} settings x {n_nights} nights")
    search_rows = []
    for ci, (kind, cfg) in enumerate(candidates):
        s = scores(y, ev.lono(kind, cfg, seed=0))
        search_rows.append({"kind": kind, "cfg": cfg, **s})
        left = sum(1 for k, _ in candidates[ci + 1:] if k == kind) * n_nights
        eta = ev.eta(kind, left)
        log(f"   [{ci + 1}/{len(candidates)}] {kind:6s} balanced acc {pct(s['balanced_accuracy'])}  "
            f"want-food {pct(s['recall_Straws_Want_Food'])}  {json.dumps(cfg)}"
            + (f"  (~{eta / 60:.0f} min left for {kind})" if eta and left else ""))
    search = pd.DataFrame(search_rows).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    search.assign(cfg=search.cfg.map(json.dumps)).to_csv(out / "search_results.csv", index=False)

    # ---- Stage 2: confirm the best with more seeds --------------------------
    top = search.head(budget["confirm_top"])
    log(f"Stage 2/3: re-running the best {len(top)} with {budget['seeds']} seeds each")
    confirm_rows = []
    for _, r in top.iterrows():
        runs = [ev.lono(r.kind, r.cfg, seed=s) for s in range(budget["seeds"])]
        singles = pd.DataFrame([scores(y, p) for p in runs])
        ens = scores(y, np.mean(runs, axis=0))
        confirm_rows.append({"kind": r.kind, "cfg": r.cfg, "single_mean": singles.balanced_accuracy.mean(),
                             "single_std": singles.balanced_accuracy.std(ddof=0), "ensemble": ens,
                             "ensemble_probs": np.mean(runs, axis=0)})
        log(f"   {r.kind:6s} single runs {pct(confirm_rows[-1]['single_mean'])} ± "
            f"{confirm_rows[-1]['single_std']:.0%}, averaged ensemble {pct(ens['balanced_accuracy'])}")
    best = max(confirm_rows, key=lambda r: (r["ensemble"]["balanced_accuracy"], r["ensemble"]["macro_f1"]))
    log(f"Chosen: {best['kind']} {json.dumps(best['cfg'])}")

    # ---- Stage 3: final ensemble on all clips -----------------------------------
    log(f"Stage 3/3: training {budget['ensemble']} final model(s) on all {len(df)} clips")
    members = []
    for s in range(budget["ensemble"]):
        members.append(Member(best["kind"], best["cfg"], budget["epochs_scale"])
                       .fit(X, stats, y, np.arange(len(y)), seed=1000 + s))
    n_used, latency, pi_est, size = export(members, X[0], out)
    silence = silence_check(out)

    # Count correction: the balanced training skews how often each call type is
    # counted; fit offsets so the Pi's counts match reality (see calibrate_counts.py)
    from calibrate_counts import calibrate, write_calibration
    log("Fitting the count correction...")
    bias, cnt_before, cnt_after = calibrate(best["ensemble_probs"], y, nights, PI_THRESHOLD)
    write_calibration(out / "bat_model.tflite", bias, PI_THRESHOLD,
                      f"{len(y)} clips x {len(GAINS_DB)} volumes, honest predictions of a "
                      f"{budget['seeds']}-model ensemble")
    yy = np.tile(y, len(GAINS_DB))
    count_rows = [f"| {c} | {(yy == k).sum()} | {(cnt_before == k).sum()} | {(cnt_after == k).sum()} | {b:+.2f} |"
                  for k, (c, b) in enumerate(zip(CLASSES, bias))]

    base = Path(args.baseline)
    base_scores = None
    if base.exists():
        bp = baseline_probs(base, X)
        if bp is not None:
            base_scores = scores(y, bp)

    config = {"model_type": best["kind"], "settings": best["cfg"], "ensemble_size": n_used,
              "classes": CLASSES, "sample_rate": SAMPLE_RATE, "max_length": FRAMES, "n_mels": N_MELS,
              "n_fft": N_FFT, "hop_length": HOP, "input_shape": [1, FRAMES, FEATURES],
              "unseen_night_balanced_accuracy": round(float(best["ensemble"]["balanced_accuracy"]), 4),
              "estimated_pi_seconds_per_sound": round(pi_est, 3), "trained_on_clips": int(len(df)),
              "budget": budget_name, "quick_test": bool(args.quick)}
    (out / "model_config.json").write_text(json.dumps(config, indent=2))

    # ---- Report ----------------------------------------------------------------
    hdr = ["| model | balanced accuracy | macro F1 | " + " | ".join(CLASSES) + " |",
           "|---" * (3 + len(CLASSES)) + "|"]
    lines = [f"# Bat model training report{' (QUICK TEST - not a real model)' if args.quick else ''}", "",
             f"{len(df)} clips, {n_nights} recording nights, budget `{budget_name}`"
             f"{', GPU' if gpus else ', CPU'}. Every score is on recording nights the model "
             "**never trained on**, averaged over test volumes "
             f"{', '.join(f'{g:+d} dB' for g in GAINS_DB)}. Class columns = share of that call type "
             "identified correctly.", "",
             "## Result", "", *hdr,
             score_row(f"**new: {best['kind']} ensemble**", best["ensemble"])]
    if base_scores:
        lines.append(score_row("current model*", base_scores))
    lines += ["",
              "\\* The current model was trained on all of these clips, so its scores are flattering." if base_scores else "",
              "",
              f"The exported file averages {n_used} copies of the chosen model "
              f"({size / 1e6:.1f} MB, about {pi_est:.2f} s per detected sound on a Pi Zero 2 W - estimated). "
              f"The score above is for an average of {budget['seeds']} copies, measured the same honest way.",
              "", "Settings were picked from many candidates, so the score may be a few points optimistic; "
              "re-running the best ones with fresh seeds (below) keeps that small.", "",
              f"## Confusion matrix (chosen model, unseen nights, original volume)", "",
              confusion_text(y, best["ensemble_probs"][0].argmax(1)), "",
              "## Count correction (`bat_model.calibration.json`)", "",
              f"How many calls of each type the Pi would count (confidence >= {PI_THRESHOLD:.0%}, all test "
              "volumes, unseen nights), without and with the correction. The correction is also "
              "checked on nights it wasn't fitted on. Copy the .calibration.json file to the Pi "
              "together with the model.", "",
              "| call type | true | counted, no correction | counted, corrected | offset |", "|---|---|---|---|---|",
              *count_rows, "",
              "## Best settings re-run with several seeds", "", *hdr]
    for r in sorted(confirm_rows, key=lambda r: -r["ensemble"]["balanced_accuracy"]):
        lines.append(score_row(f"{r['kind']} {json.dumps(r['cfg'])} (single runs {pct(r['single_mean'])} "
                               f"± {r['single_std']:.0%})", r["ensemble"]))
    lines += ["", "## Search leaderboard (single run each, top 15)", "", *hdr]
    for _, r in search.head(15).iterrows():
        lines.append(score_row(f"{r.kind} {json.dumps(r.cfg)}", r))
    lines += ["", "## Clips per class and night", "",
              markdown_table(df.groupby(["Label", "night"]).size().unstack(fill_value=0)), "",
              "## What the new model says about non-bat input", "",
              "| input | predicted | confidence |", "|---|---|---|", *silence, "",
              "On the Pi, quiet sound never reaches the model (the energy threshold filters it); "
              "this is only a sanity check.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]), flush=True)
    log(f"Done. Copy {out / 'bat_model.tflite'} and {out / 'label_encoder.pkl'} to the Pi (see README.md).")


if __name__ == "__main__":
    main()

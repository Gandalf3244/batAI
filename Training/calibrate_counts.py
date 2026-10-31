#!/usr/bin/env python3
"""
Correct a trained model's call counts (no retraining, runs in seconds on any computer).

The training uses balanced class weights so the rare calls get learned at all.
The side effect: in real audio, where ~90% of calls are Straws fighting, the
model labels too many sounds as the rare calls ("want food", "talking"), and
the nightly want-food rate - which drives the food prediction - comes out too
high.

This script takes the honest predictions saved during training (each clip
predicted by models that never heard its recording night, in
output/search_cache.jsonl) and finds one offset per call type so that the
number of calls the Pi counts - after its confidence threshold - matches the
true number. It checks the correction the same honest way: fit on 5 nights,
test on the 6th.

    python calibrate_counts.py --data ~/"Fresh Start" \\
        --cache output/search_cache.jsonl --model ../bat_model.tflite

Writes <model>.calibration.json next to the model (the Pi applies it
automatically; delete the file to turn it off) and calibration_report.md.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLASSES = ["Rods_Fighting", "Straws_Fighting", "Straws_Talking", "Straws_Want_Food"]
COMMON = "Straws_Fighting"
# How much each class's count error matters (want food drives the food prediction)
WEIGHTS = {"Rods_Fighting": 1.0, "Straws_Talking": 0.5, "Straws_Want_Food": 2.0}
# The common class has no offset of its own but its count must not collapse either
COMMON_WEIGHT = 0.5
GRID = np.arange(-6.0, 2.01, 0.25)


def apply_bias(probs, bias):
    """Same maths as model_inference.apply_class_bias on the Pi."""
    logp = np.log(np.clip(probs, 1e-9, 1.0)) + bias
    logp = logp - logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


def counted(probs, threshold):
    """What the Pi counts: the top class, only if its confidence reaches the threshold."""
    pred = probs.argmax(-1)
    keep = probs.max(-1) >= threshold
    return np.where(keep, pred, -1)


def fit_bias(probs, y, threshold):
    """Grid-search offsets for the rare classes (the common class stays at 0).

    probs: (rows, classes) honest probabilities, y: (rows,) true labels.
    Picks the lowest count error; among near-ties, the best balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score
    rare = [CLASSES.index(c) for c in WEIGHTS]
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    combos = np.array(list(itertools.product(GRID, repeat=len(rare))), np.float32)
    losses = np.empty(len(combos))
    for start in range(0, len(combos), 512):
        chunk = combos[start:start + 512]
        b = np.zeros((len(chunk), len(CLASSES)), np.float32)
        b[:, rare] = chunk
        l = logp[None] + b[:, None, :]
        l = l - l.max(-1, keepdims=True)
        pmax = 1.0 / np.exp(l).sum(-1)                   # top-class probability after the shift
        pred = np.where(pmax >= threshold, l.argmax(-1), -1)          # (combos, rows)
        loss = np.zeros(len(chunk))
        for c, w in [*WEIGHTS.items(), (COMMON, COMMON_WEIGHT)]:
            k = CLASSES.index(c)
            loss += w * np.log(((pred == k).sum(1) + 1) / ((y == k).sum() + 1)) ** 2
        losses[start:start + len(chunk)] = loss
    best = np.argsort(losses)[:200]
    near = [i for i in best if losses[i] <= losses[best[0]] + 0.01]

    def full(i):
        b = np.zeros(len(CLASSES), np.float32)
        b[rare] = combos[i]
        return b

    scored = [(balanced_accuracy_score(y, np.maximum(counted(apply_bias(probs, full(i)), threshold), 0)), i)
              for i in near]
    return full(max(scored)[1])


def calibrate(probs, y, nights, threshold):
    """probs: (volumes, clips, classes) honest probabilities.

    Returns (offsets to ship, counted labels before, counted labels after), where
    "after" uses offsets fitted without the night being tested."""
    G = probs.shape[0]
    P, Y, N = probs.reshape(-1, len(CLASSES)), np.tile(y, G), np.tile(nights, G)
    before = counted(P, threshold)
    after = np.full_like(before, -1)
    for night in np.unique(nights):
        tr, te = N != night, N == night
        after[te] = counted(apply_bias(P[te], fit_bias(P[tr], Y[tr], threshold)), threshold)
    return fit_bias(P, Y, threshold), before, after


def write_calibration(model: Path, bias, threshold, fitted_on: str) -> Path:
    out = model.with_name(model.stem + ".calibration.json")
    out.write_text(json.dumps({
        "model_sha1": hashlib.sha1(model.read_bytes()).hexdigest(),
        "class_bias": {c: round(float(b), 3) for c, b in zip(CLASSES, bias)},
        "confidence_threshold": threshold,
        "fitted_on": fitted_on,
        "note": "Log-probability offsets applied by model_inference.py before choosing the top class. "
                "Delete this file to turn the correction off.",
    }, indent=2))
    return out


SHEET_NAMES = ("Labeled_clips_consolidated.xlsx", "Labels", "Labels.xlsx")


def load_labels(data_dir: Path, labels=None):
    """Same rows, same order as train_bat_model.load_clips."""
    if labels:
        sheet = Path(labels) if Path(labels).is_absolute() else data_dir / labels
    else:
        sheet = next((data_dir / n for n in SHEET_NAMES if (data_dir / n).exists()), None)
    if sheet is None or not sheet.exists():
        raise SystemExit(f"No labels spreadsheet found in {data_dir} - use --labels")
    df = pd.read_excel(sheet, engine="openpyxl")
    df = df[df.Label.isin(CLASSES)].reset_index(drop=True)
    df["night"] = df.Filename.astype(str).str.extract(r"(\d{14})")[0]
    return df.Label.map(CLASSES.index).values, df.night.values


def load_honest_probs(cache: Path, kind, settings, n_clips):
    runs = {}
    for line in cache.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == kind and rec.get("cfg") == settings:
            runs[rec["seed"]] = np.array(rec["probs"], np.float32)
    if not runs:
        raise SystemExit(f"No results for the chosen model ({kind} {settings}) in {cache}")
    probs = np.mean([runs[s] for s in sorted(runs)], axis=0)        # the ensemble, as in report.md
    if probs.shape[1] != n_clips:
        raise SystemExit(f"Cache has {probs.shape[1]} clips but the spreadsheet has {n_clips} - "
                         "was it changed since training?")
    return probs, sorted(runs)


def summary(y, pred, label):
    rows = []
    for k, c in enumerate(CLASSES):
        true_n, pred_n, hit = (y == k).sum(), (pred == k).sum(), ((pred == k) & (y == k)).sum()
        rows.append({"call type": c, "true": int(true_n), f"counted {label}": int(pred_n),
                     f"found {label}": f"{hit / max(true_n, 1):.0%}",
                     f"right when counted {label}": f"{hit / pred_n:.0%}" if pred_n else "-"})
    return pd.DataFrame(rows).set_index("call type")


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |",
             "|---" * (len(df.columns) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder with the labels spreadsheet (as used for training)")
    ap.add_argument("--labels", default=None, help="Labels spreadsheet, if not the default name")
    ap.add_argument("--cache", default=str(HERE / "output" / "search_cache.jsonl"))
    ap.add_argument("--model", default=str(HERE / "output" / "bat_model.tflite"))
    ap.add_argument("--model-config", default=None, help="model_config.json from training "
                    "(default: next to the model as model_config.json or <model>_config.json)")
    ap.add_argument("--threshold", type=float, default=0.6, help="The Pi's confidence threshold")
    args = ap.parse_args()

    model = Path(args.model)
    cfg_path = Path(args.model_config) if args.model_config else next(
        (p for p in (model.with_name("model_config.json"), model.with_name(model.stem + "_config.json")) if p.exists()),
        None)
    if cfg_path is None:
        raise SystemExit("Can't find the model's config json - pass --model-config")
    mcfg = json.loads(cfg_path.read_text())
    if mcfg.get("classes", CLASSES) != CLASSES:
        raise SystemExit(f"Model classes {mcfg.get('classes')} differ from {CLASSES}")

    y, nights = load_labels(Path(args.data), args.labels)
    probs, seeds = load_honest_probs(Path(args.cache), mcfg["model_type"], mcfg["settings"], len(y))
    G = probs.shape[0]
    Y = np.tile(y, G)
    N = np.tile(nights, G)
    P = probs.reshape(-1, len(CLASSES))       # all test volumes pooled
    T = args.threshold

    # Sanity check: should reproduce report.md's confusion matrix
    from sklearn.metrics import confusion_matrix
    print("Honest predictions at original volume (should match report.md):")
    print(confusion_matrix(y, probs[0].argmax(1), labels=range(len(CLASSES))))

    final_bias, before, after = calibrate(probs, y, nights, T)

    table = summary(Y, before, "before").join(summary(Y, after, "after").drop(columns="true"))
    wf = CLASSES.index("Straws_Want_Food")
    per_night = pd.DataFrame({
        "true want-food": pd.Series(Y == wf).groupby(N).sum(),
        "counted before": pd.Series(before == wf).groupby(N).sum(),
        "counted after": pd.Series(after == wf).groupby(N).sum(),
    }).astype(int)
    per_night.index.name = "night"

    out = write_calibration(model, final_bias, T,
                            f"{len(y)} clips x {G} volumes, honest predictions of a {len(seeds)}-model ensemble")

    ratio = lambda col: table.loc["Straws_Want_Food", col] / max(table.loc["Straws_Want_Food", "true"], 1)
    lines = ["# Count correction report", "",
             f"Model `{model.name}` ({mcfg['model_type']}, {len(seeds)} seeds), confidence threshold {T:.0%}. "
             f"{len(y)} clips x {G} volumes. Every number is from models that never heard that "
             "recording night, and the \"after\" correction was also fitted without that night.", "",
             f"**Want-food count: {ratio('counted before'):.2f}x the true number before, "
             f"{ratio('counted after'):.2f}x after.**", "",
             "## Counts and accuracy", "",
             "*found* = share of that call type the Pi counts correctly. "
             "*right when counted* = share of the Pi's counts of that type that are correct.", "",
             markdown(table), "",
             "## Want-food calls per night", "", markdown(per_night), "",
             "## Offsets shipped", "",
             "| call type | offset |", "|---|---|",
             *[f"| {c} | {b:+.2f} |" for c, b in zip(CLASSES, final_bias)], "",
             f"Written to `{out.name}`. The Pi applies it automatically to this exact model file "
             "(checked by fingerprint) and ignores it for any other model. Delete the file to turn it off.",
             "", "Assumes the Pi hears roughly the same mix of calls as the labeled clips "
             "(mostly Straws fighting). If the mix changes a lot, re-run with new labeled clips.", ""]
    report = HERE / "calibration_report.md" if model.parent == HERE / "output" else model.with_name(
        model.stem + "_calibration_report.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[4:8 + len(table) + 4]))
    print(f"\nWrote {out} and {report}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    main()

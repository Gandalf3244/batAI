# Bat model training report

997 clips, 6 recording nights, budget `high`, GPU. Every score is on recording nights the model **never trained on**, averaged over test volumes +0 dB, -12 dB, +6 dB. Class columns = share of that call type identified correctly.

## Result

| model | balanced accuracy | macro F1 | Rods_Fighting | Straws_Fighting | Straws_Talking | Straws_Want_Food |
|---|---|---|---|---|---|---|
| **new: cnn1d ensemble** | 55% | 0.49 | 78% | 89% | 0% | 53% |
| current model* | 32% | 0.35 | 17% | 100% | 8% | 1% |

\* The current model was trained on all of these clips, so its scores are flattering.

The exported file averages 5 copies of the chosen model (1.7 MB, about 0.04 s per detected sound on a Pi Zero 2 W - estimated). The score above is for an average of 4 copies, measured the same honest way.

Settings were picked from many candidates, so the score may be a few points optimistic; re-running the best ones with fresh seeds (below) keeps that small.

## Confusion matrix (chosen model, unseen nights, original volume)

| true \ predicted | R_Fighting | S_Fighting | S_Talking | S_Want_Food |
|---|---|---|---|---|
| **R_Fighting** | 36 | 0 | 1 | 9 |
| **S_Fighting** | 11 | 793 | 52 | 40 |
| **S_Talking** | 8 | 8 | 0 | 1 |
| **S_Want_Food** | 10 | 5 | 3 | 20 |

## Best settings re-run with several seeds

| model | balanced accuracy | macro F1 | Rods_Fighting | Straws_Fighting | Straws_Talking | Straws_Want_Food |
|---|---|---|---|---|---|---|
| cnn1d {"filters": 64, "layers": 2, "kernel": 3, "dropout": 0.3, "l2": 0.0001, "lr": 0.0005, "epochs": 20, "specaug": true, "noise": 0.05} (single runs 55% ± 2%) | 55% | 0.49 | 78% | 89% | 0% | 53% |
| cnn1d {"filters": 96, "layers": 3, "kernel": 3, "dropout": 0.3, "l2": 0.0001, "lr": 0.0005, "epochs": 20, "specaug": true, "noise": 0.0} (single runs 55% ± 1%) | 54% | 0.48 | 67% | 90% | 0% | 61% |
| cnn1d {"filters": 64, "layers": 3, "kernel": 5, "dropout": 0.2, "l2": 0.001, "lr": 0.001, "epochs": 30, "specaug": true, "noise": 0.02} (single runs 54% ± 1%) | 54% | 0.48 | 80% | 92% | 0% | 45% |
| cnn1d {"filters": 96, "layers": 2, "kernel": 7, "dropout": 0.3, "l2": 0.0001, "lr": 0.001, "epochs": 45, "specaug": true, "noise": 0.0} (single runs 53% ± 2%) | 52% | 0.49 | 76% | 95% | 0% | 37% |

## Search leaderboard (single run each, top 15)

| model | balanced accuracy | macro F1 | Rods_Fighting | Straws_Fighting | Straws_Talking | Straws_Want_Food |
|---|---|---|---|---|---|---|
| cnn1d {"filters": 64, "layers": 2, "kernel": 3, "dropout": 0.3, "l2": 0.0001, "lr": 0.0005, "epochs": 20, "specaug": true, "noise": 0.05} | 56% | 0.49 | 85% | 88% | 0% | 53% |
| cnn1d {"filters": 64, "layers": 3, "kernel": 5, "dropout": 0.2, "l2": 0.001, "lr": 0.001, "epochs": 30, "specaug": true, "noise": 0.02} | 54% | 0.48 | 78% | 88% | 0% | 50% |
| cnn1d {"filters": 96, "layers": 3, "kernel": 3, "dropout": 0.3, "l2": 0.0001, "lr": 0.0005, "epochs": 20, "specaug": true, "noise": 0.0} | 54% | 0.48 | 65% | 90% | 0% | 61% |
| cnn1d {"filters": 96, "layers": 2, "kernel": 7, "dropout": 0.3, "l2": 0.0001, "lr": 0.001, "epochs": 45, "specaug": true, "noise": 0.0} | 53% | 0.49 | 70% | 93% | 0% | 50% |
| cnn1d {"filters": 64, "layers": 2, "kernel": 5, "dropout": 0.1, "l2": 0.0001, "lr": 0.0005, "epochs": 20, "specaug": false, "noise": 0.0} | 53% | 0.51 | 74% | 95% | 0% | 43% |
| mlp {"hidden": 32, "dropout": 0.5, "l2": 0.01, "lr": 0.001, "epochs": 150} | 53% | 0.47 | 76% | 91% | 6% | 39% |
| cnn1d {"filters": 32, "layers": 4, "kernel": 5, "dropout": 0.3, "l2": 0.001, "lr": 0.0005, "epochs": 20, "specaug": false, "noise": 0.02} | 52% | 0.46 | 72% | 92% | 0% | 46% |
| cnn1d {"filters": 64, "layers": 4, "kernel": 7, "dropout": 0.2, "l2": 0.001, "lr": 0.002, "epochs": 20, "specaug": false, "noise": 0.02} | 51% | 0.46 | 84% | 92% | 0% | 29% |
| cnn1d {"filters": 64, "layers": 2, "kernel": 7, "dropout": 0.1, "l2": 0.0001, "lr": 0.0005, "epochs": 45, "specaug": true, "noise": 0.0} | 51% | 0.47 | 70% | 93% | 0% | 39% |
| cnn1d {"filters": 96, "layers": 4, "kernel": 7, "dropout": 0.2, "l2": 0.0001, "lr": 0.002, "epochs": 30, "specaug": true, "noise": 0.0} | 51% | 0.48 | 69% | 94% | 6% | 34% |
| mlp {"hidden": 256, "dropout": 0.2, "l2": 0.01, "lr": 0.0003, "epochs": 150} | 51% | 0.47 | 73% | 94% | 6% | 29% |
| mlp {"hidden": 32, "dropout": 0.5, "l2": 0.0001, "lr": 0.0003, "epochs": 50} | 50% | 0.43 | 72% | 90% | 0% | 39% |
| cnn1d {"filters": 64, "layers": 4, "kernel": 5, "dropout": 0.1, "l2": 0.001, "lr": 0.002, "epochs": 45, "specaug": false, "noise": 0.05} | 50% | 0.45 | 76% | 90% | 0% | 34% |
| mlp {"hidden": 64, "dropout": 0.5, "l2": 0.0001, "lr": 0.001, "epochs": 80} | 50% | 0.47 | 73% | 95% | 0% | 32% |
| mlp {"hidden": 256, "dropout": 0.2, "l2": 0.001, "lr": 0.0003, "epochs": 80} | 49% | 0.45 | 57% | 93% | 0% | 48% |

## Clips per class and night

| | 20250912104642 | 20250912200554 | 20250913052507 | 20250913144419 | 20250914092244 | 20250914184156 |
|---|---|---|---|---|---|---|
| Rods_Fighting | 2 | 3 | 22 | 11 | 1 | 7 |
| Straws_Fighting | 211 | 131 | 217 | 178 | 85 | 74 |
| Straws_Talking | 9 | 0 | 7 | 1 | 0 | 0 |
| Straws_Want_Food | 7 | 1 | 19 | 8 | 2 | 1 |

## What the new model says about non-bat input

| input | predicted | confidence |
|---|---|---|
| silence | Rods_Fighting | 29% |
| quiet hiss | Rods_Fighting | 40% |

On the Pi, quiet sound never reaches the model (the energy threshold filters it); this is only a sanity check.

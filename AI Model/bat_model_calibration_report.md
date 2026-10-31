# Count correction report

Model `bat_model.tflite` (cnn1d, 4 seeds), confidence threshold 60%. 997 clips x 3 volumes. Every number is from models that never heard that recording night, and the "after" correction was also fitted without that night.

**Want-food count: 0.47x the true number before, 0.92x after.**

## Counts and accuracy

*found* = share of that call type the Pi counts correctly. *right when counted* = share of the Pi's counts of that type that are correct.

| call type | true | counted before | found before | right when counted before | counted after | found after | right when counted after |
|---|---|---|---|---|---|---|---|
| Rods_Fighting | 138 | 121 | 57% | 64% | 128 | 59% | 63% |
| Straws_Fighting | 2688 | 2230 | 82% | 99% | 2049 | 75% | 99% |
| Straws_Talking | 51 | 74 | 0% | 0% | 85 | 0% | 0% |
| Straws_Want_Food | 114 | 54 | 16% | 33% | 105 | 24% | 26% |

## Want-food calls per night

| night | true want-food | counted before | counted after |
|---|---|---|---|
| 20250912104642 | 21 | 6 | 15 |
| 20250912200554 | 3 | 0 | 6 |
| 20250913052507 | 57 | 39 | 60 |
| 20250913144419 | 24 | 3 | 12 |
| 20250914092244 | 6 | 3 | 9 |
| 20250914184156 | 3 | 3 | 3 |

## Offsets shipped

| call type | offset |
|---|---|
| Rods_Fighting | +0.75 |
| Straws_Fighting | +0.00 |
| Straws_Talking | +0.00 |
| Straws_Want_Food | +0.75 |

Written to `bat_model.calibration.json`. The Pi applies it automatically to this exact model file (checked by fingerprint) and ignores it for any other model. Delete the file to turn it off.

Assumes the Pi hears roughly the same mix of calls as the labeled clips (mostly Straws fighting). If the mix changes a lot, re-run with new labeled clips.

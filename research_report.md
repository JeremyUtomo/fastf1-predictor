# Model research report

Generated 2026-10-11 01:36 UTC by `python evaluate.py --tune`.

Every event is predicted with a model fitted only on earlier events (walk-forward).
Choices are made on 2020–2025; 2026 is used once, as a check.
A candidate is adopted when it lowers the winner/pole log loss by at least 0.003 without making the top-3 Brier score worse.

## Chosen settings

```json
{
  "race_post": {
    "features": [
      "grid",
      "grid_log",
      "form",
      "quali_form",
      "team_pace",
      "team_trend",
      "circuit_aff",
      "driver_edge",
      "quali_gap",
      "team_gap",
      "race_pace"
    ],
    "top_k": 3,
    "l2": 1.0,
    "half_life": null,
    "train_seasons": 4
  },
  "race_pre": {
    "features": [
      "form",
      "quali_form",
      "team_pace",
      "team_trend",
      "circuit_aff",
      "driver_edge",
      "quali_gap",
      "team_gap",
      "race_pace"
    ],
    "top_k": 3,
    "l2": 1.0,
    "half_life": null,
    "train_seasons": 4
  },
  "quali": {
    "base": [
      "quali_form",
      "team_pace",
      "team_trend",
      "form",
      "circuit_aff",
      "driver_edge",
      "quali_gap",
      "team_gap"
    ],
    "practice": "fp",
    "top_k": 5,
    "l2": 0.3,
    "half_life": null,
    "train_seasons": 6
  }
}
```

## Candidates tested (2020–2025)

| family | candidate | tune_logloss | tune_top3_brier | tune_spearman |
| --- | --- | --- | --- | --- |
| race_post | current model | 1.252 | 0.06846 | 0.778 |
| race_post | half-life 80 events | 1.256 | 0.06847 | 0.7786 |
| race_post | half-life 40 events | 1.26 | 0.0685 | 0.7787 |
| race_post | half-life 20 events | 1.271 | 0.06866 | 0.7767 |
| race_post | half-life 10 events | 1.295 | 0.06915 | 0.7715 |
| race_post | train on 6 seasons | 1.258 | 0.06808 | 0.7801 |
| race_post | train on 9 seasons | 1.259 | 0.06814 | 0.7805 |
| race_post | + driver edge (driver vs teammate) | 1.226 | 0.0684 | 0.78 |
| race_post | + qualifying time gaps | 1.215 | 0.06785 | 0.7794 |
| race_post | time gaps instead of positions | 1.216 | 0.06761 | 0.779 |
| race_post | + grid × circuit chaos | 1.216 | 0.06794 | 0.7786 |
| race_post | + this weekend's quali gap | 1.242 | 0.06799 | 0.7827 |
| race_post | + race pace (fastest-lap gap) | 1.192 | 0.06758 | 0.7774 |
| race_post | top_k 3, l2 0.3 | 1.191 | 0.06766 | 0.7752 |
| race_post | top_k 3, l2 3.0 | 1.196 | 0.06749 | 0.7785 |
| race_post | top_k 5, l2 0.3 | 1.26 | 0.06782 | 0.7798 |
| race_post | top_k 5, l2 1.0 | 1.262 | 0.06775 | 0.78 |
| race_post | top_k 5, l2 3.0 | 1.264 | 0.06762 | 0.7806 |
| race_post | top_k 10, l2 0.3 | 1.377 | 0.06901 | 0.7819 |
| race_post | top_k 10, l2 1.0 | 1.377 | 0.06895 | 0.7824 |
| race_post | top_k 10, l2 3.0 | 1.377 | 0.06885 | 0.7824 |
| race_pre | current model | 1.608 | 0.08383 | 0.7224 |
| race_pre | half-life 80 events | 1.609 | 0.08391 | 0.7237 |
| race_pre | half-life 40 events | 1.612 | 0.08406 | 0.7243 |
| race_pre | half-life 20 events | 1.619 | 0.08447 | 0.7235 |
| race_pre | half-life 10 events | 1.632 | 0.08532 | 0.7212 |
| race_pre | train on 6 seasons | 1.609 | 0.0836 | 0.725 |
| race_pre | train on 9 seasons | 1.614 | 0.08373 | 0.7245 |
| race_pre | + driver edge (driver vs teammate) | 1.551 | 0.0831 | 0.7225 |
| race_pre | + qualifying time gaps | 1.531 | 0.08217 | 0.7184 |
| race_pre | time gaps instead of positions | 1.531 | 0.08201 | 0.7191 |
| race_pre | + race pace (fastest-lap gap) | 1.498 | 0.08156 | 0.7196 |
| race_pre | top_k 3, l2 0.3 | 1.498 | 0.08162 | 0.7178 |
| race_pre | top_k 3, l2 3.0 | 1.5 | 0.08146 | 0.7212 |
| race_pre | top_k 5, l2 0.3 | 1.568 | 0.0809 | 0.7231 |
| race_pre | top_k 5, l2 1.0 | 1.569 | 0.08082 | 0.7236 |
| race_pre | top_k 5, l2 3.0 | 1.572 | 0.08071 | 0.7237 |
| race_pre | top_k 10, l2 0.3 | 1.669 | 0.08218 | 0.721 |
| race_pre | top_k 10, l2 1.0 | 1.669 | 0.08214 | 0.7217 |
| race_pre | top_k 10, l2 3.0 | 1.669 | 0.08209 | 0.7216 |
| quali | current model | 1.554 | 0.0664 | 0.7617 |
| quali | half-life 80 events | 1.553 | 0.06637 | 0.7619 |
| quali | half-life 40 events | 1.554 | 0.06639 | 0.7615 |
| quali | half-life 20 events | 1.561 | 0.0665 | 0.7608 |
| quali | half-life 10 events | 1.579 | 0.06701 | 0.7583 |
| quali | train on 6 seasons | 1.551 | 0.0665 | 0.7631 |
| quali | train on 9 seasons | 1.554 | 0.06651 | 0.7629 |
| quali | + driver edge (driver vs teammate) | 1.539 | 0.06599 | 0.764 |
| quali | + qualifying time gaps | 1.528 | 0.06575 | 0.7646 |
| quali | time gaps instead of positions | 1.535 | 0.06546 | 0.7642 |
| quali | practice as time gap | 1.531 | 0.06659 | 0.755 |
| quali | top_k 3, l2 0.3 | 1.528 | 0.06635 | 0.7636 |
| quali | top_k 3, l2 3.0 | 1.528 | 0.06514 | 0.7662 |
| quali | top_k 5, l2 0.3 | 1.521 | 0.06553 | 0.7646 |
| quali | top_k 5, l2 1.0 | 1.526 | 0.0653 | 0.7644 |
| quali | top_k 5, l2 3.0 | 1.534 | 0.06507 | 0.7647 |
| quali | top_k 10, l2 0.3 | 1.602 | 0.06833 | 0.763 |
| quali | top_k 10, l2 1.0 | 1.608 | 0.06829 | 0.7635 |
| quali | top_k 10, l2 3.0 | 1.618 | 0.06829 | 0.7631 |

## Final comparison

Lower is better for logloss, top3_brier and pos_mae; higher for fav_hit, top3_hits and spearman. Baselines: grid order (race, sprint) and recent qualifying form (qualifying, race before qualifying). Probabilities are uncalibrated here; the app adds the learned calibration on top.

| kind | model | period | events | logloss | fav_hit | top3_hits | top3_brier | spearman | pos_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| race | baseline | 2020–2025 | 131 | 1.476 | 0.5573 | 2.053 | 0.0709 | 0.7334 | 2.669 |
| race | baseline | 2026 | 16 | 1.19 | 0.6875 | 1.875 | 0.0693 | 0.7826 | 2.496 |
| race | previous model | 2020–2025 | 131 | 1.252 | 0.5954 | 2.023 | 0.0685 | 0.778 | 2.378 |
| race | previous model | 2026 | 16 | 1.181 | 0.6875 | 1.812 | 0.0704 | 0.7986 | 2.343 |
| race | new model | 2020–2025 | 131 | 1.192 | 0.5878 | 2.03 | 0.0676 | 0.7774 | 2.4 |
| race | new model | 2026 | 16 | 1.203 | 0.6875 | 1.938 | 0.0708 | 0.8067 | 2.271 |
| sprint | baseline | 2020–2025 | 24 | 1.265 | 0.5833 | 2.375 | 0.0575 | 0.8323 | 2.433 |
| sprint | baseline | 2026 | 5 | 0.745 | 0.8 | 2.2 | 0.0555 | 0.9233 | 1.985 |
| sprint | previous model | 2020–2025 | 24 | 1.208 | 0.5417 | 2.333 | 0.0535 | 0.8266 | 2.313 |
| sprint | previous model | 2026 | 5 | 0.7979 | 0.8 | 2.2 | 0.0554 | 0.9005 | 1.766 |
| sprint | new model | 2020–2025 | 24 | 1.12 | 0.5833 | 2.292 | 0.0534 | 0.8203 | 2.304 |
| sprint | new model | 2026 | 5 | 0.7624 | 0.8 | 2.2 | 0.0586 | 0.9122 | 1.824 |
| race_pre | baseline | 2020–2025 | 131 | 1.689 | 0.4198 | 1.672 | 0.0853 | 0.7074 | 2.82 |
| race_pre | baseline | 2026 | 16 | 2.006 | 0.1875 | 1.188 | 0.0926 | 0.7225 | 2.714 |
| race_pre | previous model | 2020–2025 | 131 | 1.608 | 0.458 | 1.664 | 0.0838 | 0.7224 | 2.713 |
| race_pre | previous model | 2026 | 16 | 2.035 | 0.125 | 1.312 | 0.0937 | 0.7374 | 2.67 |
| race_pre | new model | 2020–2025 | 131 | 1.498 | 0.4809 | 1.763 | 0.0816 | 0.7196 | 2.71 |
| race_pre | new model | 2026 | 16 | 2.08 | 0.125 | 1.25 | 0.0946 | 0.748 | 2.562 |
| quali | baseline | 2020–2025 | 131 | 1.668 | 0.374 | 1.954 | 0.0738 | 0.7196 | 3.025 |
| quali | baseline | 2026 | 16 | 2.184 | 0.0625 | 1.25 | 0.0954 | 0.8321 | 2.487 |
| quali | previous model | 2020–2025 | 131 | 1.554 | 0.4504 | 2.023 | 0.0664 | 0.7617 | 2.768 |
| quali | previous model | 2026 | 16 | 1.884 | 0.25 | 1.562 | 0.0862 | 0.8881 | 2.053 |
| quali | new model | 2020–2025 | 131 | 1.521 | 0.5038 | 2.03 | 0.0655 | 0.7646 | 2.739 |
| quali | new model | 2026 | 16 | 1.909 | 0.375 | 1.562 | 0.0822 | 0.8947 | 1.984 |

## Is the improvement consistent?

Winner/pole log-loss change of the new vs. the previous model per season (negative = better), the 2026 change with a bootstrap 95% interval, and the same comparison after the app's walk-forward calibration.

| kind | better seasons | Δ per season | 2026 Δ | 2026 95% interval | calibrated tune old→new | calibrated 2026 old→new |
| --- | --- | --- | --- | --- | --- | --- |
| race | 4/6 | 2020: -0.214, 2021: +0.043, 2022: -0.114, 2023: -0.129, 2024: +0.030, 2025: -0.024, 2026: +0.022 | 0.022 | -0.043 to +0.082 | 1.244 → 1.187 | 1.152 → 1.171 |
| race_pre | 5/6 | 2020: -0.214, 2021: -0.024, 2022: -0.157, 2023: -0.232, 2024: +0.018, 2025: -0.087, 2026: +0.046 | 0.046 | -0.081 to +0.165 | 1.596 → 1.478 | 2.040 → 2.113 |
| quali | 3/6 | 2020: -0.047, 2021: +0.005, 2022: -0.133, 2023: +0.035, 2024: +0.061, 2025: -0.119, 2026: +0.026 | 0.026 | -0.031 to +0.087 | 1.601 → 1.512 | 1.887 → 1.903 |
| sprint | 5/5 | 2021: -0.103, 2022: -0.141, 2023: -0.084, 2024: -0.006, 2025: -0.137, 2026: -0.035 | -0.035 | -0.120 to +0.042 | 1.160 → 1.071 | 0.710 → 0.636 |

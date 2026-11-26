# Step 4 — frozen hypothesis before confirmation

Written on 26/11/2026, **before** looking at any new participant. Code: `latence/arousal_pipeline.py` (frozen version of `latence/hugo_arousal.ipynb`, 4.1–4.3).
Nothing below may be changed after the confirmation data have been seen; any later change is reported as exploratory.

## Hypothesis

Each body signal responds to the video with its own delay. Reading the skin (GSR) in a window adapted to its delay, instead of
the video window (0–10 s), improves the prediction of self-reported arousal.

## Data

- **Development set** (used to build the pipeline, 4.1–4.3): Maroua's 20 participants, `latence/participants_dev20.txt`.
- **Confirmation set**: 20 other participants drawn at random among those with ≥ 20 arousal ratings on day 1,
  development set excluded: `download_data.py --random 20 --seed 7 --exclude latence/participants_dev20.txt`
  → `latence/participants_confirm20.txt`.
- Day 1, emotion-video (+ resting-state), PREPROCESSED.
- Then the full dataset (all labelled participants, day 1), as a second, larger confirmation.

## Target and evaluation

- High vs low arousal: thirds of each participant's own ratings (`tertile_cuts`; middle third left out). Flagged outliers removed.
- Leave-one-participant-out logistic regression; score = AUC within each test participant.
- Features: mean of each signal over a window, baseline-corrected (−5…−4 s), z-scored within participant over all their videos (no ratings used).

## Primary comparison (the only confirmatory test)

| | Features |
|---|---|
| **PRIMARY** | HR, pupil, respiration on 0–10 s + GSR on the window chosen inside each training fold (start 0–5 s, end 8–15 s, ≥ 4 s long) |
| **REFERENCE** | HR, GSR, pupil, respiration on 0–10 s |

- Test: Wilcoxon signed-rank on the per-participant AUC difference (PRIMARY − REFERENCE), two-sided, α = 0.05.
- **Confirmed** if the mean difference is > 0 and p < 0.05. Development set: +0.028, 12/16 participants, p = 0.018.
- Also reported: both sets against chance (ratings shuffled within participant, 200 draws, 95th percentile of the mean AUC).

## Exclusions (fixed in advance, participant level)

- HR: > 10 % suspect RR intervals. GSR: range of the 1 Hz low-passed GSR (p99 − p1) < 0.05 µS (flat sensor).
  Pupil: > 30 % missing after blink removal. A participant needs all four signals for the comparison.
- Fewer than 5 high or 5 low videos: not scored.
- Videos whose −5…+15 s epoch falls outside the recording are dropped.

## Secondary (exploratory, not confirmatory)

GSR alone (0–10 s, 2–12 s as in Maroua's lag, learned window); each signal alone; adding SCR features (`scr_any`, `scr_amp`, onset 1–11 s).
Normalisation relative to rest (step 4.3) is not retested: with a within-participant target and score it cannot change much by construction.

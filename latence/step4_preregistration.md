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

## Confirmation run (26/11/2026) — result

- **Deviation (data availability, not a choice)**: 6 of the 20 drawn participants (P062, P063, P073, P076, P094, P144) have ratings in
  `labels_train.csv` but no recording in the release (404 on Hugging Face, PREPROCESSED and CURATED). The test ran on the 14 available;
  2 were excluded by the fixed quality rules (P096 flat GSR, P319 pupil missing) → **12 participants scored**, 483 rated videos.
  No replacement was drawn: the full-dataset run is the larger confirmation.
- **Primary test: not confirmed.** PRIMARY 0.530 vs REFERENCE 0.531 (mean difference −0.001, better in 4/12, Wilcoxon p = 0.92).
  Neither set is above chance (95th percentile of the shuffled mean AUC: 0.554 / 0.558).
- Exploratory: GSR alone is the only signal close to chance threshold (0–10 s 0.541; 2–12 s 0.551; learned window 0.553, p = 0.064).
  HR (0.411) and respiration (0.428) alone are below 0.5, consistent with the negative bias of leave-one-out cross-validation when a feature carries no information.
- Reading: the +0.028 gain of the development set was most likely due to chance (about ten variants compared on 16 participants).

## Completion to 20 participants (decided on 26/11/2026, after the 12-participant result above)

- 6 replacements drawn with the same rule (≥ 20 day-1 ratings, development and first confirmation draws excluded), now keeping only
  participants that have an emotion-video recording on Hugging Face: `download_data.py --random 6 --seed 8 --exclude
  latence/participants_dev20.txt latence/participants_confirm20.txt` → P016, P049, P212, P292, P306, P312 (P112 and P139 skipped: no recording).
- Final confirmation set: the 14 available + these 6 → `latence/participants_confirm20_final.txt`.
- Pipeline, primary comparison, test and exclusion rules unchanged. Because this completion was decided after seeing the first result,
  both results are reported (12 participants above, and the final set).

## Final confirmation result (26/11/2026, 20 drawn, 17 scored)

- Excluded by the fixed rules: P096 (flat GSR), P212 (no eye tracking, HR 14 % suspect RR), P319 (pupil missing). 630 rated videos.
- **Primary test: not confirmed.** PRIMARY 0.540 vs REFERENCE 0.525 (mean difference +0.015, better in 8/17, Wilcoxon p = 0.12).
  Neither 4-signal set is above chance (95th percentile of the shuffled mean AUC: 0.550).
- Exploratory, but consistent with the development set:
  - **GSR alone is above chance**: 0–10 s 0.561 (p = 0.044), 2–12 s 0.577 (p = 0.006), learned window 0.573 (p = 0.009).
  - **GSR read 2–12 s beats GSR 0–10 s**: +0.016, 11/17, p = 0.031 (development set: +0.017, p = 0.039).
  - Adding HR, pupil and respiration **lowers** the AUC (PRIMARY − GSR 0–10 s: −0.021): HR (0.391) and respiration (0.439) alone are
    below 0.5, the leave-one-out bias of uninformative features.
- Candidate hypothesis for the full dataset (to be frozen before that run): GSR alone 2–12 s vs GSR alone 0–10 s.

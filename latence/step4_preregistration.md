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

---

# Full-dataset hypothesis (frozen 26/11/2026, before the machine run)

The two first sets (development 20 + confirmation 20, 37 with recordings) become the **exploration set**: they are excluded
from the full-dataset test, which thus runs on participants never looked at.

## Data
- Every participant with day-1 arousal ratings and an emotion-video recording, **cohorts 1–2** (fully randomised video order),
  exploration set excluded (`--exclude latence/participants_dev20.txt latence/participants_confirm20_final.txt`).
- Why cohorts 1–2: in cohorts 3–5 the videos come in emotion blocks (Dataset Paper, Table 2; 31–55 % of consecutive videos share
  their default emotion), so the previous video and slow drifts can be confounded with arousal. In the exploration set the skin
  effect was weaker there (GSR 2–12 s: 0.549 vs 0.599). This choice is therefore partly informed by the exploration data; cohorts
  3–5 are reported as a secondary analysis with the same pipeline.
- Extraction: `latence/process_step4.py` (HABS machine) = `arousal_pipeline.extract`; evaluation: `arousal_pipeline.py evaluate`.

## Primary comparison (the only confirmatory test)
- **GSR alone, 2–12 s** vs **GSR alone, 0–10 s** (baseline −5…−4 s, z-scored within participant over all their videos).
- Participants: usable GSR (range ≥ 0.05 µS) and ≥ 5 high and ≥ 5 low videos. Leave-one-participant-out logistic regression,
  AUC within each test participant; Wilcoxon signed-rank on the paired AUC difference, two-sided, α = 0.05.
- **Confirmed** if the mean difference is > 0 and p < 0.05. Exploration set: +0.016 to +0.019, p = 0.03 in each of the two sets.
- Also reported: both sets against chance (ratings shuffled within participant, 200 draws).

## Secondary (exploratory)
1. GSR 2–12 s + SCR features vs GSR 2–12 s (exploration: +0.009 to +0.014).
2. 4 signals (GSR 2–12 s + HR, pupil, respiration 0–10 s) vs GSR 2–12 s alone, on participants with all four signals usable
   (exploration: adding the other signals lowers the AUC).
3. Each other signal alone; GSR window learned in each fold.
4. Same analyses on cohorts 3–5 (emotion blocks): `--cohorts 3 4 5`.
5. **Day 2 replication** (added 26/11/2026, before the machine run): the same primary comparison on day-2 sessions
   (`--days 2`, cohorts 1–2, exploration set excluded), as a replication on other sessions of the same people.
   Features are z-scored within each session; quality control is applied per session.

## Change before evaluation: team arousal definition (26/11/2026)

Decided after the extraction job and its inventory, **before any AUC of the full dataset was computed**:
the high / low split is now the team definition (`latence/arousal_split.py`, `split_all`, default method "tiers"):
thirds of each participant's ratings over all their rated videos (all days together); participants with a rating SD < 1
or fewer than 8 videos in the high or low group are dropped; flagged outliers are then removed from the evaluated trials.
The previous per-session tertiles remain available (`evaluate --split session`). Micha may still propose another
definition: any later change will be reported next to this one.

Extraction inventory (job of 26/11/2026): 533 rated sessions, 434 extracted, 99 without an emotion-video recording in the
release (no read error). Sessions available before the split rules: primary (day 1, cohorts 1–2, outside the exploration set,
usable GSR, ≥ 5 high and ≥ 5 low videos) 107; day-2 replication 111; day 1 cohorts 3–5: 71.

## Full-dataset result (26/11/2026, team definition, exploration set excluded)

| Analysis | Participants | GSR 2–12 s | GSR 0–10 s | Difference (better in) | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| **PRIMARY: day 1, cohorts 1–2** | 98 | 0.569 | 0.562 | **+0.007** (52/98) | **0.020** |
| Replication: day 2, cohorts 1–2 | 109 | 0.541 | 0.536 | +0.005 (58/109) | 0.072 |
| Secondary: day 1, cohorts 3–5 | 66 | 0.570 | 0.553 | +0.016 (42/66) | 0.0001 |

- **Primary hypothesis confirmed**, with a small effect (+0.007 AUC). Same direction in all three analyses; not significant on day 2.
- GSR alone is well above chance everywhere (shuffled 95th percentile ≈ 0.52–0.53).
- **Cohorts 3–5 do not show a weaker effect** (0.570 vs 0.569): the exploration-set observation (0.549 vs 0.599) did not replicate;
  the restriction to cohorts 1–2 is therefore not needed for this effect.
- Secondary: adding SCR features to GSR 2–12 s: +0.014 (p = 0.030) day 1, +0.013 (p = 0.039) day 2, +0.005 (p = 0.59) cohorts 3–5.
  4 signals vs GSR alone: +0.004 (p = 0.47), +0.006 (p = 0.34), −0.011 (p = 0.61): the other signals add nothing.
- Respiration alone day 1 (0.420) and HR alone in cohorts 3–5 (0.439) fall below 0.5: their direct within-participant AUC is ≈ 0.50
  and the pooled coefficient ≈ 0, so the sign learned without the test participant is set by that participant's absence
  (leave-one-out bias of an uninformative feature), not a real inverse effect.

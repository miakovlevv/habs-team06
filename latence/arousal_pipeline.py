"""Step 4 pipeline: per-trial physiological features, then arousal prediction (frozen version of hugo_arousal.ipynb).

Two separate stages, so that the heavy part runs once (e.g. on the HABS machine) and the modelling runs locally:

    # 1. extract: one CSV per participant/day (resumable; a failing participant is logged, not fatal)
    python latence/arousal_pipeline.py extract --data-root $HABS_DATA --out ../habs_results/step4 --participants P001 P020
    python latence/arousal_pipeline.py extract --data-root /data --out <out_dir> --all --workers 8   # every labelled participant
    # 2. evaluate: frozen comparison of step4_preregistration.md (full-dataset section) + secondary analyses
    python latence/arousal_pipeline.py evaluate --data-root $HABS_DATA --features <out_dir> --cohorts 1 2 \
        --exclude latence/participants_dev20.txt latence/participants_confirm20_final.txt

Extraction (per participant and day; emotion-video + resting-state, PREPROCESSED):
- HR (R peaks after 50 Hz notch), GSR (1 Hz low-pass, µS), pupil (mean of both eyes, blinks removed), respiration
  (0.05–1 Hz, inspiration up), all on a 20 Hz grid, in native units.
- For every video of the metadata: baseline = mean on −5…−4 s (first second of the black screen); stored = the baseline level
  and the baseline-corrected mean of each 1-s bin from −5 to +15 s (any integer window is then a mean of bins);
  SCR features (onset 1–11 s after video onset): scr_any, scr_amp.
- Per participant: quality (RR suspects, GSR range, pupil missing) and resting-state measures (mean / SD of each signal).
Ratings are NOT read at this stage: they are joined in `evaluate` from labels_train.csv (start_marker = marker_id),
so the high / low split can change without re-extracting.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from physio_utils import (FS, bdf_path, hr_series, load_bdf, load_metadata, notch, r_peaks,  # noqa: E402
                          resp_clean, resp_polarity, rr_suspects, task_windows)

VERSION = "PREPROCESSED"
SIGNALS = ["HR", "GSR", "pupil", "resp"]
FG = 20.0                              # Hz, common grid
BINS = np.arange(-5, 15)               # 1-s bins [b, b+1[ around video onset
BASE = (-5.0, -4.0)                    # s, baseline window (first second of the black screen)
SCR_WIN = (1.0, 11.0)                  # s, SCR onsets counted for the video (step 3.2c)

# Quality thresholds (participant level, per signal) — fixed before the confirmation test
QC_RR_SUSPECT_PCT = 10.0               # HR unusable above
QC_GSR_RANGE_US = 0.05                 # GSR flat (sensor not connected) below
QC_PUPIL_MISSING_PCT = 30.0            # pupil unusable above
MIN_PER_CLASS = 5                      # high and low videos needed to score a participant


# =========================================================================== extraction

def lowpass(x, hz, fs):
    return sosfiltfilt(butter(2, hz, fs=fs, output="sos"), x)


def pupil_clean(sig):
    """Mean of both eyes on the 1000 Hz timeline (stored at ~166 Hz) -> blinks removed -> 20 Hz. Returns (signal, missing).
    No eye-tracking channels (e.g. P212): all NaN, missing = 1 (the quality rule then excludes the pupil)."""
    cols = [c for c in ["pupildiameter_le", "pupildiameter_ri"] if c in sig.columns]
    if not cols:
        return np.full(len(sig) // int(FS / FG), np.nan), 1.0
    x = sig[cols].mean(axis=1).to_numpy().copy()
    x[(x < 1.5) | (x > 9)] = np.nan
    s = pd.Series(x).interpolate(limit_area="inside")
    med = s.rolling(int(FS), center=True, min_periods=1).median()
    bad = ((s - med).abs() > 0.4).rolling(int(0.2 * FS), center=True, min_periods=1).max().astype(bool)
    s[bad] = np.nan
    missing = float(s.isna().mean())
    if missing == 1.0:
        return np.full(len(s) // int(FS / FG), np.nan), missing
    s = s.interpolate(limit_area="inside").bfill().ffill().to_numpy()
    return lowpass(s, 4.0, FS)[::int(FS / FG)], missing


def load_signals(subj, day, data, task):
    """The four signals on the 20 Hz grid (native units) + quality info."""
    _, sig, _ = load_bdf(bdf_path(task, subj, day, data=data, version=VERSION))
    pk = r_peaks(notch(sig.ECG.to_numpy()), verbose=False)
    _, hr = hr_series(pk, len(sig), fs_out=FG)
    gsr = lowpass(sig.GSR.to_numpy(), 1.0, FS)[::int(FS / FG)]
    resp = resp_clean(sig.Resp.to_numpy())
    pol = resp_polarity(resp)
    resp = (-resp if pol["fall_shorter"] else resp)[::int(FS / FG)]
    pupil, pupil_missing = pupil_clean(sig)
    m = min(len(hr), len(gsr), len(resp), len(pupil))
    x = {"HR": hr[:m], "GSR": gsr[:m], "pupil": pupil[:m], "resp": resp[:m]}
    qc = {"duration_s": m / FG, "rr_suspect_pct": 100 * float(rr_suspects(pk).mean()),
          "hr_median": float(np.median(hr)),
          "gsr_range_uS": float(np.percentile(gsr, 99) - np.percentile(gsr, 1)),
          "pupil_missing_pct": 100 * pupil_missing, "resp_inspiration_down": bool(pol["fall_shorter"])}
    return x, qc


def scr_detect(g, fs=FG, min_amp=0.02, max_rise=5.0):
    """SCRs (onset_s, amp_uS), spike artefacts removed; same detector as step 3 (hugo_brain_skin.ipynb)."""
    pk, _ = find_peaks(g, prominence=min_amp)
    rows, prev = [], 0
    for b in pk:
        lo = max(prev, b - int(max_rise * fs))
        a = lo + int(np.argmin(g[lo:b + 1]))
        prev = b
        amp = g[b] - g[a]
        if amp < min_amp or a == b or g[a] <= 1.001:
            continue
        drop = g[b] - g[min(b + int(fs), len(g) - 1)]
        if (b - a) / fs < 1.0 and drop > 0.5 * amp:
            continue
        rows.append((a / fs, amp))
    return pd.DataFrame(rows, columns=["onset_s", "amp_uS"])


def video_onsets(subj, day, data):
    """{marker_id: onset (s, in the emotion-video segment)} for every video event of the metadata."""
    m = load_metadata(subj, day, data=data, version=VERSION)
    seg = m["dataset_export"]["segments"]["emotion-video"]
    t0, dur = seg["metadata_time_origin_s"], seg["duration_s"]
    return {e["marker_id"]: e["onset_s"] - t0 for e in m["events"]
            if e.get("trial_type") == "emotion_video" and 0 <= e["onset_s"] - t0 < dur}


def extract_participant(subj, day, data):
    """Returns (trial DataFrame, participant dict)."""
    x, qc = load_signals(subj, day, data, "emotion-video")
    n = len(x["HR"])
    scr = scr_detect(x["GSR"]) if qc["gsr_range_uS"] >= QC_GSR_RANGE_US else None
    rows = []
    for marker, v in video_onsets(subj, day, data).items():
        i0 = int(round((v + BINS[0]) * FG))
        i1 = i0 + int(len(BINS) * FG)
        if i0 < 0 or i1 > n:
            continue
        row = {"participant": subj, "day": day, "marker_id": int(marker), "onset_s": round(v, 3)}
        for s in SIGNALS:
            seg = x[s][i0:i1]
            base = seg[:int((BASE[1] - BASE[0]) * FG)].mean()
            row[f"{s}_base"] = base
            for b, val in zip(BINS, (seg - base).reshape(len(BINS), int(FG)).mean(axis=1)):
                row[f"{s}_b{b:+d}"] = val
        if scr is not None:
            w = scr[(scr.onset_s >= v + SCR_WIN[0]) & (scr.onset_s < v + SCR_WIN[1])]
            row["scr_any"], row["scr_amp"] = float(len(w) > 0), float(w.amp_uS.max()) if len(w) else 0.0
        else:
            row["scr_any"] = row["scr_amp"] = np.nan
        rows.append(row)

    part = {"participant": subj, "day": day, **qc, "n_videos": len(rows)}
    try:
        r, rq = load_signals(subj, day, data, "resting-state")
        eyes_open = [(on, off) for on, off, txt in task_windows("resting-state", subj, day, data=data, version=VERSION)
                     if txt and "fermes" not in txt and "ferm" not in txt]
        sl = slice(int(eyes_open[0][0] * FG), int(eyes_open[0][1] * FG)) if eyes_open else slice(0, int(30 * FG))
        for s in SIGNALS:
            v = r[s][sl] if s == "pupil" else r[s]
            part[f"rest_{s}_mean"], part[f"rest_{s}_sd"] = float(np.nanmean(v)), float(np.nanstd(v))
        part.update({f"rest_{k}": rq[k] for k in ["rr_suspect_pct", "gsr_range_uS", "pupil_missing_pct", "duration_s"]})
    except FileNotFoundError:
        part["rest_missing"] = True
    return pd.DataFrame(rows), part


def read_ids(items):
    """Participant IDs from a mix of IDs and .txt/.csv files (column participant_id)."""
    out = []
    for it in items or []:
        f = Path(it)
        if f.is_file():
            out += (pd.read_csv(f)["participant_id"].astype(str).tolist() if f.suffix == ".csv"
                    else f.read_text().replace(",", " ").split())
        else:
            out.append(it)
    return out


def participants_from_labels(data, days, cohorts=None):
    """Participants with arousal ratings on `days` (optionally only some cohorts), ordered by cohort then ID."""
    lab = pd.read_csv(Path(data) / "labels_train.csv", na_values=["TESTSUBJECT"])
    lab = lab[lab.participant_id.astype(str).str.fullmatch(r"P\d+") & lab.rating_arousal.notna() & lab.day.isin(days)]
    if cohorts:
        lab = lab[lab.cohort.isin(cohorts)]
    first = lab.groupby("participant_id").cohort.min()
    return sorted(first.index, key=lambda q: (first[q], int(q[1:])))


def _extract_one(job):
    """Worker: extract one participant/day and write its files. Returns a log line and an optional error row."""
    p, day, data, out, k, n = job
    stem = f"{p}_day-{day}"
    if not bdf_path("emotion-video", p, day, data=data, version=VERSION).is_file():
        return f"[{k}/{n}] {stem}: no emotion-video recording, skipped", \
            {"participant": p, "day": day, "error": "no emotion-video recording", "trace": ""}
    t0 = time.time()
    try:
        trials, part = extract_participant(p, day, data)
        (out / "participants" / f"{stem}.json").write_text(json.dumps(part, indent=1))
        trials.to_csv(out / "trials" / f"{stem}.csv", index=False)   # written last: marks the participant as done
        return f"[{k}/{n}] {stem}: {len(trials)} videos ({time.time() - t0:.0f} s)", None
    except Exception as e:                                           # noqa: BLE001 — log and go on
        return f"[{k}/{n}] {stem}: FAILED {e!r}", \
            {"participant": p, "day": day, "error": repr(e), "trace": traceback.format_exc(limit=3)}


def rated_pairs(data, days, cohorts=None):
    """(participant, day) pairs with arousal ratings, day by day (all of day 1 before day 2), cohort then ID."""
    lab = pd.read_csv(Path(data) / "labels_train.csv", na_values=["TESTSUBJECT"])
    lab = lab[lab.participant_id.astype(str).str.fullmatch(r"P\d+") & lab.rating_arousal.notna() & lab.day.isin(days)]
    if cohorts:
        lab = lab[lab.cohort.isin(cohorts)]
    g = lab.groupby(["participant_id", "day"]).cohort.min().reset_index()
    g["num"] = g.participant_id.str[1:].astype(int)
    g = g.sort_values(["day", "cohort", "num"])
    return [(p, int(d)) for p, d in zip(g.participant_id, g.day)]


def run_extract(data, out, subjects, days=(1,), overwrite=False, workers=1, time_budget_min=None, pairs=None):
    """Extract every participant/day into `out` (resumable). Several workers run in parallel processes;
    with a time budget, no new participant is started once it is spent (the files already written are kept).
    pairs: explicit (participant, day) list, in processing order (otherwise all of `days` for each subject, day by day)."""
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    data, out = Path(data).expanduser().resolve(), Path(out).expanduser().resolve()
    if out == data or data in out.parents:
        sys.exit("Refusing to write inside the dataset: choose another output folder.")
    (out / "trials").mkdir(parents=True, exist_ok=True)
    (out / "participants").mkdir(parents=True, exist_ok=True)
    pairs = pairs if pairs is not None else [(p, day) for day in days for p in subjects]
    jobs = [(p, day, data, out, k, len(pairs)) for k, (p, day) in enumerate(pairs, start=1)
            if overwrite or not (out / "trials" / f"{p}_day-{day}.csv").is_file()]
    print(f"{len(pairs)} participant-days, {len(jobs)} to extract, {workers} worker(s)", flush=True)
    t_start, errors, done = time.time(), [], 0
    over = lambda: time_budget_min is not None and time.time() - t_start > 60 * time_budget_min
    with ProcessPoolExecutor(max_workers=workers) as pool:
        todo, running = list(jobs), set()
        while todo or running:
            while todo and len(running) < workers and not over():
                running.add(pool.submit(_extract_one, todo.pop(0)))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                line, err = f.result()
                done += 1
                print(line, flush=True)
                if err:
                    errors.append(err)
        if todo:
            print(f"TIME BUDGET SPENT: {len(todo)} participant-days not started "
                  f"(first: {todo[0][0]}); re-run with the same output folder to finish.", flush=True)
    if errors:
        log = out / "errors.csv"
        pd.DataFrame(errors).to_csv(log, mode="a", header=not log.is_file(), index=False)
    print(f"done in {(time.time() - t_start) / 60:.1f} min: {done} processed, {len(errors)} failure(s) or missing", flush=True)
    return errors


def cmd_extract(args):
    if args.all:
        subjects = participants_from_labels(args.data_root, args.days, args.cohorts)
    else:
        subjects = read_ids(args.participants or [args.participants_file])
    excluded = set(read_ids(args.exclude))
    subjects = [p for p in subjects if p not in excluded]
    run_extract(args.data_root, args.out, subjects, args.days, args.overwrite, args.workers, args.time_budget_min)


# =========================================================================== evaluation

def tertile_cuts(scores, frac=1 / 3):
    """(low, high) cut points on a discrete scale, groups as close as possible to a third each (same as hep_utils)."""
    s = np.asarray(scores, float)
    s = s[~np.isnan(s)]
    vals = np.unique(s)
    if len(vals) < 2:
        return np.nan, np.nan
    lo = vals[np.argmin(np.abs(np.array([(s <= v).mean() for v in vals]) - frac))]
    hi = vals[np.argmin(np.abs(np.array([(s >= v).mean() for v in vals]) - frac))]
    if lo >= hi:
        hi = vals[vals > lo].min() if (vals > lo).any() else np.nan
    return lo, hi


SPLIT = "team"   # "team": arousal_split.py (team definition, 26/11/2026); "session": tertiles per session (before)


def team_levels(data):
    """High / low / mid from the team definition (latence/arousal_split.py, Laura): thirds of each participant's
    ratings over ALL their rated videos (all days); participants with rating SD < 1 or < 8 videos in a group
    are dropped. Returns labels with participant_id, day, marker_id, rating_arousal, cohort, level, flagged."""
    from arousal_split import load_ratings, split_all
    lab = split_all(load_ratings(data))
    lab = lab[lab.discriminant & lab.day.notna()]
    lab = lab.assign(level=lab.cond.map({"haute": "high", "basse": "low", "moyenne": "mid"}))
    return lab


def load_table(features_dir, data, days, split=None):
    """Trials with ratings (labels_train.csv) and the high / low / mid level; participant table with usable signals."""
    f = Path(features_dir)
    if (f / "all_trials.csv").is_file():                       # combined files written by process_step4.py
        trials, parts = pd.read_csv(f / "all_trials.csv"), pd.read_csv(f / "all_participants.csv")
    else:
        trials = pd.concat([pd.read_csv(p) for p in sorted((f / "trials").glob("*.csv"))], ignore_index=True)
        parts = pd.DataFrame([json.loads(p.read_text()) for p in sorted((f / "participants").glob("*.json"))])
    if (split or SPLIT) == "team":
        lab = team_levels(data)                                # thirds over all days, then flagged trials removed
        lab = lab[(lab.flagged_outliers != 1) & lab.day.isin(days)]
        lab = lab.assign(day=lab.day.astype(int), marker_id=lab.start_marker.astype(int))
    else:
        lab = pd.read_csv(Path(data) / "labels_train.csv", na_values=["TESTSUBJECT"])
        lab = lab[lab.rating_arousal.notna() & (lab.flagged_outliers != 1) & lab.day.isin(days)]
        lab = lab.assign(day=lab.day.astype(int), marker_id=lab.start_marker.astype(int))
        level = []
        for (p, d), g in lab.groupby(["participant_id", "day"]):
            lo, hi = tertile_cuts(g.rating_arousal)            # cut points on the session's rated videos
            level.append(g.assign(level=np.where(g.rating_arousal >= hi, "high",
                                                 np.where(g.rating_arousal <= lo, "low", "mid"))))
        lab = pd.concat(level)
    lab = lab[["participant_id", "day", "marker_id", "rating_arousal", "level", "cohort"]]
    trials = trials.merge(lab.rename(columns={"participant_id": "participant"}), on=["participant", "day", "marker_id"])
    parts["ok_HR"] = parts.rr_suspect_pct <= QC_RR_SUSPECT_PCT
    parts["ok_GSR"] = parts.gsr_range_uS >= QC_GSR_RANGE_US
    parts["ok_pupil"] = parts.pupil_missing_pct <= QC_PUPIL_MISSING_PCT
    parts["ok_resp"] = True
    return trials, parts


def window(trials, s, a, b):
    """Baseline-corrected mean of signal s on [a, b[ s (integer bounds)."""
    return trials[[f"{s}_b{k:+d}" for k in range(a, b)]].mean(axis=1).to_numpy()


def session(d):
    """Participant-day key: features are z-scored within each session."""
    return d.participant.astype(str) + "_day-" + d.day.astype(str)


def within_z(x, groups):
    """z-score within each participant over all their videos (no labels used)."""
    df = pd.DataFrame(x).groupby(groups.to_numpy())
    return ((pd.DataFrame(x) - df.transform("mean")) / df.transform("std").replace(0, np.nan)).fillna(0).to_numpy()


def fast_auc(y, x):
    """AUC of score x for binary y (Mann–Whitney, ties averaged)."""
    r = rankdata(x)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


GSR_WINDOWS = [(a, b) for a in range(0, 6) for b in range(8, 16) if b - a >= 4]
NAIVE = {s: (s, 0, 10) for s in SIGNALS}


def per_participant_window_auc(d, y):
    """Matrix (participants x GSR_WINDOWS) of within-participant AUCs of the GSR window mean (for fold-wise selection)."""
    lab = (d.level != "mid").to_numpy()
    out = {}
    for a, b in GSR_WINDOWS:
        x = within_z(window(d, "GSR", a, b)[:, None], session(d))[:, 0]
        dd = pd.DataFrame({"p": d.participant.to_numpy()[lab], "x": x[lab], "y": y[lab]})
        out[(a, b)] = dd.groupby("p").apply(lambda g: fast_auc(g.y.to_numpy(), g.x.to_numpy()), include_groups=False)
    return pd.DataFrame(out)


def loso(d, spec_fn, y):
    """Leave-one-participant-out logistic regression; spec_fn(test_participant) -> list of (signal, a, b) or (column,).
    Returns the within-participant AUC of each test participant."""
    from sklearn.linear_model import LogisticRegression
    feats = {}
    aucs = {}
    lab = (d.level != "mid").to_numpy()
    for p in d.participant.unique():
        spec = spec_fn(p)
        key = tuple(spec)
        if key not in feats:
            cols = [d[it[0]].to_numpy(float) if len(it) == 1 else window(d, *it) for it in spec]
            feats[key] = within_z(np.column_stack(cols), session(d))
        X = feats[key]
        te = (d.participant == p).to_numpy()
        tr = ~te & lab
        model = LogisticRegression().fit(X[tr], y[tr])
        k = te & lab
        aucs[p] = fast_auc(y[k], model.predict_proba(X[k])[:, 1])
    return pd.Series(aucs)


GSR_PRIMARY = ("GSR", 2, 12)      # skin read ~2 s later than the video (heart -> skin delay)
GSR_NAIVE = ("GSR", 0, 10)        # the video window
SCR = [("scr_any",), ("scr_amp",)]


def scores(d, y, sets):
    """AUC per participant (rows) for each named set (columns). sets: {name: spec_fn(test_participant)}."""
    return pd.DataFrame({name: loso(d, fn, y) for name, fn in sets.items()})


def paired(a, b):
    """Mean difference, participants better, Wilcoxon p (two-sided)."""
    diff = a - b
    return diff.mean(), f"{(diff > 0).sum()}/{len(diff)}", wilcoxon(diff).pvalue


def scorable(d):
    """Keep participants with at least MIN_PER_CLASS high and low videos."""
    c = d.groupby("participant")["level"].value_counts().unstack(fill_value=0)
    keep = c.index[(c.get("high", 0) >= MIN_PER_CLASS) & (c.get("low", 0) >= MIN_PER_CLASS)]
    return d[d.participant.isin(keep)].reset_index(drop=True), sorted(set(c.index) - set(keep))


def summary(auc):
    return pd.DataFrame({"mean AUC": auc.mean(), "median AUC": auc.median(),
                         "AUC > 0.5": [f"{(auc[c] > 0.5).sum()}/{len(auc)}" for c in auc],
                         "p vs 0.5": [wilcoxon(auc[c] - 0.5).pvalue for c in auc]})


def cmd_evaluate(args):
    pd.set_option("display.width", 200)
    trials, parts = load_table(args.features, args.data_root, args.days, split=args.split)
    print(f"high / low definition: {args.split}")
    excluded = set(read_ids(args.exclude))
    keep = read_ids(args.participants_file and [args.participants_file])
    t = trials[~trials.participant.isin(excluded)]
    if keep:
        t = t[t.participant.isin(keep)]
    if args.cohorts:
        t = t[t.cohort.isin(args.cohorts)]
    parts = parts[parts.day.isin(args.days) & parts.participant.isin(t.participant.unique())]
    print(f"cohorts {args.cohorts or 'all'}, {len(excluded)} participants excluded beforehand (exploration set), "
          f"{parts.participant.nunique()} participants with features")
    qc = parts[["participant", "day"] + [f"ok_{s}" for s in SIGNALS]]
    usable = lambda mask: t.merge(qc.loc[mask, ["participant", "day"]], on=["participant", "day"])  # QC per session
    print(f"quality control: GSR unusable {(~qc.ok_GSR).sum()}, HR {(~qc.ok_HR).sum()}, pupil {(~qc.ok_pupil).sum()}")

    # ---- PRIMARY: GSR 2-12 s vs GSR 0-10 s, on every participant with a usable GSR
    d, small = scorable(usable(qc.ok_GSR))
    y = (d.level == "high").to_numpy()
    print(f"\nPRIMARY set: {d.participant.nunique()} participants, {len(d)} rated videos "
          f"({(d.level == 'high').sum()} high / {(d.level == 'low').sum()} low / {(d.level == 'mid').sum()} mid); "
          f"{len(small)} with < {MIN_PER_CLASS} high or low videos")
    win_auc = per_participant_window_auc(d, y)
    gsr_sets = {
        "GSR 2-12 s (PRIMARY)": lambda p: [GSR_PRIMARY],
        "GSR 0-10 s (REFERENCE)": lambda p: [GSR_NAIVE],
        "GSR 2-12 s + SCR": lambda p: [GSR_PRIMARY] + SCR,
        "GSR window learned": lambda p: [("GSR", *win_auc.drop(index=p).median().idxmax())],
    }
    auc1 = scores(d, y, gsr_sets)
    print(summary(auc1).round(3).to_string())
    rng = np.random.default_rng(0)
    null = []
    for _ in range(args.n_null):
        lev = d.groupby("participant")["level"].transform(lambda l: pd.Series(rng.permutation(l.to_numpy()), index=l.index))
        null.append(scores(d.assign(level=lev), (lev == "high").to_numpy(),
                           {k: gsr_sets[k] for k in list(gsr_sets)[:2]}).mean())
    if null:
        null = pd.DataFrame(null)
        print("chance level (ratings shuffled within participant), 95th percentile of the mean AUC: "
              + ", ".join(f"{c}: {np.percentile(null[c], 95):.3f}" for c in null))
    m, n, pv = paired(auc1.iloc[:, 0], auc1.iloc[:, 1])
    print(f"\n>>> PRIMARY TEST  GSR 2-12 s vs GSR 0-10 s: mean difference {m:+.3f}, better in {n}, Wilcoxon p = {pv:.4g}")
    m, n, pv = paired(auc1["GSR 2-12 s + SCR"], auc1.iloc[:, 0])
    print(f"    secondary     GSR 2-12 s + SCR vs GSR 2-12 s: {m:+.3f}, better in {n}, p = {pv:.4g}")

    # ---- SECONDARY: 4 signals vs GSR alone, each signal alone (participants with all four signals usable)
    d4, _ = scorable(usable(qc[[f"ok_{s}" for s in SIGNALS]].all(axis=1)))
    y4 = (d4.level == "high").to_numpy()
    sets4 = {"GSR 2-12 s": lambda p: [GSR_PRIMARY],
             "4 signals (GSR 2-12 s, others 0-10 s)": lambda p: [GSR_PRIMARY, NAIVE["HR"], NAIVE["pupil"], NAIVE["resp"]],
             **{f"{s} alone, 0-10 s": (lambda p, s=s: [NAIVE[s]]) for s in ["HR", "pupil", "resp"]}}
    auc4 = scores(d4, y4, sets4)
    print(f"\nSECONDARY set (all four signals usable): {d4.participant.nunique()} participants")
    print(summary(auc4).round(3).to_string())
    m, n, pv = paired(auc4.iloc[:, 1], auc4.iloc[:, 0])
    print(f"    secondary     4 signals vs GSR 2-12 s alone: {m:+.3f}, better in {n}, p = {pv:.4g}")

    if args.save:
        tag = ("cohorts-" + "".join(map(str, args.cohorts)) if args.cohorts else "all-cohorts") \
            + "_day-" + "".join(map(str, args.days))
        out = Path(args.features) / "results" / tag
        out.mkdir(parents=True, exist_ok=True)
        auc1.to_csv(out / "auc_primary_set.csv")
        auc4.to_csv(out / "auc_four_signal_set.csv")
        print("saved in", out)


# =========================================================================== CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="per-trial features (heavy: reads the BDF files)")
    ex.add_argument("--data-root", required=True, help="dataset root (contains PREPROCESSED/ and labels_train.csv)")
    ex.add_argument("--out", required=True, help="output folder (outside the dataset)")
    who = ex.add_mutually_exclusive_group(required=True)
    who.add_argument("--participants", nargs="+")
    who.add_argument("--participants-file")
    who.add_argument("--all", action="store_true", help="every participant with arousal ratings on --days")
    ex.add_argument("--cohorts", nargs="+", type=int, help="with --all: only these cohorts")
    ex.add_argument("--exclude", nargs="+", default=[], metavar="ID_OR_FILE", help="participants to skip")
    ex.add_argument("--days", nargs="+", type=int, default=[1])
    ex.add_argument("--workers", type=int, default=1, help="parallel processes")
    ex.add_argument("--time-budget-min", type=float, help="start no new participant after this many minutes")
    ex.add_argument("--overwrite", action="store_true")
    ev = sub.add_parser("evaluate", help="arousal prediction from the extracted features (light)")
    ev.add_argument("--data-root", required=True, help="folder with labels_train.csv")
    ev.add_argument("--features", required=True, help="--out folder of extract")
    ev.add_argument("--days", nargs="+", type=int, default=[1])
    ev.add_argument("--cohorts", nargs="+", type=int, help="only these cohorts (e.g. 1 2)")
    ev.add_argument("--exclude", nargs="+", default=[], metavar="ID_OR_FILE",
                    help="participants left out, e.g. the exploration set (IDs and/or files)")
    ev.add_argument("--participants-file", help="score only these participants")
    ev.add_argument("--split", choices=["team", "session"], default="team",
                    help="high / low definition: team (arousal_split.py, default) or session (tertiles per session)")
    ev.add_argument("--n-null", type=int, default=200, help="shuffles for the chance level (0 to skip)")
    ev.add_argument("--save", action="store_true", help="write results/<cohorts>/ in the features folder")
    args = parser.parse_args()
    {"extract": cmd_extract, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

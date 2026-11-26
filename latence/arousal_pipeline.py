"""Step 4 pipeline: per-trial physiological features, then arousal prediction (frozen version of hugo_arousal.ipynb).

Two separate stages, so that the heavy part runs once (e.g. on the HABS machine) and the modelling runs locally:

    # 1. extract: one CSV per participant/day (resumable; a failing participant is logged, not fatal)
    python latence/arousal_pipeline.py extract --data-root $HABS_DATA --out ../habs_results/step4 --participants P001 P020
    python latence/arousal_pipeline.py extract --data-root /data --out <out_dir> --all           # every labelled participant
    # 2. evaluate: frozen comparison of step4_preregistration.md (+ secondary analyses)
    python latence/arousal_pipeline.py evaluate --data-root $HABS_DATA --features ../habs_results/step4

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
    """Mean of both eyes on the 1000 Hz timeline (stored at ~166 Hz) -> blinks removed -> 20 Hz. Returns (signal, missing)."""
    x = sig[["pupildiameter_le", "pupildiameter_ri"]].mean(axis=1).to_numpy().copy()
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


def participants_from_labels(data, days):
    lab = pd.read_csv(Path(data) / "labels_train.csv", na_values=["TESTSUBJECT"])
    lab = lab[lab.participant_id.astype(str).str.fullmatch(r"P\d+") & lab.rating_arousal.notna() & lab.day.isin(days)]
    return sorted(lab.participant_id.unique())


def cmd_extract(args):
    out = Path(args.out).expanduser().resolve()
    data = Path(args.data_root).expanduser().resolve()
    if out == data or data in out.parents:
        sys.exit("Refusing to write inside the dataset: choose another --out.")
    (out / "trials").mkdir(parents=True, exist_ok=True)
    (out / "participants").mkdir(parents=True, exist_ok=True)
    if args.all:
        subjects = participants_from_labels(data, args.days)
    elif args.participants_file:
        subjects = Path(args.participants_file).read_text().replace(",", " ").split()
    else:
        subjects = args.participants
    errors = []
    t_start = time.time()
    for k, p in enumerate(subjects, start=1):
        for day in args.days:
            stem = f"{p}_day-{day}"
            done = out / "trials" / f"{stem}.csv"
            if done.is_file() and not args.overwrite:
                continue
            if not bdf_path("emotion-video", p, day, data=data, version=VERSION).is_file():
                continue                                     # day not recorded for this participant
            t0 = time.time()
            try:
                trials, part = extract_participant(p, day, data)
                (out / "participants" / f"{stem}.json").write_text(json.dumps(part, indent=1))
                trials.to_csv(done, index=False)             # written last: marks the participant as done
                print(f"[{k}/{len(subjects)}] {stem}: {len(trials)} videos ({time.time() - t0:.0f} s)", flush=True)
            except Exception as e:                           # noqa: BLE001 — log and go on
                errors.append({"participant": p, "day": day, "error": repr(e), "trace": traceback.format_exc(limit=3)})
                print(f"[{k}/{len(subjects)}] {stem}: FAILED {e!r}", flush=True)
    if errors:
        log = out / "errors.csv"
        pd.DataFrame(errors).to_csv(log, mode="a", header=not log.is_file(), index=False)
    print(f"done in {(time.time() - t_start) / 60:.1f} min, {len(errors)} failure(s)")


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


def load_table(features_dir, data, days):
    """Trials with ratings (labels_train.csv) and the high / low / mid level; participant table with usable signals."""
    f = Path(features_dir)
    trials = pd.concat([pd.read_csv(p) for p in sorted((f / "trials").glob("*.csv"))], ignore_index=True)
    parts = pd.DataFrame([json.loads(p.read_text()) for p in sorted((f / "participants").glob("*.json"))])
    lab = pd.read_csv(Path(data) / "labels_train.csv", na_values=["TESTSUBJECT"])
    lab = lab[lab.rating_arousal.notna() & (lab.flagged_outliers != 1) & lab.day.isin(days)]
    lab = lab.assign(day=lab.day.astype(int), marker_id=lab.start_marker.astype(int))
    level = []
    for (p, d), g in lab.groupby(["participant_id", "day"]):
        lo, hi = tertile_cuts(g.rating_arousal)                # cut points on all the participant's rated videos
        level.append(g.assign(level=np.where(g.rating_arousal >= hi, "high", np.where(g.rating_arousal <= lo, "low", "mid"))))
    lab = pd.concat(level)[["participant_id", "day", "marker_id", "rating_arousal", "level", "cohort"]]
    trials = trials.merge(lab.rename(columns={"participant_id": "participant"}), on=["participant", "day", "marker_id"])
    parts["ok_HR"] = parts.rr_suspect_pct <= QC_RR_SUSPECT_PCT
    parts["ok_GSR"] = parts.gsr_range_uS >= QC_GSR_RANGE_US
    parts["ok_pupil"] = parts.pupil_missing_pct <= QC_PUPIL_MISSING_PCT
    parts["ok_resp"] = True
    return trials, parts


def window(trials, s, a, b):
    """Baseline-corrected mean of signal s on [a, b[ s (integer bounds)."""
    return trials[[f"{s}_b{k:+d}" for k in range(a, b)]].mean(axis=1).to_numpy()


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
        x = within_z(window(d, "GSR", a, b)[:, None], d.participant)[:, 0]
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
            feats[key] = within_z(np.column_stack(cols), d.participant)
        X = feats[key]
        te = (d.participant == p).to_numpy()
        tr = ~te & lab
        model = LogisticRegression().fit(X[tr], y[tr])
        k = te & lab
        aucs[p] = fast_auc(y[k], model.predict_proba(X[k])[:, 1])
    return pd.Series(aucs)


def evaluate_sets(d, y):
    """AUC per participant for the primary and secondary sets."""
    win_auc = per_participant_window_auc(d, y)

    def learned(p):                                          # GSR window chosen on the training participants only
        return [("GSR", *win_auc.drop(index=p).median().idxmax())]

    others = [NAIVE["HR"], NAIVE["pupil"], NAIVE["resp"]]
    sets = {
        "PRIMARY: 4 signals, skin window learned": lambda p: others + learned(p),
        "REFERENCE: 4 signals, naive 0-10 s": lambda p: list(NAIVE.values()),
        "GSR alone, naive 0-10 s": lambda p: [NAIVE["GSR"]],
        "GSR alone, Maroua's lag 2-12 s": lambda p: [("GSR", 2, 12)],
        "GSR alone, window learned": learned,
        "HR alone, naive": lambda p: [NAIVE["HR"]],
        "pupil alone, naive": lambda p: [NAIVE["pupil"]],
        "resp alone, naive": lambda p: [NAIVE["resp"]],
        "4 signals, skin window learned + SCR": lambda p: others + learned(p) + [("scr_any",), ("scr_amp",)],
    }
    chosen = pd.Series([learned(p)[0][1:] for p in d.participant.unique()]).value_counts()
    return pd.DataFrame({name: loso(d, fn, y) for name, fn in sets.items()}), chosen


def cmd_evaluate(args):
    trials, parts = load_table(args.features, args.data_root, args.days)
    ok = parts[parts[[f"ok_{s}" for s in SIGNALS]].all(axis=1)]
    excluded = parts.loc[~parts.index.isin(ok.index), ["participant", "day"] + [f"ok_{s}" for s in SIGNALS]]
    d = trials.merge(ok[["participant", "day"]], on=["participant", "day"])
    counts = d.groupby("participant")["level"].value_counts().unstack(fill_value=0)
    enough = counts.index[(counts.get("high", 0) >= MIN_PER_CLASS) & (counts.get("low", 0) >= MIN_PER_CLASS)]
    d = d[d.participant.isin(enough)].reset_index(drop=True)
    if args.participants_file:
        keep = Path(args.participants_file).read_text().replace(",", " ").split()
        d = d[d.participant.isin(keep)].reset_index(drop=True)
    y = (d.level == "high").to_numpy()
    print(f"{d.participant.nunique()} participants scored, {len(d)} rated videos "
          f"({(d.level == 'high').sum()} high / {(d.level == 'low').sum()} low / {(d.level == 'mid').sum()} mid)")
    if len(excluded):
        print("excluded by quality control:\n" + excluded.to_string(index=False))
    small = sorted(set(counts.index) - set(enough))
    if small:
        print(f"excluded, fewer than {MIN_PER_CLASS} high or low videos: {small}")

    auc, chosen = evaluate_sets(d, y)
    print("GSR windows learned in the folds:", {f"{a}-{b} s": int(n) for (a, b), n in chosen.items()})

    # chance level: ratings shuffled within participant (primary and reference sets)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(args.n_null):
        lev = d.groupby("participant")["level"].transform(lambda l: pd.Series(rng.permutation(l.to_numpy()), index=l.index))
        dd = d.assign(level=lev)
        a, _ = evaluate_sets_subset(dd, (lev == "high").to_numpy())
        null.append(a.mean())
    null = pd.DataFrame(null)

    rows = []
    for c in auc:
        rows.append({"set": c, "mean AUC": auc[c].mean(), "median AUC": auc[c].median(),
                     "AUC > 0.5": f"{(auc[c] > 0.5).sum()}/{len(auc)}", "p vs 0.5": wilcoxon(auc[c] - 0.5).pvalue})
    table = pd.DataFrame(rows).set_index("set")
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    if len(null):
        print("chance level, 95th percentile of the mean AUC (shuffled ratings): "
              + ", ".join(f"{c}: {np.percentile(null[c], 95):.3f}" for c in null))
    prim, ref = auc.columns[0], auc.columns[1]
    diff = auc[prim] - auc[ref]
    print(f"\nPRIMARY TEST ({prim} vs {ref}): mean difference {diff.mean():+.3f}, "
          f"better in {(diff > 0).sum()}/{len(diff)} participants, Wilcoxon p = {wilcoxon(diff).pvalue:.4f}")
    if args.save:
        out = Path(args.features) / "results"
        out.mkdir(exist_ok=True)
        auc.to_csv(out / "auc_per_participant.csv")
        table.to_csv(out / "summary.csv")
        print("saved in", out)


def evaluate_sets_subset(d, y):
    """Primary and reference sets only (for the permutation null)."""
    win_auc = per_participant_window_auc(d, y)
    others = [NAIVE["HR"], NAIVE["pupil"], NAIVE["resp"]]
    sets = {"PRIMARY": lambda p: others + [("GSR", *win_auc.drop(index=p).median().idxmax())],
            "REFERENCE": lambda p: list(NAIVE.values())}
    return pd.DataFrame({name: loso(d, fn, y) for name, fn in sets.items()}), None


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
    ex.add_argument("--days", nargs="+", type=int, default=[1])
    ex.add_argument("--overwrite", action="store_true")
    ev = sub.add_parser("evaluate", help="arousal prediction from the extracted features (light)")
    ev.add_argument("--data-root", required=True, help="folder with labels_train.csv")
    ev.add_argument("--features", required=True, help="--out folder of extract")
    ev.add_argument("--days", nargs="+", type=int, default=[1])
    ev.add_argument("--participants-file", help="score only these participants (e.g. the confirmation set)")
    ev.add_argument("--n-null", type=int, default=200, help="shuffles for the chance level (0 to skip)")
    ev.add_argument("--save", action="store_true", help="write results/ in the features folder")
    args = parser.parse_args()
    {"extract": cmd_extract, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

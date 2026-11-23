"""Étape 2 — HEP au niveau du groupe : traitement participant par participant.

Pour chaque participant, sur tous ses jours disponibles (emotion-video) :
  1. notes de toutes ses vidéos → tiers bas / haut (add_arousal_split, provisoire) ;
  2. jour par jour (pour limiter la mémoire) : nettoyage EEG, époques autour des pics R,
     on accumule les moyennes par condition (toutes électrodes), l'ECG, les faux battements ;
  3. on enregistre un fichier <subj>.npz + une ligne de résumé.

Les fichiers servent ensuite au notebook laura_hep_groupe.ipynb (tests de groupe).

    python latence/hep_group.py P001 P006 ...      # participants donnés
    python latence/hep_group.py --all              # tous ceux présents dans HABS_DATA/CURATED
    options : --out DOSSIER (défaut HABS_DATA/derivatives/hep), --force (recalcule)
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hep_utils import (DATA, FC_CLUSTER, add_arousal_split, bdf_path, cluster_wave,  # noqa: E402
                       ecg_epochs, hep_epochs, load_labels, load_video_session,
                       surrogate_peaks, video_trials)

CONDS = ["basse", "moyenne", "haute"]
MIN_VIDEOS = 8          # vidéos minimum par condition (basse et haute) pour garder un participant


def available_days(subj, data=DATA):
    return [d for d in (1, 2, 3) if bdf_path("emotion-video", subj, d, data).is_file()]


def process_subject(subj, data=DATA, labels=None):
    """Retourne (arrays à enregistrer, résumé) pour un participant."""
    labels = load_labels(data) if labels is None else labels
    days = available_days(subj, data)
    if not days:
        raise FileNotFoundError("aucun enregistrement emotion-video")

    # 1. découpage haute / basse sur l'ensemble des vidéos notées du participant
    trials = pd.concat([video_trials(subj, d, data, labels=labels) for d in days], ignore_index=True)
    trials = add_arousal_split(trials)
    lo, hi = trials.attrs["cuts"][subj]
    n_vid = trials.cond.value_counts()
    summary = {"subj": subj, "days": days, "n_videos_notees": int(trials.arousal.notna().sum()),
               "seuil_basse": lo, "seuil_haute": hi,
               **{f"videos_{c}": int(n_vid.get(c, 0)) for c in CONDS}}
    if not np.isfinite(hi) or min(n_vid.get("basse", 0), n_vid.get("haute", 0)) < MIN_VIDEOS:
        summary["exclu"] = "pas assez de vidéos en haute ou basse"
        return None, summary
    cond_of = {(r.day, r.trial): r.cond for r in trials.itertuples()}

    # 2. jour par jour : on accumule sommes et comptes
    sums, ecg_sums, counts = {}, {}, {c: 0 for c in CONDS}
    sur_sum, sur_n = None, 0
    waves, rows, bads, hr = [], [], {}, []
    for d in days:
        seg = load_video_session(subj, d, data, verbose=False)
        seg["trials"] = seg["trials"].assign(cond=[cond_of[(d, t)] for t in seg["trials"].trial])
        bads[d] = seg["bads"]
        hr.append(60 / np.median(np.diff(seg["rpk"]) / 1000.0))

        ep = hep_epochs(seg)
        meta = ep.metadata.reset_index(drop=True)
        meta["cond"] = [cond_of[(d, t)] for t in meta.trial]
        x = ep.get_data() * 1e6                                    # µV
        e = ecg_epochs(seg, ep)
        for c in CONDS:
            m = (meta.cond == c).to_numpy()
            if m.any():
                sums[c] = sums.get(c, 0) + x[m].sum(0)
                ecg_sums[c] = ecg_sums.get(c, 0) + e[m].sum(0)
                counts[c] += int(m.sum())
        waves.append(cluster_wave(ep).astype(np.float32))
        rows.append(meta[["day", "trial", "arousal", "cond", "rr_s", "next_rr_s", "t_in_video_s"]])
        times, ch_names = ep.times, ep.ch_names

        sp = hep_epochs(seg, peaks=surrogate_peaks(seg, seed=1))
        s = sp.get_data() * 1e6
        sur_sum = s.sum(0) if sur_sum is None else sur_sum + s.sum(0)
        sur_n += len(s)
        del seg, ep, x, sp, s

    meta = pd.concat(rows, ignore_index=True)
    rr_vid = meta[meta.cond.isin(["basse", "haute"])].groupby(["day", "trial"]).agg(
        rr=("rr_s", "mean"), cond=("cond", "first"))
    summary.update({f"battements_{c}": counts[c] for c in CONDS})
    summary.update({"battements_faux": sur_n, "hr_mediane_bpm": float(np.mean(hr)),
                    "rr_basse_s": float(rr_vid.rr[rr_vid.cond == "basse"].mean()),
                    "rr_haute_s": float(rr_vid.rr[rr_vid.cond == "haute"].mean()),
                    "interpolees": json.dumps({str(k): v for k, v in bads.items()}, ensure_ascii=False)})

    arrays = {"times": times, "ch_names": np.array(ch_names), "cluster": np.array(FC_CLUSTER),
              "surrogate": (sur_sum / sur_n).astype(np.float32),
              "waves": np.concatenate(waves), "wave_cond": meta.cond.to_numpy().astype(str),
              "wave_video": (meta.day.astype(str) + "_" + meta.trial.astype(str)).to_numpy().astype(str),
              "wave_rr": meta.rr_s.to_numpy(np.float32)}
    for c in CONDS:
        if counts[c]:
            arrays[f"evoked_{c}"] = (sums[c] / counts[c]).astype(np.float32)   # (électrodes, temps) µV
            arrays[f"ecg_{c}"] = (ecg_sums[c] / counts[c]).astype(np.float32)
    return arrays, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(DATA) / "derivatives" / "hep")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    subjects = args.subjects
    if args.all:
        subjects = sorted(p.name for p in (Path(DATA) / "CURATED").glob("P*") if available_days(p.name))
    args.out.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    summary_file = args.out / "resume.csv"
    done = pd.read_csv(summary_file) if summary_file.is_file() else pd.DataFrame(columns=["subj"])

    for i, subj in enumerate(subjects, 1):
        if not args.force and (args.out / f"{subj}.npz").is_file():
            print(f"[{i}/{len(subjects)}] {subj} : déjà fait")
            continue
        t = time.time()
        try:
            arrays, summary = process_subject(subj, labels=labels)
            if arrays is not None:
                np.savez_compressed(args.out / f"{subj}.npz", **arrays)
            msg = summary.get("exclu") or (f"basse {summary['battements_basse']} / haute "
                                          f"{summary['battements_haute']} battements")
        except Exception as err:   # un participant qui échoue ne bloque pas les autres
            summary = {"subj": subj, "exclu": f"erreur : {err!r}"}
            msg = summary["exclu"]
            traceback.print_exc()
        print(f"[{i}/{len(subjects)}] {subj} ({time.time() - t:.0f} s) : {msg}", flush=True)
        done = pd.concat([done[done.subj != subj], pd.DataFrame([summary])], ignore_index=True)
        done.to_csv(summary_file, index=False)


if __name__ == "__main__":
    main()

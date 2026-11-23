"""Étape 2, piste 1 — HEP au repos, à comparer au HEP pendant les vidéos.

Pour chaque participant et chaque jour, on prend les fenêtres de repos :
  - yeux ouverts : « repos calme sans clignements… » (resting-state, 30 s)
                   + « repos calme yeux ouverts » (tâche artefact, 2 × 60 s) ;
  - yeux fermés  : « repos calme yeux fermes » (resting-state, 60 s) — comparaison secondaire,
                   l'état cérébral yeux fermés étant très différent du visionnage d'une vidéo.
Même nettoyage que pour les vidéos (hep_utils) ; les électrodes interpolées pendant la vidéo du
même jour sont aussi interpolées ici, pour comparer les mêmes électrodes.

Résultats : HABS_DATA/derivatives/hep_repos/<subj>.npz + resume_repos.csv.
Les fichiers des vidéos (derivatives/hep/) ne sont pas modifiés.

    python latence/hep_repos.py --all          (ou une liste de participants)
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physio_utils import DATA, FS, bandpass, bdf_path, ecg_polarity, load_bdf, notch, r_peaks, task_windows  # noqa: E402
from hep_utils import (_montage, ecg_epochs, flat_or_noisy, hep_epochs,  # noqa: E402
                       pulse_channels)

CONDS = {"repos_yo": ("repos calme sans clignements", "repos calme yeux ouverts"),
         "repos_yf": ("repos calme yeux fermes",)}
TASKS = ["resting-state", "artefact"]
VIDEO_OUT = Path(DATA) / "derivatives" / "hep"


def video_bads(subj, day):
    """Électrodes interpolées pendant la vidéo du même jour (resume.csv de hep_group.py)."""
    f = VIDEO_OUT / "resume.csv"
    if not f.is_file():
        return []
    r = pd.read_csv(f).set_index("subj")
    if subj not in r.index or pd.isna(r.loc[subj, "interpolees"]):
        return []
    b = json.loads(r.loc[subj, "interpolees"]).get(str(day), {})
    return sum(b.values(), [])


def rest_windows(subj, day, task):
    """Fenêtres de repos d'une tâche, au format attendu par hep_epochs (une ligne par fenêtre)."""
    rows = []
    for on, off, txt in task_windows(task, subj, day):
        cond = next((c for c, keys in CONDS.items() if txt and txt.startswith(keys)), None)
        if task == "artefact" and txt == "repos calme":
            # Libellé court chez certains participants (ex. P020) ; mêmes fenêtres de 60 s, au même
            # endroit du protocole que « repos calme yeux ouverts » chez les autres → yeux ouverts.
            cond = "repos_yo"
        if cond:
            rows.append({"subj": subj, "day": day, "on_s": on, "off_s": off, "video": cond,
                         "arousal": np.nan, "cond": cond})
    df = pd.DataFrame(rows)
    df["trial"] = np.arange(len(df))
    return df


def load_rest_session(subj, day, task, extra_bads=(), l_freq=0.5, h_freq=30.0):
    """Même chaîne que hep_utils.load_video_session, pour une tâche de repos."""
    _, sig, ch = load_bdf(bdf_path(task, subj, day))
    eeg_names = ch.loc[ch.modality == "EEG", "bdf_label"].tolist()
    ecg_raw = notch(sig["ECG"].to_numpy())
    pol, _ = ecg_polarity(ecg_raw)
    rpk = r_peaks(ecg_raw)
    ecg = pol * bandpass(ecg_raw, 0.5, 45)

    raw = mne.io.RawArray(sig[eeg_names].to_numpy().T * 1e-6, mne.create_info(eeg_names, FS, "eeg"))
    del sig
    raw.set_montage(_montage(), on_missing="warn")
    raw.filter(l_freq, h_freq, phase="zero")
    bads = {"plates/bruitées": flat_or_noisy(raw)}
    bads["pouls"] = pulse_channels(raw, rpk, exclude=bads["plates/bruitées"])
    bads["vidéo du même jour"] = [b for b in extra_bads if b not in sum(bads.values(), [])]
    raw.info["bads"] = sorted(set(sum(bads.values(), [])))
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)
    return {"subj": subj, "day": day, "raw": raw, "ecg": ecg, "rpk": rpk,
            "trials": rest_windows(subj, day, task), "bads": bads}


def process_subject(subj):
    sums, ecg_sums, counts, rr = {}, {}, {c: 0 for c in CONDS}, {c: [] for c in CONDS}
    bads, times, ch_names = {}, None, None
    for day in (1, 2, 3):
        for task in TASKS:
            if not bdf_path(task, subj, day).is_file():
                continue
            seg = load_rest_session(subj, day, task, extra_bads=video_bads(subj, day))
            bads[f"{day}-{task}"] = seg["bads"]
            if seg["trials"].empty:
                continue
            ep = hep_epochs(seg)
            if len(ep) == 0:
                continue
            meta = ep.metadata.reset_index(drop=True)
            x, e = ep.get_data() * 1e6, ecg_epochs(seg, ep)
            for c in CONDS:
                m = (meta.video == c).to_numpy()
                if m.any():
                    sums[c] = sums.get(c, 0) + x[m].sum(0)
                    ecg_sums[c] = ecg_sums.get(c, 0) + e[m].sum(0)
                    counts[c] += int(m.sum())
                    rr[c] += meta.rr_s[m].dropna().tolist()
            times, ch_names = ep.times, ep.ch_names
            del seg, ep, x
    if times is None:
        raise FileNotFoundError("aucun enregistrement de repos utilisable")
    arrays = {"times": times, "ch_names": np.array(ch_names)}
    for c in CONDS:
        if counts[c]:
            arrays[f"evoked_{c}"] = (sums[c] / counts[c]).astype(np.float32)
            arrays[f"ecg_{c}"] = (ecg_sums[c] / counts[c]).astype(np.float32)
    summary = {"subj": subj, **{f"battements_{c}": counts[c] for c in CONDS},
               **{f"rr_{c}_s": float(np.mean(rr[c])) if rr[c] else np.nan for c in CONDS},
               "interpolees": json.dumps(bads, ensure_ascii=False)}
    return arrays, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(DATA) / "derivatives" / "hep_repos")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    subjects = args.subjects or []
    if args.all:   # les participants déjà traités pour les vidéos
        subjects = sorted(f.stem for f in VIDEO_OUT.glob("P*.npz"))
    args.out.mkdir(parents=True, exist_ok=True)
    summary_file = args.out / "resume_repos.csv"
    done = pd.read_csv(summary_file) if summary_file.is_file() else pd.DataFrame(columns=["subj"])
    for i, subj in enumerate(subjects, 1):
        if not args.force and (args.out / f"{subj}.npz").is_file():
            print(f"[{i}/{len(subjects)}] {subj} : déjà fait", flush=True)
            continue
        t = time.time()
        try:
            arrays, summary = process_subject(subj)
            np.savez_compressed(args.out / f"{subj}.npz", **arrays)
            msg = ", ".join(f"{c} {summary[f'battements_{c}']}" for c in CONDS)
        except Exception as err:
            summary = {"subj": subj, "exclu": f"erreur : {err!r}"}
            msg = summary["exclu"]
            traceback.print_exc()
        print(f"[{i}/{len(subjects)}] {subj} ({time.time() - t:.0f} s) : {msg}", flush=True)
        done = pd.concat([done[done.subj != subj], pd.DataFrame([summary])], ignore_index=True)
        done.to_csv(summary_file, index=False)


if __name__ == "__main__":
    main()

"""Fonctions réutilisables pour l'analyse des latences entre signaux physiologiques.

Extraites du notebook de validation de Hugo (latence/hugo_latency_validation.ipynb),
sans changement de logique. Seule différence : le dossier des données se règle
via la variable d'environnement HABS_DATA (sinon : <repo>/sample).

    from physio_utils import r_peaks, task_windows, load_physio, ...

Ajouts de Hugo après fusion : notch (déplacé depuis hep_utils), resp_clean, breath_rate,
resp_polarity, option verbose de r_peaks, durée d'événement absente tolérée dans task_windows.
Auto-test (reproduit l'étape 1.5 sur le sample) : python latence/physio_utils.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import (butter, coherence, correlate, correlation_lags, csd, filtfilt, iirnotch,
                          find_peaks, sosfiltfilt, welch)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starting_notebooks"))
from bdf_loader import load_bdf  # noqa: E402

DATA = Path(os.environ.get("HABS_DATA", ROOT / "sample"))   # dataset en lecture seule
VERSION = "CURATED"   # signaux bruts : on maîtrise nous-mêmes le filtrage
SUBJ, DAY = "P001", 1
FS = 1000.0           # Hz, fréquence des signaux BDF
FS_HR = 4.0           # Hz, grille régulière pour HR et Resp


# --------------------------------------------------------------------------- chargement

def bdf_path(task, subj=SUBJ, day=DAY, data=DATA, version=VERSION):
    return Path(data) / version / subj / f"{subj}_day-{day}_task-{task}_multimodal.bdf"


def load_physio(task, version=VERSION, subj=SUBJ, day=DAY, data=DATA):
    """Charge ECG + Resp d'un segment. Retourne (DataFrame time_s/ECG/Resp, table des canaux)."""
    _, signals, channels = load_bdf(bdf_path(task, subj, day, data, version))
    return signals[["time_s", "ECG", "Resp"]].copy(), channels


def load_metadata(subj=SUBJ, day=DAY, data=DATA, version=VERSION):
    return json.loads((Path(data) / version / subj / f"{subj}_day-{day}_metadata.json").read_text(encoding="utf-8"))


def task_windows(task, subj=SUBJ, day=DAY, data=DATA, version=VERSION):
    """Fenêtres des événements (en s, relatives au segment) d'après le metadata.

    Retourne une liste de (début_s, fin_s, instruction_text)."""
    meta = load_metadata(subj, day, data, version)
    seg = meta["dataset_export"]["segments"][task]
    t0, dur = seg["metadata_time_origin_s"], seg["duration_s"]
    out = []
    for ev in meta["events"]:
        on = ev["onset_s"] - t0
        if 0 <= on < dur:
            off = min(on + (ev.get("duration_s") or 0), dur)
            out.append((round(on, 3), round(off, 3), ev.get("instruction_text")))
    return out


# --------------------------------------------------------------------------- filtrage

def bandpass(x, lo, hi, fs=FS, order=2):
    """Filtre passe-bande Butterworth à phase nulle (aller-retour)."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def notch(x, freqs=(50.0, 100.0), fs=FS, q=30.0):
    """Coupe-bande à phase nulle sur le secteur (50 Hz) et son harmonique.

    À appliquer à l'ECG avant r_peaks : sinon, sur un ECG bruité, le pic R peut se caler
    sur une crête du 50 Hz (P001 jour 2 : 4 ms en médiane, 67 % des battements).
    """
    for f0 in freqs:
        b, a = iirnotch(f0, q, fs)
        x = filtfilt(b, a, x)
    return x


# --------------------------------------------------------------------------- respiration

def resp_clean(resp, fs=FS):
    """Respiration filtrée 0,05–1 Hz, signe NON corrigé (voir resp_polarity)."""
    return bandpass(resp, 0.05, 1.0, fs)

def breath_cycles(x, fs=FS):
    """Pics et creux respiratoires d'un signal filtré."""
    prom = 0.6 * np.std(x)
    peaks, _ = find_peaks(x, distance=int(1.5 * fs), prominence=prom)
    troughs, _ = find_peaks(-x, distance=int(1.5 * fs), prominence=prom)
    return peaks, troughs


def breath_rate(x, fs=FS):
    """Fréquence respiratoire (Hz) par la période médiane entre pics.

    Plus robuste que le pic du spectre, que les soupirs attirent vers les basses fréquences.
    """
    peaks, _ = breath_cycles(x, fs)
    return 1 / np.median(np.diff(peaks) / fs)


def resp_polarity(x, fs=FS):
    """Indices du sens du signal respiratoire filtré (le capteur ne le documente pas).

    Les deux indices pointent vers « inspiration vers le bas » (comme sur P001) si :
      - fall_shorter : la phase descendante est plus courte que la montante (inspiration < expiration) ;
      - pause_high   : les pauses (pente faible) sont au-dessus de la moyenne (fin d'expiration en haut).
    Une pause proche de la moyenne (|pause_level_z| < 0,2) suggère un signal de type vitesse et non volume.
    """
    pk, tr = breath_cycles(x, fs)
    rise = np.median([(pk[pk > q][0] - q) / fs for q in tr if (pk > q).any()])
    fall = np.median([(tr[tr > q][0] - q) / fs for q in pk if (tr > q).any()])
    slope = np.abs(np.gradient(x))
    pause = slope < np.percentile(slope, 20)
    level = ((x[pause] - x.mean()) / x.std()).mean()
    return {"rise_s": rise, "fall_s": fall, "fall_shorter": fall < rise,
            "pause_level_z": level, "pause_high": level > 0}


# --------------------------------------------------------------------------- ECG

def ecg_polarity(ecg, fs=FS):
    """+1 si les pics R pointent vers le haut, -1 s'ils pointent vers le bas.

    Retourne (polarité, ratio max/|min| médian). Ratio proche de 1 = ambigu → vérifier à l'œil.
    """
    e = np.abs(bandpass(ecg, 5, 30, fs))
    beats, _ = find_peaks(e, distance=int(0.33 * fs), height=0.4 * np.percentile(e, 99))
    raw = bandpass(ecg, 0.5, 45, fs)
    w = int(0.06 * fs)
    beats = beats[(beats > w) & (beats < len(raw) - w)]
    ratio = np.median([raw[b - w:b + w].max() for b in beats]) / np.median([-raw[b - w:b + w].min() for b in beats])
    return (1 if ratio >= 1 else -1), ratio


def r_peaks(ecg, fs=FS, verbose=True):
    """Indices des pics R (échantillons). Le signal est remis à l'endroit si besoin.

    Passer l'ECG par notch() avant (50 Hz).
    """
    pol, ratio = ecg_polarity(ecg, fs)
    if verbose and 0.8 < ratio < 1.25:
        print(f"⚠️ polarité ECG ambiguë (ratio max/|min| = {ratio:.2f}) : vérifier le battement moyen")
    ecg = pol * ecg
    e = bandpass(ecg, 5, 30, fs)
    pk, _ = find_peaks(e, distance=int(0.33 * fs), height=0.4 * np.percentile(e, 99))
    raw = bandpass(ecg, 0.5, 45, fs)              # ECG brut sans dérive
    w = int(0.03 * fs)
    starts = np.maximum(pk - w, 0)
    return np.array([a + np.argmax(raw[a:p + w]) for a, p in zip(starts, pk)])


def rr_suspects(pk, fs=FS, tol=0.2):
    """Booléen par intervalle RR : True si écart > tol à la médiane locale (9 battements)."""
    rr = np.diff(pk) / fs
    med = pd.Series(rr).rolling(9, center=True, min_periods=1).median().to_numpy()
    return np.abs(rr - med) / med > tol


def hr_series(pk, n_samples, fs=FS, fs_out=FS_HR):
    """HR instantanée (bpm) interpolée sur une grille régulière couvrant le segment."""
    rr = np.diff(pk) / fs
    ok = ~rr_suspects(pk, fs)
    t_beat, hr = pk[1:][ok] / fs, 60 / rr[ok]
    t = np.arange(0, n_samples / fs, 1 / fs_out)
    return t, np.interp(t, t_beat, hr)


def hr_fast(h, lo=0.15, hi=1.0, fs=FS_HR):
    """Garde seulement les oscillations rapides de HR (bande de la RSA), phase nulle."""
    return bandpass(h, lo, hi, fs=fs)


# --------------------------------------------------------------------------- outils temps / spectre

def on_grid(x, t_grid, fs=FS):
    """Valeurs du signal x (échantillonné à fs depuis t=0) aux instants t_grid (s)."""
    return np.interp(t_grid, np.arange(len(x)) / fs, x)


def crop(t, x, on, off):
    """Morceau de x (et de son temps t) compris dans [on, off[ secondes."""
    keep = (t >= on) & (t < off)
    return t[keep], x[keep]


def spectrum(x, fs=FS_HR, nperseg_s=16):
    """Densité spectrale de puissance (Welch), tendance linéaire retirée par segment."""
    n = min(int(nperseg_s * fs), len(x))
    return welch(x, fs=fs, nperseg=n, detrend="linear")


def band_peak(f, p, lo=0.15, hi=0.5):
    """Fréquence et valeur du maximum de p dans la bande [lo, hi] Hz."""
    band = (f >= lo) & (f <= hi)
    i = np.argmax(p[band])
    return f[band][i], p[band][i]


def resp_hr_coherence(resp, hr, fs=FS_HR, nperseg_s=16):
    """Cohérence Resp–HR et phase de l'interspectre ; phase > 0 si HR en retard sur Resp."""
    n = min(int(nperseg_s * fs), len(hr))
    f, coh = coherence(resp, hr, fs=fs, nperseg=n, detrend="linear")
    _, pxy = csd(resp, hr, fs=fs, nperseg=n, detrend="linear")
    return f, coh, -np.angle(pxy)


# --------------------------------------------------------------------------- latence

def latence(x, y, fs, max_lag_s=3, sign=None):
    """Délai (s) qui maximise la corrélation entre x et y. Lag > 0 : y est en retard sur x.

    sign : None = plus grand |corrélation|, +1 = plus grande corrélation positive,
           -1 = plus grande corrélation négative.
    """
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    c = correlate(y, x, mode="full") / len(x)
    lags = correlation_lags(len(y), len(x), mode="full") / fs
    keep = np.abs(lags) <= max_lag_s
    score = np.abs(c[keep]) if sign is None else sign * c[keep]
    best = np.argmax(score)
    return lags[keep][best], c[keep][best]


# --------------------------------------------------------------------------- auto-test

if __name__ == "__main__":
    # Reproduit la sous-étape 1.5 (Resp → HR, fenêtres de repos) sur P001 jour 1.
    fs_lag = 20.0
    for task in ["resting-state", "artefact"]:
        df, _ = load_physio(task, subj="P001", day=1)
        resp = -resp_clean(df.Resp.to_numpy())                 # P001 : inspiration vers le bas
        pk = r_peaks(notch(df.ECG.to_numpy()))
        t_hr, h = hr_series(pk, len(df), fs_out=fs_lag)
        h_f = hr_fast(h, fs=fs_lag)
        print(task, resp_polarity(-resp), "polarité ECG", ecg_polarity(df.ECG.to_numpy()))
        for on, off, txt in task_windows(task, subj="P001", day=1):
            if not txt.startswith("repos"):
                continue
            t_w, hr_w = crop(t_hr, h_f, on, off)
            lag, r = latence(on_grid(resp, t_w), hr_w, fs_lag, max_lag_s=1.5, sign=+1)
            print(f"  {on:6.1f}–{off:6.1f} s : HR en retard de {lag:+.2f} s (r = {r:.2f})")

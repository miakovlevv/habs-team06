"""Fonctions réutilisables pour l'analyse des latences entre signaux physiologiques.

Extraites du notebook de validation de Hugo (latence/hugo_latency_validation.ipynb),
sans changement de logique. Seule différence : le dossier des données se règle
via la variable d'environnement HABS_DATA (sinon : <repo>/sample).

    from physio_utils import r_peaks, task_windows, load_physio, ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import (butter, coherence, correlate, correlation_lags, csd,
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
            off = min(on + ev["duration_s"], dur)
            out.append((round(on, 3), round(off, 3), ev.get("instruction_text")))
    return out


# --------------------------------------------------------------------------- filtrage

def bandpass(x, lo, hi, fs=FS, order=2):
    """Filtre passe-bande Butterworth à phase nulle (aller-retour)."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


# --------------------------------------------------------------------------- respiration

def breath_cycles(x, fs=FS):
    """Pics et creux respiratoires d'un signal filtré."""
    prom = 0.6 * np.std(x)
    peaks, _ = find_peaks(x, distance=int(1.5 * fs), prominence=prom)
    troughs, _ = find_peaks(-x, distance=int(1.5 * fs), prominence=prom)
    return peaks, troughs


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


def r_peaks(ecg, fs=FS):
    """Indices des pics R (échantillons). Le signal est remis à l'endroit si besoin."""
    pol, ratio = ecg_polarity(ecg, fs)
    if 0.8 < ratio < 1.25:
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

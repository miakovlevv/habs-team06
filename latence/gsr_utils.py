"""Peau (GSR) : lecture, lissage, contrôle qualité et détection des réponses (SCR).

Fonctions reprises SANS CHANGEMENT DE LOGIQUE du notebook de Hugo
(branche hugo/brain-skin, latence/hugo_brain_skin.ipynb, section 3.1–3.2c), pour pouvoir les
réutiliser ailleurs. Seule différence : la version des données est un paramètre (Hugo utilise
PREPROCESSED ; l'analyse HEP utilise CURATED). Si Hugo modifie sa détection, mettre ce fichier à jour.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starting_notebooks"))
from bdf_loader import uV_to_uS  # noqa: E402
from physio_utils import DATA, FS, VERSION, bdf_path  # noqa: E402

FG = 20.0          # Hz, grille commune pour la GSR
FLAT_RANGE = 0.05  # µS : en dessous, capteur plat (aucune SCR possible) — critère de Hugo


def read_gsr(subj, task="emotion-video", day=1, data=DATA, version=VERSION):
    """Voie GSR seule, en µS, à FS."""
    with pyedflib.EdfReader(str(bdf_path(task, subj, day, data=data, version=version))) as r:
        return uV_to_uS(r.readSignal(r.getSignalLabels().index("GSR")))


def smooth_gsr(g, fs=FS, fs_out=FG):
    """Passe-bas 1 Hz (phase nulle), puis sous-échantillonnage à fs_out."""
    return sosfiltfilt(butter(2, 1.0, fs=fs, output="sos"), g)[::int(fs / fs_out)]


def gsr_qc(g):
    """Étendue de la GSR lissée (µS) et % du temps bloqué au plancher de 1 µS."""
    return {"range_uS": np.percentile(g, 99) - np.percentile(g, 1), "median_uS": np.median(g),
            "floor_pct": 100 * np.mean(g <= 1.001)}


def scr_detect(g, fs=FG, min_amp=0.02, max_rise=5.0):
    """SCR sous forme de DataFrame (onset_s, peak_s, amp_uS, artefact).

    Pic = maximum local de proéminence >= min_amp ; début = point le plus bas dans les max_rise s
    précédentes (pas avant le pic précédent) ; artefact = pointe : montée < 1 s et plus de la moitié
    de l'amplitude perdue en 1 s. Débuts sur le plancher de 1 µS écartés (vrai début masqué)."""
    pk, _ = find_peaks(g, prominence=min_amp)
    rows, prev = [], 0
    for b in pk:
        lo = max(prev, b - int(max_rise * fs))
        a = lo + np.argmin(g[lo:b + 1])
        prev = b
        amp = g[b] - g[a]
        if amp < min_amp or a == b or g[a] <= 1.001:
            continue
        drop = g[b] - g[min(b + int(fs), len(g) - 1)]
        rows.append((a / fs, b / fs, amp, (b - a) / fs < 1.0 and drop > 0.5 * amp))
    return pd.DataFrame(rows, columns=["onset_s", "peak_s", "amp_uS", "artefact"])


def scr_driver(g, fs=FG):
    """Signal phasique : partie positive de la pente de la GSR (µS/s)."""
    return np.clip(np.gradient(g) * fs, 0, None)


def scr_per_video(scr, onsets, lo=1.0, hi=11.0):
    """Pour chaque début de vidéo : y a-t-il une SCR (hors artefacts) qui démarre entre +lo et +hi s ?
    Et l'amplitude de la plus grande (0 sinon). Fenêtre 1–11 s = « scr_any » de Hugo (3.2c)."""
    s = scr[~scr.artefact]
    out = []
    for v in onsets:
        w = s[(s.onset_s >= v + lo) & (s.onset_s < v + hi)]
        out.append((len(w) > 0, w.amp_uS.max() if len(w) else 0.0))
    return out

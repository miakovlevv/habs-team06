"""Étape 2 — HEP (heartbeat-evoked potential) : réponse EEG à chaque battement cardiaque.

Pipeline, pour un participant et un jour :
    seg = load_video_session(subj, day)          # EEG (MNE Raw filtré), ECG, pics R, vidéos + notes
    ep  = hep_epochs(seg)                        # époques EEG autour des pics R, dans les vidéos
puis, sur plusieurs jours concaténés :
    ep  = add_arousal_split(ep)                  # tiers bas / haut d'activation, par participant
    hep_metrics(...), permutation_test(...)      # amplitude et latence, haute vs basse

Contrôles prévus (artefact du champ cardiaque, CFA) :
    - battement moyen de l'ECG par condition  (ecg_epochs)
    - fréquence cardiaque par condition        (colonne rr_s des métadonnées d'époques)
    - faux battements tirés au hasard          (surrogate_peaks)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from physio_utils import (DATA, FS, VERSION, bandpass, bdf_path, ecg_polarity,
                          load_bdf, load_metadata, notch, r_peaks)

mne.set_log_level("WARNING")

# Groupe fronto-central : là où le HEP est classiquement le plus visible (pas de Cz dans ce montage).
FC_CLUSTER = ["Fz", "F1", "F2", "FCz", "FC1", "FC2", "C1", "C2"]

TMIN, TMAX = -0.2, 0.8        # s autour du pic R
BASELINE = (-0.2, -0.05)      # avant l'onde R (évite le QRS)
HEP_WIN = (0.25, 0.45)        # fenêtre d'amplitude moyenne
LAT_WIN = (0.20, 0.60)        # fenêtre de recherche pour la latence
MIN_NEXT_RR = 0.6             # s : on exige que le battement suivant arrive après +600 ms
REJECT_UV = 150.0             # amplitude crête-à-crête max par époque (µV)
VIDEO_MARGIN = 0.0            # s ignorées au début de chaque vidéo (réponse d'apparition)


# --------------------------------------------------------------------------- chargement

def load_labels(data=DATA):
    """labels_train.csv avec les notes en nombres (« TESTSUBJECT » = sujet de test → NaN)."""
    lab = pd.read_csv(Path(data) / "labels_train.csv")
    for c in ["rating_arousal", "rating_valence", "flagged_outliers"]:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    return lab


def video_trials(subj, day, data=DATA, version=VERSION, labels=None):
    """Vidéos du segment emotion-video : début/fin (s, relatifs au segment), fichier, notes.

    Notes : on prend `rating_arousal` de labels_train.csv, relié par
    marker_id (metadata) = start_marker (labels). Vidéos absentes de labels_train → NaN.

    Attention : le metadata.json contient aussi des notes (`arousal_score`, identiques à
    rating_arousal quand les deux existent — vérifié sur P001), y compris pour des vidéos
    absentes de labels_train (28/108 pour P001), probablement réservées à l'évaluation.
    On ne les utilise pas (colonne `arousal_meta`, gardée pour contrôle seulement)."""
    meta = load_metadata(subj, day, data, version)
    seg = meta["dataset_export"]["segments"]["emotion-video"]
    t0, dur = seg["metadata_time_origin_s"], seg["duration_s"]
    rows = []
    for ev in meta["events"]:
        on = ev["onset_s"] - t0
        if ev.get("trial_type") == "emotion_video" and 0 <= on < dur:
            rows.append({"subj": subj, "day": day, "marker": ev["marker_id"], "on_s": on,
                         "off_s": min(on + ev["duration_s"], dur),
                         "video": ev.get("media_file"),
                         "arousal_meta": ev.get("arousal_score")})
    df = pd.DataFrame(rows)
    df["trial"] = np.arange(len(df))

    labels = load_labels(data) if labels is None else labels
    lab = labels[(labels.participant_id == subj) & (labels.day == day)]
    lab = lab[["start_marker", "video", "rating_arousal", "rating_valence", "flagged_outliers"]]
    df = df.merge(lab.rename(columns={"video": "video_label"}), how="left",
                  left_on="marker", right_on="start_marker").drop(columns="start_marker")
    bad = df.video_label.notna() & (df.video_label != df.video)
    if bad.any():
        raise ValueError(f"{subj} jour {day} : vidéo différente entre metadata et labels pour {bad.sum()} essais")
    return df.drop(columns="video_label").rename(columns={"rating_arousal": "arousal",
                                                          "rating_valence": "valence"})


def load_video_session(subj, day, data=DATA, version=VERSION,
                       l_freq=0.5, h_freq=30.0, ica=False, verbose=True):
    """Charge EEG + ECG de la tâche vidéo, détecte les pics R, nettoie l'EEG.

    Étapes EEG : filtre [l_freq, h_freq] → électrodes plates/bruitées → électrodes qui
    captent le pouls (pulse_channels) → interpolation → (option) ICA cardiaque → réf. moyenne.

    Retourne un dict : raw (MNE, EEG seul), ecg (filtré, à l'endroit), rpk (pics R),
    trials (vidéos + notes), bads (dict raison -> électrodes), ica_excluded."""
    _, sig, ch = load_bdf(bdf_path("emotion-video", subj, day, data, version))
    eeg_names = ch.loc[ch.modality == "EEG", "bdf_label"].tolist()

    # Secteur 50 Hz retiré AVANT r_peaks : le passe-bas 45 Hz d'ordre 2 de r_peaks l'atténue
    # peu, et sur un ECG bruité (P001 jour 2) le maximum cherché à ±30 ms se cale sur une
    # crête du 50 Hz au lieu du sommet du R (décalage de quelques ms, battement moyen « ondulé »).
    ecg_raw = notch(sig["ECG"].to_numpy())
    pol, _ = ecg_polarity(ecg_raw)
    rpk = r_peaks(ecg_raw)
    ecg = pol * bandpass(ecg_raw, 0.5, 45)

    x = sig[eeg_names].to_numpy().T * 1e-6                # µV -> V pour MNE
    del sig
    raw = mne.io.RawArray(x, mne.create_info(eeg_names, FS, "eeg"))
    raw.set_montage(_montage(), on_missing="warn")
    raw.filter(l_freq, h_freq, phase="zero")

    bads = {"plates/bruitées": flat_or_noisy(raw)}
    bads["pouls"] = pulse_channels(raw, rpk, exclude=bads["plates/bruitées"])
    all_bads = sum(bads.values(), [])
    raw.info["bads"] = all_bads
    if all_bads:   # reconstruites à partir des voisines → mêmes électrodes pour tous les jours/sujets
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)

    excluded = []
    if ica:
        excluded = remove_cardiac_ica(raw, ecg)

    trials = video_trials(subj, day, data, version)
    if verbose:
        hr = 60 / np.median(np.diff(rpk) / FS)
        print(f"{subj} jour {day} : {len(trials)} vidéos, {len(rpk)} battements "
              f"(HR médiane {hr:.0f} bpm) | interpolées : {bads}"
              + (f" | ICA : composantes cardiaques retirées {excluded}" if ica else ""))
    return {"subj": subj, "day": day, "raw": raw, "ecg": ecg, "rpk": rpk,
            "trials": trials, "bads": bads, "ica_excluded": excluded}


def pulse_channels(raw, rpk, exclude=(), z=5.0, win=(0.0, 0.6)):
    """Électrodes dont la réponse moyenne au battement est anormalement forte.

    Une électrode posée près d'une artère enregistre l'onde de pouls (~200–400 ms après R),
    souvent bien plus grande que le vrai HEP : elle domine alors toute moyenne de groupe.
    On compare, entre électrodes, la RMS du potentiel moyen (réf. moyenne provisoire)."""
    tmp = raw.copy().pick([c for c in raw.ch_names if c not in exclude])
    tmp.set_eeg_reference("average", projection=False)
    samp = rpk[(rpk + int(TMIN * FS) > 0) & (rpk + int(TMAX * FS) < tmp.n_times)]
    ev = mne.Epochs(tmp, np.c_[samp, 0 * samp, 1 + 0 * samp], tmin=TMIN, tmax=TMAX,
                    baseline=BASELINE, reject=dict(eeg=REJECT_UV * 1e-6), preload=True).average()
    m = (ev.times >= win[0]) & (ev.times <= win[1])
    rms = np.sqrt((ev.data[:, m] ** 2).mean(1))
    med, mad = np.median(rms), np.median(np.abs(rms - np.median(rms))) * 1.4826
    return [ev.ch_names[i] for i in np.where((rms - med) / mad > z)[0]]


def remove_cardiac_ica(raw, ecg, n_components=30, seed=97):
    """ICA : retire les composantes corrélées à l'ECG (artefact du champ cardiaque).

    Ajuste l'ICA sur une copie filtrée à 1 Hz et sous-échantillonnée (plus rapide, plus
    stable), puis applique la correction au signal complet. Modifie `raw` sur place."""
    raw.add_channels([mne.io.RawArray(ecg[None] * 1e-6, mne.create_info(["ECG"], FS, "ecg"))],
                     force_update_info=True)
    fit = raw.copy().filter(1.0, None).resample(250)
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=seed, max_iter="auto")
    ica.fit(fit, picks="eeg")
    idx, _ = ica.find_bads_ecg(raw, ch_name="ECG", method="ctps")
    ica.exclude = idx
    ica.apply(raw)
    raw.drop_channels(["ECG"])
    return idx


def _montage():
    for name in ("colin27_1005", "standard_1005"):   # nom selon la version de MNE
        try:
            return mne.channels.make_standard_montage(name)
        except ValueError:
            continue


def flat_or_noisy(raw, z=4.0):
    """Électrodes plates ou très bruitées (écart-type robuste hors normes)."""
    sd = raw.get_data().std(axis=1)
    med, mad = np.median(sd), np.median(np.abs(sd - np.median(sd))) * 1.4826
    bad = (sd < 1e-7) | (np.abs(sd - med) / mad > z)
    return [raw.ch_names[i] for i in np.where(bad)[0]]


# --------------------------------------------------------------------------- époques

def beats_in_videos(rpk, trials, n_samples, fs=FS, margin=VIDEO_MARGIN):
    """Pics R utilisables à l'intérieur des vidéos.

    Garde un battement si : l'époque [TMIN, TMAX] tient dans la vidéo (après `margin`),
    et le battement suivant arrive au moins MIN_NEXT_RR s plus tard.
    Retourne (échantillons, DataFrame métadonnées par battement)."""
    nxt = np.r_[np.diff(rpk), np.inf] / fs
    prv = np.r_[np.inf, np.diff(rpk)] / fs
    t = rpk / fs
    rows, keep = [], []
    for tr in trials.itertuples():
        inside = (t + TMIN >= tr.on_s + margin) & (t + TMAX <= tr.off_s) & (nxt >= MIN_NEXT_RR)
        inside &= (rpk + int(TMIN * fs) >= 0) & (rpk + int(TMAX * fs) < n_samples)
        for i in np.where(inside)[0]:
            keep.append(rpk[i])
            rows.append({"subj": tr.subj, "day": tr.day, "trial": tr.trial, "video": tr.video,
                         "arousal": tr.arousal, "t_s": t[i], "t_in_video_s": t[i] - tr.on_s,
                         "rr_s": prv[i] if np.isfinite(prv[i]) else np.nan, "next_rr_s": nxt[i]})
    return np.array(keep, dtype=int), pd.DataFrame(rows)


def hep_epochs(seg, peaks=None, reject_uv=REJECT_UV):
    """Époques EEG autour des pics R (ou de `peaks`, p. ex. faux battements) dans les vidéos."""
    raw = seg["raw"]
    peaks = seg["rpk"] if peaks is None else peaks
    samp, meta = beats_in_videos(peaks, seg["trials"], raw.n_times)
    events = np.c_[samp, np.zeros_like(samp), np.ones_like(samp)]
    ep = mne.Epochs(raw, events, {"R": 1}, tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                    metadata=meta, reject=dict(eeg=reject_uv * 1e-6), preload=True)
    return ep


def ecg_epochs(seg, epochs):
    """ECG découpé autour des mêmes battements que `epochs` (contrôle du champ cardiaque)."""
    samp = epochs.events[:, 0]
    a, b = int(TMIN * FS), int(TMAX * FS)
    x = np.array([seg["ecg"][s + a:s + b + 1] for s in samp])
    bl = (np.arange(a, b + 1) / FS >= BASELINE[0]) & (np.arange(a, b + 1) / FS <= BASELINE[1])
    return x - x[:, bl].mean(1, keepdims=True)


def surrogate_peaks(seg, seed=0):
    """Faux battements : autant d'instants que de vrais battements dans chaque vidéo,
    mais tirés au hasard (uniformément) dans la vidéo, sans lien avec le pic R.

    Si le « HEP » obtenu avec ces faux battements ressemble au vrai, ce qu'on mesure
    n'est pas lié au cœur (dérive lente, réponse à la vidéo…)."""
    rng = np.random.default_rng(seed)
    rpk = seg["rpk"]
    out = []
    for tr in seg["trials"].itertuples():
        a, b = int(tr.on_s * FS), int(tr.off_s * FS)
        n = int(((rpk >= a) & (rpk < b)).sum())
        out.append(np.sort(rng.integers(a, b, n)))
    return np.sort(np.concatenate(out)) if out else np.array([], int)


# --------------------------------------------------------------------------- activation haute / basse

def tertile_cuts(scores, frac=1 / 3):
    """Seuils (bas, haut) sur une échelle discrète, pour des groupes aussi proches que
    possible d'un tiers chacun : « basse » = note ≤ bas, « haute » = note ≥ haut, bas < haut.

    Avec des ex æquo, un quantile simple peut déborder (ex. P001 : q33 = 5 mettait 53 % des
    vidéos en « basse »). Ici on choisit, parmi les notes possibles, le seuil dont la
    proportion est la plus proche de `frac`. Retourne (nan, nan) si on ne peut pas séparer."""
    s = np.asarray(scores, float)
    s = s[~np.isnan(s)]
    vals = np.unique(s)
    if len(vals) < 2:
        return np.nan, np.nan
    p_lo = np.array([(s <= v).mean() for v in vals])
    p_hi = np.array([(s >= v).mean() for v in vals])
    lo = vals[np.argmin(np.abs(p_lo - frac))]
    hi = vals[np.argmin(np.abs(p_hi - frac))]
    if lo >= hi:   # échelle très peu utilisée : on garde au moins une note d'écart
        hi = vals[vals > lo].min() if (vals > lo).any() else np.nan
    return lo, hi


def add_arousal_split(meta, by="subj"):
    """Ajoute la colonne `cond` (« basse » / « haute » / « moyenne » / « sans note »).

    Version provisoire (en attendant la définition de Micha) : tiers bas et tiers haut des
    notes de CHAQUE participant (tertile_cuts). Les seuils sont calculés sur les vidéos, pas
    sur les battements, pour qu'une vidéo avec plus de battements ne pèse pas plus."""
    meta = meta.copy()
    vids = meta.drop_duplicates(["subj", "day", "trial"])
    cond = pd.Series("moyenne", index=meta.index)
    cuts = {}
    for s, v in vids.groupby(by):
        lo, hi = tertile_cuts(v.arousal)
        cuts[s] = (lo, hi)
        m = meta[by] == s
        cond[m & (meta.arousal <= lo)] = "basse"
        cond[m & (meta.arousal >= hi)] = "haute"
    meta["cond"] = cond.where(meta.arousal.notna(), "sans note")
    meta.attrs["cuts"] = cuts
    return meta


# --------------------------------------------------------------------------- mesures

def cluster_wave(epochs, picks=FC_CLUSTER):
    """Moyenne sur le groupe d'électrodes, par époque : (n_époques, n_temps) en µV."""
    picks = [p for p in picks if p in epochs.ch_names and p not in epochs.info["bads"]]
    return epochs.get_data(picks=picks).mean(1) * 1e6


def mean_amp(wave, times, win=HEP_WIN):
    m = (times >= win[0]) & (times <= win[1])
    return wave[..., m].mean(-1)


def frac_area_latency(wave, times, win=LAT_WIN, frac=0.5, polarity=None):
    """Latence à laquelle on atteint `frac` de l'aire (signée) de la courbe moyenne dans `win`.

    Plus robuste au bruit que la latence du pic. `polarity` : +1 (aire positive),
    -1 (aire négative) ; None = celle qui domine dans la fenêtre."""
    m = (times >= win[0]) & (times <= win[1])
    w, t = wave[m], times[m]
    if polarity is None:
        polarity = 1 if w.sum() >= 0 else -1
    a = np.clip(polarity * w, 0, None)
    if a.sum() == 0:
        return np.nan
    c = np.cumsum(a) / a.sum()
    return t[np.searchsorted(c, frac)]


def peak_latency(wave, times, win=LAT_WIN, polarity=None):
    m = (times >= win[0]) & (times <= win[1])
    w, t = wave[m], times[m]
    if polarity is None:
        polarity = 1 if w.sum() >= 0 else -1
    return t[np.argmax(polarity * w)]


def hep_metrics(waves, times, polarity=None):
    """Amplitude moyenne, latence (aire fractionnelle 50 %) et latence du pic d'une courbe moyenne."""
    w = waves.mean(0)
    return {"n": len(waves), "amp_uV": mean_amp(w, times),
            "lat_50area_s": frac_area_latency(w, times, polarity=polarity),
            "lat_pic_s": peak_latency(w, times, polarity=polarity)}


def jackknife_latency_diff(wa, wb, times, polarity=None):
    """Différence de latence (b - a) et son erreur-type par jackknife (Miller et al., 1998)."""
    def lat(w): return frac_area_latency(w, times, polarity=polarity)
    d_full = lat(wb.mean(0)) - lat(wa.mean(0))
    n = min(len(wa), len(wb), 60)
    ia = np.array_split(np.arange(len(wa)), n)
    ib = np.array_split(np.arange(len(wb)), n)
    d = np.array([lat(np.delete(wb, ib[k], 0).mean(0)) - lat(np.delete(wa, ia[k], 0).mean(0))
                  for k in range(n)])
    se = np.sqrt((n - 1) / n * np.sum((d - d.mean()) ** 2))
    return d_full, se


def permutation_test(waves, groups, stat, n_perm=2000, seed=0, block=None):
    """Test de permutation de stat(haute) - stat(basse).

    `groups` : tableau de 'basse'/'haute' par époque. `block` : identifiant de vidéo par
    époque ; si fourni, on permute les étiquettes par VIDÉO (les battements d'une même vidéo
    ne sont pas indépendants). Retourne (différence observée, p bilatéral)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)

    def diff(g):
        return stat(waves[g == "haute"]) - stat(waves[g == "basse"])

    obs = diff(groups)
    if block is None:
        null = np.array([diff(rng.permutation(groups)) for _ in range(n_perm)])
    else:
        block = np.asarray(block)
        ub = pd.unique(block)
        lab = np.array([groups[block == b][0] for b in ub])
        idx = [np.where(block == b)[0] for b in ub]
        null = []
        for _ in range(n_perm):
            pl = rng.permutation(lab)
            g = np.empty_like(groups)
            for b, l in zip(idx, pl):
                g[b] = l
            null.append(diff(g))
        null = np.array(null)
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1)
    return obs, p

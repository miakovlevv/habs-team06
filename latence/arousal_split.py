"""Définition de l'activation « haute » et « basse » à partir des notes post-vidéo (1–9).

Analyse et justification : latence/arousal_definition.ipynb.

Définition retenue (DEFAULT_METHOD = "tiers") : pour chaque participant, son tiers de vidéos
les moins activantes = « basse », son tiers le plus activant = « haute », le reste = « moyenne ».
Un participant est « non discriminant » (exclu) s'il utilise trop peu l'échelle (écart-type de
ses notes < MIN_SD) ou s'il a moins de MIN_VIDEOS vidéos dans un des deux groupes.

    from arousal_split import load_ratings, split_all
    lab = split_all(load_ratings())        # ajoute les colonnes cond, seuil_basse, seuil_haute
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("HABS_DATA", Path(__file__).resolve().parent.parent / "sample"))
DEFAULT_METHOD = "tiers"
MIN_SD = 1.0          # écart-type minimal des notes d'un participant
MIN_VIDEOS = 8        # vidéos minimum par groupe (basse et haute)
METHODS = ["seuils fixes 3/7", "tiers", "z ±0,5", "médiane"]


def load_ratings(data=DATA):
    """labels_train.csv, notes converties en nombres ; ne garde que les vidéos notées.

    « TESTSUBJECT » (participants de test) → NaN, donc retirés."""
    lab = pd.read_csv(Path(data) / "labels_train.csv", na_values=["TESTSUBJECT"])
    for c in ["rating_arousal", "rating_valence", "flagged_outliers", "day"]:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    return lab[lab.rating_arousal.notna()].copy()


def tertile_cuts(scores, frac=1 / 3):
    """Seuils (bas, haut) : « basse » = note ≤ bas, « haute » = note ≥ haut, bas < haut.

    Avec des ex æquo (échelle entière de la cohorte 1), un quantile simple peut déborder
    (ex. P001 : q33 = 5 mettait 53 % des vidéos en « basse »). On choisit donc, parmi les
    notes existantes, le seuil dont la proportion est la plus proche de `frac`.
    Marche aussi pour les notes continues (curseur, cohortes 2 à 5)."""
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


def split_subject(x, method=DEFAULT_METHOD):
    """Conditions (« basse » / « moyenne » / « haute ») pour les notes x d'UN participant.

    Retourne (tableau de conditions, seuil bas, seuil haut)."""
    x = np.asarray(x, float)
    if method == "seuils fixes 3/7":
        lo, hi = 3.0, 7.0
    elif method == "tiers":
        lo, hi = tertile_cuts(x)
    elif method == "z ±0,5":
        lo, hi = x.mean() - 0.5 * x.std(ddof=1), x.mean() + 0.5 * x.std(ddof=1)
    elif method == "médiane":
        m = np.median(x)
        return np.where(x < m, "basse", np.where(x > m, "haute", "moyenne")), m, m
    else:
        raise ValueError(f"méthode inconnue : {method}")
    return np.where(x <= lo, "basse", np.where(x >= hi, "haute", "moyenne")), lo, hi


def split_all(lab, method=DEFAULT_METHOD, by="participant_id", min_sd=MIN_SD, min_videos=MIN_VIDEOS):
    """Ajoute cond, seuil_basse, seuil_haute et discriminant (bool) à chaque vidéo notée.

    Les seuils sont calculés sur toutes les vidéos notées du participant (tous jours confondus)."""
    lab = lab.copy()
    lab["cond"], lab["seuil_basse"], lab["seuil_haute"] = "moyenne", np.nan, np.nan
    for _, idx in lab.groupby(by).groups.items():
        x = lab.loc[idx, "rating_arousal"].to_numpy()
        cond, lo, hi = split_subject(x, method)
        lab.loc[idx, "cond"], lab.loc[idx, "seuil_basse"], lab.loc[idx, "seuil_haute"] = cond, lo, hi
    n = lab.groupby(by).cond.value_counts().unstack(fill_value=0)
    sd = lab.groupby(by).rating_arousal.std()
    ok = (sd >= min_sd) & (n.get("basse", 0) >= min_videos) & (n.get("haute", 0) >= min_videos)
    lab["discriminant"] = lab[by].map(ok).fillna(False)
    return lab


def scale_usage(lab, by="participant_id"):
    """Comment chaque participant utilise l'échelle : une ligne par participant."""
    g = lab.groupby(by).rating_arousal
    return pd.DataFrame({
        "cohorte": lab.groupby(by).cohort.first(),
        "n_videos": g.size(), "moyenne": g.mean(), "ecart_type": g.std(),
        "etendue": g.max() - g.min(),
        "part_note_la_plus_frequente": g.apply(lambda x: x.round().value_counts(normalize=True).max()),
        "notes_entieres": g.apply(lambda x: (x % 1 == 0).mean()),
    })


def video_norm(lab, by="participant_id"):
    """Activation « de référence » de chaque vidéo pour chaque essai : moyenne des notes des
    AUTRES participants pour la même vidéo (on retire le participant lui-même)."""
    s = lab.groupby("video").rating_arousal.agg(["sum", "count"])
    own = lab.groupby([by, "video"]).rating_arousal.agg(["sum", "count"])
    tot = s.loc[lab.video].to_numpy()
    mine = own.loc[list(zip(lab[by], lab.video))].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series((tot[:, 0] - mine[:, 0]) / (tot[:, 1] - mine[:, 1]), index=lab.index)


def evaluate(lab, methods=METHODS, by="participant_id"):
    """Compare les méthodes : participants utilisables, taille des groupes, contraste des notes,
    écart d'activation de référence des vidéos, cohérence sur les vidéos vues deux fois."""
    norm = video_norm(lab, by)
    rows, per_subject = [], {}
    for m in methods:
        d = split_all(lab, m, by).assign(norm=norm)
        hb = d[d.cond.isin(["basse", "haute"])]
        per = hb.groupby([by, "cond"]).agg(n=("rating_arousal", "size"), note=("rating_arousal", "mean"),
                                           norme=("norm", "mean")).unstack()
        per = per[d.groupby(by).discriminant.first()]
        rep = hb[hb.duplicated([by, "video"], keep=False)]
        cons = rep.groupby([by, "video"]).cond.nunique().eq(1).mean() if len(rep) else np.nan
        rows.append({"méthode": m, "participants utilisables": len(per),
                     "vidéos par groupe (médiane)": np.median(per["n"].to_numpy()),
                     "écart de notes haute − basse": (per["note"]["haute"] - per["note"]["basse"]).median(),
                     "écart d'activation de référence": (per["norme"]["haute"] - per["norme"]["basse"]).median(),
                     "cohérence des répétitions": cons})
        per_subject[m] = per
    return pd.DataFrame(rows), per_subject

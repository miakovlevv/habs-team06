"""
Calibration des notes d'arousal entre participants (design creux, notes continues).

IDÉE
----
Chaque participant p transforme son arousal ressenti en note par une relation affine
qui lui est propre (un décalage et un étalement) :

    note_pv = m + delta_p + s * exp(gamma_p) * (u_v + e_pv)

    u_v     : effet "vidéo" partagé entre participants, u_v ~ N(0, 1)   (fixe l'échelle)
    e_pv    : écart individuel autour de l'effet vidéo, e ~ N(0, sigma_e^2)
    delta_p : décalage du participant (>0 : il note plus haut que le participant moyen)
    gamma_p : étalement du participant (>0 : il utilise une plage plus large)

Les paramètres (m, s, sigma_e, delta_p, gamma_p) sont estimés par maximum de vraisemblance
pénalisée. Les u_v sont INTÉGRÉS analytiquement (formule fermée), ce qui évite la
dégénérescence classique d'un ajustement conjoint quand beaucoup de vidéos ont peu d'observateurs.

SORTIE : la note recalée "à la manière du participant moyen" :

    a_cal = m + (note - m - delta_p) * exp(-gamma_p)

C'est une valeur exprimée dans l'unité des notes (ex. échelle 1-9), comparable entre
participants. On la découpe ensuite en classes avec des seuils COMMUNS.

HYPOTHÈSE D'IDENTIFICATION : ce sont les vidéos vues par plusieurs participants qui
permettent de séparer "biais du participant" et "intensité de la vidéo". Par défaut,
seules les vidéos avec >= 2 observateurs servent à l'estimation (min_raters=2) ; les
paramètres obtenus sont ensuite appliqués à TOUTES les notes. Sans cela, une vidéo vue par
un seul participant reviendrait à supposer qu'il a vu un échantillon "typique" de vidéos.

Dépendances : numpy, pandas, scipy
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
def matrix_to_long(matrix: pd.DataFrame) -> pd.DataFrame:
    """Matrice participants x vidéos (CSV) -> format long ['pid', 'video', 'rating'].
    Les cellules vides (vidéo non vue) sont supprimées."""
    long = matrix.copy().rename_axis("pid").reset_index().melt(
        id_vars="pid", var_name="video", value_name="rating")
    return long.dropna(subset=["rating"]).reset_index(drop=True)


def load_matrix_csv(path: str) -> pd.DataFrame:
    """Lit le CSV produit par build_ratings_csv.py et le renvoie au format long."""
    return matrix_to_long(pd.read_csv(path, index_col=0))


def to_long(data) -> pd.DataFrame:
    """
    Accepte n'importe quel format d'entrée et renvoie le format long ['pid', 'video', 'rating'] :
      - chemin vers le CSV (participants en lignes, vidéos en colonnes)
      - DataFrame "large" : participants en index (ou dans une colonne 'participant'/'pid'),
        vidéos en colonnes, notes dans les cellules (vides = vidéo non vue)
      - DataFrame "long" avec les colonnes pid, video, rating
    """
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Attendu : chemin CSV ou DataFrame (large ou long).")
    if {"pid", "video", "rating"}.issubset(data.columns):
        return data[["pid", "video", "rating"]].dropna(subset=["rating"]).reset_index(drop=True)
    for col in ("participant", "pid", "participant_id"):      # CSV lu sans index_col=0
        if col in data.columns:
            data = data.set_index(col)
            break
    return matrix_to_long(data.apply(pd.to_numeric, errors="coerce"))


# --------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------
@dataclass
class Hyper:
    sigma_delta: float = 1.5   # écart-type a priori des décalages (en points de note)
    sigma_gamma: float = 0.5   # écart-type a priori des log-étalements (0.5 ~ facteur x1.65)
    min_raters: int = 2        # une vidéo doit avoir >= min_raters observateurs pour servir à l'ajustement


@dataclass
class Calibration:
    labels: pd.DataFrame        # une ligne par (pid, video) : rating, a_cal, censored, n_raters, in_fit
    participants: pd.DataFrame  # une ligne par participant : offset, spread, notes recalées de référence
    videos: pd.DataFrame        # une ligne par vidéo : arousal consensus (échelle des notes), n_raters
    m: float                    # note moyenne du participant de référence
    s: float                    # écart-type (points de note) de l'effet vidéo pour le participant de référence
    sigma_e: float              # écart individuel, relatif à l'écart-type de l'effet vidéo
    hyper: Hyper = field(default_factory=Hyper)
    converged: bool = True
    nll: float = float("nan")

    @property
    def icc_video(self) -> float:
        """Part de variance (en arousal recalé) expliquée par l'effet vidéo partagé."""
        return 1.0 / (1.0 + self.sigma_e ** 2)


def _nll_and_grad(theta, pi, vi, r, n_v, P, hyper: Hyper):
    """Log-vraisemblance négative marginale (u_v intégrés) + prior, et son gradient exact."""
    m, log_s, log_se = theta[0], theta[1], theta[2]
    delta, gamma = theta[3:3 + P], theta[3 + P:3 + 2 * P]
    V, N = len(n_v), len(r)
    se2 = np.exp(2 * log_se)

    log_g = log_s + gamma[pi]
    g = np.exp(log_g)
    z = (r - m - delta[pi]) / g                      # note ramenée à l'échelle latente
    S1 = np.bincount(vi, z, V)
    S2 = np.bincount(vi, z * z, V)

    F = 0.5 * np.log1p(n_v / se2) + (S2 - S1 ** 2 / (se2 + n_v)) / (2 * se2)
    nll = (log_g.sum() + N * log_se + F.sum()
           + 0.5 * (delta ** 2).sum() / hyper.sigma_delta ** 2
           + 0.5 * (gamma ** 2).sum() / hyper.sigma_gamma ** 2)

    w = (z - (S1 / (se2 + n_v))[vi]) / se2           # dF/dz_i
    dz_dloc = -1.0 / g                                # dz/dm = dz/ddelta_p
    d_logg = 1.0 - w * z                              # d(nll)/d(log g_i)
    dF_dse2 = (-n_v / (2 * se2 * (se2 + n_v)) + S1 ** 2 / (2 * se2 * (se2 + n_v) ** 2)
               - (S2 - S1 ** 2 / (se2 + n_v)) / (2 * se2 ** 2))

    grad = np.empty_like(theta)
    grad[0] = (w * dz_dloc).sum()
    grad[1] = d_logg.sum()
    grad[2] = N + (2 * se2 * dF_dse2).sum()
    grad[3:3 + P] = np.bincount(pi, w * dz_dloc, P) + delta / hyper.sigma_delta ** 2
    grad[3 + P:] = np.bincount(pi, d_logg, P) + gamma / hyper.sigma_gamma ** 2
    return nll, grad


def fit_calibration(ratings: pd.DataFrame, hyper: Hyper | None = None,
                    rating_bounds: tuple[float, float] | None = (1.0, 9.0)) -> Calibration:
    """
    ratings       : chemin CSV, matrice participants x vidéos, ou DataFrame long
                    ['pid', 'video', 'rating'] (voir to_long) ; notes continues acceptées
    rating_bounds : bornes de l'échelle, uniquement pour marquer les notes "plafonnées"
                    (colonne 'censored'). Une note à la borne est un minimum/maximum de
                    l'intensité, pas une valeur exacte. None = pas de marquage.
    """
    hyper = hyper or Hyper()
    d = to_long(ratings).copy()
    d["rating"] = d["rating"].astype(float)
    if d.duplicated(["pid", "video"]).any():
        warnings.warn("Couples (pid, video) en double : moyenne des notes utilisée")
        d = d.groupby(["pid", "video"], as_index=False)["rating"].mean()

    d["n_raters"] = d.groupby("video")["pid"].transform("nunique")
    d["in_fit"] = d["n_raters"] >= hyper.min_raters
    fit_d = d[d["in_fit"]]
    if len(fit_d) < 10:
        raise ValueError("Pas assez de vidéos partagées entre participants pour calibrer.")

    pids = sorted(d["pid"].unique(), key=str)
    vids_fit = sorted(fit_d["video"].unique(), key=str)
    pmap = {p: i for i, p in enumerate(pids)}
    vmap = {v: i for i, v in enumerate(vids_fit)}
    P, V = len(pids), len(vids_fit)
    pi = fit_d["pid"].map(pmap).to_numpy()
    vi = fit_d["video"].map(vmap).to_numpy()
    r = fit_d["rating"].to_numpy()
    n_v = np.bincount(vi, minlength=V).astype(float)

    theta0 = np.zeros(3 + 2 * P)
    theta0[:3] = [r.mean(), np.log(r.std() / 1.2 + 1e-6), np.log(0.7)]
    bounds = [(None, None), (-8, 8), (-4, 3)] + [(None, None)] * (2 * P)
    res = minimize(_nll_and_grad, theta0, args=(pi, vi, r, n_v, P, hyper), jac=True,
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
    m = float(res.x[0])
    s, sigma_e = float(np.exp(res.x[1])), float(np.exp(res.x[2]))
    delta, gamma = res.x[3:3 + P], res.x[3 + P:]

    # ---- Notes recalées (toutes les observations, y compris hors ajustement) ----
    pidx = d["pid"].map(pmap).to_numpy()
    d["a_cal"] = m + (d["rating"].to_numpy() - m - delta[pidx]) * np.exp(-gamma[pidx])
    if rating_bounds is not None:
        lo, hi = rating_bounds
        d["censored"] = (d["rating"] <= lo + 1e-9) | (d["rating"] >= hi - 1e-9)
    else:
        d["censored"] = False
    labels = d[["pid", "video", "rating", "a_cal", "censored", "n_raters", "in_fit"]].reset_index(drop=True)

    # ---- Table participants ----
    parts = pd.DataFrame({
        "pid": pids,
        "n_ratings": d.groupby("pid").size().reindex(pids).to_numpy(),
        "n_in_fit": fit_d.groupby("pid").size().reindex(pids).fillna(0).astype(int).to_numpy(),
        "offset": delta,            # >0 : note plus haut que le participant moyen
        "spread": np.exp(gamma),    # >1 : plage de notes plus large que la moyenne
    })
    for rv in (1.0, 5.0, 9.0):      # que vaut sa note rv une fois recalée ?
        parts[f"cal_if_rating_{rv:g}"] = m + (rv - m - delta) * np.exp(-gamma)

    # ---- Arousal consensus par vidéo (moyenne a posteriori de l'effet vidéo) ----
    d["_z"] = (d["rating"].to_numpy() - m - delta[pidx]) / (s * np.exp(gamma[pidx]))
    agg = d.groupby("video").agg(S1=("_z", "sum"), n=("_z", "size"))
    videos = pd.DataFrame({
        "video": agg.index,
        "video_arousal": m + s * (agg["S1"] / (sigma_e ** 2 + agg["n"])).to_numpy(),
        "n_raters": agg["n"].to_numpy(),
    }).reset_index(drop=True)

    return Calibration(labels=labels, participants=parts, videos=videos, m=m, s=s,
                       sigma_e=sigma_e, hyper=hyper, converged=bool(res.success), nll=float(res.fun))


# --------------------------------------------------------------------------
# Classes (seuils communs à tous les participants)
# --------------------------------------------------------------------------
def fit_cutpoints(values, n_classes: int = 3) -> np.ndarray:
    """Seuils par quantiles de `values` (à calculer sur le TRAIN uniquement)."""
    return np.quantile(np.asarray(values), [i / n_classes for i in range(1, n_classes)])


def apply_cutpoints(values, cutpoints) -> np.ndarray:
    """0 = niveau le plus faible, ..., len(cutpoints) = niveau le plus élevé."""
    return np.digitize(np.asarray(values), np.asarray(cutpoints))


def _class_names(k: int, names: list[str] | None = None) -> list[str]:
    if names is not None:
        if len(names) != k:
            raise ValueError(f"{k} classes attendues, {len(names)} noms fournis")
        return list(names)
    return {3: ["low", "mid", "high"], 4: ["low", "mid", "high", "very_high"]}.get(
        k, [str(i) for i in range(k)])


def add_classes(labels: pd.DataFrame, n_classes: int = 3, cutpoints=None,
                names: list[str] | None = None):
    """
    Ajoute 'class' et 'class_name' à labels (colonne a_cal). Retourne (labels, cutpoints).
    cutpoints : seuils en unités de note recalée (ex. [4, 6.5]) ; sinon quantiles de a_cal.
    """
    if cutpoints is None:
        cutpoints = fit_cutpoints(labels["a_cal"], n_classes)
    names = _class_names(len(cutpoints) + 1, names)
    out = labels.copy()
    out["class"] = apply_cutpoints(out["a_cal"], cutpoints)
    out["class_name"] = [names[c] for c in out["class"]]
    return out, np.asarray(cutpoints)


# --------------------------------------------------------------------------
# Échelle individuelle : que signifie chaque note pour CHAQUE participant ?
# --------------------------------------------------------------------------
def participant_scales(cal: Calibration, n_classes: int = 4, cutpoints=None,
                       class_names: list[str] | None = None, rating_grid=None):
    """
    Table "échelle par participant" : pour chaque participant, la classe d'arousal
    (low / mid / high / very_high) associée à chaque note brute.

    Colonnes :
      pid
      alpha : décalage du participant (>0 : note plus haut que le participant moyen)
      beta  : étalement (>1 : utilise une plage de notes plus large que la moyenne)
      note_<k>->classe : classe correspondant à la note brute k pour CE participant
      seuil_<a>-><b>   : note brute (continue) à partir de laquelle il passe de la classe a à b
                         (les notes sont continues : ces seuils donnent la frontière exacte)

    cutpoints   : seuils COMMUNS en unités de note recalée (ex. [3, 5.5, 7]) ;
                  par défaut, quantiles de toutes les notes recalées (classes équilibrées).
    rating_grid : notes brutes à décrire (défaut : entiers couvrant l'étendue observée ;
                  passer range(0, 11) pour une échelle 0-10).
    Retourne (table, cutpoints).
    """
    if cutpoints is None:
        cutpoints = fit_cutpoints(cal.labels["a_cal"], n_classes)
    cutpoints = np.asarray(cutpoints, dtype=float)
    names = _class_names(len(cutpoints) + 1, class_names)
    if rating_grid is None:
        r = cal.labels["rating"]
        rating_grid = range(int(np.floor(r.min())), int(np.ceil(r.max())) + 1)
    rating_grid = list(rating_grid)

    p = cal.participants
    delta, spread = p["offset"].to_numpy(), p["spread"].to_numpy()
    table = pd.DataFrame({"pid": p["pid"], "alpha": delta, "beta": spread})

    cols = {}
    for k in rating_grid:
        a_cal = cal.m + (k - cal.m - delta) / spread            # note k recalée
        cols[f"{k:g}"] = [names[c] for c in apply_cutpoints(a_cal, cutpoints)]
    # frontière brute : a_cal = c  <=>  note = m + delta + (c - m) * spread
    for i, c in enumerate(cutpoints):
        cols[f"seuil_{names[i]}->{names[i + 1]}"] = cal.m + delta + (c - cal.m) * spread
    table = pd.concat([table, pd.DataFrame(cols)], axis=1)
    return table, cutpoints


# --------------------------------------------------------------------------
# Protocole leave-one-subject-out pour les labels
# --------------------------------------------------------------------------
def loso_labels(ratings: pd.DataFrame, n_classes: int = 3, hyper: Hyper | None = None):
    """
    Pour chaque participant test, génère (pid, labels_train, labels_test, cutpoints) :
      - labels_train : calibration ajustée SANS le participant test ; seuils appris dessus
      - labels_test  : notes du participant test recalées par une calibration sur TOUS les
                       participants (sa "vérité terrain" : il faut ses notes pour estimer son
                       biais), classées avec les seuils du train
    Le modèle physiologique ne doit voir que labels_train à l'entraînement.
    """
    ratings = to_long(ratings)
    full = fit_calibration(ratings, hyper)
    for pid in sorted(ratings["pid"].unique(), key=str):
        train = fit_calibration(ratings[ratings["pid"] != pid], hyper)
        tr, cuts = add_classes(train.labels, n_classes)
        te, _ = add_classes(full.labels[full.labels["pid"] == pid], cutpoints=cuts)
        yield pid, tr, te, cuts


# --------------------------------------------------------------------------
# Validation : la calibration prédit-elle mieux des notes non vues ?
# --------------------------------------------------------------------------
def predict_ratings(cal: Calibration, pairs: pd.DataFrame) -> np.ndarray:
    """Prédit la note brute de (pid, video) : m + delta_p + spread_p * (arousal consensus - m)."""
    p = cal.participants.set_index("pid")
    v = cal.videos.set_index("video")["video_arousal"]
    delta = p["offset"].reindex(pairs["pid"]).to_numpy()
    spread = p["spread"].reindex(pairs["pid"]).to_numpy()
    va = v.reindex(pairs["video"]).fillna(cal.m).to_numpy()   # vidéo inconnue -> moyenne
    return cal.m + delta + spread * (va - cal.m)


def cv_check(ratings: pd.DataFrame, n_folds: int = 5, seed: int = 0,
             hyper: Hyper | None = None) -> pd.DataFrame:
    """
    Validation croisée en masquant des notes individuelles. RMSE (en points de note) de trois
    prédicteurs de la note d'un participant pour une vidéo :
      moyenne_video : moyenne des notes brutes des autres participants (aucune calibration)
      zscore        : moyenne des z-scores par participant, remise à l'échelle du participant
      calibration   : ce modèle
    Si 'calibration' ne bat pas les deux autres, elle n'apporte rien sur vos données.
    """
    rng = np.random.default_rng(seed)
    d = to_long(ratings)
    fold = rng.integers(0, n_folds, len(d))
    err = {"moyenne_video": [], "zscore": [], "calibration": []}
    for k in range(n_folds):
        tr, te = d[fold != k], d[fold == k]
        y = te["rating"].to_numpy()

        cal = fit_calibration(tr, hyper, rating_bounds=None)
        err["calibration"].append((y - predict_ratings(cal, te)) ** 2)

        vm = tr.groupby("video")["rating"].mean()
        err["moyenne_video"].append((y - vm.reindex(te["video"]).fillna(tr["rating"].mean()).to_numpy()) ** 2)

        st = tr.groupby("pid")["rating"].agg(["mean", "std"])
        mu_tr, sd_tr = st["mean"].reindex(tr["pid"]).to_numpy(), st["std"].reindex(tr["pid"]).to_numpy()
        zv = pd.Series((tr["rating"].to_numpy() - mu_tr) / sd_tr).groupby(tr["video"].to_numpy()).mean()
        pred = (st["mean"].reindex(te["pid"]).to_numpy()
                + st["std"].reindex(te["pid"]).to_numpy() * zv.reindex(te["video"]).fillna(0).to_numpy())
        err["zscore"].append((y - pred) ** 2)
    return pd.DataFrame({"rmse": {k: float(np.sqrt(np.nanmean(np.concatenate(v)))) for k, v in err.items()}})


# --------------------------------------------------------------------------
# Données simulées (même structure creuse que vos données)
# --------------------------------------------------------------------------
def simulate_ratings(n_participants=20, n_videos=220, n_per_participant=40, sigma_e=0.6,
                     bounds=(1.0, 9.0), seed=0):
    rng = np.random.default_rng(seed)
    m, s = 5.0, 1.8
    u = rng.normal(0, 1, n_videos)
    delta = rng.normal(0, 1.0, n_participants)
    gamma = rng.normal(0, 0.4, n_participants)
    rows = []
    for p in range(n_participants):
        for v in rng.choice(n_videos, n_per_participant, replace=False):
            a = u[v] + rng.normal(0, sigma_e)
            r = np.clip(m + delta[p] + s * np.exp(gamma[p]) * a, *bounds)
            rows.append((f"P{p:03d}", f"v{v:03d}", r, m + s * a))
    df = pd.DataFrame(rows, columns=["pid", "video", "rating", "a_true"])
    return df, dict(m=m, s=s, delta=delta, gamma=gamma, sigma_e=sigma_e)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ratings = to_long(sys.argv[1])
        print(f"{ratings.pid.nunique()} participants, {ratings.video.nunique()} vidéos, {len(ratings)} notes")
    else:
        ratings, truth = simulate_ratings()
        print("Mode simulation (passez un CSV en argument pour vos données)")
    cal = fit_calibration(ratings)
    print(f"convergence: {cal.converged} | m={cal.m:.2f} s={cal.s:.2f} "
          f"sigma_e={cal.sigma_e:.2f} | part de variance 'vidéo' (ICC)={cal.icc_video:.2f}")
    print(cal.participants.round(2).to_string(index=False))
    print("\nValidation croisée (RMSE en points de note, plus bas = mieux) :")
    print(cv_check(ratings).round(3))
"""
Construit un CSV participants (lignes) x vidéos (colonnes) à partir des JSON de session.

Usage en Python :
    from build_ratings_csv import build_ratings_csv
    df = build_ratings_csv("data/json/", "arousal_matrix.csv")

Usage en ligne de commande :
    python build_ratings_csv.py data/json/ arousal_matrix.csv
    python build_ratings_csv.py data/json/ valence_matrix.csv --score valence_score
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd


def _collect_paths(source) -> list[Path]:
    """Accepte un dossier (recherche récursive de *.json), un fichier ou une liste."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        return sorted(p.rglob("*.json")) if p.is_dir() else [p]
    paths: list[Path] = []
    for s in source:
        paths.extend(_collect_paths(s))
    return paths


def read_session_ratings(json_path: Path, score_field: str = "arousal_score",
                         trial_type: str = "emotion_video") -> tuple[dict, list[tuple]]:
    """Retourne (infos session, [(video, note), ...]) pour un fichier JSON."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("dataset_export", {})
    info = {
        "participant_id": meta.get("participant_id") or json_path.stem,
        "day": meta.get("day"),
        "protocol": meta.get("protocol"),
    }
    rows = [
        (ev["media_file"], ev.get(score_field))
        for ev in data.get("events", [])
        if ev.get("trial_type") == trial_type and ev.get("media_file")
    ]
    return info, rows


def build_ratings_csv(source, output_csv: str | Path | None = None,
                      score_field: str = "arousal_score",
                      strip_extension: bool = True,
                      sort_columns: bool = True,
                      include_day: bool | None = None) -> pd.DataFrame:
    """
    source          : dossier de JSON, fichier JSON, ou liste de chemins
    output_csv      : chemin du CSV à écrire (None = pas d'écriture)
    score_field     : 'arousal_score' (défaut) ou 'valence_score'
    strip_extension : 'accident_1.mp4' -> 'accident_1'
    sort_columns    : colonnes triées par ordre alphabétique (sinon ordre d'apparition)
    include_day     : True  -> une ligne par (participant, jour), index 'P001_day1'
                      False -> une ligne par participant
                      None  -> automatique : bascule sur True s'il y a plusieurs jours
                               pour un même participant (avec avertissement)

    Les notes manquantes (vidéo non vue, note nulle) sont laissées vides (NaN).
    Si une même vidéo apparaît plusieurs fois pour une session, la moyenne est utilisée.
    """
    paths = _collect_paths(source)
    if not paths:
        raise FileNotFoundError(f"Aucun fichier JSON trouvé dans {source!r}")

    sessions = []
    for p in paths:
        try:
            info, rows = read_session_ratings(p, score_field)
        except (json.JSONDecodeError, KeyError) as e:
            warnings.warn(f"{p.name} ignoré ({type(e).__name__}: {e})")
            continue
        if not rows:
            warnings.warn(f"{p.name} : aucune vidéo notée trouvée, fichier ignoré")
            continue
        sessions.append((p, info, rows))

    # Doit-on séparer les jours ?
    pids = [s[1]["participant_id"] for s in sessions]
    has_duplicates = len(pids) != len(set(pids))
    if include_day is None:
        include_day = has_duplicates
        if has_duplicates:
            warnings.warn("Plusieurs sessions pour un même participant : "
                          "une ligne par (participant, jour) est utilisée.")
    elif not include_day and has_duplicates:
        raise ValueError("Doublons de participant_id avec include_day=False : "
                         "des sessions s'écraseraient. Utilisez include_day=True.")

    records, col_order = {}, []
    for p, info, rows in sessions:
        key = info["participant_id"]
        if include_day:
            key = f"{key}_{info['day'] or p.stem}"
        if key in records:
            raise ValueError(f"Clé en double '{key}' ({p.name}). Vérifiez les métadonnées.")

        s = pd.Series({(Path(v).stem if strip_extension else v): score for v, score in rows},
                      dtype="float64")
        # Vidéos répétées : moyenne (le dict ci-dessus garderait seulement la dernière)
        names = [(Path(v).stem if strip_extension else v) for v, _ in rows]
        if len(names) != len(set(names)):
            warnings.warn(f"{p.name} : vidéos répétées, moyenne des notes utilisée")
            s = (pd.DataFrame(rows, columns=["v", "s"])
                 .assign(v=lambda d: d.v.map(lambda x: Path(x).stem if strip_extension else x))
                 .groupby("v", sort=False)["s"].mean())
        records[key] = s
        col_order += [c for c in s.index if c not in col_order]

    df = pd.DataFrame.from_dict(records, orient="index")
    df = df[sorted(col_order) if sort_columns else col_order]
    df.index.name = "participant"

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="dossier contenant les JSON (ou un fichier)")
    ap.add_argument("--output", help="CSV de sortie")
    ap.add_argument("--score", default="arousal_score",
                    choices=["arousal_score", "valence_score"])
    ap.add_argument("--keep-extension", action="store_true", help="garder le .mp4")
    ap.add_argument("--no-sort", action="store_true", help="colonnes dans l'ordre d'apparition")
    args = ap.parse_args()

    out = build_ratings_csv(args.source, args.output, score_field=args.score,
                            strip_extension=not args.keep_extension,
                            sort_columns=not args.no_sort)
    print(f"{out.shape[0]} participants x {out.shape[1]} vidéos -> {args.output}")
    print(f"Valeurs manquantes : {int(out.isna().sum().sum())}")
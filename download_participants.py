"""Télécharge d'autres participants pour le notebook multimodal (et la partie autoencodeur).

À lancer depuis la racine du repo, environnement activé, après `hf auth login` :

    python download_participants.py            # 25 participants de plus (par défaut)
    python download_participants.py --n 10     # un autre nombre
    python download_participants.py --dry-run  # affiche la sélection sans rien télécharger

Le script :
1. lit sample/labels_train.csv (les participants de test, marqués TESTSUBJECT, sont ignorés) ;
2. garde les participants qui ont au moins --min-ratings notes d'arousal le jour 1 ;
3. retire ceux déjà présents dans sample/PREPROCESSED/ ;
4. en tire --n au hasard (graine fixe, donc la même sélection pour tout le monde) ;
5. télécharge pour chacun le .bdf et le .record_map de la tâche emotion-video du jour 1.

Le notebook utilise ensuite automatiquement tous les participants présents
(PARTICIPANT_IDS = None) : il suffit de le relancer en entier.
"""

import argparse
import os
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ID = "HABSteam/multimodal-eeg-emotion"
MB_PER_RECORDING = 220  # taille moyenne approximative d'un enregistrement emotion-video


def eligible_participants(labels_path: Path, min_ratings: int) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, na_values=["TESTSUBJECT"])
    labels["participant_id"] = labels["participant_id"].astype(str).str.strip()
    labels["day"] = pd.to_numeric(
        labels["day"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    labels["rating_arousal"] = pd.to_numeric(labels["rating_arousal"], errors="coerce")
    day1 = labels[(labels["day"] == 1) & labels["rating_arousal"].notna()]
    day1 = day1[day1["participant_id"].str.fullmatch(r"P\d+")]
    counts = day1.groupby("participant_id").agg(
        n_ratings=("rating_arousal", "size"),
        cohort=("cohort", "first") if "cohort" in day1 else ("rating_arousal", "size"),
    )
    return counts[counts["n_ratings"] >= min_ratings].sort_index()


def already_downloaded(bdf_root: Path) -> set:
    found = bdf_root.glob("P*/P*_day-1_task-emotion-video_multimodal.bdf")
    return {path.parent.name for path in found}


def find_token():
    # Le token enregistré par `hf auth login`, ou la variable HF_TOKEN.
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_file = home / "token"
    return token_file.read_text(encoding="utf-8").strip() if token_file.is_file() else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download_file(remote: str, local: Path, token: str) -> None:
    """Télécharge un fichier par morceaux de 1 Mo (très peu de mémoire), avec reprise si interrompu."""
    if local.is_file():
        print("    déjà présent")
        return
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{urllib.parse.quote(remote)}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise
        location = urllib.parse.urljoin(url, error.headers["Location"])
        response = None
    else:
        location = None
    part = local.with_name(local.name + ".part")
    local.parent.mkdir(parents=True, exist_ok=True)
    done = part.stat().st_size if part.is_file() else 0
    if location is not None:  # le fichier est servi par un autre serveur (CDN), sans le token
        if response is not None:
            response.close()
        headers = {"Range": f"bytes={done}-"} if done else {}
        response = urllib.request.urlopen(urllib.request.Request(location, headers=headers), timeout=60)
        if done and response.status != 206:
            done = 0
    elif done:
        done = 0  # pas de reprise possible sans redirection : on repart de zéro
    total = done + int(response.headers.get("Content-Length") or 0)
    with response, open(part, "ab" if done else "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} Mo", end="", flush=True)
    print()
    part.replace(local)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=25, help="nombre de participants à ajouter")
    parser.add_argument("--min-ratings", type=int, default=20, help="notes d'arousal minimum au jour 1")
    parser.add_argument("--seed", type=int, default=42, help="graine du tirage (reproductible)")
    parser.add_argument("--data-root", type=Path, default=Path("sample"))
    parser.add_argument("--dry-run", action="store_true", help="afficher la sélection sans télécharger")
    args = parser.parse_args()

    labels_path = args.data_root / "labels_train.csv"
    if not labels_path.is_file():
        raise SystemExit(f"{labels_path} introuvable : télécharge d'abord les CSV (voir le guide d'installation).")

    eligible = eligible_participants(labels_path, args.min_ratings)
    present = already_downloaded(args.data_root / "PREPROCESSED")
    candidates = [p for p in eligible.index if p not in present]
    print(f"{len(eligible)} participants éligibles, {len(present)} déjà présents, {len(candidates)} candidats.")

    if not candidates and not (args.data_root / "download_selection.txt").is_file():
        raise SystemExit("Aucun nouveau participant à télécharger.")
    selection_file = args.data_root / "download_selection.txt"
    if selection_file.is_file():
        chosen = selection_file.read_text(encoding="utf-8").split()
        print(f"Sélection reprise de {selection_file} (supprime ce fichier pour en tirer une nouvelle).")
    else:
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(candidates, min(args.n, len(candidates))), key=lambda p: int(re.sub(r"\D", "", p)))

    print(f"Sélection ({len(chosen)}) : {', '.join(chosen)}")
    if "cohort" in eligible:
        print("Cohortes :", eligible.loc[chosen, "cohort"].value_counts().sort_index().to_dict())
    print(f"Espace disque estimé : environ {len(chosen) * MB_PER_RECORDING / 1000:.1f} Go")
    if args.dry_run:
        return
    selection_file.write_text("\n".join(chosen), encoding="utf-8")

    token = find_token()
    if token is None:
        raise SystemExit("Token Hugging Face introuvable : lance `hf auth login` (ou définis HF_TOKEN), puis relance.")
    files = []
    for participant in chosen:
        stem = f"PREPROCESSED/{participant}/{participant}_day-1_task-emotion-video_multimodal"
        files += [f"{stem}.record_map", f"{stem}.bdf"]
    for i, remote in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {remote}")
        try:
            download_file(remote, args.data_root / remote, token)
        except urllib.error.HTTPError as error:
            print(f"    échec ({error.code}) : {'fichier absent' if error.code == 404 else 'accès refusé ou erreur serveur'}")

    now_present = already_downloaded(args.data_root / "PREPROCESSED")
    missing = [p for p in chosen if p not in now_present]
    print(f"Téléchargés : {len(chosen) - len(missing)} / {len(chosen)}")
    if missing:
        print("Sans enregistrement emotion-video du jour 1 sur Hugging Face :", ", ".join(missing))
    print("Relance maintenant le notebook en entier (Run > Run All Cells).")


if __name__ == "__main__":
    main()
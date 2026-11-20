"""Download participants of the HABS dataset from Hugging Face (generalises Maroua's download_participants.py).

Run from the repo root, after `hf auth login`. The data folder comes from --data-root,
otherwise from the HABS_DATA environment variable; writing into the repo's sample/ is refused.

    # explicit list of participants
    python latence/download_data.py --participants P001 P017 P020 --tasks emotion-video resting-state
    # list in a file: .txt (IDs separated by spaces / new lines) or .csv (column participant_id)
    python latence/download_data.py --participants-file my_list.txt --tasks emotion-video resting-state
    # random draw (fixed seed: same selection for everyone)
    python latence/download_data.py --random 10 --seed 42
    # show what would be downloaded, without downloading
    python latence/download_data.py --participants P001 --dry-run

For each participant, day, version and task: the .bdf and its .record_map (always kept together),
plus the session metadata.json (needed for event windows). The tables (labels_train.csv,
participants_train.csv, markers.csv) are fetched once if missing. Existing files are skipped,
so the script can be re-run safely; an interrupted download resumes.
"""

import argparse
import os
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ID = "HABSteam/multimodal-eeg-emotion"
ROOT = Path(__file__).resolve().parent.parent
TASKS = ["emotion-video", "resting-state", "artefact"]
VERSIONS = ["PREPROCESSED", "CURATED"]
TABLES = ["labels_train.csv", "participants_train.csv", "markers.csv"]
MB = {"emotion-video": 220, "resting-state": 20, "artefact": 45}   # approximate size per recording


# --------------------------------------------------------------------------- selection

def read_participant_file(path):
    """IDs from a .csv (column participant_id) or a text file (IDs separated by spaces / new lines)."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)["participant_id"].astype(str).str.strip().tolist()
    return path.read_text(encoding="utf-8").replace(",", " ").split()


def eligible_participants(labels_path, days, min_ratings):
    """Training participants with at least min_ratings arousal ratings on each requested day."""
    labels = pd.read_csv(labels_path, na_values=["TESTSUBJECT"])
    labels["participant_id"] = labels["participant_id"].astype(str).str.strip()
    labels["day"] = pd.to_numeric(labels["day"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    labels = labels[labels["rating_arousal"].notna() & labels["participant_id"].str.fullmatch(r"P\d+")]
    counts = labels[labels["day"].isin(days)].groupby(["participant_id", "day"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=days, fill_value=0)
    return sorted(counts.index[(counts >= min_ratings).all(axis=1)])


def by_number(ids):
    return sorted(set(ids), key=lambda p: int(re.sub(r"\D", "", p) or 0))


# --------------------------------------------------------------------------- download (from Maroua's script)

def find_token():
    """Token saved by `hf auth login`, or the HF_TOKEN variable."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_file = home / "token"
    return token_file.read_text(encoding="utf-8").strip() if token_file.is_file() else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download_file(remote, local, token):
    """Download in 1 MB chunks (little memory), resuming a partial .part file when possible."""
    if local.is_file():
        print("    already there")
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
    if location is not None:  # file served by a CDN, without the token
        if response is not None:
            response.close()
        headers = {"Range": f"bytes={done}-"} if done else {}
        response = urllib.request.urlopen(urllib.request.Request(location, headers=headers), timeout=60)
        if done and response.status != 206:
            done = 0
    elif done:
        done = 0  # no resume without redirect: start again
    total = done + int(response.headers.get("Content-Length") or 0)
    with response, open(part, "ab" if done else "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    part.replace(local)


def fetch(remote, data_root, token):
    """Download one file; returns False if it does not exist on Hugging Face."""
    try:
        download_file(remote, data_root / remote, token)
        return True
    except urllib.error.HTTPError as error:
        print(f"    failed ({error.code}): {'file not found' if error.code == 404 else 'access denied or server error'}")
        return False


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--participants", nargs="+", metavar="ID", help="participant IDs, e.g. P001 P017")
    who.add_argument("--participants-file", type=Path, help=".txt (IDs) or .csv (column participant_id)")
    who.add_argument("--random", type=int, metavar="N", help="draw N training participants at random")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=["emotion-video"])
    parser.add_argument("--days", nargs="+", type=int, default=[1])
    parser.add_argument("--versions", nargs="+", choices=VERSIONS, default=["PREPROCESSED"])
    parser.add_argument("--seed", type=int, default=42, help="seed of the random draw")
    parser.add_argument("--min-ratings", type=int, default=20, help="random mode: min arousal ratings per day")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("HABS_DATA"),
                        help="destination folder (default: $HABS_DATA)")
    parser.add_argument("--dry-run", action="store_true", help="show the plan without downloading")
    args = parser.parse_args()

    if not args.data_root:
        sys.exit("No destination: pass --data-root or set HABS_DATA (e.g. export HABS_DATA=../habs_data).")
    data_root = Path(args.data_root).expanduser().resolve()
    if data_root == ROOT or ROOT / "sample" in [data_root, *data_root.parents]:
        sys.exit("Refusing to write into the repo root or its sample/ folder: choose another --data-root.")

    token = find_token()
    if token is None and not args.dry_run:
        sys.exit("Hugging Face token not found: run `hf auth login` (or set HF_TOKEN), then re-run.")

    # tables (needed for random mode and for the analyses)
    for table in TABLES:
        if not (data_root / table).is_file():
            if args.dry_run:
                print(f"would download {table}")
            else:
                print(table)
                fetch(table, data_root, token)

    if args.random is not None:
        labels_path = data_root / "labels_train.csv"
        if not labels_path.is_file():
            sys.exit(f"{labels_path} missing: run once without --dry-run to fetch the tables.")
        pool = eligible_participants(labels_path, args.days, args.min_ratings)
        chosen = by_number(random.Random(args.seed).sample(pool, min(args.random, len(pool))))
        print(f"{len(pool)} eligible participants (>= {args.min_ratings} ratings on day(s) {args.days}), seed {args.seed}")
    else:
        chosen = by_number(args.participants or read_participant_file(args.participants_file))

    files = []
    for p in chosen:
        for version in args.versions:
            for day in args.days:
                folder = f"{version}/{p}"
                files.append(f"{folder}/{p}_day-{day}_metadata.json")
                for task in args.tasks:
                    stem = f"{folder}/{p}_day-{day}_task-{task}_multimodal"
                    files += [f"{stem}.record_map", f"{stem}.bdf"]

    n_rec = len(chosen) * len(args.versions) * len(args.days)
    size_gb = n_rec * sum(MB[t] for t in args.tasks) / 1000
    print(f"Participants ({len(chosen)}): {', '.join(chosen)}")
    print(f"Tasks: {', '.join(args.tasks)} | days: {args.days} | versions: {', '.join(args.versions)}")
    print(f"Destination: {data_root} | estimated size: ~{size_gb:.1f} GB (before skipping existing files)")
    if args.dry_run:
        return

    missing = []
    for i, remote in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {remote}")
        if not fetch(remote, data_root, token):
            missing.append(remote)

    print(f"Done: {len(files) - len(missing)} / {len(files)} files present.")
    if missing:   # tasks and days are not available for every participant
        print("Not available on Hugging Face:")
        for remote in missing:
            print("   ", remote)


if __name__ == "__main__":
    main()

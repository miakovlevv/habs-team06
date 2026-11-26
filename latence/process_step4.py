"""Entry point for the HABS machine run of step 4 (feature extraction on the whole dataset).

To use it, process.py needs a single line (the platform calls process.process):

    from latence.process_step4 import process  # noqa: E402,F401

What it does: `arousal_pipeline.extract` on every participant-day with arousal ratings, day 1 first (cohorts 1–2,
primary analysis, then 3–5, secondary), then day 2 (replication on other sessions), in parallel processes,
with a time budget so the job ends cleanly.
Writes into out_dir/step4/ (one CSV + one JSON per participant, errors.csv), then two combined files,
all_trials.csv and all_participants.csv, which are all we need to download. The evaluation runs locally
(`arousal_pipeline.py evaluate`), so sklearn is not needed on the machine.

Environment variables (optional): HABS_WORKERS (default: number of CPUs, at most 8),
HABS_TIME_BUDGET_MIN (default 75), HABS_LIMIT (only the first N participant-days, for a test run),
HABS_DAYS (default "1 2").
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arousal_pipeline as ap  # noqa: E402


def process(data_dir: Path, out_dir: Path) -> None:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out = out_dir / "step4"
    days = [int(d) for d in os.environ.get("HABS_DAYS", "1 2").split()]
    pairs = ap.rated_pairs(data_dir, days)                          # day 1 (cohorts 1-2, then 3-5), then day 2
    if os.environ.get("HABS_LIMIT"):
        pairs = pairs[:int(os.environ["HABS_LIMIT"])]
    workers = int(os.environ.get("HABS_WORKERS", min(8, os.cpu_count() or 1)))
    budget = float(os.environ.get("HABS_TIME_BUDGET_MIN", 75))
    print(f"step 4 extraction: {len(pairs)} rated participant-days (days {days}), {workers} workers, "
          f"budget {budget} min", flush=True)
    ap.run_extract(data_dir, out, [], workers=workers, time_budget_min=budget, pairs=pairs)

    # combined files: the only thing to download
    trials = [pd.read_csv(f) for f in sorted((out / "trials").glob("*.csv"))]
    parts = [json.loads(f.read_text()) for f in sorted((out / "participants").glob("*.json"))]
    if trials:
        pd.concat(trials, ignore_index=True).to_csv(out_dir / "all_trials.csv", index=False)
        pd.DataFrame(parts).to_csv(out_dir / "all_participants.csv", index=False)
    print(f"wrote all_trials.csv ({sum(len(t) for t in trials)} videos) and all_participants.csv "
          f"({len(parts)} participants) in {out_dir}", flush=True)

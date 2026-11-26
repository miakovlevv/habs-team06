"""Export the exact trials of the step-4 full-dataset test, so another model (e.g. Maroua's autoencoder) can be
scored on the same participants, videos and high / low labels.

    python latence/export_step4_sets.py --features ../habs_results/full_run

Writes latence/step4_sets/:
  - trials_<set>.csv: participant, day, marker_id, rating_arousal, level (high / low / mid) of every video used.
    Folds = leave-one-participant-out (one fold per participant). Mid videos are used only for the within-session
    z-score of the features (no labels involved), never to train or score.
  - sessions.csv: every rated session of the extraction, with the set(s) it enters or the reason it is excluded.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

import arousal_pipeline as ap

SETS = {"cohorts-12_day-1": ([1], [1, 2]),       # primary test
        "cohorts-12_day-2": ([2], [1, 2]),       # replication on other sessions
        "cohorts-345_day-1": ([1], [3, 4, 5])}   # other cohorts (emotion blocks)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--features", required=True, help="folder with all_trials.csv and all_participants.csv")
    a.add_argument("--data-root", default=os.environ.get("HABS_DATA"))
    a.add_argument("--out", default=str(ap.HERE / "step4_sets"))
    args = a.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    excluded = set(ap.read_ids([str(ap.HERE / "participants_dev20.txt"),
                                str(ap.HERE / "participants_confirm20_final.txt")]))
    trials, parts = ap.load_table(args.features, args.data_root, [1, 2])
    team = ap.team_levels(args.data_root)
    sess = parts[["participant", "day", "ok_GSR", "ok_HR", "ok_pupil"]].copy()
    sess["sets"], sess["reason"] = "", ""

    def note(mask, col, text):
        sess.loc[mask & (sess[col] == ""), col] = text

    note(sess.participant.isin(excluded), "reason", "exploration set (dev20 / confirm20)")
    note(~sess.participant.isin(team.participant_id), "reason", "team split: rating SD < 1 or < 8 videos per group")
    note(~sess.ok_GSR, "reason", "GSR flat (range < 0.05 uS)")
    for name, (days, cohorts) in SETS.items():
        t = trials[~trials.participant.isin(excluded) & trials.day.isin(days) & trials.cohort.isin(cohorts)]
        t = t.merge(sess.loc[sess.ok_GSR, ["participant", "day"]], on=["participant", "day"])
        d, small = ap.scorable(t)
        d = d[["participant", "day", "marker_id", "rating_arousal", "level"]].sort_values(["participant", "marker_id"])
        d.to_csv(out / f"trials_{name}.csv", index=False)
        inset = sess.participant.isin(d.participant) & sess.day.isin(days)
        sess.loc[inset, "sets"] += name + " "
        note(sess.participant.isin(small) & sess.day.isin(days), "reason", f"< {ap.MIN_PER_CLASS} high or low videos")
        print(f"{name}: {d.participant.nunique()} participants, {len(d)} videos "
              f"({(d.level == 'high').sum()} high / {(d.level == 'low').sum()} low / {(d.level == 'mid').sum()} mid)")
    cohort = trials.drop_duplicates("participant").set_index("participant").cohort
    sess["cohort"] = sess.participant.map(cohort)
    sess["sets"] = sess.sets.str.strip()
    note((sess.sets == "") & sess.cohort.isna(), "reason", "no labelled trial left after the team split")
    sess.sort_values(["day", "participant"]).to_csv(out / "sessions.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

from arousal_utils import *
from build_ratings_csv import *
import argparse
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="dossier contenant les JSON (ou un fichier)")
    ap.add_argument("--output", help="CSV de sortie")
    args = ap.parse_args()

    out = build_ratings_csv(args.source)

    cal = fit_calibration(out)
    table, cuts = participant_scales(cal, n_classes=4)

    table.to_csv(args.output)
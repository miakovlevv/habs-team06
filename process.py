"""The only file you need to change.

This one is the notebook's example feature, run over many recordings instead
of one: load a resting-state recording with the dataset's own loader, take the
Welch power spectrum of every electrode, and integrate 8-12 Hz to get absolute
alpha power in uV^2. Explore in starting_notebooks/bdf_exploration.ipynb, then
change this file -- it is the same computation, so the numbers agree.

It is a starting point, not a suggestion about what to study.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: The loader that comes with the notebook travels with your job, because
#: `habs submit` uploads this repository. It reads the .record_map beside a
#: recording, so it knows which of the 84 channels are electrodes and which
#: are ECG, respiration, skin conductance or gaze -- and it puts the 166 Hz
#: gaze samples on the same timeline as the 1 kHz signals.
sys.path.insert(0, str(Path(__file__).resolve().parent / "starting_notebooks"))
from bdf_loader import load_bdf  # noqa: E402

#: BDF headers do not agree on a unit. MNE wants volts.
TO_VOLTS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6, "nV": 1e-9}

BAND = (8.0, 12.0)          # alpha
RECORDINGS = 3              # raise it once you know how long yours take


def alpha_power(bdf: Path) -> pd.DataFrame:
    """Absolute alpha power per electrode, in uV^2. Notebook section 6."""
    import mne
    from scipy.integrate import trapezoid

    raw, signals, channels = load_bdf(bdf)
    electrodes = channels.loc[channels.mne_type == "eeg", "bdf_label"].tolist()
    units = channels.set_index("bdf_label")["header_unit"].to_dict()
    rate = float(raw.info["sfreq"])

    volts = np.vstack([signals[ch].to_numpy() * TO_VOLTS[units[ch]] for ch in electrodes])
    eeg = mne.io.RawArray(volts, mne.create_info(electrodes, rate, "eeg"), verbose="ERROR")

    # A two-second window: long enough to resolve the alpha peak, short enough
    # that Welch averages over many of them.
    window = min(round(2 * rate), eeg.n_times)
    spectrum = eeg.compute_psd(method="welch", fmin=0.5, fmax=min(48.0, rate / 2),
                               n_fft=window, n_per_seg=window, n_overlap=window // 2,
                               verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    psd = psd * 1e12                                   # V^2/Hz -> uV^2/Hz
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])

    return pd.DataFrame({
        "channel": electrodes,
        "alpha_uV2": trapezoid(psd[:, band], x=freqs[band], axis=1),
        "seconds": eeg.n_times / rate,
    })


def process(data_dir: Path, out_dir: Path) -> None:
    """Read recordings from data_dir, write your results into out_dir.

    data_dir is the dataset root, read-only and identical for every team:
    data_dir/CURATED/P001/..., and on the platform also labels_train.csv,
    markers.csv and participants_train.csv. Anything you write into out_dir
    is kept.

    The same code runs against sample/, which has the same layout.
    """
    # Named rather than globbed over everything. The dataset is 351 GB: the
    # emotion-video recordings are 150-280 MB each, resting-state about 20.
    recordings = sorted(data_dir.glob("CURATED/P*/P*_task-resting-state_*.bdf"))
    print(f"{len(recordings)} resting-state recording(s) found; "
          f"using up to {RECORDINGS}")

    per_channel, summary = [], []
    for bdf in recordings[:RECORDINGS]:
        table = alpha_power(bdf)
        table.insert(0, "recording", bdf.stem)
        per_channel.append(table)

        strongest = table.loc[table.alpha_uV2.idxmax()]
        summary.append({
            "recording": bdf.stem,
            "participant": bdf.parent.name,
            "seconds": round(float(table.seconds.iloc[0]), 1),
            "electrodes": len(table),
            "alpha_uV2_mean": float(table.alpha_uV2.mean()),
            "strongest_channel": strongest.channel,
            "strongest_uV2": float(strongest.alpha_uV2),
        })
        print(f"  {bdf.stem}: {summary[-1]['alpha_uV2_mean']:.1f} uV^2 mean, "
              f"strongest {strongest.channel} at {strongest.alpha_uV2:.1f}")

    pd.DataFrame(summary).to_csv(out_dir / "alpha.tsv", sep="\t", index=False)
    pd.concat(per_channel).to_csv(out_dir / "alpha_by_channel.tsv", sep="\t", index=False)
    print(f"\nwrote alpha.tsv ({len(summary)} recordings) and alpha_by_channel.tsv")

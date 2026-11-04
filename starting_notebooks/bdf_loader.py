import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pyedflib


def uV_to_uS(voltage, calibration_table=None):
    """Convert microvolts to microsiemens using the GSR calibration table.

    Calibration voltages are in volts and resistances are in kiloohms.
    Values outside the calibration range are clamped to its endpoints.
    """
    if calibration_table is None:
        calibration_table = {
            "voltage_reading": [-2.14, -1.99755, -1.59528, 0.0167, 1.70074, 2.00199],
            "resistance": [1000, 500, 250, 125, 50, 25],
        }
    resistance = np.interp(
        np.asarray(voltage) / 1e6,
        calibration_table["voltage_reading"],
        calibration_table["resistance"],
    )
    return 1000 / resistance


def load_bdf(bdf_path: str | Path, fallback_eeg: list[str] | None = None):
    """Load a BDF recording, check the units and align the Gaze data thanks to
    the record_map infos.

    Parameters
    ----------
    bdf_path : str or pathlib.Path
        Path to the BDF. A same-name .record_map is used when available.
    fallback_eeg : list of str, optional
        EEG channel names to use when there is no record map. Defaults to the
        example dataset's electrode names. Other channels are marked as misc.

    Returns
    -------
    raw : mne.io.Raw
        BDF metadata and annotations; samples are not preloaded.
    data_df : pandas.DataFrame
        Full recording on the master timeline. GSR values are converted from
        microvolts to microsiemens; other signals use BDF header units.
        Lower-rate gaze positions are reconstructed when possible;
        unmapped lower-rate channels are NaN.
    channels : pandas.DataFrame
        Header names, units for data_df, sample rates, and sidecar channel
        metadata. Converted GSR channels have header_unit set to uS.

    The full table can require substantial memory for long recordings. Gaze
    timestamps are estimated, not the original per-sample XDF timestamps.
    Copy the module, including its imports, to reuse it in another project.
    """
    bdf_path = Path(bdf_path)
    if not bdf_path.is_file():
        raise FileNotFoundError(f"BDF file not found: {bdf_path}")

    # MNE provides the annotations; pyEDFlib reads the stored physical units
    # and each channel at its native sampling rate.
    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="ERROR")
    with pyedflib.EdfReader(str(bdf_path)) as reader:
        labels = reader.getSignalLabels()
        units = [reader.getPhysicalDimension(i) for i in range(reader.signals_in_file)]
        rates = [reader.getSampleFrequency(i) for i in range(reader.signals_in_file)]
    if labels != raw.ch_names:
        raise ValueError(
            "MNE and BDF header channel labels differ; inspect the channel order."
        )

    channels = pd.DataFrame(
        {"bdf_label": labels, "header_unit": units, "header_rate_hz": rates}
    )
    channels["map_name"] = None
    channels["modality"] = "unknown"
    channels["map_unit"] = None
    channels["mne_type"] = "misc"

    map_path = bdf_path.with_suffix(".record_map")
    record_map: dict | None = (
        json.loads(map_path.read_text(encoding="utf-8")) if map_path.is_file() else None
    )
    stream_map: dict[str, dict[str, dict]] = (
        record_map["stream_map"] if record_map is not None else {}
    )
    groups = (
        [
            (name, spec)
            for stream in stream_map.values()
            for name, spec in stream.items()
        ]
        if record_map
        else []
    )

    def indices_for(spec: dict):
        if "channel_ranges" in spec:
            first, last = spec["channel_ranges"]
            return list(range(int(first), int(last) + 1))  # last index is inclusive
        return spec.get("channel_indices", [])

    if record_map:
        seen = set()
        for group_name, spec in groups:
            if group_name.lower() in {"bdf annotations", "edf annotations"}:
                continue
            indices = indices_for(spec)
            correction: dict = spec.get("correction", {})
            details: list[dict[str, str]] = correction.get("Channels", [])
            if len(indices) != len(details):
                raise ValueError(
                    f"Channel count mismatch for {group_name}; inspect the record map."
                )
            modality = correction.get("RecordType", group_name)
            mne_type = {"EEG": "eeg", "ECG": "ecg", "RESPIRATION": "resp"}.get(
                modality.upper(), "misc"
            )
            for idx, detail in zip(indices, details, strict=True):
                if (
                    not isinstance(idx, int)
                    or idx < 0
                    or idx >= len(channels)
                    or idx in seen
                ):
                    raise ValueError(
                        f"Invalid or duplicate channel index {idx} in record map."
                    )
                seen.add(idx)
                name, label = detail.get("name", ""), labels[idx]
                if not name or not (
                    name == label or (len(label) == 16 and name.startswith(label))
                ):
                    raise ValueError(
                        f"Record map channel {name!r} does not match BDF label {label!r} at index {idx}."
                    )
                channels.loc[idx, ["map_name", "modality", "map_unit", "mne_type"]] = [
                    name,
                    modality,
                    detail.get("units"),
                    mne_type,
                ]
        print(f"Channel metadata: {map_path.name}")
    else:
        if fallback_eeg is None:
            # The fallback list is for the example dataset; edit it for other recordings.
            fallback_eeg = [
                "Fp1",
                "Fp2",
                "F9",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "F10",
                "FC5",
                "FC1",
                "FC2",
                "FC6",
                "T9",
                "T7",
                "C3",
                "C4",
                "T8",
                "T10",
                "CP5",
                "CP1",
                "CP2",
                "CP6",
                "P9",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "P10",
                "O1",
                "O2",
                "AF7",
                "AF3",
                "AF4",
                "AF8",
                "F5",
                "F1",
                "F2",
                "F6",
                "FC3",
                "FCz",
                "FC4",
                "C5",
                "C1",
                "C2",
                "C6",
                "CP3",
                "CP4",
                "P5",
                "P1",
                "P2",
                "P6",
                "PO5",
                "PO3",
                "PO4",
                "PO6",
                "FT7",
                "FT8",
                "TP7",
                "TP8",
                "PO7",
                "PO8",
                "POz",
            ]
        print(
            "No record map found: using fallback EEG names; verify channel types and units."
        )
        channels.loc[
            channels.bdf_label.isin(fallback_eeg), ["modality", "mne_type"]
        ] = ["EEG (fallback)", "eeg"]

    # This changes MNE channel metadata, not the physical values in data_df.
    raw.set_channel_types(
        dict(zip(labels, channels.mne_type, strict=True)), verbose="ERROR"
    )
    eeg = channels[channels.mne_type == "eeg"]
    if eeg.empty:
        raise ValueError(
            "No EEG channels identified; inspect the record map or fallback_eeg."
        )
    if not eeg.header_unit.isin({"V", "mV", "uV", "µV", "μV", "nV"}).all():
        raise ValueError(
            "An EEG channel has an unrecognized BDF header unit; verify its scaling."
        )

    sfreq = float(raw.info["sfreq"])
    total_samples = record_map.get("TotalSamples") if record_map else None
    n_master = (
        min(raw.n_times, total_samples)
        if isinstance(total_samples, int) and total_samples > 0
        else raw.n_times
    )

    # The sidecar can describe how lower-rate gaze samples were packed.
    gaze_specs = [
        spec
        for _, spec in groups
        if spec.get("correction", {}).get("RecordType", "").lower() == "oculometry"
    ]
    if len(gaze_specs) > 1:
        raise ValueError("Multiple oculometry groups found; inspect the record map.")
    gaze_indices = []
    if gaze_specs:
        spec = gaze_specs[0]
        correction = spec["correction"]
        gaze_indices = indices_for(spec)
        gaze_rate = float(
            correction.get(
                "EffectiveSamplingFrequency", correction.get("SamplingFrequency", 0)
            )
        )
        onset = float(correction.get("OnsetSeconds", 0.0))
        if (
            not gaze_indices
            or not (0 < gaze_rate < sfreq)
            or not np.isfinite(onset)
            or onset < 0
        ):
            raise ValueError(
                "Check gaze channels, effective rate and onset in the record map."
            )
        if any(not (0 < rates[i] < sfreq) for i in gaze_indices):
            raise ValueError(
                "Gaze channels must have a lower stored rate than the master timeline."
            )
        if any(not np.isclose(rates[i], rates[gaze_indices[0]]) for i in gaze_indices):
            raise ValueError(
                "Gaze channels have different stored rates; inspect them separately."
            )
        native_count = round(n_master * gaze_rate / sfreq)
        onset_idx = min(max(round(onset * sfreq), 0), n_master - 1)
        filler_count = round(onset * gaze_rate)
        if not (0 < native_count < n_master and filler_count < native_count):
            raise ValueError(
                "Unexpected gaze sample count or onset; inspect the recording and map."
            )
        gaze_rows = np.round(
            np.linspace(onset_idx, n_master - 1, native_count - filler_count)
        ).astype(int)

    signal_columns = {}
    unmapped = []
    with pyedflib.EdfReader(str(bdf_path)) as reader:
        sample_counts = reader.getNSamples()
        for idx, label in enumerate(labels):
            if idx in gaze_indices:
                packed = reader.readSignal(
                    idx, 0, min(native_count, int(sample_counts[idx]))
                )
                # native_count is the expected number of samples, not the numnber of really
                # stored samples in the BDF. If the effective sampling rate from the record
                # map is higher than the rounded one stored in the BDF, native_count > len(packed)
                packed = np.pad(
                    packed, (0, native_count - len(packed)), constant_values=np.nan
                )
                values = np.full(n_master, np.nan)
                values[gaze_rows] = packed[filler_count:]
            elif np.isclose(rates[idx], sfreq):
                # This is a defensive measure for a shorter-than-expected channel.
                # Normally those channels are not padded.
                values = reader.readSignal(
                    idx, 0, min(n_master, int(sample_counts[idx]))
                )
                values = np.pad(
                    values, (0, n_master - len(values)), constant_values=np.nan
                )
            else:
                values = np.full(n_master, np.nan)
                unmapped.append(label)
            is_gsr = str(channels.loc[idx, "modality"]).lower() == "gsr" or (
                not record_map and label.strip().lower() == "gsr"
            )
            if is_gsr:
                values = uV_to_uS(values)
                channels.loc[idx, "header_unit"] = "uS"
            signal_columns[label] = values

    data_df = pd.DataFrame(signal_columns)
    data_df.insert(0, "time_s", np.arange(n_master) / sfreq)
    if gaze_indices:
        print(
            f"Gaze: stored {rates[gaze_indices[0]]:g} Hz, effective {gaze_rate:.3f} Hz, onset {onset:.3f} s"
        )
    if unmapped:
        print("Lower-rate channels without an alignment rule remain NaN:", unmapped)
    return raw, data_df, channels

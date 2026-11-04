# Starting notebooks

The `starting_notebooks/` folder contains examples for exploring the data and
building a decoder. Both use the notebook environment described below.

## Exploring a BDF recording

[`starting_notebooks/bdf_exploration.ipynb`](bdf_exploration.ipynb)
is a starting point for inspecting a recording: it reads channel metadata and
annotations, builds a DataFrame, plots a selected segment, and estimates EEG
power. The reusable [`bdf_loader.py`](bdf_loader.py) also
illustrates how gaze samples can be aligned using a `.record_map` sidecar.

GSR is stored in microvolts (µV) in the BDF. When loading the BDF with
`load_bdf` from `bdf_loader.py`, GSR values are converted to microsiemens (µS).

The notebooks have their own dependencies, separate from the repository root
`requirements.txt` used for submitted jobs. From the repository root, create an
environment using either Conda:

```bash
conda env create -f starting_notebooks/environment.yml
conda activate hackathon-eeg
```

or a Python 3.12 (tested with 3.12.10) virtual environment, activated according to your platform:

```bash
python -m venv .venv
# Activate .venv, then:
python -m pip install -r starting_notebooks/requirements.txt
```

Open the notebook in your IDE's notebook interface and select the new
environment as its kernel. Start the kernel in the repository root or
`starting_notebooks/`. Run the cells in order; the default path points to the
included curated resting-state sample. Adjust the file path, segment and marker
settings to explore other recordings. The optional interactive viewer opens a
separate Qt window and requires a graphical desktop; the static plots work
without it.

If you prefer a browser, install Jupyter Notebook in the notebook environment
(`python -m pip install notebook`), then run `jupyter notebook` from the
repository root and open the notebook.

## Real-time decoding from a recording

[`starting_notebooks/real-time_decoding.ipynb`](real-time_decoding.ipynb)
trains a subject-specific EEG classifier to distinguish video viewing from
rating periods. It uses event markers to label the periods, trains on earlier
cycles and evaluates on later cycles, then replays the held-out recording in
chunks to demonstrate causal, online-style predictions. The notebook also
inspects model performance, spatial patterns and processing times. The replay
uses a recorded BDF; it is not a live acquisition system.

Use the same notebook environment and start the kernel in
`starting_notebooks/` so it can import `bdf_loader.py`. Before running the
notebook, download the preprocessed P001 day-1 emotion-video BDF and its
matching `.record_map` from the [Hugging Face dataset](https://huggingface.co/datasets/HABSteam/multimodal-eeg-emotion).
Access is per person: sign in to Hugging Face, accept the dataset terms, and
wait for HABS to approve your request. Once approved, run these commands from
the repository root:

```bash
pip install huggingface_hub
hf auth login
hf download HABSteam/multimodal-eeg-emotion --repo-type dataset --local-dir ./sample --include "PREPROCESSED/P001/P001_day-1_task-emotion-video_multimodal.bdf" "PREPROCESSED/P001/P001_day-1_task-emotion-video_multimodal.record_map"
```

This places both files in `sample/PREPROCESSED/P001/`, where the notebook
expects them. Alternatively, change `BDF` in the notebook to point to the
downloaded BDF, keeping its sidecar next to it. See the [main README](../README.md)
for more download options.

The notebook loads the whole recording into memory and can require substantial
RAM. The optional final cell replays one minute at recording speed; it can be
interrupted. The result is a single-subject demonstration and may reflect eye
or muscle activity as well as brain signals.

Three tables describe the dataset: `markers.csv` (marker codes), `labels_train.csv`
(trial annotations) and `participants_train.csv` (participant information). Only `markers.csv`
is already in this repository, as it's needed for [`starting_notebooks/real-time_decoding.ipynb`](real-time_decoding.ipynb) — the three are part of the dataset, so take them
with `--include "*.csv"`, or read them from `/data` inside a job. See
[DATASET.md](../DATASET.md) for details.

## Using the loader in a job

To reuse the loader in a submitted job, copy the module including its imports
and check its dependencies (`mne`, `numpy`, `pandas`, and `pyedflib`) are in the
root `requirements.txt`. Times are relative to the start of the BDF. The loader
reads the whole recording into memory, so long files can require substantial
memory. This is exploratory analysis, not a finished preprocessing pipeline.


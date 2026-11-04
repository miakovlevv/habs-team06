# HABS Neuro-AI Hackathon

**Build advanced AI models on real EEG data and contribute to the next generation of neurotechnology.**

Université Paris-Saclay · 2–30 November 2026 · [hackathon.habs.ai](https://hackathon.habs.ai/)

You write `process.py`. Run it on your own machine against data you download, or
send it to one of our H100s when you want one. **The GPU is optional** — a lot of
good work in this hackathon will never need it.

---

## 0 · Three accounts

All free, all quick. Make them before anything else.

|   |   |
| - | - |
| [**GitHub**](https://github.com/signup) | Your team's code lives here, and it is what a GPU run is authorised against. |
| [**Hugging Face**](https://huggingface.co/join) | Where you download the dataset from. |
| **Slack** | Mentors, announcements, and where your team key comes back. We email you the invitation — there is no public join link. |

## 1 · Setup, once

|   |   |
| - | - |
| **Copy this repo** | **Use this template** → *Create a new repository*. Call it anything, public or private. |
| **Join Slack** | Accept the invitation we email you. Mentors, announcements and everyone else are there. Nothing to find: if it has not arrived, ask us. |
| **DM `@HABS Bot` your repo** | Send your repo link, or `your-name/your-repo`. One message per team. Your **team key** and **results link** come back in that DM. Tell us even if you never plan to use a GPU: it is also how we know where your results belong. |
| **Paste the key** | In [`.github/workflows/submit.yml`](.github/workflows/submit.yml), replace `PASTE-YOUR-TEAM-KEY-HERE`, keeping the quotes. Only needed for GPU runs. |

> **Do not rename or move the repository afterwards.** Your key is bound to its exact
> address, so renaming it silently breaks GPU runs. If you must, tell `@HABS Bot` and
> we will issue a new key.

## 2 · The data

📖 **[DATASET.md](DATASET.md)** — 355 participants, three tasks, EEG with ECG, skin
conductance, respiration, gaze and pupil size, plus trial labels.
**Read it before you write anything.**

|   |   |
| - | - |
| [`sample/`](sample) | P001, day 1 — already here. Enough to write code against. |
| [🤗 the full dataset](https://huggingface.co/datasets/HABSteam/multimodal-eeg-emotion) | Pull what you want onto your own machine: one file, one folder, or all 351 GB. |
| `/data` | The whole dataset, mounted read-only on the GPU. Nothing to download. |

Access is per person. Open the dataset page with a Hugging Face account, accept
the terms, and HABS approves your request — so ask early, not the night before.

```bash
pip install huggingface_hub && hf auth login

hf download HABSteam/multimodal-eeg-emotion --repo-type dataset --local-dir ./data \
    --include "CURATED/P001/*"                 # one participant
#   --include "CURATED/*/*resting-state*"      # one task, everyone
#   --include "*.csv"                          # the tables only
#   (no --include)                             # everything, 351 GB
```

This download is for working on your own machine. A GPU run needs none of it —
`/data` is already mounted there.

## 3 · Your code

[`process.py`](process.py) is the only file you have to change.

```python
def process(data_dir: Path, out_dir: Path) -> None:
```

We call it once. `data_dir` is the dataset root, read-only and identical for every
team. Everything you write into `out_dir` is kept and is yours.

Put dependencies in [`requirements.txt`](requirements.txt) — CUDA and PyTorch are
already in the image. For examples and setup instructions, see the
[starting notebooks](starting_notebooks/README.md).

## 4 · Run it on your machine

This is the normal way to work, and for many projects it is the only way you need.

```bash
pip install -r requirements.txt
pip install "habs-hack[storage] @ git+https://github.com/HABS-BRAIN/habs-hack@main"

habs run --data ./sample --out ./out
```

Same contract, no queue. Worth doing before every submission.

## 5 · Run it on a GPU — optional

Only if your work needs one: heavy training, or the whole 351 GB at once. Everything
else belongs on your own machine, where the loop is faster anyway.

**Actions** → **HABS** → **Run workflow**. That is the submission — pushing does not
spend GPU time, so commit as often as you like. Jobs share a pool of H100s with every
other team.

|   |   |   |
| - | - | - |
| **build your image** | red here means `requirements.txt` | seconds |
| **run on a GPU** | red here means your code; the log is your own output | minutes to an hour |


## 6 · Your results

Open them with your **results link**:

- **[Azure Storage Explorer](https://azure.microsoft.com/products/storage/storage-explorer)**
  → *Connect to a resource* → *Blob container* → *SAS URL* → paste.
  `out/` holds a folder per job, `logs/` the full output.
- or `export HABS_SAS='<your link>'`, then `habs status` · `habs watch` · `habs ls` ·
  `habs logs` · `habs get`.

---

**Stuck?** Contact your mentor on Slack.

Good luck!

# biorxiv-vizcarra-2026

Evaluating the Transferability of Pathology Foundation Models Across Cancer-related H&E versus Neurodegeneration-related Immunohistochemical Classification Tasks

This repository is the code archive for that preprint. It loads pathology tiles into [Pixeltable](https://pixeltable.com/), computes foundation-model embeddings, and evaluates frozen-embedding linear probes against ResNet CNN fine-tuning.

The snapshot covers the **four datasets used in the paper**:

| Dataset | CLI name | Pixeltable slug | Task |
| --- | --- | --- | --- |
| ABeta (Tang) | `abeta` | `abeta-dataset` | IHC: cored, diffuse, CAA |
| Vizcarra-2023 tau | `tau` | `tau-dataset` | IHC: pretangle (`pre_nft`) vs iNFT (`inft`) |
| ICIAR 2018 BACH | `bach` | `bach-dataset` | H&E: Benign, InSitu, Invasive, Normal |
| TCGA TILs | `till` | `till-dataset` | H&E: TIL-positive vs TIL-negative |

The TIL loader is spelled `till` (matching the module name). Pass either the CLI name or the Pixeltable slug to `--dataset`.

## Setup

Python **3.14**. From the repo root:

```bash
uv sync
uv sync --extra embeddings --extra cnn
# Gated pathology models (UNI, CONCH, Virchow, …):
uv sync --extra embeddings --extra pathology
# Analysis notebooks:
uv sync --extra notebooks
```

Copy `.env.example` to `.env` and set `HF_TOKEN` for Hugging Face gated models. Accept each model’s license on Hugging Face before running (for example [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) and [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/conch)).

A pip-oriented lockfile is in `requirements.txt`. Non-uv users can `pip install -r requirements.txt`.

## Pipeline

```
[optional] vizcarra-2023-create-tiles.py   # tau ROIs → classification tiles
python -m load_tiles <dataset> --source …  # tiles → Pixeltable `tiles`
dataset-embeddings.py                      # embeddings → Pixeltable `embeddings`
dataset-linear-probe.py                    # frozen embeddings + linear probe
dataset-cnn-classifier.py                  # ResNet fine-tune on tile images
dataset-stats.py                           # tile / WSI / class / embedding counts
results-analysis.ipynb                     # paper figures from reports/
```

Loaders are idempotent on `(dataset, tileName)`. Embedding is incremental on `(dataset, tileName, model_id)`.

### 1. Load tiles

```bash
uv run python -m load_tiles --help
uv run python -m load_tiles tau --help

uv run python -m load_tiles abeta --source /path/to/abeta-tiles
uv run python -m load_tiles tau --source /path/to/vizcarra-2023-tiles
uv run python -m load_tiles bach --source /path/to/bach
uv run python -m load_tiles till --source /path/to/tcga-tils
```

Expected on-disk layouts are in each loader docstring (`load_tiles/abeta.py`, `tau.py`, `bach.py`, `till.py`).

Tau tiles are produced from the Vizcarra-2023 ROI dataset:

```bash
uv run python vizcarra-2023-create-tiles.py \
  --source /path/to/vizcarra-2023 \
  --output /path/to/vizcarra-2023-tiles
```

### 2. Embed

```bash
uv run python dataset-embeddings.py --list-models
uv run python dataset-embeddings.py --dataset tau
uv run python dataset-embeddings.py --dataset abeta --models openai/clip-vit-base-patch32
```

Some models (UPath and related WSU checkpoints) expect a sibling `ImageEmbeddingModelComparison` checkout, or `IMAGE_EMBEDDING_COMPARISON_ROOT`.

### 3. Evaluate

```bash
uv run python dataset-linear-probe.py --dataset tau --output-name tau-dataset-single-label
uv run python dataset-cnn-classifier.py --dataset tau --output-name tau-dataset-single-label

uv run python dataset-stats.py --dataset tau
uv run python dataset-stats.py --dataset abeta --split 60:20:20 --seed 0
```

Reports are written under `reports/linear-probe/<output-name>/` and `reports/cnn/<output-name>/` (gitignored). `results-analysis.ipynb` reads those folders.

Optional Pixeltable dashboard (localhost only; use SSH port-forwarding if needed):

```bash
PIXELTABLE_DASHBOARD_PORT=<PORT> uv run python -c "import pixeltable as pxt; pxt.dashboard.serve(open_browser=False); input('Dashboard running — press Enter to stop')"
```

## License

Copyright 2026 The Authors.

This code is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

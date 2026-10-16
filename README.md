# EN4554 — Assignment 2 on CNNs

CIFAR-10 / PyTorch assignment: backpropagation from scratch, optimizers,
regularization and residual networks.

- Notebook: `EN4554_a02.ipynb`
- Cached experiment histories: `results/*.json`
- Dataset (`data/`) is gitignored; it re-downloads automatically on first run.

## Environment

Conda env `cnn_gpu` — Python 3.10, torch 2.7.1+cu118, run on an RTX 4050 Laptop (6 GB).
Select the **Python (cnn_gpu)** kernel in Jupyter/VS Code.

## Progress

- [x] Setup — splits, normalization, seeds
- [x] Task 1 — Backpropagation
- [ ] Task 2 — Optimizers
- [ ] Task 3 — Regularization and residual connections
- [ ] Task 4 — Best CNN under 500k parameters

## Notes

- The CIFAR-10 host (`cs.toronto.edu`, which redirects to `cave.cs.toronto.edu`)
  is slow — expect ~2 hours for the 170 MB archive on a first run. It is
  downloaded once and then cached in `data/`.
- Python 3.10 on Windows can throw `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`
  when loading the system certificate store, which breaks the download. The
  setup cell works around this by pointing SSL at certifi's CA bundle.

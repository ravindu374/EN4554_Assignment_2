# EN4554 — Assignment 2 on CNNs

CIFAR-10 / PyTorch assignment: backpropagation from scratch, optimizers,
regularization and residual networks.

- Notebook: `EN4554_a02.ipynb` — the single submission artifact. All explanations and
  interpretations live in the notebook itself; export it to `220355M_a02.pdf`.
- Cached experiment histories: `results/*.json`
- Dataset (`data/`) is gitignored; it re-downloads automatically on first run.

## Environment

Conda env `cnn_gpu` — Python 3.10, torch 2.7.1+cu118, run on an RTX 4050 Laptop (6 GB).
Select the **Python (cnn_gpu)** kernel in Jupyter/VS Code.

## Progress

- [x] Setup — splits, normalization, seeds
- [x] Task 1 — Backpropagation (verified: val acc 50.6%, grad check 3.7e-9)
- [x] Task 2 — Optimizers (verified: Adam 80.4% best, optimizers match torch.optim)
- [x] Task 3 — Regularization and residual connections — plain-56 degrades (0.7719) vs residual-56 (0.0009)
- [x] Task 4 — Best CNN under 500k parameters — 90.05% test accuracy, 463,627 params, 20 epochs

## Notes

- The CIFAR-10 host (`cs.toronto.edu`, which redirects to `cave.cs.toronto.edu`)
  is slow — expect ~2 hours for the 170 MB archive on a first run. It is
  downloaded once and then cached in `data/`.
- Python 3.10 on Windows can throw `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`
  when loading the system certificate store, which breaks the download. The
  setup cell works around this by pointing SSL at certifi's CA bundle.

## Submission

Per the lecturer: *"For assignment 2 you should add all explanations and interpretations
to the jupyter notebook itself, export it to a pdf and submit. No need to make a separate
report other than that."*

So `EN4554_a02.ipynb` is the whole submission. Restart-and-Run-All, then export to
`220355M_a02.pdf`.

`report/` is no longer needed for submission and is kept only as a by-product
(`report/figures/*.pdf` are still used to rebuild the explainer).

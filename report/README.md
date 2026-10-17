# Report

`report.tex` — the written report. Compile with:

    pdflatex report.tex
    pdflatex report.tex     # second pass for references

Or drop `report.tex` + `figures/` into Overleaf.

## Figures

`figures/*.pdf` are vector figures regenerated from the cached results by:

    python ../explainer/figs.py

The `\fig{name}{width}{caption}` macro uses `\IfFileExists`: if the PDF is
present it is included, otherwise a labelled grey placeholder box is drawn, so
the document always compiles.

| macro call | figure |
|---|---|
| `\fig{f_t1c}` | Task 1(c) loss and accuracy curves |
| `\fig{f_t1d}` | Task 1(d) per-layer gradient norms |
| `\fig{f_t2b}` | Task 2(b) optimizer trajectories |
| `\fig{f_t2c}` | Task 2(c) accuracy vs learning rate |
| `\fig{f_augment}` | Task 3(a) augmentation examples |
| `\fig{f_t3a}` | Task 3(a) train/val and gap |
| `\fig{f_t3b}` | Task 3(b) training loss per epoch |
| `\fig{f_t3b_summary}` | Task 3(b) final loss vs depth |
| `\fig{f_t4}` | Task 4 ablation |

To swap in your own image, drop it in `figures/` under the same name, or change
the first argument of the `\fig` call.

`notes_for_report.md` holds the longer prose that was removed from the notebook,
kept as source material.

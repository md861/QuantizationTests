# Quantization Lab

![ParoQuant project journey](docs/assets/paroquant-journey.png)

Quantization Lab is a research-oriented educational sandbox for seeing how low-bit
quantization changes matrices, spectra, and reconstruction error.

The project starts at matrix level before scaling toward transformer models.
The current focus is Milestone 1: generating synthetic matrices, applying
symmetric INT8 and INT4 quantization, measuring distortion, and making the
failure modes visible.

## Progress

| Area | Status |
| --- | --- |
| Project scaffold and environment | Complete |
| Synthetic matrix generation | Complete |
| Symmetric INT8 and INT4 quantization | Complete |
| Reconstruction and spectrum metrics | Complete |
| Baseline experiment | Complete |
| Outlier-severity sweep | Complete |
| Consolidated comparison plots | Complete |
| Rotation and scaling experiments | Next |
| Transformer integration | Later |

## Current Milestone

Milestone 1 builds the quantization sandbox:

- matrix generation for Gaussian, heavy-tailed, and outlier-heavy data
- symmetric INT8 and INT4 quantization
- reconstruction metrics such as MSE, MAE, cosine similarity, relative
  Frobenius error, and SNR
- singular-value spectrum diagnostics
- comparison plots showing original matrices, residuals, spectra, and metrics

This repository prioritizes clarity, modularity, reproducibility, and
mathematical transparency over production inference performance.

This project is being developed with the help of AI-assisted coding, with the
lab book and project summary used to keep the research process inspectable.

## Run Tests

Use the project virtual environment:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Current expected test state:

```text
59 passed
```

## Reproduce Artifacts

Generated experiment outputs are intentionally ignored by Git. Recreate them
locally with:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
```

These commands write CSV files under `results/` and comparison figures under
`plots/`.

## Project Notes

- `project_summary.md` is the compact handoff for resuming work.
- `lab_book/project_journey.md` is the chronological development record.
- `plots/` and `results/` are local generated artifacts, not tracked source.

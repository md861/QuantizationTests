# Quantization Lab

![ParoQuant project journey](docs/assets/paroquant-journey.png)

Quantization Lab is a research-oriented educational sandbox for seeing how low-bit
quantization changes matrices, spectra, and reconstruction error.

The project starts at matrix level before scaling toward transformer models.
Milestone 1 (quantization sandbox) and Milestone 2 (ParoQuant core) are
complete: pairwise Givens rotations, per-channel scaling, column-grouped and
row-grouped quantization, the rotation/scaling experiment, and a full
comparative sweep across all quantization paths are implemented and tested.
Milestone 3 (tiny transformer integration) is the next research step.

## Project Roadmap

| Milestone | Focus | Status |
| --- | --- | --- |
| 1. Quantization Sandbox | Matrix generation, INT8/INT4 quantization, metrics, spectra, and visual diagnostics | Complete |
| 2. ParoQuant Core | Givens rotations, channel scaling, grouped quantization, and outlier suppression | Complete |
| 3. Tiny Transformer Integration | Apply the quantizer to tiny-gpt2 and DistilGPT2, then measure perplexity and drift | Next |
| 4. Real LLM Benchmarking | Scale to larger open-source LLMs and compare against GPTQ, AWQ, and bitsandbytes | Later |

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
| Integration and hygiene tests | Complete |
| Histogram visualizations | Complete |
| Results analysis helper | Complete |
| Pairwise Givens rotation module | Complete |
| Per-channel scaling | Complete |
| Grouped quantization | Complete |
| Rotation/scaling experiment | Complete |
| Comparative sweep experiment | Complete |
| Transformer integration | Next |

## Completed Milestone 2

Milestone 2 implemented the ParoQuant core: transformations applied before
quantization to reduce outlier pressure:

- **Pairwise Givens rotations** (`quant/rotations.py`): rotate any column pair
  by an angle that minimises the joint max-abs, redistributing outlier energy
  across channels before INT4/INT8 quantization.  Key functions:
  `rotation_matrix`, `apply_rotation`, `optimal_angle`, `rotate_channel_pair`,
  `apply_sequential_rotations`.
- Verified via entry-zeroing (arctan2 angle analytically zeros a target entry),
  full Givens QR decomposition (Q@R=A cross-checked against `numpy.linalg.qr`),
  and column orthogonalisation (Jacobi angle drives inner product to machine zero).
- **Per-channel scaling** (`quant/scaling.py`): compute one reversible positive
  factor per column so nonzero channel max-abs values share a target before
  quantization. Key functions: `column_max_abs`, `compute_channel_scaling`,
  `apply_channel_scaling`, `invert_channel_scaling`, `balance_channel_max_abs`.
- **Grouped quantization** (`quant/quantizer.py`): two strategies — (1) column-grouped:
  contiguous column blocks share one scale per block; (2) row-grouped: each column
  is split into row-groups each with their own scale (the industry-standard GPTQ/AWQ
  approach, giving tighter precision when outliers are localised within a column).
  Key functions: `grouped_symmetric_quantize`, `row_grouped_symmetric_quantize`,
  `quantize_int8_grouped`, `quantize_int4_grouped`,
  `quantize_int8_row_grouped`, `quantize_int4_row_grouped`.
- **Rotation/scaling experiment** (`experiments/rotation_experiment.py`):
  compares baseline INT4, rotation-only INT4, scaling-only INT4, and
  rotation+scaling INT4 on one controlled outlier-heavy matrix.
- **Comparative sweep experiment** (`experiments/sweep_experiment.py`):
  sweeps 12 quantization paths (global, col-grouped, row-grouped, scale+global,
  rotate+global, rotate+scale+global, rotate+scale+row-grouped) across a grid
  of seeds, outlier fractions, and outlier scales. Outputs `results/sweep_metrics.csv`
  and a 4-panel `plots/sweep_dashboard.png`. Key finding: row-grouped and
  rotate+scale+row-grouped are the only paths that consistently beat global INT4
  for row-localised outliers; column-grouped gives no improvement at any group size.

The next milestone is tiny transformer integration: apply the best current
matrix-level path to tiny-gpt2 or DistilGPT2, then measure perplexity,
activation drift, and output similarity.

Milestone 1 built the quantization sandbox:

- matrix generation for Gaussian, heavy-tailed, and outlier-heavy data
- symmetric INT8 and INT4 quantization
- reconstruction metrics: MSE, MAE, cosine similarity, relative Frobenius error, SNR
- singular-value spectrum diagnostics
- value, residual, and quantized-code histogram diagnostics
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
157 passed
```

## Reproduce Artifacts

Generated experiment outputs are intentionally ignored by Git. Recreate them
locally with:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/rotation_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/sweep_experiment.py
```

These commands write CSV files under `results/` and comparison figures under
`plots/`.

The analysis helper also writes a collated benchmark-style visual:

- `plots/analysis_dashboard.png`

The rotation/scaling experiment writes:

- `results/rotation_metrics.csv`
- `plots/rotation_scaling_comparison.png`

The sweep experiment writes:

- `results/sweep_metrics.csv` and `plots/sweep_dashboard.png` (32×32, seeds 0–4)
- `results/sweep_metrics_320x320.csv` and `plots/sweep_dashboard_320x320.png` (320×320, seeds 5–9)

To reproduce the large-matrix sweep:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python - << 'EOF'
from pathlib import Path
from experiments.sweep_experiment import SweepConfig, run_sweep_experiment, print_summary
config = SweepConfig(
    shape=(320, 320), seeds=[5,6,7,8,9],
    outlier_fractions=[0.02, 0.07, 0.15], outlier_scales=[7.5, 15.0, 30.0],
    row_group_sizes=[4, 8, 16, 32], col_group_sizes=[4, 8, 16],
    csv_name="sweep_metrics_320x320.csv", plot_name="sweep_dashboard_320x320.png",
)
records = run_sweep_experiment(config)
print_summary(records)
EOF
```

Milestone 2 development visuals include:

- `plots/channel_scaling_dashboard.png`

## Project Notes

- `project_summary.md` is the compact handoff for resuming work.
- `lab_book/project_journey.md` is the chronological development record.
- `docs/research_draft.md` is the living paper-style draft of findings,
  examples, figures, and current claims.
- `docs/figures/` contains tracked figures used by the research draft.
- When a committed change affects the research story, update the draft and
  commit any required `docs/figures/` resources with it.
- `plots/` and `results/` are local generated artifacts, not tracked source.

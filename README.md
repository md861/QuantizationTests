# Quantization Lab

![ParoQuant project journey](docs/assets/paroquant-journey.png)

Quantization Lab is a research-oriented educational sandbox for seeing how low-bit
quantization changes matrices, spectra, and reconstruction error.

The project starts at matrix level before scaling toward transformer models.
Milestone 1 (quantization sandbox) is complete. Milestone 2 (ParoQuant core)
is now underway, with pairwise Givens rotations, per-channel scaling, grouped
quantization, and the first rotation/scaling experiment all in place.

## Project Roadmap

| Milestone | Focus | Status |
| --- | --- | --- |
| 1. Quantization Sandbox | Matrix generation, INT8/INT4 quantization, metrics, spectra, and visual diagnostics | Complete |
| 2. ParoQuant Core | Givens rotations, channel scaling, grouped quantization, and outlier suppression | Active |
| 3. Tiny Transformer Integration | Apply the quantizer to tiny-gpt2 and DistilGPT2, then measure perplexity and drift | Planned |
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
| Transformer integration | Later |

## Current Milestone

Milestone 2 implements the ParoQuant core — transformations applied before
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
146 passed
```

## Reproduce Artifacts

Generated experiment outputs are intentionally ignored by Git. Recreate them
locally with:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/rotation_experiment.py
```

These commands write CSV files under `results/` and comparison figures under
`plots/`.

The analysis helper also writes a collated benchmark-style visual:

- `plots/analysis_dashboard.png`

The rotation/scaling experiment writes:

- `results/rotation_metrics.csv`
- `plots/rotation_scaling_comparison.png`

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

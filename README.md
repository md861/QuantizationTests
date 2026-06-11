# Quantization Lab

![ParoQuant project journey](docs/assets/paroquant-journey.png)

Quantization Lab is a research-oriented educational sandbox for seeing how low-bit
quantization changes matrices, spectra, and reconstruction error.

The project starts at matrix level before scaling toward transformer models.
Milestone 1 (quantization sandbox) and Milestone 2 (ParoQuant core) are
complete: pairwise Givens rotations, per-channel scaling, column-grouped and
row-grouped quantization, the rotation/scaling experiment, and a full
comparative sweep across all quantization paths are implemented and tested.
Milestone 3 (tiny transformer integration) is underway: the transformer
harness (`experiments/transformer_experiment.py`) is implemented and tested,
covering weight reconstruction, activation drift, logit/loss, and perplexity
across INT4 and INT8 paths on any HuggingFace causal LM. Benchmark
runs on `sshleifer/tiny-gpt2`, `roneneldan/TinyStories-1M`, and
`EleutherAI/pythia-14m` are complete and documented in the research draft.

## Project Roadmap

| Milestone | Focus | Status |
| --- | --- | --- |
| 1. Quantization Sandbox | Matrix generation, INT8/INT4 quantization, metrics, spectra, and visual diagnostics | Complete |
| 2. ParoQuant Core | Givens rotations, channel scaling, grouped quantization, and outlier suppression | Complete |
| 3. Tiny Transformer Integration | Apply INT4/INT8 quantizer to `sshleifer/tiny-gpt2` ✓, `roneneldan/TinyStories-1M` ✓, `EleutherAI/pythia-14m` ✓, `EleutherAI/pythia-70m` ✓, `distilgpt2` ✓; rotation presets next | Active |
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
| Top-width sparse rotation selection | Complete |
| Per-channel scaling | Complete |
| Grouped quantization | Complete |
| Rotation/scaling experiment | Complete |
| Comparative sweep experiment | Complete |
| Transformer harness (weight + activation + logit metrics) | Active |

## Completed Milestone 2

Milestone 2 implemented the ParoQuant core: transformations applied before
quantization to reduce outlier pressure:

- **Pairwise Givens rotations** (`quant/rotations.py`): rotate any column pair
  by an angle that minimises the joint max-abs, redistributing outlier energy
  across channels before INT4/INT8 quantization.  Key functions:
  `rotation_matrix`, `apply_rotation`, `optimal_angle`, `rotate_channel_pair`,
  `channel_widths`, `top_width_channel_pairs`, `rotate_top_width_pairs`,
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
  Optional `top_width_pair_fractions` add ParoQuant-style sparse rotation paths
  such as `top_width_rotate_p10_global` and
  `top_width_rotate_scale_p10_row_g4`, selecting independent channel pairs from
  the top percentage of max-abs width differences.
  Sweep CSV rows include `rotation_count`, `rotation_pair_fraction`, and
  `rotation_candidate_fraction` so rotation-heavy comparisons remain interpretable.
  Summary tables report condition-wise mean and standard deviation for MSE ratio
  and zero fraction; dashboard error bars show the same cross-condition spread.

## Active Milestone 3

Milestone 3 applies the ParoQuant INT4/INT8 pipeline to real transformer weights.
The harness (`experiments/transformer_experiment.py`) supports single-layer and
all-layer modes, comparing global, row-grouped, scale+row-grouped, and top-width
rotate+scale+row-grouped paths across both bitwidths on any HuggingFace causal LM.
It measures weight reconstruction (MSE, cosine similarity, SNR), activation drift
(MSE, cosine similarity, relative error), and full-model logit/loss quality
(logit MSE, top-5 token overlap, next-token loss delta, perplexity, and
perplexity ratio).

Completed all-layer runs:

- `sshleifer/tiny-gpt2`: eight compatible transformer layers quantized; all
  tested paths preserved top-5 token overlap on the built-in calibration batch,
  and perplexity ratios stayed within about six parts per million of 1.0. This
  is treated as harness validation because the model's linear layers are
  extremely small.
- `roneneldan/TinyStories-1M`: 48 compatible transformer layers quantized. INT4
  global quantization damaged full-model behavior on the built-in calibration
  batch (perplexity ratio 16.1x), while INT4 row-grouped g4 reduced the hit to
  1.21x and capped top-width rotation+scale+row g4 reduced it further to 1.14x.
  INT8 paths stayed close to the original model, with logit MSE far below INT4.
- `EleutherAI/pythia-14m`: 45 compatible transformer layers quantized (INT8 and
  INT4 baselines). First model where INT8 global is not lossless (perplexity
  ratio 1.24); INT8 row-grouped g4 restores losslessness (0.994). INT4 global is
  catastrophic (perplexity ratio 15,074x); INT4 row-grouped g4 gives 1.33x.
  Group size 4 vs 32 is a >2x quality difference at INT4.
- `EleutherAI/pythia-70m`: 45 compatible transformer layers quantized (INT8 and
  INT4 baselines, ~13 min each). INT8 global degrades further (PPL ratio 1.44);
  INT8 g4 remains lossless (0.971). INT4 global catastrophic (~501 trillion PPLx);
  INT4 row-grouped g4 gives 7.52x — a qualitative jump from 14m's 1.33x.
  Group size effect at INT4: g4 vs g128 is a 478x quality gap. INT8 and INT4
  take identical wall-clock time (~13 min), confirming runtime is dominated by
  weight passes not bitwidth arithmetic.

- `distilgpt2`: 24 compatible transformer layers quantized (INT8 and INT4
  baselines, ~11–12 min each). INT8 g4 fully lossless (PPLx 0.999, top-5 1.000).
  INT4 g4 gives **1.058x** — best INT4 result of any real model in this study.
  INT4 global: 48.85x (catastrophic but less so than Pythia models). Key finding:
  architecture and distillation training dominate over parameter count for INT4
  quality; distilgpt2 quantizes 7x better than Pythia-70m at INT4 g4 despite
  being larger.

All planned baseline models complete. Next: rotation presets on Pythia-14m,
Pythia-70m, and distilgpt2.

Use the safer benchmark runner (`experiments/run_transformer_benchmark.py`) for
all remaining models. Always launch from a detached tmux session with
`; tmux kill-session -t bench` appended so the session self-destructs on
completion. Pre-download each model with `--download-only` before the heavy run:

```bash
tmux new-session -d -s bench && tmux send-keys -t bench \
  "MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/run_transformer_benchmark.py \
  pythia-70m-int8-baseline --download-only 2>&1 | tee /tmp/pythia70m_download.log \
  ; tmux kill-session -t bench" Enter
```

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
206 passed, 1 warning
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
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/transformer_experiment.py
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
- `results/sweep_metrics_top_width_32x32.csv` and `plots/sweep_dashboard_top_width_32x32.png`
- `results/sweep_metrics_top_width_320x320.csv` and `plots/sweep_dashboard_top_width_320x320.png`

The top-width sparse-rotation sweeps use `top_width_pair_fractions=[0.05, 0.10, 0.20]`.

The transformer harness writes:

- `results/transformer_weight_metrics.csv`
- `results/transformer_activation_metrics.csv`
- `results/transformer_logit_metrics.csv`
- `plots/transformer_dashboard.png`

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

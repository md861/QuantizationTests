# Quantization Lab Project Summary

This is the compact handoff document for resuming work on the Quantization Lab research sandbox. For the full chronological history, see `lab_book/project_journey.md`.

## Current State

Milestone 1 and Milestone 2 are complete at matrix level. Milestone 2
(ParoQuant Core) now includes pairwise Givens rotations, per-channel scaling,
top-width channel-pair selection for sparse rotations, column-grouped
quantization, row-grouped quantization, the rotation/scaling experiment, and
comparative sweeps across the implemented quantization paths.
Milestone 3 (tiny transformer integration) is underway. The transformer harness
(`experiments/transformer_experiment.py`) is implemented and tested: it loads any
HuggingFace causal LM, runs INT4 and INT8 paths on each linear layer, and measures
weight reconstruction, activation drift, logit/loss quality, and perplexity.
The first all-layer runs on `sshleifer/tiny-gpt2` and
`roneneldan/TinyStories-1M` are complete and documented in
`docs/research_draft.md` with tracked dashboard figures in `docs/figures/`.

Implemented so far:

- synthetic matrix generation
- symmetric INT8 and INT4 quantization
- reconstruction and spectrum metrics
- heatmap, spectrum, and quantization-summary visualizations
- baseline experiment comparing INT8 and INT4 across matrix families
- outlier-severity sweep comparing INT8 and INT4 across controlled outlier fractions and scales
- histogram visualizations for values, residuals, and quantized codes
- results-analysis helper comparing INT4 against INT8 from generated CSVs, including a collated benchmark-style dashboard
- integration and repository-hygiene tests
- pairwise Givens rotation utilities (`quant/rotations.py`)
- ParoQuant-style top-width-difference channel-pair selection for independent sparse rotations
- per-channel scaling utilities (`quant/scaling.py`)
- grouped symmetric quantization utilities, including column-grouped and row-grouped paths
- rotation/scaling experiment comparing four INT4 preprocessing paths
- comparative sweep experiment comparing global, grouped, scaled, rotated, and combined INT4 paths
- living research-paper draft in `docs/research_draft.md`
- tracked paper figures in `docs/figures/`
- tests for all implemented modules
- Milestone 3 transformer harness (`experiments/transformer_experiment.py`): loads
  any HuggingFace causal LM, quantizes linear layers at INT4 and INT8 across
  global, row-grouped, scale+row-grouped, and top-width rotate+scale+row-grouped
  paths; measures weight reconstruction, activation drift, full-model logit/loss,
  and perplexity; writes three CSVs and a 4-panel dashboard with log-scale MSE
  panels and symmetric-log loss deltas;
  supports single-layer and all-layer modes;
  `delete_hf_cache_after=True` evicts the model after each run
- first all-layer `sshleifer/tiny-gpt2` transformer run: eight compatible
  layers, 196 weight records, 196 activation records, and 22 logit/loss records;
  top-5 overlap stayed 1.0 and perplexity ratios stayed within about six parts
  per million of 1.0 on the built-in calibration batch
- first all-layer `roneneldan/TinyStories-1M` transformer run: 48 compatible
  layers, 1008 weight records, 1008 activation records, and 14 all-layer
  logit/loss records; INT4 global had a 16.1x perplexity ratio on the built-in
  calibration batch, INT4 row-grouped g4 reduced the ratio to 1.21x, and capped
  top-width rotate+scale+row g4 reduced it to 1.14x; INT8 paths stayed close to
  the original model
- top-width rotation fractions are capped model-wide for transformer runs:
  requested p5/p10/p20 paths are lowered when needed so the widest selected layer
  stays within `max_rotation_pairs=1000`; for TinyStories this produced one
  common p3.0637% rotation path instead of skipping the 256-output MLP expansion
  layers

Resume reminder: `quant/rotations.py`, `quant/scaling.py`, grouped quantization (both column-grouped and row-grouped in `quant/quantizer.py`), `experiments/rotation_experiment.py`, and `experiments/sweep_experiment.py` are all complete. The sweep experiment compares 12 baseline quantization paths (global, col-grouped, row-grouped, scale, rotate, rotate+scale, rotate+scale+row-grouped) across a grid of seeds, outlier fractions, and outlier scales, writing `results/sweep_metrics.csv` and `plots/sweep_dashboard.png`. It can also opt into top-width sparse-rotation paths via `SweepConfig.top_width_pair_fractions`, e.g. `top_width_rotate_p10_global` and `top_width_rotate_scale_p10_row_g4`. Key findings from the historical sweeps: 32×32 sweep (45 cond, 12 methods) — row_grouped_g4 MSE ratio 0.112 (~9×); scale_global 0.531; rotation alone 0.902. 320×320 sweep (45 cond, 15 methods, new seeds/conditions) — row_grouped_g4 MSE ratio 0.143 (~7×); rotation adds zero measurable benefit over row-grouped at this scale; scale_global collapses to 0.845 (random scatter means every column has outliers); column-grouped converges toward global. New top-width p5/p10/p20 sweeps show sparse rotations improve global rotation paths, especially 320×320 rotate+scale_global (best p20 ratio 0.820 vs single-pair 0.844), but do not beat row-grouped quantization; row_grouped_g4 remains 0.112 on 32×32 and 0.143 on 320×320. Group size remains the dominant variable across both scales. Current Milestone 3 work is to run the implemented transformer harness across the remaining planned small-model benchmark set, then compare whether the TinyStories finding survives on Pythia-14M, Pythia-70M, and DistilGPT2.

## Environment

Repo root:

```bash
/home/mynk/PQ_project
```

Use the project virtual environment:

```bash
.venv/bin/python
```

Run tests with:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Current known passing test state:

```text
200 passed
```

Matplotlib note: use `MPLCONFIGDIR=/tmp/paroquant-mpl` because the default home config path may be read-only.

Git note: this folder is now a valid Git repo on branch `main`, tracking the private GitHub repository:

```text
https://github.com/md861/QuantizationTests
```

Generated artifacts under `plots/` and `results/`, the virtual environment, caches, and local tools are ignored by Git.

## Working Protocol For Future Agents

If another coding agent resumes this project, the safest order is:

1. Read this file first, then skim the latest entry in `lab_book/project_journey.md`.
2. Use `.venv/bin/python` for project commands and set `MPLCONFIGDIR=/tmp/paroquant-mpl` for Matplotlib-backed tests or scripts.
3. Treat `plots/` and `results/` as disposable generated artifacts unless the task explicitly says otherwise.
4. Keep the current branch clean before making unrelated changes; check `git status --short --branch` first.
5. Do not commit or push unless the user explicitly asks, or it is the agreed end-of-day checkpoint.
6. When commit or push is requested, use the existing private remote `origin` for `main` in `QuantizationTests`.
7. Keep docs in sync with code changes, especially this summary and the lab book, so handoff remains easy.
8. **Always update `README.md` before every commit and push.** The README is the public-facing entry point on GitHub and must never be stale. At minimum check: milestone statuses, progress table rows, current-milestone description, and the expected test count. Treat a stale README as a broken handoff.
9. **Keep the research draft current before every commit and push when the work changes the research story.** Update `docs/research_draft.md` with new findings, examples, caveats, and figure references. Copy any paper-ready figure resources into `docs/figures/` and commit them with the draft. Do not rely on ignored `plots/` artifacts for GitHub-visible paper figures.
10. **Always record rotation metadata for rotation experiments.** Any experiment, CSV, table, figure caption, or research-draft claim involving rotations must state the number of pair rotations applied per matrix/layer and the actual percentage/fraction of possible channel pairs used. For non-rotation baselines, record `rotation_count=0` and `rotation_pair_fraction=0.0`. For top-width sparse rotations, also record the configured candidate percentage as `rotation_candidate_fraction` and distinguish it from the actual independent rotation count/fraction.
11. **Handover Diagnostic shorthand.** When the user says "handover diagnostic", do this checklist: read `project_summary.md`, skim the latest `lab_book/project_journey.md` entry, check `git status --short --branch`, inspect recent commits with `git log --oneline -12`, verify the test suite, search docs for stale milestone/test-count/output references, then report what changed since the last session, current stale states, and the next recommended step.

Typical publish flow:

```bash
git status --short --branch
# update README.md, project_summary.md, lab_book/project_journey.md as needed
# update docs/research_draft.md and docs/figures/ when findings or paper figures change
git add <files>
git commit -m "<short message>"
git push origin main
```

## Implemented Modules

### `quant/matrix_factory.py`

Creates reproducible synthetic matrices for controlled quantization tests. These matrices are the input data used by the quantizer, metrics, and experiments.

- `gaussian_matrix(...)`
  - Creates independent Gaussian entries.
  - Formula: $x_{ij} \sim \mathcal{N}(\mu, \sigma^2)$
- `heavy_tailed_matrix(...)`
  - Creates Student-t entries for outlier-prone, heavy-tailed data.
  - Formula: $x_{ij} = s \cdot t_{\nu}$, where $\nu$ is the degrees of freedom and $s$ is a scale factor.
- `outlier_matrix(...)`
  - Creates a Gaussian base matrix, then replaces a controlled fraction of entries with large signed outliers.
  - Approximate rule: choose $k = \mathrm{round}(f \cdot mn)$ entries and set them near $\mu \pm |\mathcal{N}(\alpha\sigma, \sigma^2)|$, where $f$ is `outlier_fraction` and $\alpha$ is `outlier_scale`.
- `make_matrix(...)`
  - Dispatches by `MatrixKind` or string name for experiment-friendly matrix creation.
- `MatrixKind`

All generators support `shape`, `seed`, and `dtype`.

### `quant/quantizer.py`

Implements symmetric full-matrix quantization and stores both integer codes and dequantized reconstructions.

- `QuantizationResult`
  - Stores `quantized`, `dequantized`, `scale`, `bitwidth`, `qmin`, and `qmax`.
- `symmetric_quantize(matrix, bitwidth=...)`
  - Computes a single full-matrix scale from the maximum absolute value.
  - Scale: $s = \max(|W|) / (2^{b-1} - 1)$
  - Quantize: $Q = \mathrm{clip}(\mathrm{round}(W / s), q_{\min}, q_{\max})$
  - Dequantize: $\hat{W} = sQ$
- `quantize_int8(matrix)`
- `quantize_int4(matrix)`
- `grouped_symmetric_quantize(matrix, bitwidth=..., group_size=...)`
  - Quantizes contiguous **column** groups with one symmetric scale per group.
  - For group $g$, scale is $s_g = \max(|W_g|) / (2^{b-1}-1)$.
  - Dequantization uses $\hat{W}_g = s_g Q_g$.
- `quantize_int8_grouped(matrix, group_size=...)`
- `quantize_int4_grouped(matrix, group_size=...)`
- `row_grouped_symmetric_quantize(matrix, bitwidth=..., row_group_size=...)`
  - Quantizes contiguous **row** groups within each column independently (the GPTQ/AWQ approach).
  - Gives $n_{\mathrm{cols}} \times \lceil n_{\mathrm{rows}} / g \rceil$ scales in total.
  - Outliers in one row-group only inflate that group's scale; all other groups keep tight precision.
- `quantize_int8_row_grouped(matrix, row_group_size=...)`
- `quantize_int4_row_grouped(matrix, row_group_size=...)`

Ranges:

- INT8: `[-127, 127]`
- INT4: `[-7, 7]`, stored as NumPy `int8`

Zero matrices use `scale=1.0` and reconstruct exactly to zeros.

Column-grouped results store scales in `QuantizationResult.scales` (shape: `(n_col_groups,)`) and `group_size`.
Row-grouped results store scales in `QuantizationResult.scales` (shape: `(n_cols, n_row_groups)`) and `row_group_size`.
The scalar `scale` field is the mean of all group scales for summary compatibility.

### `quant/metrics.py`

Computes reconstruction, similarity, spectrum, and integer-code diagnostics comparing an original matrix $W$ with a dequantized reconstruction $\hat{W}$. Let $E = \hat{W} - W$.

- MSE: $\mathrm{mean}(E^2)$
- MAE: $\mathrm{mean}(|E|)$
- relative Frobenius error: $\|W - \hat{W}\|_F / \|W\|_F$
- cosine similarity: $\langle W, \hat{W} \rangle / (\|W\|_2\|\hat{W}\|_2)$ after flattening
- SNR in dB: $10\log_{10}(\sum W^2 / \sum (W-\hat{W})^2)$
- max absolute error: $\max(|E|)$
- mean signed error: $\mathrm{mean}(E)$
- error standard deviation: $\mathrm{std}(E)$
- spectrum L2 error: $\|\sigma(W) - \sigma(\hat{W})\|_2$
- relative spectrum L2 error: $\|\sigma(W) - \sigma(\hat{W})\|_2 / \|\sigma(W)\|_2$
- rank and stable-rank diagnostics
- optional saturation fraction: fraction of codes equal to $q_{\min}$ or $q_{\max}$
- optional zero fraction: fraction of codes equal to 0

Main API:

- `QuantizationMetrics`
- `compute_quantization_metrics(...)`

### `quant/spectrum.py`

Computes singular-value analysis for understanding how quantization changes matrix geometry. For a matrix $W$, singular values are $\sigma_1 \ge \sigma_2 \ge \dots$.

- `SpectrumStats`
  - Stores singular values, rank, spectral/nuclear/Frobenius norms, condition number, stable rank, and explained energy.
- `singular_values(...)`
  - Computes $\sigma(W)$ using SVD.
- `explained_energy(...)`
  - Computes cumulative squared singular-value energy.
  - Formula: $\mathrm{energy}_k = \sum_{i=1}^{k}\sigma_i^2 / \sum_i \sigma_i^2$
- `analyze_spectrum(...)`
  - Computes rank, norms, condition number, stable rank, and explained energy.
  - Stable rank: $\|W\|_F^2 / \|W\|_2^2 = \sum_i\sigma_i^2 / \sigma_1^2$
- `compare_spectra(...)`
  - Compares singular-value spectra and stable-rank changes between reference and candidate matrices.

### `quant/rotations.py`

Implements pairwise Givens channel rotations for outlier redistribution before quantization.

- `GivensRotation`
  - Frozen dataclass storing `i`, `j`, and `theta` for one rotation.
- `rotation_matrix(n, i, j, theta)`
  - Returns an $n \times n$ identity matrix with the $(i, j)$ subblock replaced by the Givens rotation:
    $R[i,i] = R[j,j] = \cos\theta$, $R[i,j] = -\sin\theta$, $R[j,i] = \sin\theta$.
  - Right-multiplying: $W' = W R$ rotates columns $i$ and $j$ of $W$.
- `apply_rotation(matrix, i, j, theta)`
  - Applies the rotation directly to columns $i$ and $j$ without constructing the full $n \times n$ matrix:
    $w_i' = \cos\theta \cdot w_i + \sin\theta \cdot w_j$,
    $w_j' = -\sin\theta \cdot w_i + \cos\theta \cdot w_j$.
  - Preserves input dtype; computes in float64 internally.
- `optimal_angle(matrix, i, j, *, n_search=360)`
  - Grid-searches $\theta \in [0, \pi)$ for the angle minimising $\max(|w_i'|_\infty, |w_j'|_\infty)$.
  - The objective is $\pi$-periodic, so $[0, \pi)$ covers the full search space.
- `rotate_channel_pair(matrix, i, j, *, n_search=360)`
  - Convenience wrapper returning `(rotated_matrix, theta)`.
- `apply_sequential_rotations(matrix, rotations)`
  - Applies a list of `GivensRotation` objects in order.

Key invariant: Givens rotations are orthogonal, so $\|W'\|_F = \|W\|_F$ exactly.

### `quant/scaling.py`

Implements reversible per-channel scaling for balancing column magnitudes before quantization.

- `ChannelScaling`
  - Frozen dataclass storing `factors` and `target_max_abs`.
- `column_max_abs(matrix)`
  - Returns $\max_i |W_{ij}|$ for each column $j$.
- `compute_channel_scaling(matrix, *, target_max_abs=None)`
  - Computes one positive factor per column.
  - Default target: mean of nonzero column max-abs values.
  - Formula for nonzero columns: $d_j = \tau / \max_i |W_{ij}|$, where $\tau$ is the target max-abs.
  - Zero columns receive factor 1.0.
- `apply_channel_scaling(matrix, scaling)`
  - Applies $W' = W D$, where $D = \mathrm{diag}(d_1,\dots,d_n)$.
- `invert_channel_scaling(matrix, scaling)`
  - Applies $W = W' D^{-1}$.
- `balance_channel_max_abs(matrix, *, target_max_abs=None)`
  - Convenience wrapper returning `(scaled_matrix, scaling)`.

Key invariant: scaling is exactly reversible up to floating-point rounding when the same `ChannelScaling` metadata is used for inversion.

### `quant/visualize.py`

Provides Matplotlib visualizations:

- `plot_matrix_heatmap(...)`
- `plot_matrix_grid(...)`
- `plot_singular_values(...)`
- `plot_value_histogram(...)`
- `plot_residual_histogram(...)`
- `plot_quantized_code_histogram(...)`
- `plot_spectrum_comparison(...)`
- `plot_quantization_summary(...)`
- `plot_quantization_comparison(...)`
- `plot_quantization_histograms(...)`
- `plot_channel_scaling_quantization_dashboard(...)`

Spectrum styling convention:

- `Original`: thicker dotted line
- quantized/dequantized curves: thinner solid lines

`plot_quantization_summary(...)` includes:

- original heatmap
- quantized-code heatmap
- dequantized heatmap
- residual heatmap
- spectrum comparison
- metrics panel with error and spectrum diagnostics
- data representation fields:
  - original dtype and bitwidth
  - quantized storage dtype and storage bitwidth

Default experiment plots use `plot_quantization_comparison(...)` to reduce plot count. Each comparison figure includes:

- original matrix
- one residual heatmap per quantization method
- singular-value spectra comparing original and dequantized data per method
- side-by-side metric summaries per method

`plot_channel_scaling_quantization_dashboard(...)` compares global quantization with per-channel scaled quantization. It includes:

- original matrix, global residual, and channel-scaled residual heatmaps
- original and scaled column max-abs bars
- residual max-abs per column
- singular-value spectra for original, global INT4, and channel-scaled INT4
- per-column MSE bars
- summary metrics for global and channel-scaled paths

### `experiments/baseline_experiment.py`

Runs the first end-to-end Milestone 1 experiment:

- generates Gaussian, heavy-tailed, and outlier matrices
- quantizes each with INT8 and INT4
- computes metrics
- writes `results/baseline_metrics.csv`
- writes three quantization comparison plots:
  - `plots/baseline_gaussian_comparison.png`
  - `plots/baseline_heavy_tailed_comparison.png`
  - `plots/baseline_outlier_comparison.png`

### `experiments/outlier_experiment.py`

Runs an outlier-severity sweep:

- sweeps controlled outlier fractions and outlier scales
- quantizes each matrix with INT8 and INT4
- computes the same reconstruction, spectrum, zero-fraction, and saturation diagnostics as baseline
- writes `results/outlier_metrics.csv`
- optionally writes one quantization comparison plot per fraction/scale condition:
  - `plots/outlier_fraction_<fraction>_scale_<scale>_comparison.png`

### `experiments/analyze_results.py`

Analyzes generated baseline and outlier CSVs:

- compares INT4 against INT8 for each experiment condition
- computes MSE ratios, relative-Frobenius ratios, SNR deltas, zero-fraction deltas, and saturation deltas
- prints a compact summary
- optionally writes:
  - `results/baseline_analysis.csv`
  - `results/outlier_analysis.csv`
- optionally writes a collated benchmark-style dashboard:
  - `plots/analysis_dashboard.png`

### `experiments/sweep_experiment.py`

Comparative sweep across all quantization paths and outlier conditions (Milestone 2).

- **Paths compared** (for a 32×32 matrix by default):
  - `global`: standard full-matrix INT4
  - `col_grouped_g{g}`: column-grouped INT4 for each configured group size
  - `row_grouped_g{g}`: row-grouped INT4 for each configured row group size
  - `scale_global`: per-channel scale then global INT4
  - `rotate_global`: pairwise Givens rotation then global INT4
  - `rotate_scale_global`: rotation + scaling then global INT4
  - `rotate_scale_row_g{g}`: rotation + scaling then row-grouped INT4
  - `top_width_rotate_p{pct}_global`: opt-in sparse independent rotations
    selected from the top percentage of channel width differences, then global INT4
  - `top_width_rotate_scale_p{pct}_global`: top-width rotations + scaling + global INT4
  - `top_width_rotate_scale_p{pct}_row_g{g}`: top-width rotations + scaling + row-grouped INT4
- **Condition grid**: seeds × outlier_fractions × outlier_scales (configurable via `SweepConfig`)
- **Outputs**: `results/sweep_metrics.csv` and `plots/sweep_dashboard.png`
- **Rotation metadata**: every CSV row includes `rotation_count`, actual `rotation_pair_fraction`, and configured `rotation_candidate_fraction`; non-rotation baselines use 0, 0.0, and 0.0
- **Aggregation**: summary tables use condition-wise MSE ratios (`method_mse / global_mse` on the same matrix), then report mean and standard deviation across sweep conditions. Standard deviation is spread across seeds/outlier severities, not a confidence interval.
- **Dashboard**: 4 panels — mean MSE ratio per method with std error bars, mean zero fraction per method with std error bars, MSE ratio vs outlier severity, effect of row group size
- **Sweep results — 32×32** (45 cond, 12 methods): row_grouped_g4=0.112, rotate_scale_row_g4=0.111, scale_global=0.531, col_grouped_g4=0.766, rotate_global=0.902. Column-grouped gives no improvement for row-localised outliers. Rotation adds small benefit through scaling.
- **Sweep results — 320×320** (45 cond, 15 methods, new seeds/conditions/group sizes): row_grouped_g4=0.143, rotate_scale_row_g4=0.143, scale_global=0.845, col_grouped_g4=0.898, rotate_global=0.965. At large scale with random scatter: rotation adds zero measurable benefit; per-channel scaling loses effectiveness because random scatter means every column has outliers; only row-grouped remains effective.
- **Top-width sparse sweep results**: p5/p10/p20 top-width rotations improve rotation-only global paths and 320×320 rotate+scale_global, but they do not improve the best row-grouped results. On 32×32, p20 rotation-only global improves to 0.811 vs single-pair rotate_global 0.902, while rotate_scale_row_g4 remains best at 0.111. On 320×320, p20 rotate+scale_global improves to 0.820 vs single-pair rotate_scale_global 0.844, but row_grouped_g4 and rotate_scale_row_g4 remain best at 0.143.

API:

- `SweepConfig` — dataclass with grid parameters and output settings; `csv_name` and `plot_name` fields allow multiple sweeps to coexist without overwriting; `top_width_pair_fractions` enables opt-in top-width sparse-rotation paths
- `SweepRecord` — frozen dataclass with metrics for one (condition, method) pair
- `run_sweep_experiment(config) -> list[SweepRecord]`
- `methods_in_config(config) -> list[str]`
- `print_summary(records) -> None`

Two sweeps have been run:
- 32×32, seeds 0–4, fractions [0.01,0.05,0.10], scales [5,10,20] → `results/sweep_metrics.csv`, `docs/figures/sweep_dashboard.png`
- 320×320, seeds 5–9, fractions [0.02,0.07,0.15], scales [7.5,15,30] → `results/sweep_metrics_320x320.csv`, `docs/figures/sweep_dashboard_320x320.png`
- 32×32 with top-width p5/p10/p20 rotations → `results/sweep_metrics_top_width_32x32.csv`, `docs/figures/sweep_dashboard_top_width_32x32.png`
- 320×320 with top-width p5/p10/p20 rotations → `results/sweep_metrics_top_width_320x320.csv`, `docs/figures/sweep_dashboard_top_width_320x320.png`

### `experiments/transformer_experiment.py`

Milestone 3 transformer quantization harness.

- **Config**: `TransformerConfig` — `model_name`, `calibration_texts`,
  `single_layer_name` (None = all layers), `bitwidths` ([4, 8] by default),
  `row_group_sizes` (fixed sizes applied to every layer, e.g. [4]),
  `row_group_fractions` (sizes relative to each layer's n_rows: 0.5 → n/2,
  0.25 → n/4, 0.0625 → n/16; computed as `max(1, round(n_rows × f))` then
  merged with fixed sizes and deduplicated, so a 64-row weight produces group
  sizes [4, 32, 16] while a 256-row weight produces [4, 128, 64, 16]),
  `top_width_pair_fractions` ([0.05, 0.10, 0.20]),
  `max_rotation_pairs` (safety cap for large models; transformer runs lower
  configured top-width fractions model-wide when needed so every selected layer
  can run the same capped rotation paths),
  `delete_hf_cache_after` (evict HF model cache after run).
- **Records**: `WeightRecord`, `ActivationRecord`, `LogitRecord` — each carries
  a `bitwidth` field (4 or 8) alongside `method`.
- **Weight experiment**: quantizes each linear layer weight at each configured
  bitwidth with four path families — global, row-grouped (all resolved group
  sizes), scale+row-grouped, top-width-rotate+scale+row-grouped — and computes
  MSE, relative Frobenius error, cosine similarity, SNR, zero fraction,
  saturation fraction, and rotation metadata per layer per (method, bitwidth).
- **Top-width cap policy**: before an all-layer run, the harness computes the
  widest selected layer's possible channel-pair count and lowers requested
  `top_width_pair_fractions` as needed so `round(total_pairs * fraction)` never
  exceeds `max_rotation_pairs`. Duplicate effective fractions are deduplicated.
  This keeps rotation method names common across layers for full-model logit/loss
  comparisons.
- **Activation experiment**: registers a forward hook to capture layer inputs once,
  then analytically computes the output drift for each quantized weight without
  re-running the full model. Measures activation MSE, cosine similarity, and
  relative error.
- **Logit/loss experiment**: temporarily swaps all selected layer weights per
  method, but regenerates one method's dequantized weights just-in-time instead
  of storing every layer/method reconstruction. This reduces temp disk pressure
  for larger Milestone 3 models such as Pythia while preserving the same output
  metrics: logit MSE, cosine similarity, top-5 token overlap, next-token loss
  delta, perplexity, original perplexity, and perplexity ratio.
- **Outputs**: `results/transformer_weight_metrics.csv`,
  `results/transformer_activation_metrics.csv`,
  `results/transformer_logit_metrics.csv`, and `plots/transformer_dashboard.png`.
- **HF Conv1D note**: GPT-2-style Conv1D stores weights as (in, out); nn.Linear
  stores (out, in). The harness normalises both to (in, out) internally via
  `_extract_weight` / `_set_weight`.
- **Hardware note**: `lm_head` is excluded from quantization (embedding-tied,
  not a typical quantization target). Set `max_rotation_pairs` (default 1000) to
  avoid slow top-width rotation on large weight matrices in bigger models.

### `experiments/rotation_experiment.py`

Runs the first Milestone 2 transformation experiment on one controlled outlier-heavy matrix.

Compared INT4 paths:

- `baseline`: $W \rightarrow \mathrm{INT4} \rightarrow \hat{W}$
- `rotation_only`: $W \rightarrow W R \rightarrow \mathrm{INT4} \rightarrow \widehat{WR} \rightarrow \widehat{W} = \widehat{WR} R^{-1}$
- `scaling_only`: $W \rightarrow W D \rightarrow \mathrm{INT4} \rightarrow \widehat{WD} \rightarrow \widehat{W} = \widehat{WD}D^{-1}$
- `rotation_scaling`: $W \rightarrow W R \rightarrow W R D \rightarrow \mathrm{INT4} \rightarrow \widehat{WRD} \rightarrow \widehat{W} = \widehat{WRD}D^{-1}R^{-1}$

The rotation pair is selected from the two columns with largest max-abs values, then `rotate_channel_pair(...)` chooses the angle that minimises the joint max-abs over that pair.

Default outputs:

- `results/rotation_metrics.csv`
- `plots/rotation_scaling_comparison.png`

The dashboard includes:

- transformed matrix heatmaps
- final residual heatmaps against the original matrix
- transformed column max-abs bars
- per-column MSE bars
- singular-value spectra against the original matrix
- summary metrics for all four paths

## Tests

Current test files:

- `tests/test_matrix_factory.py`
- `tests/test_quantizer.py`
- `tests/test_metrics.py`
- `tests/test_spectrum.py`
- `tests/test_visualize.py`
- `tests/test_baseline_experiment.py`
- `tests/test_outlier_experiment.py`
- `tests/test_analyze_results.py`
- `tests/test_rotations.py`
- `tests/test_scaling.py`
- `tests/test_rotation_experiment.py`
- `tests/test_sweep_experiment.py`
- `tests/test_integration.py`
- `tests/test_transformer_experiment.py`

Run all tests:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Current known passing test state:

```text
Focused transformer suite: 37 passed, 1 warning
Full suite before the streaming-logit refactor: 200 passed
```

## Design Conventions

- Prioritize clarity, reproducibility, and mathematical transparency over performance.
- Keep numerical logic separate from plotting:
  - generation: `matrix_factory.py`
  - quantization: `quantizer.py`
  - reconstruction metrics: `metrics.py`
  - singular-value analysis: `spectrum.py`
  - plotting: `visualize.py`
- Expose intermediate tensors and metadata for debugging.
- Use seeded NumPy generators for reproducibility.
- Keep APIs small and experiment-friendly.
- Update both this file and `lab_book/project_journey.md` as the project evolves.
- Update `docs/research_draft.md` when a result becomes part of the research story.
- Copy paper-ready figure resources into `docs/figures/` and commit them with the draft.

## Research Draft

The living paper-style draft is:

```text
docs/research_draft.md
```

It currently summarizes the matrix-level sandbox, INT8/INT4 examples, metrics, outlier failure modes, result-analysis dashboards, rotation/scaling tests, row-grouped quantization, top-width sparse rotation selection, and Milestone 2 sweep findings. Tracked paper figures live under `docs/figures/`. Keep claims cautious unless they are supported by sweeps or repeated evidence.

## Current Baseline Result

Run:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
```

## Current Outlier Sweep

Run:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
```

Key observation from the first run:

- INT8 preserves Gaussian, heavy-tailed, and outlier matrices well.
- INT4 degrades much more on heavy-tailed and outlier-heavy matrices.
- Heavy-tailed INT4 produced a very high zero fraction, matching the expected low-bit outlier-pressure failure mode.

## Next Recommended Step

The transformer harness is implemented and now uses streaming logit/loss
evaluation to avoid storing a full dequantized layer/method grid.

Temporary priority before continuing Milestone 3: harden the Pythia benchmark
workflow to mitigate VS Code/Codex disconnects in WSL2. Implement a dedicated
runner/preflight path before attempting more Pythia runs:

1. Add `experiments/run_transformer_benchmark.py` with conservative named presets
   such as `pythia-14m-int8-baseline`, `pythia-14m-int4-baseline`, and
   `pythia-70m-int8-baseline`.
2. Add `--download-only`, `--local-files-only`, and `--torch-threads` options so
   model download/cache population, offline cached runs, and CPU throttling are
   explicit.
3. Add checkpointing or incremental CSV append so partial long runs preserve
   completed records.
4. Document WSL guidance: use standalone WSL terminal or `tmux`, pre-download
   models, start with INT8/no-rotation presets, and clear Hugging Face cache only
   after successful runs.
5. After the runner is in place, run the conservative Pythia-14M baseline before
   the full default benchmark: `bitwidths=[8]`, `top_width_pair_fractions=[]`,
   `save_plots=False`, and `delete_hf_cache_after=False`.
6. If that succeeds, repeat Pythia-14M with `bitwidths=[4]` and rotations still
   disabled. Add capped rotations only after the baseline paths are stable.
7. Continue the remaining planned benchmark models one at a time:
   `EleutherAI/pythia-70m`, `distilgpt2`.
8. Add a larger held-out text batch for loss/perplexity evaluation.
9. Extend the Milestone 3 research section in `docs/research_draft.md` to compare whether
   the matrix-level findings (row-grouped dominates, scaling degrades with large
   models, rotation adds marginal benefit) survive on real weights.
10. Commit tracked figures to `docs/figures/` when the section is ready.

Acceptance check for Milestone 2 artifacts:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/rotation_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/sweep_experiment.py
```

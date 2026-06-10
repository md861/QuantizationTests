# Quantization Lab Project Summary

This is the compact handoff document for resuming work on the Quantization Lab research sandbox. For the full chronological history, see `lab_book/project_journey.md`.

## Current State

Milestone 1 is complete. Milestone 2 (ParoQuant Core) is underway: pairwise Givens rotations are implemented and tested.

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
- tests for all implemented modules

Resume reminder: `quant/rotations.py` is complete. Next is `quant/scaling.py` (per-channel scaling), then `experiments/rotation_experiment.py`.

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
72 passed
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

Typical publish flow:

```bash
git status --short --branch
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

Ranges:

- INT8: `[-127, 127]`
- INT4: `[-7, 7]`, stored as NumPy `int8`

Zero matrices use `scale=1.0` and reconstruct exactly to zeros.

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
- `tests/test_integration.py`

Run all tests:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Current known passing test state:

```text
103 passed
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

Start the first rotation/scaling experiment for Milestone 2.

Acceptance check:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
```

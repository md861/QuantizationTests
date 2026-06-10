# ParoQuant Project Summary

This is the compact handoff document for resuming work on the ParoQuant research sandbox. For the full chronological history, see `lab_book/project_journey.md`.

## Current State

The project is in Milestone 1: a matrix-level quantization sandbox for understanding symmetric INT8/INT4 quantization, reconstruction error, spectra, and visual distortions.

Implemented so far:

- synthetic matrix generation
- symmetric INT8 and INT4 quantization
- reconstruction and spectrum metrics
- heatmap, spectrum, and quantization-summary visualizations
- baseline experiment comparing INT8 and INT4 across matrix families
- outlier-severity sweep comparing INT8 and INT4 across controlled outlier fractions and scales
- tests for all implemented modules
- generated example plots in `plots/`

The next recommended implementation step is a small results-analysis helper or first rotation/scaling experiment.

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
59 passed
```

Matplotlib note: use `MPLCONFIGDIR=/tmp/paroquant-mpl` because the default home config path may be read-only.

Git note: Git is installed, but this folder is not currently a valid Git repo. An empty `.git/` directory exists.

## Implemented Modules

### `quant/matrix_factory.py`

Generates reproducible synthetic matrices:

- `gaussian_matrix(...)`
- `heavy_tailed_matrix(...)`
- `outlier_matrix(...)`
- `make_matrix(...)`
- `MatrixKind`

All generators support `shape`, `seed`, and `dtype`.

### `quant/quantizer.py`

Implements symmetric full-matrix quantization:

- `QuantizationResult`
- `symmetric_quantize(matrix, bitwidth=...)`
- `quantize_int8(matrix)`
- `quantize_int4(matrix)`

Ranges:

- INT8: `[-127, 127]`
- INT4: `[-7, 7]`, stored as NumPy `int8`

Zero matrices use `scale=1.0` and reconstruct exactly to zeros.

### `quant/metrics.py`

Computes error and diagnostic metrics comparing original vs dequantized matrices:

- MSE
- MAE
- relative Frobenius error
- cosine similarity
- SNR in dB
- max absolute error
- mean signed error
- error standard deviation
- spectrum L2 error
- relative spectrum L2 error
- rank and stable-rank diagnostics
- optional saturation fraction
- optional zero fraction

Main API:

- `QuantizationMetrics`
- `compute_quantization_metrics(...)`

### `quant/spectrum.py`

Computes singular-value analysis:

- `SpectrumStats`
- `singular_values(...)`
- `explained_energy(...)`
- `analyze_spectrum(...)`
- `compare_spectra(...)`

### `quant/visualize.py`

Provides Matplotlib visualizations:

- `plot_matrix_heatmap(...)`
- `plot_matrix_grid(...)`
- `plot_singular_values(...)`
- `plot_spectrum_comparison(...)`
- `plot_quantization_summary(...)`
- `plot_quantization_comparison(...)`

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

## Tests

Current test files:

- `tests/test_matrix_factory.py`
- `tests/test_quantizer.py`
- `tests/test_metrics.py`
- `tests/test_spectrum.py`
- `tests/test_visualize.py`
- `tests/test_baseline_experiment.py`
- `tests/test_outlier_experiment.py`

Run all tests:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

## Generated Example Plots

Matrix factory examples:

- `plots/matrix_factory_3x3_heatmaps.png`
- `plots/matrix_factory_3x3_spectra.png`
- `plots/matrix_factory_20x20_heatmaps.png`
- `plots/matrix_factory_20x20_spectra.png`
- `plots/matrix_factory_5x20_heatmaps.png`
- `plots/matrix_factory_5x20_spectra.png`

Quantization summary examples:

- `plots/outlier_5x20_int8_quantization_summary.png`
- `plots/outlier_5x20_int4_quantization_summary.png`

Baseline experiment artifacts:

- `results/baseline_metrics.csv`
- `plots/baseline_gaussian_comparison.png`
- `plots/baseline_heavy_tailed_comparison.png`
- `plots/baseline_outlier_comparison.png`

Outlier experiment artifacts:

- `results/outlier_metrics.csv`
- `plots/outlier_fraction_<fraction>_scale_<scale>_comparison.png`

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

Implement one of:

- `experiments/outlier_experiment.py`: sweep `outlier_fraction` and `outlier_scale` to quantify when INT4 starts failing.
- baseline-results analysis helper: read `results/baseline_metrics.csv` and produce compact comparison tables/plots.

Acceptance check:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
```

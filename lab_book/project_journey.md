# Quantization Lab Book

This lab book records the development journey for the ParoQuant research project from a blank workspace toward a working quantization sandbox. It is intended for future engineers or AI agents to understand what was done, why it was done, which commands were run, what failed, and how issues were resolved.

## 2026-05-19 — Project Bootstrap And Environment Setup

### Starting Context

- Workspace: `/home/mynk/PQ_project`
- Shell: `bash`
- Project brief source: `context.md`
- Goal: create an educational, research-oriented ParoQuant implementation, beginning with Milestone 1: a matrix-level quantization sandbox.

### 1. Confirmed `context.md` Exists

Command:

```bash
rg --files -g 'context.md' -g 'CONTEXT.md' -g 'Context.md'
```

Output:

```text
context.md
```

Result: `context.md` exists in the project root.

### 2. Read Project Context

Commands:

```bash
sed -n '1,240p' context.md
sed -n '241,520p' context.md
```

Key findings:

- Project is an educational/research implementation of ParoQuant-style quantization for LLMs.
- Milestone 1 is a quantization sandbox.
- Immediate modules requested:
  - `matrix_factory.py`
  - `quantizer.py`
  - `metrics.py`
  - `baseline_experiment.py`
  - `visualize.py`
- Design priorities:
  - clarity
  - modularity
  - reproducibility
  - mathematical transparency
  - minimal dependencies

### 3. Inspected Initial Workspace

Commands:

```bash
find . -maxdepth 3 -type f | sort
find . -maxdepth 3 -type d | sort
```

Observed files:

```text
./context.md
./paroquant_codex_context_md.md:Zone.Identifier
```

Observed directories:

```text
.
./.agents
./.codex
./.git
```

Important note: `.git/` existed but later proved to be empty and not a valid Git repository.

### 4. Created Initial Repository Scaffold

Command:

```bash
mkdir -p quant experiments notebooks plots results papers notes
```

Files added:

```text
quant/__init__.py
quant/matrix_factory.py
quant/quantizer.py
quant/metrics.py
quant/visualize.py
quant/rotations.py
quant/scaling.py
experiments/__init__.py
experiments/baseline_experiment.py
experiments/outlier_experiment.py
experiments/rotation_experiment.py
README.md
requirements.txt
main.py
```

File content policy:

- Python implementation files were created as lightweight stubs with module docstrings only.
- `README.md` describes the project goal and Milestone 1.
- `requirements.txt` contains:

```text
numpy
matplotlib
pytest
```

### 5. Verified Scaffold

Command:

```bash
find . -maxdepth 3 -type f | sort
```

Output:

```text
./README.md
./context.md
./experiments/__init__.py
./experiments/baseline_experiment.py
./experiments/outlier_experiment.py
./experiments/rotation_experiment.py
./main.py
./paroquant_codex_context_md.md:Zone.Identifier
./quant/__init__.py
./quant/matrix_factory.py
./quant/metrics.py
./quant/quantizer.py
./quant/rotations.py
./quant/scaling.py
./quant/visualize.py
./requirements.txt
```

Initial import check:

```bash
python -c "import quant; import experiments"
```

Output:

```text
/bin/bash: line 1: python: command not found
```

Resolution:

```bash
python3 -c "import quant; import experiments"
```

Output: no output, command succeeded.

Result: system uses `python3`, not `python`.

### 6. First Prerequisite Check

Commands and results:

```bash
python3 --version
```

```text
Python 3.10.6
```

```bash
python3 -m pip --version
```

```text
/usr/bin/python3: No module named pip
```

```bash
git --version
```

```text
git version 2.34.1
```

```bash
python3 -m venv --help
```

Result: `venv` command existed, but later failed because `ensurepip` support was missing.

Dependency checks:

```bash
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "import matplotlib; print(matplotlib.__version__)"
python3 -m pytest --version
python3 -c "import torch; print(torch.__version__)"
```

Outputs:

```text
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'matplotlib'
/usr/bin/python3: No module named pytest
ModuleNotFoundError: No module named 'torch'
```

Optional tooling checks:

```bash
python3 -m ruff --version
python3 -m mypy --version
python3 -m notebook --version
python3 -m ipykernel --version
```

Outputs:

```text
/usr/bin/python3: No module named ruff
/usr/bin/python3: No module named mypy
/usr/bin/python3: No module named notebook
/usr/bin/python3: No module named ipykernel
```

Virtual environment check:

```bash
python3 -m venv /tmp/paroquant-venv-check
```

Output:

```text
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.10-venv

You may need to use sudo with that command.  After installing the python3-venv
package, recreate your virtual environment.

Failing command: ['/tmp/paroquant-venv-check/bin/python3', '-Im', 'ensurepip', '--upgrade', '--default-pip']
```

Package manager checks:

```bash
apt --version
uv --version
conda --version
pipx --version
```

Outputs:

```text
apt 2.4.8 (amd64)
/bin/bash: line 1: uv: command not found
/bin/bash: line 1: conda: command not found
/bin/bash: line 1: pipx: command not found
```

Git repository check:

```bash
git status --short
```

Output:

```text
fatal: not a git repository (or any of the parent directories): .git
```

`.git` inspection:

```bash
ls -la .git
find .git -maxdepth 2 -type f | sort
```

Output:

```text
total 4
dr-xr-xr-x  2 mynk mynk   40 May 19 13:42 .
drwxr-xr-x 12 mynk mynk 4096 May 19 13:42 ..
```

Result: `.git/` exists but is empty; this is not a valid Git repository.

### 7. Attempted To Install Python Packaging Prerequisites

Recommended command:

```bash
sudo apt install -y python3-pip python3.10-venv
```

Attempt from the assistant execution environment failed because sudo could not prompt interactively:

```text
sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper
sudo: a password is required
```

Direct apt attempt also failed due to missing root privileges:

```bash
apt install -y python3-pip python3.10-venv
```

Output:

```text
E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
```

User then ran the install locally.

### 8. Resolved Apt 404 Errors

User reported apt install errors with multiple `404 Not Found` package URLs. Diagnosis: stale apt package lists.

Relevant environment check:

```bash
lsb_release -a
```

Output:

```text
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.2 LTS
Release:        22.04
Codename:       jammy
```

Recommended fix:

```bash
sudo apt update
sudo apt install -y python3-pip python3.10-venv
```

Fallback if needed:

```bash
sudo apt update --fix-missing
sudo apt install -y --fix-missing python3-pip python3.10-venv
```

User later confirmed success with:

```text
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
pip 22.0.2 from /tmp/venv-test/lib/python3.10/site-packages/pip (python 3.10)
```

### 9. Created Project Virtual Environment

Command:

```bash
python3 -m venv .venv
```

Result: succeeded.

### 10. Installed Project Dependencies

Initial command:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Initial output showed network/DNS failure:

```text
WARNING: The directory '/home/mynk/.cache/pip' or its parent directory is not owned or is not writable by the current user. The cache has been disabled.
WARNING: Retrying ... Failed to establish a new connection: [Errno -2] Name or service not known
ERROR: Could not find a version that satisfies the requirement numpy (from versions: none)
ERROR: No matching distribution found for numpy
```

Diagnosis: sandboxed execution blocked network/DNS access, not a package issue.

Retried with network approval:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Successful installs:

```text
Successfully installed contourpy-1.3.2 cycler-0.12.1 exceptiongroup-1.3.1 fonttools-4.63.0 iniconfig-2.3.0 kiwisolver-1.5.0 matplotlib-3.10.9 numpy-2.2.6 packaging-26.2 pillow-12.2.0 pluggy-1.6.0 pygments-2.20.0 pyparsing-3.3.2 pytest-9.0.3 python-dateutil-2.9.0.post0 six-1.17.0 tomli-2.4.1 typing-extensions-4.15.0
```

### 11. Verified Installed Dependencies

Command:

```bash
.venv/bin/python -c "import numpy, matplotlib, pytest; print('numpy', numpy.__version__); print('matplotlib', matplotlib.__version__); print('pytest', pytest.__version__)"
```

Output:

```text
mkdir -p failed for path /home/mynk/.config/matplotlib: [Errno 30] Read-only file system: '/home/mynk/.config/matplotlib'
Matplotlib created a temporary cache directory at /tmp/matplotlib-eqw4f8k5 because there was an issue with the default path (/home/mynk/.config/matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
numpy 2.2.6
matplotlib 3.10.9
pytest 9.0.3
```

Matplotlib warning fix:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -c "import matplotlib; print(matplotlib.__version__)"
```

Output:

```text
3.10.9
```

Project import check:

```bash
.venv/bin/python -c "import quant; import experiments; print('project imports ok')"
```

Output:

```text
project imports ok
```

Pytest version check:

```bash
.venv/bin/python -m pytest --version
```

Output:

```text
pytest 9.0.3
```

Pytest smoke run:

```bash
.venv/bin/python -m pytest
```

Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/mynk/PQ_project
collected 0 items

============================ no tests ran in 0.01s =============================
```

Note: exit code was `5`, which is normal for pytest when no tests are collected.

### 12. Re-ran Prerequisite Sweep

Commands:

```bash
python3 --version
.venv/bin/python --version
.venv/bin/python -m pip --version
git --version
```

Outputs:

```text
Python 3.10.12
Python 3.10.12
pip 22.0.2 from /home/mynk/PQ_project/.venv/lib/python3.10/site-packages/pip (python 3.10)
git version 2.34.1
```

Core package checks:

```bash
.venv/bin/python -c "import numpy; print(numpy.__version__)"
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -c "import matplotlib; print(matplotlib.__version__)"
.venv/bin/python -m pytest --version
.venv/bin/python -c "import quant; import experiments; print('imports ok')"
```

Outputs:

```text
2.2.6
3.10.9
pytest 9.0.3
imports ok
```

Optional tooling checks:

```bash
.venv/bin/python -m ruff --version
.venv/bin/python -m mypy --version
.venv/bin/python -c "import torch; print(torch.__version__)"
.venv/bin/python -m ipykernel --version
```

Outputs:

```text
/home/mynk/PQ_project/.venv/bin/python: No module named ruff
/home/mynk/PQ_project/.venv/bin/python: No module named mypy
ModuleNotFoundError: No module named 'torch'
/home/mynk/PQ_project/.venv/bin/python: No module named ipykernel
```

Virtual environment recheck:

```bash
python3 -m venv /tmp/paroquant-venv-recheck
```

Result: succeeded.

Git repo recheck:

```bash
git status --short
```

Output:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Current state:

- Milestone 1 prerequisites are ready.
- Optional tools are not installed yet:
  - `ruff`
  - `mypy`
  - `ipykernel`
  - `torch`
- This folder is still not a valid Git repository.

## Current Recommended Next Steps

1. Initialize Git cleanly if version control is desired.
2. Add a `.gitignore` to exclude `.venv/`, caches, generated plots, and results.
3. Implement Milestone 1 modules:
   - `quant/matrix_factory.py`
   - `quant/quantizer.py`
   - `quant/metrics.py`
   - `quant/visualize.py`
   - `experiments/baseline_experiment.py`
4. Add focused tests for quantizers, metrics, and matrix generation.
5. Consider adding optional dev dependencies later:
   - `ruff`
   - `mypy`
   - `ipykernel`
   - `torch`, when transformer integration begins.

## 2026-05-19 — Milestone 1: Matrix Factory Implementation

### Goal

Start Milestone 1 with `quant/matrix_factory.py`, because quantizers, metrics, and experiments all need reproducible synthetic matrices.

### Implementation Summary

Updated `quant/matrix_factory.py` from a docstring-only stub into a small typed API:

- `MatrixKind`: enum for supported matrix families.
- `gaussian_matrix(...)`: independent Gaussian entries.
- `heavy_tailed_matrix(...)`: scaled Student-t distribution.
- `outlier_matrix(...)`: Gaussian base matrix with controlled synthetic outliers.
- `make_matrix(...)`: dispatcher for experiment code.

Design choices:

- All generators accept a `seed` for reproducibility.
- All generators accept a `dtype`, defaulting to `np.float32`.
- Validation raises `ValueError` for invalid shapes or distribution parameters.
- Outlier matrices inject a rounded `outlier_fraction * matrix.size` count of entries.

### Tests Added

Created `tests/test_matrix_factory.py`.

Covered:

- reproducibility for seeded Gaussian and heavy-tailed matrices
- shape and dtype behavior
- outlier injection count
- string and enum dispatch via `make_matrix`
- invalid parameter validation

### Verification

Command:

```bash
.venv/bin/python -m pytest tests/test_matrix_factory.py
```

Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/mynk/PQ_project
collected 13 items

tests/test_matrix_factory.py .............                               [100%]

============================== 13 passed in 0.22s ==============================
```

### Manual 3x3 Matrix Factory Smoke Test

Command:

```bash
.venv/bin/python -c "from quant.matrix_factory import gaussian_matrix, heavy_tailed_matrix, outlier_matrix
import numpy as np
np.set_printoptions(precision=4, suppress=True)
print('Gaussian 3x3:')
print(gaussian_matrix((3, 3), seed=42))
print('\nHeavy-tailed 3x3:')
print(heavy_tailed_matrix((3, 3), seed=42))
print('\nOutlier 3x3:')
print(outlier_matrix((3, 3), outlier_fraction=0.33, outlier_scale=10.0, seed=42))"
```

Output:

```text
Gaussian 3x3:
[[ 0.3047 -1.04    0.7505]
 [ 0.9406 -1.951  -1.3022]
 [ 0.1278 -0.3162 -0.0168]]

Heavy-tailed 3x3:
[[ 0.1994  1.4187 -1.6188]
 [ 0.0723 -0.0164  0.8427]
 [ 0.1062  1.1922  0.6568]]

Outlier 3x3:
[[ 0.3047 -1.04    0.7505]
 [ 9.1407 11.1272 10.4675]
 [ 0.1278 -0.3162 -0.0168]]
```

Notes:

- The Gaussian and outlier matrices share the same base seed, so the non-outlier entries match.
- The outlier matrix used `outlier_fraction=0.33`; for 9 entries this rounds to 3 injected outliers.

## 2026-05-19 — Milestone 1: Initial Visualizer Implementation

### Goal

Use the three sample `3x3` matrices from the matrix factory as a first practical test for `quant/visualize.py`.

### Implementation Summary

Updated `quant/visualize.py` from a docstring-only stub into a small Matplotlib-based plotting module:

- `plot_matrix_heatmap(...)`: plots one 2D matrix as a heatmap.
- `plot_matrix_grid(...)`: plots multiple named matrices side by side and can save the figure to disk.

Design choices:

- The visualizer returns Matplotlib `Axes` or `Figure` objects so experiments can further customize plots.
- `plot_matrix_grid(...)` creates parent directories for saved output paths.
- Validation rejects non-2D arrays.

### Tests Added

Created `tests/test_visualize.py`.

Covered:

- single heatmap creation
- grid plot PNG saving
- empty grid validation
- non-2D input validation

### Verification

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/mynk/PQ_project
collected 17 items

tests/test_matrix_factory.py .............                               [ 76%]
tests/test_visualize.py ....                                             [100%]

============================== 17 passed in 1.20s ==============================
```

### Generated 3x3 Matrix Heatmap

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -c "from quant.matrix_factory import gaussian_matrix, heavy_tailed_matrix, outlier_matrix
from quant.visualize import plot_matrix_grid
import matplotlib.pyplot as plt
matrices = {
    'Gaussian': gaussian_matrix((3, 3), seed=42),
    'Heavy-tailed': heavy_tailed_matrix((3, 3), seed=42),
    'Outlier': outlier_matrix((3, 3), outlier_fraction=0.33, outlier_scale=10.0, seed=42),
}
fig = plot_matrix_grid(matrices, output_path='plots/matrix_factory_3x3_heatmaps.png', cmap='coolwarm')
plt.close(fig)
print('saved plots/matrix_factory_3x3_heatmaps.png')"
```

Output:

```text
saved plots/matrix_factory_3x3_heatmaps.png
```

File check:

```bash
ls -lh plots/matrix_factory_3x3_heatmaps.png
file plots/matrix_factory_3x3_heatmaps.png
```

Output:

```text
-rw-r--r-- 1 mynk mynk 36K May 19 14:06 plots/matrix_factory_3x3_heatmaps.png
plots/matrix_factory_3x3_heatmaps.png: PNG image data, 1895 x 567, 8-bit/color RGBA, non-interlaced
```

## 2026-05-19 — Milestone 1: Singular-Value Spectrum Analyzer

### Goal

Add a singular-value analyzer so matrices can be compared before and after quantization. This is useful for checking whether quantization changes matrix rank structure, dominant directions, condition number, or energy concentration.

### Implementation Summary

Created `quant/spectrum.py` with:

- `SpectrumStats`: dataclass for spectrum summary values.
- `singular_values(...)`: returns singular values sorted largest to smallest.
- `explained_energy(...)`: returns cumulative squared singular-value energy.
- `analyze_spectrum(...)`: computes singular values, rank, spectral norm, nuclear norm, Frobenius norm, condition number, stable rank, and explained energy.
- `compare_spectra(...)`: compares two matrices by rank, stable rank, absolute spectrum L2 error, and relative spectrum L2 error.

Extended `quant/visualize.py` with:

- `plot_singular_values(...)`
- `plot_spectrum_comparison(...)`

### Tests Added

Created `tests/test_spectrum.py` and extended `tests/test_visualize.py`.

Covered:

- SVD values against NumPy
- rank and norm summary statistics
- explained energy behavior
- spectrum comparison behavior
- validation for invalid inputs
- spectrum plot creation and PNG saving

### Verification

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/mynk/PQ_project
collected 25 items

tests/test_matrix_factory.py .............                               [ 52%]
tests/test_spectrum.py ......                                            [ 76%]
tests/test_visualize.py .....                                            [100%]

============================== 25 passed in 1.11s ==============================
```

### 3x3 Spectrum Smoke Test

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -c "from quant.matrix_factory import gaussian_matrix, heavy_tailed_matrix, outlier_matrix
from quant.spectrum import analyze_spectrum
from quant.visualize import plot_spectrum_comparison
import matplotlib.pyplot as plt
matrices = {
    'Gaussian': gaussian_matrix((3, 3), seed=42),
    'Heavy-tailed': heavy_tailed_matrix((3, 3), seed=42),
    'Outlier': outlier_matrix((3, 3), outlier_fraction=0.33, outlier_scale=10.0, seed=42),
}
for name, matrix in matrices.items():
    stats = analyze_spectrum(matrix)
    print(name)
    print('  singular_values:', [round(float(x), 4) for x in stats.singular_values])
    print('  rank:', stats.rank)
    print('  condition_number:', round(stats.condition_number, 4))
    print('  stable_rank:', round(stats.stable_rank, 4))
fig = plot_spectrum_comparison(matrices, output_path='plots/matrix_factory_3x3_spectra.png')
plt.close(fig)
print('saved plots/matrix_factory_3x3_spectra.png')"
```

Output:

```text
Gaussian
  singular_values: [2.6165, 1.1812, 0.0017]
  rank: 3
  condition_number: 1568.1135
  stable_rank: 1.2038
Heavy-tailed
  singular_values: [2.2734, 1.4411, 0.0871]
  rank: 3
  condition_number: 26.1037
  stable_rank: 1.4033
Outlier
  singular_values: [17.8034, 1.3437, 0.1604]
  rank: 3
  condition_number: 111.0145
  stable_rank: 1.0058
saved plots/matrix_factory_3x3_spectra.png
```

Generated plot:

```text
plots/matrix_factory_3x3_spectra.png: PNG image data, 1022 x 616, 8-bit/color RGBA, non-interlaced
```

### 20x20 Heatmap And Spectrum Examples

Generated larger examples for Gaussian, heavy-tailed, and outlier matrices using shape `(20, 20)` and `seed=42`.

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -c "from quant.matrix_factory import gaussian_matrix, heavy_tailed_matrix, outlier_matrix
from quant.spectrum import analyze_spectrum
from quant.visualize import plot_matrix_grid, plot_spectrum_comparison
import matplotlib.pyplot as plt
matrices = {
    'Gaussian 20x20': gaussian_matrix((20, 20), seed=42),
    'Heavy-tailed 20x20': heavy_tailed_matrix((20, 20), seed=42),
    'Outlier 20x20': outlier_matrix((20, 20), outlier_fraction=0.03, outlier_scale=10.0, seed=42),
}
heatmap_fig = plot_matrix_grid(matrices, output_path='plots/matrix_factory_20x20_heatmaps.png', cmap='coolwarm', figsize=(13.5, 4.2))
plt.close(heatmap_fig)
spectrum_fig = plot_spectrum_comparison(matrices, output_path='plots/matrix_factory_20x20_spectra.png', figsize=(7.0, 4.5))
plt.close(spectrum_fig)
for name, matrix in matrices.items():
    stats = analyze_spectrum(matrix)
    print(name)
    print('  top_5_singular_values:', [round(float(x), 4) for x in stats.singular_values[:5]])
    print('  rank:', stats.rank)
    print('  condition_number:', round(stats.condition_number, 4))
    print('  stable_rank:', round(stats.stable_rank, 4))
print('saved plots/matrix_factory_20x20_heatmaps.png')
print('saved plots/matrix_factory_20x20_spectra.png')"
```

Output:

```text
Gaussian 20x20
  top_5_singular_values: [7.7281, 7.0113, 6.4819, 6.2851, 5.7763]
  rank: 20
  condition_number: 103.4419
  stable_rank: 6.0594
Heavy-tailed 20x20
  top_5_singular_values: [25.482, 22.107, 18.3928, 14.986, 12.7619]
  rank: 20
  condition_number: 179.1357
  stable_rank: 3.8699
Outlier 20x20
  top_5_singular_values: [19.1629, 15.6759, 14.3909, 12.7316, 12.0352]
  rank: 20
  condition_number: 577.9024
  stable_rank: 4.1987
saved plots/matrix_factory_20x20_heatmaps.png
saved plots/matrix_factory_20x20_spectra.png
```

Generated files:

```text
plots/matrix_factory_20x20_heatmaps.png: PNG image data, 2141 x 632, 8-bit/color RGBA, non-interlaced
plots/matrix_factory_20x20_spectra.png:  PNG image data, 1101 x 696, 8-bit/color RGBA, non-interlaced
```

### 5x20 Heatmap And Spectrum Examples

Generated rectangular examples for Gaussian, heavy-tailed, and outlier matrices using shape `(5, 20)` and `seed=42`. This shape is useful as an early proxy for non-square weight matrices.

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -c "from quant.matrix_factory import gaussian_matrix, heavy_tailed_matrix, outlier_matrix
from quant.spectrum import analyze_spectrum
from quant.visualize import plot_matrix_grid, plot_spectrum_comparison
import matplotlib.pyplot as plt
matrices = {
    'Gaussian 5x20': gaussian_matrix((5, 20), seed=42),
    'Heavy-tailed 5x20': heavy_tailed_matrix((5, 20), seed=42),
    'Outlier 5x20': outlier_matrix((5, 20), outlier_fraction=0.05, outlier_scale=10.0, seed=42),
}
heatmap_fig = plot_matrix_grid(matrices, output_path='plots/matrix_factory_5x20_heatmaps.png', cmap='coolwarm', figsize=(13.5, 3.6))
plt.close(heatmap_fig)
spectrum_fig = plot_spectrum_comparison(matrices, output_path='plots/matrix_factory_5x20_spectra.png', figsize=(7.0, 4.5))
plt.close(spectrum_fig)
for name, matrix in matrices.items():
    stats = analyze_spectrum(matrix)
    print(name)
    print('  singular_values:', [round(float(x), 4) for x in stats.singular_values])
    print('  rank:', stats.rank)
    print('  condition_number:', round(stats.condition_number, 4))
    print('  stable_rank:', round(stats.stable_rank, 4))
print('saved plots/matrix_factory_5x20_heatmaps.png')
print('saved plots/matrix_factory_5x20_spectra.png')"
```

Output:

```text
Gaussian 5x20
  singular_values: [4.4209, 3.6035, 3.5281, 2.9997, 2.4487]
  rank: 5
  condition_number: 1.8054
  stable_rank: 3.0685
Heavy-tailed 5x20
  singular_values: [8.5703, 7.6359, 7.3644, 6.0016, 3.8874]
  rank: 5
  condition_number: 2.2047
  stable_rank: 3.2283
Outlier 5x20
  singular_values: [15.6561, 10.9266, 9.9997, 8.7712, 2.9899]
  rank: 5
  condition_number: 5.2364
  stable_rank: 2.2454
saved plots/matrix_factory_5x20_heatmaps.png
saved plots/matrix_factory_5x20_spectra.png
```

Generated files:

```text
plots/matrix_factory_5x20_heatmaps.png: PNG image data, 2142 x 546, 8-bit/color RGBA, non-interlaced
plots/matrix_factory_5x20_spectra.png:  PNG image data, 1101 x 696, 8-bit/color RGBA, non-interlaced
```

## 2026-05-19 — Milestone 1: Symmetric Quantizer Implementation

### Goal

Implement the baseline symmetric quantization method from `context.md`, starting with INT8 and INT4 matrix-level quantization.

### Implementation Summary

Updated `quant/quantizer.py` with:

- `QuantizationResult`: dataclass exposing quantized values, dequantized reconstruction, scale, bitwidth, and integer range.
- `symmetric_quantize(matrix, bitwidth=...)`: shared implementation for signed symmetric quantization.
- `quantize_int8(matrix)`: convenience wrapper using range `[-127, 127]`.
- `quantize_int4(matrix)`: convenience wrapper using range `[-7, 7]`, stored in NumPy `int8` because NumPy has no native `int4` dtype.

Important behavior:

- Uses one scale for the full matrix:
  `scale = max(abs(matrix)) / (2 ** (bitwidth - 1) - 1)`.
- Zero matrices use `scale=1.0` and reconstruct exactly to zeros.
- Inputs must be 2D floating-point NumPy arrays.

### Tests Added

Created `tests/test_quantizer.py`.

Covered:

- INT8 and INT4 integer ranges
- scale formula on a simple known matrix
- zero matrix behavior
- dequantized shape and dtype preservation
- INT8 reconstruction error less than or equal to INT4 on the same matrix
- invalid bitwidth, non-2D input, and integer input validation

### Verification

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/mynk/PQ_project
collected 34 items

tests/test_matrix_factory.py .............                               [ 38%]
tests/test_quantizer.py .........                                        [ 64%]
tests/test_spectrum.py ......                                            [ 82%]
tests/test_visualize.py .....                                            [100%]

============================== 34 passed in 1.17s ==============================
```

### 3x3 Quantization Smoke Test

Command:

```bash
.venv/bin/python -c "from quant.matrix_factory import gaussian_matrix
from quant.quantizer import quantize_int8, quantize_int4
import numpy as np
np.set_printoptions(precision=4, suppress=True)
matrix = gaussian_matrix((3, 3), seed=42)
for name, result in [('INT8', quantize_int8(matrix)), ('INT4', quantize_int4(matrix))]:
    mse = float(np.mean((matrix - result.dequantized) ** 2))
    print(name)
    print('  scale:', round(result.scale, 6))
    print('  qmin/qmax:', result.qmin, result.qmax)
    print('  quantized:')
    print(result.quantized)
    print('  dequantized:')
    print(result.dequantized)
    print('  mse:', round(mse, 8))"
```

Output:

```text
INT8
  scale: 0.015362
  qmin/qmax: -127 127
  quantized:
[[  20  -68   49]
 [  61 -127  -85]
 [   8  -21   -1]]
  dequantized:
[[ 0.3072 -1.0446  0.7528]
 [ 0.9371 -1.951  -1.3058]
 [ 0.1229 -0.3226 -0.0154]]
  mse: 1.396e-05
INT4
  scale: 0.278719
  qmin/qmax: -7 7
  quantized:
[[ 1 -4  3]
 [ 3 -7 -5]
 [ 0 -1  0]]
  dequantized:
[[ 0.2787 -1.1149  0.8362]
 [ 0.8362 -1.951  -1.3936]
 [ 0.     -0.2787  0.    ]]
  mse: 0.00565798
```

## 2026-05-19 — Milestone 1: Quantization Metrics Implementation

### Goal

Implement `quant/metrics.py` so quantized matrices can be evaluated with the required reconstruction metrics from `context.md` plus additional diagnostics useful for studying low-bit quantization behavior.

### Implementation Summary

Updated `quant/metrics.py` with:

- `QuantizationMetrics`: dataclass holding reconstruction, spectrum, and optional integer diagnostics.
- `mean_squared_error(...)`
- `mean_absolute_error(...)`
- `relative_frobenius_error(...)`
- `cosine_similarity(...)`
- `signal_to_noise_ratio_db(...)`
- `max_absolute_error(...)`
- `compute_quantization_metrics(...)`

`compute_quantization_metrics(...)` compares an original matrix with its dequantized reconstruction and returns:

- MSE
- MAE
- relative Frobenius error
- cosine similarity
- SNR in dB
- max absolute error
- mean signed error / bias
- error standard deviation
- singular-value spectrum L2 error
- relative spectrum L2 error
- original and reconstructed rank
- original and reconstructed stable rank
- stable rank change
- optional quantized zero fraction
- optional quantized saturation fraction

Conventions:

- Exact reconstruction has infinite SNR.
- Two all-zero matrices have cosine similarity `1.0`.
- A zero original with nonzero reconstruction has infinite relative Frobenius error and negative-infinite SNR.
- Saturation fraction is only computed when `quantized`, `qmin`, and `qmax` are provided.

### Tests Added

Created `tests/test_metrics.py`.

Covered:

- known-value MSE, MAE, and max absolute error
- relative Frobenius error
- cosine similarity
- exact-reconstruction SNR
- zero-matrix conventions
- optional quantized zero and saturation diagnostics
- spectrum diagnostic presence
- shape and type validation

### Verification

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/mynk/PQ_project
collected 44 items

tests/test_matrix_factory.py .............                               [ 29%]
tests/test_metrics.py ..........                                         [ 52%]
tests/test_quantizer.py .........                                        [ 72%]
tests/test_spectrum.py ......                                            [ 86%]
tests/test_visualize.py .....                                            [100%]

============================== 44 passed in 1.22s ==============================
```

### INT4 Metrics Smoke Test

Command:

```bash
.venv/bin/python -c "from dataclasses import asdict
from quant.matrix_factory import outlier_matrix
from quant.quantizer import quantize_int4
from quant.metrics import compute_quantization_metrics
matrix = outlier_matrix((5, 20), outlier_fraction=0.05, outlier_scale=10.0, seed=42)
result = quantize_int4(matrix)
metrics = compute_quantization_metrics(matrix, result.dequantized, quantized=result.quantized, qmin=result.qmin, qmax=result.qmax)
for key, value in asdict(metrics).items():
    if value is None:
        print(f'{key}: None')
    else:
        print(f'{key}: {value:.8f}')"
```

Output:

```text
mse: 0.22982830
mae: 0.42819295
relative_frobenius_error: 0.20435010
cosine_similarity: 0.97992444
snr_db: 13.79250271
max_abs_error: 0.77082354
mean_error: -0.06815278
error_std: 0.47453504
spectrum_l2_error: 1.28259719
relative_spectrum_l2_error: 0.05467180
reference_rank: 5.00000000
candidate_rank: 5.00000000
reference_stable_rank: 2.24537289
candidate_stable_rank: 2.25999452
stable_rank_change: 0.01462164
saturation_fraction: 0.03000000
zero_fraction: 0.62000000
```

## 2026-05-19 — Milestone 1: Quantization Summary Visualization

### Goal

Add a visualization that shows the original matrix, quantized integer codes, dequantized reconstruction, residual error, and metrics summary in one figure.

### Implementation Summary

Extended `quant/visualize.py` with:

- `plot_quantization_summary(...)`

The figure contains:

- original matrix heatmap
- quantized-code heatmap
- dequantized matrix heatmap
- residual heatmap
- text panel with quantizer settings and metrics

The function accepts a `QuantizationResult` and optionally precomputed `QuantizationMetrics`. If metrics are not provided, it computes them internally with `compute_quantization_metrics(...)`.

### Tests Added

Extended `tests/test_visualize.py`.

Covered:

- PNG saving for quantization summary plots
- shape validation between original, quantized, and dequantized matrices

### Verification

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/mynk/PQ_project
collected 46 items

tests/test_matrix_factory.py .............                               [ 28%]
tests/test_metrics.py ..........                                         [ 50%]
tests/test_quantizer.py .........                                        [ 69%]
tests/test_spectrum.py ......                                            [ 82%]
tests/test_visualize.py .......                                         [100%]

============================== 46 passed in 1.70s ==============================
```

### Example Plots

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -c "from quant.matrix_factory import outlier_matrix
from quant.quantizer import quantize_int8, quantize_int4
from quant.visualize import plot_quantization_summary
import matplotlib.pyplot as plt
matrix = outlier_matrix((5, 20), outlier_fraction=0.05, outlier_scale=10.0, seed=42)
for name, quantizer in [('int8', quantize_int8), ('int4', quantize_int4)]:
    result = quantizer(matrix)
    path = f'plots/outlier_5x20_{name}_quantization_summary.png'
    fig = plot_quantization_summary(matrix, result, output_path=path, title=f'Outlier 5x20 {name.upper()} Quantization Summary')
    plt.close(fig)
    print(f'saved {path}')"
```

Output:

```text
saved plots/outlier_5x20_int8_quantization_summary.png
saved plots/outlier_5x20_int4_quantization_summary.png
```

Generated files:

```text
plots/outlier_5x20_int8_quantization_summary.png: PNG image data, 2399 x 1046, 8-bit/color RGBA, non-interlaced
plots/outlier_5x20_int4_quantization_summary.png: PNG image data, 2398 x 1046, 8-bit/color RGBA, non-interlaced
```

### Spectrum Upgrade

Updated `plot_quantization_summary(...)` to include singular-value spectra directly in the summary figure.

The upgraded figure now contains:

- original matrix heatmap
- quantized-code heatmap
- dequantized matrix heatmap
- residual heatmap
- spectrum comparison for original, quantized codes, and dequantized matrix
- metrics panel with additional spectrum analysis:
  - relative spectrum error
  - stable rank change
  - rank before and after dequantization
  - stable rank for original, quantized-code, and dequantized matrices
  - top singular values for original, quantized-code, and dequantized matrices

Verification:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/mynk/PQ_project
collected 46 items

tests/test_matrix_factory.py .............                               [ 28%]
tests/test_metrics.py ..........                                         [ 50%]
tests/test_quantizer.py .........                                        [ 69%]
tests/test_spectrum.py ......                                            [ 82%]
tests/test_visualize.py .......                                         [100%]

============================== 46 passed in 1.97s ==============================
```

Regenerated example files:

```text
plots/outlier_5x20_int8_quantization_summary.png: PNG image data, 2856 x 1280, 8-bit/color RGBA, non-interlaced
plots/outlier_5x20_int4_quantization_summary.png: PNG image data, 2855 x 1280, 8-bit/color RGBA, non-interlaced
```

### Spectrum Styling Update

Updated `plot_spectrum_comparison(...)` so spectra are easier to read when curves overlap:

- `Original` uses a thicker dotted line.
- Quantized-code and dequantized spectra use thinner solid lines.
- This convention applies both to standalone spectrum plots and to the quantization summary spectrum subplot.

Verification:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================== 46 passed in 1.61s ==============================
```

Regenerated files using the new styling:

```text
plots/matrix_factory_3x3_spectra.png
plots/matrix_factory_20x20_spectra.png
plots/matrix_factory_5x20_spectra.png
plots/outlier_5x20_int8_quantization_summary.png
plots/outlier_5x20_int4_quantization_summary.png
```

## 2026-06-10 — Quantization Summary Data Bitwidth Update

### Goal

Make the quantization summary panel show the bitwidth of the original data, not only the quantizer bitwidth. This helps distinguish the original floating-point matrix representation from the low-bit quantization target.

### Implementation Summary

Updated `quant/visualize.py` so `plot_quantization_summary(...)` now includes a `Data` section in the summary panel:

- `original_dtype`
- `original_bits`
- `quantized_dtype`
- `storage_bits`

This makes the INT4 storage nuance explicit: INT4 uses a 4-bit quantization range, but the quantized code array is stored in NumPy `int8` because NumPy has no native `int4` dtype.

### Verification

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================== 46 passed in 1.82s ==============================
```

Regenerated files:

```text
plots/outlier_5x20_int8_quantization_summary.png: PNG image data, 2853 x 1343, 8-bit/color RGBA, non-interlaced
plots/outlier_5x20_int4_quantization_summary.png: PNG image data, 2873 x 1343, 8-bit/color RGBA, non-interlaced
```

## 2026-06-10 — Milestone 1: Baseline Experiment Implementation

### Goal

Implement the first end-to-end experiment tying together matrix generation, INT8/INT4 quantization, metrics, spectra, and summary plots.

### Implementation Summary

Updated `experiments/baseline_experiment.py` with:

- `BaselineConfig`
- `BaselineRecord`
- `run_baseline_experiment(...)`
- `print_summary(...)`
- CLI entrypoint via `main()`

Default experiment behavior:

- Generates `64x64` matrices with `seed=42`.
- Matrix families:
  - Gaussian
  - heavy-tailed Student-t
  - synthetic outlier
- Quantizers:
  - INT8
  - INT4
- Saves metrics to `results/baseline_metrics.csv`.
- Saves quantization summary plots to `plots/baseline_<matrix_kind>_<quantizer>.png`.

Added direct-script path handling so this works:

```bash
.venv/bin/python experiments/baseline_experiment.py
```

### Tests Added

Created `tests/test_baseline_experiment.py`.

Covered:

- experiment produces 6 records: 3 matrix families times 2 quantizers
- CSV writing
- optional plot suppression
- plot generation into temporary directories

### Verification

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================== 48 passed in 6.57s ==============================
```

### Experiment Run

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
```

Output:

```text
Baseline quantization experiment
matrix_kind, quantizer, mse, rel_frob, cosine, snr_db, zero_frac, sat_frac
gaussian, int8, 0.00006977, 0.00837273, 0.99996495, 41.5427, 0.014160, 0.000244
gaussian, int4, 0.02305315, 0.15219306, 0.98862548, 16.3521, 0.210449, 0.000244
heavy_tailed, int8, 0.01473174, 0.04021331, 0.99919261, 27.9126, 0.161621, 0.000244
heavy_tailed, int4, 2.02861545, 0.47189208, 0.88722913, 6.5231, 0.936523, 0.000244
outlier, int8, 0.00079497, 0.01990441, 0.99980196, 34.0210, 0.039062, 0.000244
outlier, int4, 0.25421618, 0.35594003, 0.94144610, 8.9725, 0.623047, 0.000488
Saved metrics to results/baseline_metrics.csv
Saved plots to plots/baseline_<matrix_kind>_<quantizer>.png
```

Generated files:

```text
results/baseline_metrics.csv
plots/baseline_gaussian_int8.png
plots/baseline_gaussian_int4.png
plots/baseline_heavy_tailed_int8.png
plots/baseline_heavy_tailed_int4.png
plots/baseline_outlier_int8.png
plots/baseline_outlier_int4.png
```

Initial observation:

- INT8 preserves all three matrix families well.
- INT4 degrades much more on heavy-tailed and outlier-heavy matrices.
- Heavy-tailed INT4 has very high zero fraction, matching the expected outlier-pressure failure mode.

## 2026-05-19 — Compact Project Summary Handoff

### Goal

Create a compact handoff document that another engineer or AI coding agent can read quickly to resume the project without walking through the full chronological lab book.

### Implementation Summary

Created `project_summary.md` at the repo root.

The summary includes:

- current project state
- environment and test commands
- implemented modules
- generated plot artifacts
- design conventions
- known environment notes
- next recommended implementation step

Maintenance convention:

- `lab_book/project_journey.md` remains the full chronological development record.
- `project_summary.md` should be updated as a compact current-state handoff whenever major features, commands, conventions, or next steps change.

## 2026-06-10 — Milestone 1: Outlier Severity Experiment

### Goal

Resume the project from the compact handoff and implement the recommended next experiment: a controlled sweep over outlier-heavy matrices to study how rare large values affect full-matrix symmetric INT8 and INT4 quantization.

### Starting Point

Read first, per resume instructions:

```bash
sed -n '1,220p' project_summary.md
sed -n '1,260p' lab_book/project_journey.md
```

Confirmed:

- Project is still in Milestone 1.
- `experiments/outlier_experiment.py` was only a docstring stub.
- The folder still is not a valid Git repository; `.git/` exists but is empty.

### Implementation Summary

Updated `experiments/outlier_experiment.py` with:

- `OutlierExperimentConfig`
- `OutlierRecord`
- `run_outlier_experiment(...)`
- `print_summary(...)`
- CLI entrypoint via `main()`

Default experiment behavior:

- Generates `64x64` outlier matrices with `seed=123`.
- Sweeps outlier fractions:
  - `0.001`
  - `0.005`
  - `0.01`
  - `0.02`
- Sweeps outlier scales:
  - `4.0`
  - `10.0`
  - `20.0`
- Quantizers:
  - INT8
  - INT4
- Saves metrics to `results/outlier_metrics.csv`.
- Saves quantization summary plots to `plots/outlier_fraction_<fraction>_scale_<scale>_<quantizer>.png`.

The output records include the usual reconstruction and spectrum diagnostics, plus:

- outlier fraction
- outlier scale
- outlier count
- quantization scale
- quantizer/code range metadata
- dtype/storage metadata

### Tests Added

Created `tests/test_outlier_experiment.py`.

Covered:

- experiment record count across fraction, scale, and quantizer combinations
- CSV writing
- optional plot suppression
- plot generation into temporary directories
- config validation
- sanity check that harsher outlier scale increases INT4 quantization scale and error for a fixed seed/fraction

### Verification

Focused test command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_outlier_experiment.py
```

Output:

```text
============================== 9 passed in 2.36s ===============================
```

### Handoff Update

Updated `project_summary.md` to include:

- outlier experiment module
- new test file
- outlier result and plot artifact names
- updated expected full-suite test count
- next recommended step: a small results-analysis helper or first rotation/scaling experiment

## 2026-06-10 — Default Experiment Plot Consolidation

### Goal

Reduce the number of plots generated by default in the baseline and outlier experiments. Instead of one plot per quantization method, each matrix condition should produce a single comparison figure containing:

- original matrix
- residual heatmap for each quantization method
- singular-value spectra with original and dequantized data for each method
- side-by-side summary columns for each method's calculated metrics

### Implementation Summary

Added `plot_quantization_comparison(...)` to `quant/visualize.py`.

The comparison helper:

- accepts one original matrix
- accepts a mapping of quantizer labels to `QuantizationResult`
- accepts optional precomputed metric mappings
- validates shapes and matching labels
- renders a three-row comparison figure:
  - top row: original matrix and residual heatmaps
  - middle row: per-method original-vs-dequantized spectra
  - bottom row: original metadata and per-method metrics

Updated `experiments/baseline_experiment.py`:

- now writes one comparison plot per matrix family
- default plot count changed from 6 to 3
- plot names are now:
  - `plots/baseline_gaussian_comparison.png`
  - `plots/baseline_heavy_tailed_comparison.png`
  - `plots/baseline_outlier_comparison.png`

Updated `experiments/outlier_experiment.py`:

- now writes one comparison plot per fraction/scale condition
- default plot count changed from 24 to 12
- plot names are now:
  - `plots/outlier_fraction_<fraction>_scale_<scale>_comparison.png`

CSV behavior is unchanged: each quantization method still gets its own metrics row. Related rows now share the same comparison `plot_path`.

### Tests Updated

Updated:

- `tests/test_visualize.py`
- `tests/test_baseline_experiment.py`
- `tests/test_outlier_experiment.py`

Covered:

- comparison helper saves a PNG
- comparison helper validates matching shapes
- baseline default plotting produces 3 unique comparison paths for 6 metric rows
- outlier default plotting produces one unique comparison path per fraction/scale condition

### Verification

Focused test command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_visualize.py tests/test_baseline_experiment.py tests/test_outlier_experiment.py
```

Output:

```text
============================== 21 passed in 6.14s ==============================
```

Full-suite command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================== 59 passed in 6.17s ==============================
```

## 2026-06-10 — Bookkeeping And Handoff Refresh

### Goal

Update stale project bookkeeping before continuing with implementation work.

### Changes

Updated `project_summary.md`:

- renamed the compact handoff from ParoQuant Project Summary to Quantization Lab Project Summary
- corrected the Git note to reflect the current valid Git repository
- recorded the private GitHub remote:
  - `https://github.com/md861/QuantizationTests`
- clarified that generated `plots/` and `results/` artifacts are local and ignored by Git
- replaced the outdated outlier-experiment next step with the remaining Milestone 1 polish items:
  - histogram visualization helpers
  - results-analysis helper
  - first rotation/scaling experiment as the next research step after Milestone 1 polish

Updated `main.py`:

- replaced the scaffold-era baseline placeholder
- now prints current test and experiment commands for the implemented Milestone 1 sandbox

Updated `lab_book/project_journey.md`:

- renamed the lab-book title to Quantization Lab Book
- preserved earlier historical notes, including the old invalid-Git state, as chronology rather than current status

## 2026-06-10 — Integration And Hygiene Test Layer

### Goal

Add a lightweight integration test layer to ensure new work keeps fitting into the stable Milestone 1 code, and add a guard against stale scaffold comments/placeholders returning to current-facing files.

### Implementation Summary

Created `tests/test_integration.py`.

Covered:

- direct end-to-end pipeline:
  - generate each matrix family
  - quantize with INT8 and INT4
  - compute quantization metrics
  - verify shape, dtype, and basic error contracts
- experiment-output contract:
  - run baseline and outlier experiments with tiny matrices
  - write CSVs to a temporary directory
  - verify expected rows and stable key columns
- repository hygiene:
  - scan current-facing source/docs for stale scaffold markers
  - intentionally exclude historical lab-book entries from this stale-marker check

### Verification

Focused test command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_integration.py
```

Output:

```text
============================== 3 passed in 0.58s ===============================
```

Full-suite command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================== 62 passed in 5.82s ==============================
```

## 2026-06-10 — Milestone 1 Polish: Histograms And Result Analysis

### Goal

Complete the remaining Milestone 1 polish items before moving to rotation/scaling work:

- histogram visualizations for value, residual, and quantized-code distributions
- a compact results-analysis helper for generated baseline and outlier CSVs

### Histogram Implementation

Updated `quant/visualize.py` with:

- `plot_value_histogram(...)`
- `plot_residual_histogram(...)`
- `plot_quantized_code_histogram(...)`
- `plot_quantization_histograms(...)`

The combined histogram figure compares:

- original matrix value distribution
- residual distribution for each quantization method
- integer code distribution for each quantization method

### Histogram Tests

Updated `tests/test_visualize.py`.

Focused command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_visualize.py
```

Output:

```text
============================== 16 passed in 3.48s ==============================
```

### Results Analysis Implementation

Created `experiments/analyze_results.py`.

The helper:

- reads `results/baseline_metrics.csv`
- reads `results/outlier_metrics.csv`
- compares INT4 against INT8 by condition
- computes MSE ratios, relative-Frobenius ratios, SNR deltas, zero-fraction deltas, and saturation deltas
- prints a compact summary
- writes optional analysis CSVs:
  - `results/baseline_analysis.csv`
  - `results/outlier_analysis.csv`

### Results Analysis Tests

Created `tests/test_analyze_results.py`.

Focused command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_analyze_results.py
```

Output:

```text
============================== 3 passed in 0.39s ===============================
```

### Analysis Run

Command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
```

Result:

- printed compact INT4-vs-INT8 comparisons
- identified the largest INT4/INT8 MSE ratio in the local generated results
- wrote ignored local analysis artifacts under `results/`

### Handoff Update

Updated `README.md` and `project_summary.md` to mark histogram visualizations and results analysis as complete Milestone 1 polish items. The next recommended implementation step is now the first rotation/scaling experiment for Milestone 2.

### Full Verification

Final pre-checkpoint command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================== 71 passed in 7.01s ==============================
```

### Resume Checkpoint

Milestone 1 polish is complete. On resume, the next planned research implementation is the first rotation/scaling experiment for Milestone 2.

Reminder for next resume: check the last implemented changes before moving on:

- histogram visualization helpers in `quant/visualize.py`
- results-analysis helper in `experiments/analyze_results.py`
- integration/hygiene tests in `tests/test_integration.py`

## 2026-06-10 — Benchmark-Style Result Analysis Visuals

### Goal

Extend `experiments/analyze_results.py` so the CSV analysis can also produce visuals similar to common quantization benchmark summaries.

### Implementation Summary

Updated `experiments/analyze_results.py` with:

- `plot_baseline_analysis_bars(...)`
- `plot_outlier_mse_ratio_heatmap(...)`
- `plot_outlier_zero_delta_heatmap(...)`
- generic `plot_outlier_metric_heatmap(...)`

Initial analysis plotting wrote separate files:

- `plots/baseline_analysis_bars.png`
- `plots/outlier_mse_ratio_heatmap.png`
- `plots/outlier_zero_delta_heatmap.png`

This was then consolidated so the default analysis now writes one review figure:

- `plots/analysis_dashboard.png`

The baseline bar chart shows:

- INT4 / INT8 MSE ratio
- INT4 zero-fraction increase
- INT4 SNR drop

The outlier heatmaps show fraction-by-scale sensitivity for:

- INT4 / INT8 MSE ratio
- INT4 zero-fraction increase

The collated dashboard places the baseline bar summaries and outlier heatmaps in one figure for easier review.

### Tests

Updated `tests/test_analyze_results.py`.

Covered:

- analysis plotting files are written and non-empty
- outlier heatmaps reject non-outlier records
- existing CSV analysis behavior remains intact

### Verification

Focused command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_analyze_results.py
```

Output:

```text
============================== 4 passed in 1.15s ===============================
```

Full-suite command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
============================== 72 passed in 7.71s ==============================
```

## 2026-06-10 — Project Summary Math Documentation Refresh

### Goal

Improve the compact handoff document so it explains what the core modules create and the mathematical quantities they compute, while keeping generated plot/result artifacts out of the summary.

### Changes

Updated `project_summary.md`:

- expanded `quant/matrix_factory.py` with concise distribution descriptions:
  - Gaussian matrices
  - Student-t heavy-tailed matrices
  - Gaussian-plus-outlier matrices
- expanded `quant/quantizer.py` with symmetric quantization formulae:
  - scale
  - integer code quantization
  - dequantization
- expanded `quant/metrics.py` with readable formulae for:
  - MSE
  - MAE
  - relative Frobenius error
  - cosine similarity
  - SNR
  - spectrum errors
  - zero and saturation fractions
- expanded `quant/spectrum.py` with SVD/singular-value descriptions:
  - explained energy
  - stable rank
  - spectrum comparison
- removed the generated example plots/artifacts section from the summary

This keeps `project_summary.md` focused on current source capabilities and resume guidance rather than local generated files.

## 2026-06-10 — Milestone 2: Pairwise Givens Rotation Module

### Goal

Begin Milestone 2 by implementing the core ParoQuant primitive: pairwise Givens rotations between weight-matrix columns, which redistribute outlier energy before quantization.

### Implementation Summary

Created `quant/rotations.py` with:

- `GivensRotation`
  - Frozen dataclass storing `i`, `j`, and `theta`.
- `rotation_matrix(n, i, j, theta)`
  - Returns an $n \times n$ Givens rotation matrix. Right-multiplying $W$ by this matrix rotates columns $i$ and $j$.
- `apply_rotation(matrix, i, j, theta)`
  - Applies the rotation directly to columns $i$ and $j$ without building the full $n \times n$ matrix.
  - Computes in float64 internally; preserves input dtype on return.
  - Does not mutate the input.
- `optimal_angle(matrix, i, j, *, n_search=360)`
  - Grid-searches $\theta \in [0, \pi)$ for the angle minimising $\max(\|w_i'\|_\infty, \|w_j'\|_\infty)$.
  - The cost function is $\pi$-periodic, so searching $[0, \pi)$ covers the full optimum.
- `rotate_channel_pair(matrix, i, j, *, n_search=360)`
  - Convenience wrapper returning `(rotated_matrix, theta)`.
- `apply_sequential_rotations(matrix, rotations)`
  - Applies a list of `GivensRotation` objects in order.

Key invariant: Givens rotations are orthogonal ($R^T R = I$), so the Frobenius norm of the matrix is exactly preserved.

### Tests Added

Created `tests/test_rotations.py` — 28 tests.

Covered:

- rotation matrix is identity at zero angle
- rotation matrix is orthogonal ($R^T R = I$)
- subblock values match $(\cos\theta, \pm\sin\theta)$
- rotation matrix leaves non-target rows and columns as identity
- `apply_rotation` at zero is identity
- `apply_rotation` preserves Frobenius norm
- `apply_rotation` is invertible (rotate by $\theta$ then $-\theta$ recovers original)
- `apply_rotation` only modifies target columns
- `apply_rotation` preserves dtype (float32 and float64)
- `apply_rotation` does not mutate input
- `apply_rotation` agrees numerically with `matrix @ rotation_matrix(...)`
- `optimal_angle` does not increase max-abs after rotation
- `optimal_angle` strictly improves a single-outlier column
- `optimal_angle` returns a float in $[0, \pi)$
- `rotate_channel_pair` result matches `apply_rotation` with the returned angle
- `apply_sequential_rotations` with empty list is identity
- `apply_sequential_rotations` chains correctly against manual composition
- `apply_sequential_rotations` preserves Frobenius norm
- validation errors for mismatched indices, out-of-range indices, non-2D input, integer dtype, invalid `n_search`

### Integration Tests Updated

Added two tests to `tests/test_integration.py`:

- `test_rotation_reduces_int4_error_on_outlier_matrix`: rotates the dominant outlier column pair on a controlled outlier matrix and verifies that relative Frobenius error and zero fraction both decrease after INT4 quantization.
- `test_sequential_rotations_preserve_frobenius_norm`: applies a sequence of rotations to a float32 matrix and checks the Frobenius norm is preserved within float32 rounding tolerance.

### Verification

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
103 passed in 8.03s
```

### Next Step

Implement `quant/scaling.py` — per-channel scaling to complement the rotations.

## 2026-06-10 — Handoff Protocol And Git Workflow Notes

### Goal

Make the continuation path explicit enough that another coding agent can resume work without needing oral context.

### Notes Added To The Handoff Docs

- `project_summary.md` now includes a short working protocol for future agents.
- The protocol tells a new agent to:
  - read `project_summary.md` first
  - skim the latest lab-book entry second
  - use `.venv/bin/python`
  - set `MPLCONFIGDIR=/tmp/paroquant-mpl` for Matplotlib-backed runs
  - treat `plots/` and `results/` as disposable generated artifacts
  - check `git status --short --branch` before changing anything
  - avoid commits and pushes unless the user explicitly asks or a checkpoint is due
  - use the existing private `origin` remote on `main` when a push is requested

### Takeaway

The docs are now better suited for a handoff to a different agent, because they explain both the project state and the local working conventions that have mattered during development.

## 2026-06-10 — Rotation Verification Tests And Worked Example

### Goal

Extend `tests/test_rotations.py` with three mathematically grounded verification tests that prove the rotation machinery is correct from first principles, not just self-consistent.  Also produce a concrete numerical worked example on a chosen matrix.

### Verification Tests Added

Four tests added to `tests/test_rotations.py` (bringing module tests from 28 to 32):

**`test_exact_angle_zeros_target_entry`**

Uses the analytic result that $\theta = \arctan2(b, a)$ applied to columns $i$ and $j$ of matrix $M$ zeros entry $M[k, j]$ exactly:

$$M'[k, j] = -\sin\theta \cdot a + \cos\theta \cdot b = 0 \quad \text{iff} \quad \tan\theta = b/a$$

Also verifies $|M'[k, i]| = \sqrt{a^2 + b^2}$ (the merged magnitude lands on column $i$).

**`test_givens_qr_via_cascaded_rotations`**

Implements Givens QR decomposition entirely through `apply_rotation`, using the transpose trick for row operations:

- Row rotation on $R$: `apply_rotation(R.T, i-1, i, theta).T` $\equiv G_\mathrm{step} @ R$
- $Q$ accumulation: `apply_rotation(Q, i-1, i, theta)` $\equiv Q @ G_\mathrm{step}^T$

At each step the angle is $\theta = \arctan2(R[i, j],\, R[i-1, j])$, which analytically zeros $R[i, j]$.  After sweeping all sub-diagonal positions the test asserts $\|Q R - A\|_F < 10^{-10}$, $\|Q^T Q - I\|_F < 10^{-10}$, and every lower-triangular entry satisfies $|R[r, c]| < 10^{-10}$.

**`test_givens_qr_diagonal_magnitudes_match_numpy`**

Checks that $|\mathrm{diag}(R)|$ from the Givens QR matches `numpy.linalg.qr` up to sign conventions.  Diagonal magnitudes are unique regardless of sign choice, so this provides an independent ground-truth comparison.

**`test_exact_angle_orthogonalises_column_pair`**

Uses the Jacobi SVD angle formula.  Setting dot product of rotated columns to zero:

$$\sin(2\theta)(\|c_j\|^2 - \|c_i\|^2) + 2\cos(2\theta)(c_i \cdot c_j) = 0$$

Solving: $\theta = \tfrac{1}{2}\arctan2(2 c_i \cdot c_j,\; \|c_i\|^2 - \|c_j\|^2)$

The test applies this angle and asserts $|c_i' \cdot c_j'| < 10^{-10}$.

### Concrete Worked Example (5×4 matrix, seed 77)

```text
Matrix A (5×4):
[[ 0.4278 -0.5708  2.6545 -1.6085]
 [ 0.6617 -0.1434 -0.3545  1.0664]
 [-1.8179 -0.9847 -0.1142  1.7413]
 [ 0.0890  0.8957 -1.8633 -1.2389]
 [ 0.9695 -0.6282 -0.0630  0.7309]]

Entry-zeroing  (row 2, col_i=0, col_j=3):
  a = -1.817922,  b = +1.741274
  theta = arctan2(b, a) = 2.3777 rad
  A'[2,3] = 2.22e-16  (zeroed)
  A'[2,0] = 2.517315 = sqrt(a²+b²)  ✓

Givens QR (10 row-rotations):
  ||Q @ R - A||_F   = 1.10e-15  ✓
  ||Q^T Q - I||_F   = 3.76e-16  ✓
  max |R lower tri| = 2.85e-16  ✓
  diag |R| ours  = [2.2076  1.5292  2.5027  1.3937]
  diag |R| numpy = [2.2076  1.5292  2.5027  1.3937]  ✓

Column orthogonalisation (8×8, cols 0 and 3, seed 11):
  dot before = +3.1906
  theta = 0.5 * arctan2(2*dot, ||ci||²-||cj||²) = 0.6658 rad
  dot after  = -7.22e-16  ✓
```

### Verification

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Output:

```text
107 passed in 8.35s
```

### Next Step

Implement `quant/scaling.py` — per-channel scaling to complement the rotations.

## 2026-06-10 — README Update Policy

### Decision

The user established a standing rule: **`README.md` must be updated before every commit and push to GitHub.** The README is the public entry point for the repository and must not be allowed to drift behind the code.

### What to check on every publish

- Milestone roadmap statuses (Complete / Active / Planned / Later)
- Progress table rows — add any newly completed areas, advance "Next" items
- "Current Milestone" description — reflect what is actively being built
- Expected test count in the "Run Tests" section

This rule is also recorded in `project_summary.md` as item 8 of the working protocol, so any agent reading the handoff docs will see it before making a commit.

## 2026-06-10 — Project Summary Stale Data Cleanup

### Goal

Remove stale current-facing handoff data after the project moved from Milestone 1 into Milestone 2.

### Changes

Updated `project_summary.md`:

- corrected the environment-section test count from `72 passed` to `107 passed`
- updated the next recommended step from the broad first rotation/scaling experiment to the more precise current path:
  - implement `quant/scaling.py`
  - then build `experiments/rotation_experiment.py` using the completed rotation utilities

### Check

Searched `README.md`, `project_summary.md`, and `lab_book/project_journey.md` for old test counts and milestone wording. Historical lab-book entries still retain their original test outputs because they are chronological records, not current-state claims.

## 2026-06-10 — Milestone 2: Per-Channel Scaling Module

### Goal

Implement the first version of `quant/scaling.py`, a reversible per-channel scaling primitive to complement pairwise Givens rotations before quantization experiments.

### Implementation

Created `quant/scaling.py` with:

- `ChannelScaling`
  - frozen dataclass storing per-column `factors` and `target_max_abs`
- `column_max_abs(matrix)`
  - computes $\max_i |W_{ij}|$ for every column
- `compute_channel_scaling(matrix, target_max_abs=None)`
  - computes positive per-column factors
  - default target is the mean of nonzero column max-abs values
  - formula for nonzero columns: $d_j = \tau / \max_i |W_{ij}|$
  - zero columns receive factor 1.0
- `apply_channel_scaling(matrix, scaling)`
  - applies $W' = W D$
- `invert_channel_scaling(matrix, scaling)`
  - applies $W = W' D^{-1}$
- `balance_channel_max_abs(matrix, target_max_abs=None)`
  - convenience wrapper returning `(scaled_matrix, scaling)`

Implementation conventions match the rotation module:

- accept only 2D floating-point arrays
- compute in float64 internally
- preserve input dtype on returned matrices
- avoid mutating inputs

### Tests

Added `tests/test_scaling.py` with coverage for:

- per-column max-abs computation
- balancing nonzero columns to a shared target
- custom target values
- scaling then inverse scaling recovering the original matrix
- dtype preservation for float32
- zero-matrix identity factors
- non-mutation of inputs
- validation errors for non-2D input, integer input, invalid targets, factor-count mismatch, and nonpositive factors

Updated `tests/test_integration.py` with a scaling integration check:

- balance an outlier-heavy matrix
- invert the scaling back to the original matrix
- run INT4 quantization on the scaled matrix
- verify the scaled column max-abs spread is smaller and metrics remain finite

### Documentation

Updated:

- `README.md`
  - marks per-channel scaling as complete
  - describes the scaling module in the Milestone 2 section
  - updates expected test count
- `project_summary.md`
  - records `quant/scaling.py` as implemented
  - adds formulae and API descriptions
  - updates the next step to `experiments/rotation_experiment.py`

### Verification

Focused command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_scaling.py tests/test_integration.py -q
```

Output:

```text
18 passed in 0.55s
```

Full-suite command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest -q
```

Output:

```text
120 passed in 8.25s
```

### Next Step

Build `experiments/rotation_experiment.py` to compare baseline quantization against rotation and scaling pre-processing.

## 2026-06-10 — Channel Scaling Visual Dashboard

### Goal

Add a visual test/dashboard for comparing global INT4 quantization against per-channel scaled INT4 quantization.

### Implementation

Added `plot_channel_scaling_quantization_dashboard(...)` to `quant/visualize.py`.

The dashboard compares:

- original matrix
- global INT4 residual
- channel-scaled INT4 residual after inverse scaling
- original column max-abs values
- scaled column max-abs values
- residual max-abs per column
- singular-value spectra
- per-column MSE
- summary metrics for both paths

The channel-scaled path is:

```text
W -> per-channel scale -> global INT4 quantize scaled W -> dequantize -> inverse scale
```

### Test

Added `test_channel_scaling_quantization_dashboard_saves_png` to `tests/test_scaling.py`.

The test uses a deliberately imbalanced matrix:

```text
matrix[:, 0] *= 50
matrix[:, 1] *= 20
matrix[:, 2] *= 10
```

It saves the review dashboard to:

```text
plots/channel_scaling_dashboard.png
```

### Verification

Focused command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_scaling.py -q
```

Output:

```text
13 passed in 2.28s
```

Full-suite command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest -q
```

Output:

```text
121 passed in 9.85s
```

## 2026-06-10 — Milestone 2: Rotation/Scaling Experiment

### Goal

Implement the first end-to-end Milestone 2 experiment comparing INT4 quantization with rotation and scaling preprocessing.

### Implementation

Created `experiments/rotation_experiment.py` with:

- `RotationExperimentConfig`
  - shape, seed, outlier severity, rotation search steps, output dirs, and plot toggle
- `RotationExperimentRecord`
  - one CSV row per method with transform metadata, quantizer metadata, reconstruction metrics, spectrum metrics, zero fraction, and saturation fraction
- `run_rotation_experiment(...)`
  - generates one controlled outlier-heavy matrix
  - selects the two columns with largest max-abs values for the rotation pair
  - runs four INT4 paths:
    - `baseline`
    - `rotation_only`
    - `scaling_only`
    - `rotation_scaling`
  - writes `results/rotation_metrics.csv`
  - optionally writes `plots/rotation_scaling_comparison.png`
- `plot_rotation_experiment_dashboard(...)`
  - creates the consolidated visual comparison

The four transformation paths are:

```text
baseline:
  W -> INT4 -> dequantized W_hat

rotation_only:
  W -> W R -> INT4 -> dequantized WR_hat -> WR_hat R^{-1}

scaling_only:
  W -> W D -> INT4 -> dequantized WD_hat -> WD_hat D^{-1}

rotation_scaling:
  W -> W R -> W R D -> INT4 -> dequantized WRD_hat -> WRD_hat D^{-1} R^{-1}
```

Metrics are always computed against the original matrix after inverting the preprocessing transforms.

### Dashboard

The default dashboard shows, for each method:

- transformed matrix heatmap
- final residual heatmap against the original matrix
- transformed column max-abs bars
- per-column MSE bars
- singular-value spectra against the original matrix

The figure also includes a compact metric summary for all four methods.

### Tests

Added `tests/test_rotation_experiment.py` with coverage for:

- metrics CSV creation without plots
- one dashboard plot creation with plots enabled
- all four methods are present
- scaling methods reduce transformed column max-abs spread
- config validation

### Verification

Focused command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest tests/test_rotation_experiment.py tests/test_scaling.py -q
```

Output:

```text
22 passed in 5.08s
```

Full-suite command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest -q
```

Output:

```text
130 passed in 12.33s
```

## 2026-06-10 — Living Research Draft Started

### Goal

Start a paper-style draft that documents the research story as it develops: motivations, examples, metrics, visuals, test cases, current findings, and limitations.

### Draft

Created:

```text
docs/research_draft.md
```

The draft currently includes:

- abstract and research motivation
- synthetic matrix families
- symmetric INT8/INT4 quantization equations
- metric definitions with formulas
- Gaussian INT8/INT4 worked example
- outlier-driven INT4 failure example
- baseline/outlier analysis dashboard discussion
- mathematical test cases as evidence
- global scaling vs channel-wise scaling visual discussion
- rotation/scaling experiment discussion
- current findings, limitations, next work, and reproduction commands

### Figures Generated For The Draft

Generated flat plot outputs:

```text
plots/research_matrix_families.png
plots/research_int4_gaussian_comparison.png
plots/research_outlier_histograms.png
plots/analysis_dashboard.png
```

The draft also references existing flat Milestone 2 figures:

```text
plots/channel_scaling_dashboard.png
plots/rotation_scaling_comparison.png
```

### Current Interpretation Captured

The draft deliberately uses cautious wording:

- INT4 is clearly more sensitive to outlier pressure than INT8 in the current sandbox.
- Per-channel scaling is strongly beneficial in the current controlled examples.
- Rotation + scaling is best in the first rotation/scaling run, but the margin over scaling alone is modest.
- A broader sweep is needed before making a general "best strategy" claim.

## 2026-06-10 — Tracked Paper Figures

### Goal

Make the research draft and its figures available on GitHub while keeping `plots/` as an ignored local artifact directory.

### Change

Created:

```text
docs/figures/
```

Copied the current paper figures from flat `plots/` outputs into tracked draft assets:

```text
docs/figures/research_matrix_families.png
docs/figures/research_int4_gaussian_comparison.png
docs/figures/research_outlier_histograms.png
docs/figures/analysis_dashboard.png
docs/figures/channel_scaling_dashboard.png
docs/figures/rotation_scaling_comparison.png
```

Updated `docs/research_draft.md` so all figure links point to `docs/figures/` via relative Markdown paths.

### Note

The project still treats `plots/` and `results/` as ignored generated artifacts. Paper-ready figures should be copied into `docs/figures/` when they become part of the tracked draft.

## 2026-06-10 — Research Draft Update Policy

### Decision

The user established a standing rule: when development work changes the research story, the research draft and its required resources must be updated before commit and push.

### What Future Agents Must Do

- Update `docs/research_draft.md` when a result, example, limitation, visual, or claim becomes part of the paper narrative.
- Copy paper-ready figure resources into `docs/figures/` and update the draft links to point there.
- Commit `docs/research_draft.md` and any needed `docs/figures/` files with the related code/docs changes.
- Do not rely on ignored `plots/` artifacts for GitHub-visible paper figures.
- Continue treating `plots/` and `results/` as local generated artifacts unless the user explicitly says otherwise.

### Handoff Update

This rule is now recorded in `project_summary.md` under the future-agent working protocol and in `README.md` under Project Notes.

## 2026-06-10 — Milestone 2: Grouped Quantization

### Goal

Implement grouped symmetric quantization as a viable Milestone 2 comparison path. Grouped quantization uses multiple local scales instead of one full-matrix scale, making it a more realistic baseline for later rotation/scaling comparisons.

### Implementation

Updated `quant/quantizer.py`:

- added optional `scales` and `group_size` metadata to `QuantizationResult`
- added `grouped_symmetric_quantize(matrix, bitwidth, group_size)`
- added `quantize_int8_grouped(matrix, group_size=...)`
- added `quantize_int4_grouped(matrix, group_size=...)`

For each contiguous column group $W_g$:

$$
s_g = \max(|W_g|)/(2^{b-1}-1)
$$

and:

$$
Q_g = \mathrm{clip}(\mathrm{round}(W_g/s_g), q_{\min}, q_{\max})
$$

with dequantization:

$$
\hat{W}_g = s_g Q_g.
$$

Zero groups use scale 1.0. The returned scalar `scale` field stores the mean group scale for summary compatibility; the actual group scales are in `scales`.

### Tests

Updated `tests/test_quantizer.py` with coverage for:

- grouped INT4 range and metadata
- grouped INT8 wrapper behavior
- last partial group handling
- zero-group scale handling
- equivalence to global quantization when `group_size == n_cols`
- reduced error when an outlier column is isolated
- invalid group size validation

Updated `tests/test_integration.py` to run grouped INT4 through the matrix-generation and metrics pipeline.

### Research Draft

Updated `docs/research_draft.md` with a grouped-quantization section, including formulas and a small outlier example comparing:

- global INT4
- grouped INT4 with group size 4
- column-wise INT4 with group size 1

Current example:

```text
Global INT4              MSE=0.297624  zero_frac=0.664062
Grouped INT4 (group=4)   MSE=0.275787  zero_frac=0.652344
Column INT4 (group=1)    MSE=0.089124  zero_frac=0.339844
```

### Next Step

Compare grouped quantization against rotation/scaling paths across seeds, outlier fractions, outlier scales, and group sizes.

### Verification

Full-suite command:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest -q
```

Output:

```text
138 passed in 13.25s
```

## 2026-06-10 — Handover Diagnostics And Stale Data Sweep

### Action

Performed handover diagnostics (scan all docs, verify test suite, check git log, flag drift).

### Stale data found and corrected

- `README.md` intro paragraph named only rotations and scaling; updated to also mention grouped quantization and rotation/scaling experiment.
- `project_summary.md` resume reminder said "likely grouped quantization" as a next step; grouped quantization is complete so updated to describe the actual next step (comparative sweeps).
- `memory/project_overview.md` was significantly out of date: still showed 72 tests, M2 as "NEXT", and rotations/scaling as stubs; fully refreshed to reflect 138 tests, M2 active, and all completed modules.

### Shorthand established

The user established **"handover diagnostics"** as a shorthand for this action: read `project_summary.md`, tail the lab book, check `git log`, verify the test suite, and flag any drift between docs and code state.

### Current clean state

- 138 tests passing
- All Milestone 2 core modules complete: rotations, per-channel scaling, grouped quantization, rotation/scaling experiment
- Research draft live at `docs/research_draft.md`
- Next: comparative sweeps across seeds, outlier fractions, scales, and group sizes

## 2026-06-10 — Milestone 2: Row-Grouped Quantization

### Motivation

The existing `grouped_symmetric_quantize` groups **columns** together, giving one scale per block of columns. This was identified as a weaker baseline than the industry standard: in GPTQ and AWQ, each column is split into **row groups**, each with its own scale. A row-level outlier in the column-grouped approach inflates the scale for the entire column block; in the row-grouped approach, only the row group containing the outlier is penalised, leaving all other row groups with tight, precise scales.

### Implementation

Added to `quant/quantizer.py`:

- `row_grouped_symmetric_quantize(matrix, bitwidth, row_group_size)`
  - Iterates over each column independently, splitting rows into groups of `row_group_size`.
  - One scale per group: $s_{c,g} = \max(|W_{c,g}|) / (2^{b-1}-1)$.
  - Returns `scales` as a 2-D array of shape `(n_cols, n_row_groups)` and `row_group_size` in metadata.
  - Zero groups use scale 1.0. Partial last groups handled correctly.
- `quantize_int8_row_grouped(matrix, row_group_size=...)`
- `quantize_int4_row_grouped(matrix, row_group_size=...)`

Also added optional `row_group_size` field to `QuantizationResult` (backwards compatible; defaults to `None`).

### Comparison of quantization strategies now available

| Strategy | Scales stored | Outlier isolation |
|---|---|---|
| Global (`symmetric_quantize`) | 1 | None — worst-case element sets scale for all |
| Column-grouped (`grouped_symmetric_quantize`) | $\lceil n_{\mathrm{cols}}/g \rceil$ | Column-block level |
| Row-grouped (`row_grouped_symmetric_quantize`) | $n_{\mathrm{cols}} \times \lceil n_{\mathrm{rows}}/g \rceil$ | Row-group within each column — tightest |

### Tests

Added 8 tests to `tests/test_quantizer.py`:

- range and metadata (scales shape, `row_group_size` field, `group_size` is None)
- INT8 wrapper shape
- partial last group scale values
- zero-group unit scale
- full-row-group equivalence
- strictly lower error than global when row outlier is isolated
- strictly lower error than column-grouped for a row-outlier scenario
- invalid `row_group_size` raises ValueError

### Verification

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest -q
```

Output:

```text
146 passed in 12.75s
```

### Next Step

Implement the comparative sweep experiment to compare all quantization paths across a grid of conditions.

## 2026-06-10 — Milestone 2 Complete: Comparative Sweep Experiment

### Motivation

With all quantization paths implemented (global, column-grouped, row-grouped, scale+global, rotate+global, rotate+scale+global, rotate+scale+row-grouped), the next step was to compare them systematically under controlled conditions: varying seeds, outlier fractions, outlier scales, and group sizes. A single-condition comparison risks cherry-picking; a grid sweep gives reproducible evidence.

### Implementation

Added `experiments/sweep_experiment.py` with:

- `SweepConfig` — dataclass controlling the condition grid and output paths
- `SweepRecord` — frozen dataclass storing metrics for one (condition, method) pair
- `run_sweep_experiment(config)` — outer loop over seeds × fractions × scales; inner dispatch over all methods; writes CSV and optional dashboard
- `_quantize_all_methods(matrix, config)` — applies all paths to one matrix; computes rotation metadata once and shares it across rotate+global, rotate+scale+global, and all rotate+scale+row-grouped variants
- `_plot_dashboard(records, config)` — 4-panel figure: (1) mean MSE ratio per method (horizontal bar, green < 1.0), (2) mean zero fraction per method, (3) MSE ratio vs outlier scale for key methods, (4) MSE ratio vs row_group_size
- `_write_csv` / `_save_figure` helpers; `methods_in_config` utility; `print_summary` terminal view

Added `tests/test_sweep_experiment.py` with 11 tests covering record count, metric validity, CSV output, dashboard file, global MSE correctness, method set coverage, MSE ratio baseline, and row-grouped superiority on row-outlier scenarios.

### Actual Sweep Results (45 conditions, 12 methods)

```
Method                                MSE ratio   Zero frac
------------------------------------------------------------
rotate_scale_row_g4                      0.1110      0.1365
row_grouped_g4                           0.1115      0.1359
rotate_scale_row_g8                      0.2157      0.2191
row_grouped_g8                           0.2192      0.2192
rotate_scale_row_g16                     0.3529      0.3106
row_grouped_g16                          0.3618      0.3120
rotate_scale_global                      0.5068      0.4017
scale_global                             0.5307      0.4101
col_grouped_g4                           0.7659      0.5187
col_grouped_g8                           0.8750      0.5602
rotate_global                            0.9021      0.5702
global                                   1.0000      0.5926
```

Key observations:
- row_grouped_g4 reduces MSE to ~11% of global INT4 on average (9× improvement)
- Rotation adds only marginal benefit on top of row-grouped alone (0.111 vs 0.112 at g=4)
- scale_global (0.53×) outperforms column-grouped at any group size
- Rotation alone (0.90×) barely moves the needle; its value comes through scaling
- Group size is the dominant variable: g=4 gives 9× improvement, g=16 gives only 3×

### Test State

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest -q
```

Output:

```text
157 passed in 12.89s
```

### Milestone 2 Status

All Milestone 2 modules are implemented and tested:

- `quant/rotations.py` — pairwise Givens rotations
- `quant/scaling.py` — per-channel scaling
- `quant/quantizer.py` — global, column-grouped, and row-grouped quantization
- `experiments/rotation_experiment.py` — single-condition rotation/scaling comparison
- `experiments/sweep_experiment.py` — full multi-condition comparative sweep

Milestone 2 is complete. Next is Milestone 3: apply the ParoQuant pipeline to a tiny transformer and measure perplexity and activation drift.

## 2026-06-11 — Large-Matrix Sweep (320×320) and SweepConfig Extension

### Motivation

The 32×32 sweep used seeds 0–4, outlier fractions [0.01, 0.05, 0.10], and scales [5, 10, 20]. A second sweep on 320×320 matrices with non-overlapping seeds and conditions tests whether findings hold at larger scale and with random-scatter outliers.

### Changes

Added `csv_name: str` and `plot_name: str` fields to `SweepConfig` (both default to the original filenames, so all existing tests pass unchanged). This allows multiple sweeps to coexist without overwriting each other's outputs.

### Sweep Configuration

- shape=(320, 320), seeds=[5,6,7,8,9], fractions=[0.02,0.07,0.15], scales=[7.5,15.0,30.0]
- row_group_sizes=[4,8,16,32], col_group_sizes=[4,8,16] → 15 methods, 45 conditions, 675 records
- csv_name="sweep_metrics_320x320.csv", plot_name="sweep_dashboard_320x320.png"

### Actual Results

```
Method                                MSE ratio   Zero frac
------------------------------------------------------------
row_grouped_g4                           0.1432      0.1874
rotate_scale_row_g4                      0.1432      0.1874
row_grouped_g8                           0.2738      0.3073
rotate_scale_row_g8                      0.2738      0.3073
row_grouped_g16                          0.4274      0.4318
rotate_scale_row_g16                     0.4275      0.4318
rotate_scale_row_g32                     0.5726      0.5357
row_grouped_g32                          0.5727      0.5359
rotate_scale_global                      0.8439      0.6851
scale_global                             0.8448      0.6857
col_grouped_g4                           0.8983      0.7002
col_grouped_g8                           0.9188      0.7061
col_grouped_g16                          0.9370      0.7114
rotate_global                            0.9650      0.7213
global                                   1.0000      0.7314
```

### Key New Findings

- Row-grouped still provides ~7× MSE improvement at 320×320, confirming the finding scales
- Rotation adds **zero** measurable benefit over row-grouped or scaling at this scale — differences at the 4th decimal place suggest the 32×32 rotation advantage may have been noise
- Per-channel scaling collapses from 0.531 (32×32) to 0.845 (320×320): with random scatter, almost every column has outliers, leaving no columns for scaling to balance
- Column-grouped converges toward global (0.90–0.94) at all group sizes with random scatter
- Group size remains the dominant variable for row-grouped across both scales

"""Integration and repository-hygiene checks for the quantization sandbox."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.baseline_experiment import BaselineConfig, run_baseline_experiment
from experiments.outlier_experiment import OutlierExperimentConfig, run_outlier_experiment
from quant.matrix_factory import MatrixKind, make_matrix, outlier_matrix
from quant.metrics import compute_quantization_metrics
from quant.quantizer import quantize_int4, quantize_int8
from quant.rotations import GivensRotation, apply_sequential_rotations, rotate_channel_pair


def test_quantization_pipeline_integrates_generation_quantizers_and_metrics() -> None:
    """Exercise the core Milestone 1 pipeline without experiment wrappers."""

    for matrix_kind in MatrixKind:
        matrix = make_matrix(matrix_kind, (10, 12), seed=31)
        int8_result = quantize_int8(matrix)
        int4_result = quantize_int4(matrix)

        int8_metrics = compute_quantization_metrics(
            matrix,
            int8_result.dequantized,
            quantized=int8_result.quantized,
            qmin=int8_result.qmin,
            qmax=int8_result.qmax,
        )
        int4_metrics = compute_quantization_metrics(
            matrix,
            int4_result.dequantized,
            quantized=int4_result.quantized,
            qmin=int4_result.qmin,
            qmax=int4_result.qmax,
        )

        assert int8_result.quantized.shape == matrix.shape
        assert int4_result.quantized.shape == matrix.shape
        assert int8_result.dequantized.dtype == matrix.dtype
        assert int4_result.dequantized.dtype == matrix.dtype
        assert int8_metrics.mse <= int4_metrics.mse
        assert np.isfinite(int8_metrics.relative_frobenius_error)
        assert np.isfinite(int4_metrics.relative_frobenius_error)


def test_experiment_outputs_share_expected_contracts(tmp_path) -> None:
    """Run both experiments together and check stable CSV/record contracts."""

    baseline_config = BaselineConfig(
        shape=(8, 8),
        seed=17,
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )
    outlier_config = OutlierExperimentConfig(
        shape=(8, 8),
        seed=17,
        outlier_fractions=(0.0, 0.125),
        outlier_scales=(4.0,),
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )

    baseline_records = run_baseline_experiment(baseline_config)
    outlier_records = run_outlier_experiment(outlier_config)

    assert len(baseline_records) == 6
    assert len(outlier_records) == 4
    assert {record.quantizer for record in baseline_records} == {"int8", "int4"}
    assert {record.quantizer for record in outlier_records} == {"int8", "int4"}

    baseline_rows = _read_csv_rows(tmp_path / "results" / "baseline_metrics.csv")
    outlier_rows = _read_csv_rows(tmp_path / "results" / "outlier_metrics.csv")

    assert len(baseline_rows) == len(baseline_records)
    assert len(outlier_rows) == len(outlier_records)
    assert {"matrix_kind", "quantizer", "mse", "plot_path"} <= set(baseline_rows[0])
    assert {
        "matrix_kind",
        "quantizer",
        "outlier_fraction",
        "outlier_scale",
        "mse",
        "plot_path",
    } <= set(outlier_rows[0])


def test_current_source_files_do_not_contain_stale_scaffold_markers() -> None:
    """Keep current-facing files from drifting back to scaffold placeholders."""

    stale_markers = (
        "ParoQuant Lab scaffold ready",
        "Next: implement experiments/baseline_experiment.py",
        "docstring-only stub",
        "TODO: implement",
        "not currently a valid Git repo",
        "empty `.git/` directory exists",
    )
    checked_paths = [
        Path("README.md"),
        Path("project_summary.md"),
        Path("main.py"),
        *Path("quant").glob("*.py"),
        *Path("experiments").glob("*.py"),
    ]

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for marker in stale_markers:
            assert marker not in text, f"{path} contains stale marker: {marker}"


def test_rotation_reduces_int4_error_on_outlier_matrix() -> None:
    """Rotating the worst channel pair before INT4 quantization reduces error."""

    matrix = outlier_matrix((32, 32), outlier_fraction=0.02, outlier_scale=10.0, seed=7)

    baseline = quantize_int4(matrix)
    baseline_metrics = compute_quantization_metrics(
        matrix,
        baseline.dequantized,
        quantized=baseline.quantized,
        qmin=baseline.qmin,
        qmax=baseline.qmax,
    )

    # Find and apply the single best rotation across all column pairs
    best_col = int(np.argmax(np.max(np.abs(matrix), axis=0)))
    second_col = int(np.argmax(
        np.max(np.abs(matrix), axis=0) * np.arange(matrix.shape[1] != best_col, dtype=float)
    ))
    # Simpler: pair the outlier column with its nearest neighbour by index
    pair_j = (best_col + 1) % matrix.shape[1]
    rotated, _ = rotate_channel_pair(matrix, best_col, pair_j)

    rotated_result = quantize_int4(rotated)
    rotated_metrics = compute_quantization_metrics(
        rotated,
        rotated_result.dequantized,
        quantized=rotated_result.quantized,
        qmin=rotated_result.qmin,
        qmax=rotated_result.qmax,
    )

    assert rotated_metrics.relative_frobenius_error < baseline_metrics.relative_frobenius_error
    assert rotated_metrics.zero_fraction < baseline_metrics.zero_fraction


def test_sequential_rotations_preserve_frobenius_norm() -> None:
    """Applying any sequence of Givens rotations must not change matrix energy."""

    rng = np.random.default_rng(31)
    matrix = rng.standard_normal((16, 16)).astype(np.float32)
    rotations = [
        GivensRotation(i=0, j=4, theta=0.6),
        GivensRotation(i=1, j=7, theta=1.2),
        GivensRotation(i=3, j=9, theta=0.3),
    ]
    result = apply_sequential_rotations(matrix, rotations)
    np.testing.assert_allclose(
        np.linalg.norm(matrix, "fro"),
        np.linalg.norm(result, "fro"),
        rtol=1e-5,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))

"""Integration and repository-hygiene checks for the quantization sandbox."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.baseline_experiment import BaselineConfig, run_baseline_experiment
from experiments.outlier_experiment import OutlierExperimentConfig, run_outlier_experiment
from quant.matrix_factory import MatrixKind, make_matrix
from quant.metrics import compute_quantization_metrics
from quant.quantizer import quantize_int4, quantize_int8


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))

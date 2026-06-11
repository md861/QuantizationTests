"""Tests for the Milestone 2 sweep experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.sweep_experiment import (
    SweepConfig,
    SweepRecord,
    methods_in_config,
    run_sweep_experiment,
)
from quant.matrix_factory import outlier_matrix
from quant.quantizer import quantize_int4


def _minimal_config(tmp_path: Path, *, save_plots: bool = False) -> SweepConfig:
    return SweepConfig(
        shape=(8, 8),
        seeds=[0, 1],
        outlier_fractions=[0.05],
        outlier_scales=[5.0, 10.0],
        row_group_sizes=[4],
        col_group_sizes=[4],
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=save_plots,
    )


def test_sweep_returns_expected_record_count(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    records = run_sweep_experiment(config)
    n_conditions = 2 * 1 * 2  # seeds × fractions × scales
    n_methods = len(methods_in_config(config))
    assert len(records) == n_conditions * n_methods


def test_all_records_have_valid_metrics(tmp_path: Path) -> None:
    records = run_sweep_experiment(_minimal_config(tmp_path))
    for r in records:
        assert math.isfinite(r.mse) and r.mse >= 0.0
        assert math.isfinite(r.relative_frobenius_error) and r.relative_frobenius_error >= 0.0
        assert 0.0 <= r.zero_fraction <= 1.0
        if r.saturation_fraction is not None:
            assert 0.0 <= r.saturation_fraction <= 1.0


def test_sweep_writes_csv(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    records = run_sweep_experiment(config)
    csv_path = config.results_dir / "sweep_metrics.csv"
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(records)
    assert "method" in rows[0]
    assert "rotation_count" in rows[0]
    assert "rotation_pair_fraction" in rows[0]
    assert "rotation_candidate_fraction" in rows[0]
    assert "mse" in rows[0]


def test_sweep_writes_dashboard_when_save_plots_true(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path, save_plots=True)
    run_sweep_experiment(config)
    assert (config.plots_dir / "sweep_dashboard.png").exists()


def test_global_method_mse_matches_direct_quantization(tmp_path: Path) -> None:
    config = SweepConfig(
        shape=(8, 8),
        seeds=[7],
        outlier_fractions=[0.05],
        outlier_scales=[10.0],
        row_group_sizes=[4],
        col_group_sizes=[4],
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )
    records = run_sweep_experiment(config)
    global_rec = next(r for r in records if r.method == "global")

    matrix = outlier_matrix((8, 8), outlier_fraction=0.05, outlier_scale=10.0, seed=7)
    direct = quantize_int4(matrix)
    direct_mse = float(np.mean((matrix - direct.dequantized) ** 2))
    assert abs(global_rec.mse - direct_mse) < 1e-7


def test_methods_in_config_matches_actual_method_names(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    records = run_sweep_experiment(config)
    expected = set(methods_in_config(config))
    actual = {r.method for r in records}
    assert expected == actual


def test_sweep_includes_top_width_rotation_paths(tmp_path: Path) -> None:
    config = SweepConfig(
        shape=(8, 8),
        seeds=[0],
        outlier_fractions=[0.05],
        outlier_scales=[10.0],
        row_group_sizes=[4],
        col_group_sizes=[],
        top_width_pair_fractions=[0.10, 0.25],
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )
    method_names = set(methods_in_config(config))
    assert "top_width_rotate_p10_global" in method_names
    assert "top_width_rotate_scale_p10_row_g4" in method_names
    assert "top_width_rotate_p25_global" in method_names

    records = run_sweep_experiment(config)
    actual = {r.method for r in records}
    assert method_names == actual


def test_sweep_records_rotation_metadata(tmp_path: Path) -> None:
    config = SweepConfig(
        shape=(8, 8),
        seeds=[0],
        outlier_fractions=[0.05],
        outlier_scales=[10.0],
        row_group_sizes=[4],
        col_group_sizes=[],
        top_width_pair_fractions=[0.25],
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )
    records = run_sweep_experiment(config)

    global_record = next(r for r in records if r.method == "global")
    assert global_record.rotation_count == 0
    assert global_record.rotation_pair_fraction == 0.0
    assert global_record.rotation_candidate_fraction == 0.0

    rotate_record = next(r for r in records if r.method == "rotate_global")
    assert rotate_record.rotation_count == 1
    assert rotate_record.rotation_pair_fraction == pytest.approx(1 / 28)
    assert rotate_record.rotation_candidate_fraction == 0.0

    top_width_record = next(
        r for r in records if r.method == "top_width_rotate_p25_global"
    )
    assert top_width_record.rotation_count > 0
    assert top_width_record.rotation_pair_fraction == pytest.approx(
        top_width_record.rotation_count / 28
    )
    assert top_width_record.rotation_candidate_fraction == pytest.approx(0.25)


def test_global_mse_ratio_is_one(tmp_path: Path) -> None:
    """Global method should have MSE ratio = 1.0 relative to itself."""
    from experiments.sweep_experiment import _mse_ratios
    records = run_sweep_experiment(_minimal_config(tmp_path))
    ratios = _mse_ratios(records)
    for ratio in ratios["global"]:
        assert abs(ratio - 1.0) < 1e-10


def test_row_grouped_outperforms_global_for_row_outlier(tmp_path: Path) -> None:
    """Row-grouped INT4 must have lower MSE than global INT4 on a row-outlier matrix."""
    config = SweepConfig(
        shape=(16, 16),
        seeds=[0],
        outlier_fractions=[0.0625],  # 1 row in 16
        outlier_scales=[20.0],
        row_group_sizes=[4],
        col_group_sizes=[],
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )
    records = run_sweep_experiment(config)
    global_mse = next(r.mse for r in records if r.method == "global")
    rg_mse = next(r.mse for r in records if r.method == "row_grouped_g4")
    assert rg_mse < global_mse


def test_rotate_scale_row_outperforms_global_for_row_outlier(tmp_path: Path) -> None:
    config = SweepConfig(
        shape=(16, 16),
        seeds=[0],
        outlier_fractions=[0.0625],
        outlier_scales=[20.0],
        row_group_sizes=[4],
        col_group_sizes=[],
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )
    records = run_sweep_experiment(config)
    global_mse = next(r.mse for r in records if r.method == "global")
    rot_mse = next(r.mse for r in records if r.method == "rotate_scale_row_g4")
    assert rot_mse < global_mse


def test_record_seeds_match_config(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    records = run_sweep_experiment(config)
    actual_seeds = {r.seed for r in records}
    assert actual_seeds == set(config.seeds)


def test_all_methods_have_same_condition_count(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    records = run_sweep_experiment(config)
    n_conditions = 2 * 1 * 2
    from collections import Counter
    counts = Counter(r.method for r in records)
    for method, count in counts.items():
        assert count == n_conditions, f"{method} has {count} records, expected {n_conditions}"

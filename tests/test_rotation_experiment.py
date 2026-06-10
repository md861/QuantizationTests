"""Tests for the rotation/scaling quantization experiment."""

import csv

import pytest

from experiments.rotation_experiment import (
    RotationExperimentConfig,
    run_rotation_experiment,
)


def test_rotation_experiment_writes_metrics_without_plots(tmp_path) -> None:
    config = RotationExperimentConfig(
        shape=(8, 8),
        seed=43,
        outlier_fraction=0.125,
        outlier_scale=8.0,
        rotation_search_steps=36,
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )

    records = run_rotation_experiment(config)

    assert len(records) == 4
    assert {record.method for record in records} == {
        "baseline",
        "rotation_only",
        "scaling_only",
        "rotation_scaling",
    }
    assert {record.quantizer for record in records} == {"int4"}
    assert all(record.shape == "8x8" for record in records)
    assert all(record.mse >= 0.0 for record in records)
    assert all(record.plot_path == "" for record in records)

    csv_path = config.results_dir / "rotation_metrics.csv"
    assert csv_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 4
    assert rows[0]["method"]
    assert rows[0]["relative_frobenius_error"]


def test_rotation_experiment_writes_dashboard_plot(tmp_path) -> None:
    config = RotationExperimentConfig(
        shape=(8, 8),
        seed=47,
        outlier_fraction=0.125,
        outlier_scale=8.0,
        rotation_search_steps=36,
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=True,
    )

    records = run_rotation_experiment(config)

    assert len({record.plot_path for record in records}) == 1
    for record in records:
        assert record.plot_path
        plot_path = tmp_path / record.plot_path
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


def test_scaling_methods_reduce_transformed_column_spread(tmp_path) -> None:
    config = RotationExperimentConfig(
        shape=(16, 16),
        seed=53,
        outlier_fraction=0.0625,
        outlier_scale=12.0,
        rotation_search_steps=72,
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )

    records = {record.method: record for record in run_rotation_experiment(config)}

    assert (
        records["scaling_only"].transform_column_max_abs_spread
        < records["baseline"].transform_column_max_abs_spread
    )
    assert (
        records["rotation_scaling"].transform_column_max_abs_spread
        < records["rotation_only"].transform_column_max_abs_spread
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (0, 8)}, "shape"),
        ({"shape": (8, 1)}, "at least two columns"),
        ({"base_std": 0.0}, "base_std"),
        ({"outlier_fraction": -0.1}, "between 0 and 1"),
        ({"outlier_scale": 0.0}, "outlier_scale"),
        ({"rotation_search_steps": 0}, "rotation_search_steps"),
    ],
)
def test_rotation_experiment_validates_config(tmp_path, kwargs, message: str) -> None:
    config = RotationExperimentConfig(
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
        **kwargs,
    )

    with pytest.raises(ValueError, match=message):
        run_rotation_experiment(config)

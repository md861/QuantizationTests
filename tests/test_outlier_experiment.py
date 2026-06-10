"""Tests for the outlier-severity quantization experiment."""

import csv

import pytest

from experiments.outlier_experiment import OutlierExperimentConfig, run_outlier_experiment


def test_outlier_experiment_writes_metrics_without_plots(tmp_path) -> None:
    config = OutlierExperimentConfig(
        shape=(8, 8),
        seed=7,
        outlier_fractions=(0.0, 0.125),
        outlier_scales=(4.0, 12.0),
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )

    records = run_outlier_experiment(config)

    assert len(records) == 8
    assert {record.matrix_kind for record in records} == {"outlier"}
    assert {record.quantizer for record in records} == {"int8", "int4"}
    assert {record.shape for record in records} == {"8x8"}
    assert {record.outlier_count for record in records} == {0, 8}
    assert all(record.mse >= 0.0 for record in records)
    assert all(record.plot_path == "" for record in records)

    csv_path = config.results_dir / "outlier_metrics.csv"
    assert csv_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 8
    assert rows[0]["outlier_fraction"]
    assert rows[0]["outlier_scale"]
    assert rows[0]["mse"]


def test_outlier_experiment_writes_summary_plots(tmp_path) -> None:
    config = OutlierExperimentConfig(
        shape=(6, 6),
        seed=11,
        outlier_fractions=(0.1,),
        outlier_scales=(8.0,),
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=True,
    )

    records = run_outlier_experiment(config)

    assert len(records) == 2
    assert len({record.plot_path for record in records}) == 1
    for record in records:
        assert record.plot_path
        plot_path = tmp_path / record.plot_path
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    assert len(list((tmp_path / "plots").glob("outlier_fraction_*_comparison.png"))) == 1


def test_larger_outlier_scale_increases_int4_error_and_quantization_scale(tmp_path) -> None:
    config = OutlierExperimentConfig(
        shape=(16, 16),
        seed=19,
        outlier_fractions=(0.1,),
        outlier_scales=(2.0, 20.0),
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )

    records = run_outlier_experiment(config)
    int4_records = sorted(
        (record for record in records if record.quantizer == "int4"),
        key=lambda record: record.outlier_scale,
    )

    assert int4_records[1].quantization_scale > int4_records[0].quantization_scale
    assert int4_records[1].mse > int4_records[0].mse


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (0, 8)}, "shape"),
        ({"base_std": 0.0}, "base_std"),
        ({"outlier_fractions": ()}, "outlier_fractions"),
        ({"outlier_fractions": (-0.1,)}, "between 0 and 1"),
        ({"outlier_scales": ()}, "outlier_scales"),
        ({"outlier_scales": (0.0,)}, "outlier_scales"),
    ],
)
def test_outlier_experiment_validates_config(tmp_path, kwargs, message: str) -> None:
    config = OutlierExperimentConfig(
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
        **kwargs,
    )

    with pytest.raises(ValueError, match=message):
        run_outlier_experiment(config)

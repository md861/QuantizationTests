"""Tests for the baseline quantization experiment."""

import csv

from experiments.baseline_experiment import BaselineConfig, run_baseline_experiment


def test_baseline_experiment_writes_metrics_without_plots(tmp_path) -> None:
    config = BaselineConfig(
        shape=(8, 8),
        seed=3,
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=False,
    )

    records = run_baseline_experiment(config)

    assert len(records) == 6
    assert {record.matrix_kind for record in records} == {
        "gaussian",
        "heavy_tailed",
        "outlier",
    }
    assert {record.quantizer for record in records} == {"int8", "int4"}
    assert all(record.shape == "8x8" for record in records)
    assert all(record.mse >= 0.0 for record in records)
    assert all(record.plot_path == "" for record in records)

    csv_path = config.results_dir / "baseline_metrics.csv"
    assert csv_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 6
    assert rows[0]["matrix_kind"]
    assert rows[0]["mse"]


def test_baseline_experiment_writes_summary_plots(tmp_path) -> None:
    config = BaselineConfig(
        shape=(6, 6),
        seed=5,
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        save_plots=True,
    )

    records = run_baseline_experiment(config)

    assert len(records) == 6
    assert len({record.plot_path for record in records}) == 3
    for record in records:
        assert record.plot_path
        plot_path = tmp_path / record.plot_path
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    assert len(list((tmp_path / "plots").glob("baseline_*_comparison.png"))) == 3

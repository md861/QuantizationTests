"""Tests for result-analysis helpers."""

import csv

import pytest

from experiments.analyze_results import (
    ResultsAnalysisConfig,
    analyze_baseline_metrics,
    run_results_analysis,
    worst_by_mse_ratio,
)
from experiments.baseline_experiment import BaselineConfig, run_baseline_experiment
from experiments.outlier_experiment import OutlierExperimentConfig, run_outlier_experiment


def test_run_results_analysis_writes_comparison_csvs(tmp_path) -> None:
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    run_baseline_experiment(
        BaselineConfig(
            shape=(8, 8),
            seed=23,
            results_dir=results_dir,
            plots_dir=plots_dir,
            save_plots=False,
        )
    )
    run_outlier_experiment(
        OutlierExperimentConfig(
            shape=(8, 8),
            seed=23,
            outlier_fractions=(0.0, 0.125),
            outlier_scales=(4.0,),
            results_dir=results_dir,
            plots_dir=plots_dir,
            save_plots=False,
        )
    )

    report = run_results_analysis(
        ResultsAnalysisConfig(
            results_dir=results_dir,
            output_dir=results_dir,
            write_csv=True,
        )
    )

    assert len(report.baseline) == 3
    assert len(report.outlier) == 2
    assert report.outlier[0].condition == "fraction=0.0, scale=4.0"
    assert all(record.mse_ratio_int4_over_int8 >= 1.0 for record in report.records)
    assert all(record.zero_fraction_delta_int4_minus_int8 >= 0.0 for record in report.records)

    baseline_analysis = results_dir / "baseline_analysis.csv"
    outlier_analysis = results_dir / "outlier_analysis.csv"
    assert baseline_analysis.exists()
    assert outlier_analysis.exists()

    with baseline_analysis.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 3
    assert rows[0]["experiment"] == "baseline"
    assert rows[0]["mse_ratio_int4_over_int8"]


def test_worst_by_mse_ratio_returns_largest_condition(tmp_path) -> None:
    results_dir = tmp_path / "results"
    run_baseline_experiment(
        BaselineConfig(
            shape=(8, 8),
            seed=29,
            results_dir=results_dir,
            plots_dir=tmp_path / "plots",
            save_plots=False,
        )
    )
    records = analyze_baseline_metrics(results_dir / "baseline_metrics.csv")

    worst = worst_by_mse_ratio(records)

    assert worst is not None
    assert worst.mse_ratio_int4_over_int8 == max(
        record.mse_ratio_int4_over_int8 for record in records
    )


def test_analyze_baseline_metrics_requires_int8_and_int4_rows(tmp_path) -> None:
    csv_path = tmp_path / "baseline_metrics.csv"
    csv_path.write_text(
        "\n".join(
            [
                (
                    "matrix_kind,quantizer,mse,relative_frobenius_error,snr_db,"
                    "zero_fraction,saturation_fraction"
                ),
                "gaussian,int8,0.1,0.2,30.0,0.1,0.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both int8 and int4"):
        analyze_baseline_metrics(csv_path)

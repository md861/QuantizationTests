"""Tests for result-analysis helpers."""

import csv

import matplotlib.pyplot as plt
import pytest

from experiments.analyze_results import (
    ResultsAnalysisConfig,
    analyze_baseline_metrics,
    plot_baseline_analysis_bars,
    plot_outlier_mse_ratio_heatmap,
    plot_outlier_zero_delta_heatmap,
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
            plots_dir=tmp_path / "analysis_plots",
            write_csv=True,
            save_plots=True,
        )
    )

    assert len(report.baseline) == 3
    assert len(report.outlier) == 2
    assert report.outlier[0].condition == "fraction=0.0, scale=4.0"
    assert all(record.mse_ratio_int4_over_int8 >= 1.0 for record in report.records)
    assert all(record.zero_fraction_delta_int4_minus_int8 >= 0.0 for record in report.records)

    baseline_analysis = results_dir / "baseline_analysis.csv"
    outlier_analysis = results_dir / "outlier_analysis.csv"
    dashboard_plot = tmp_path / "analysis_plots" / "analysis_dashboard.png"
    assert baseline_analysis.exists()
    assert outlier_analysis.exists()
    assert dashboard_plot.exists()
    assert dashboard_plot.stat().st_size > 0
    assert not (tmp_path / "analysis_plots" / "baseline_analysis_bars.png").exists()
    assert not (tmp_path / "analysis_plots" / "outlier_mse_ratio_heatmap.png").exists()
    assert not (tmp_path / "analysis_plots" / "outlier_zero_delta_heatmap.png").exists()

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


def test_outlier_heatmap_requires_outlier_sweep_records(tmp_path) -> None:
    results_dir = tmp_path / "results"
    run_baseline_experiment(
        BaselineConfig(
            shape=(8, 8),
            seed=31,
            results_dir=results_dir,
            plots_dir=tmp_path / "plots",
            save_plots=False,
        )
    )
    records = analyze_baseline_metrics(results_dir / "baseline_metrics.csv")

    with pytest.raises(ValueError, match="outlier_fraction"):
        plot_outlier_mse_ratio_heatmap(records)


def test_zero_fraction_plots_use_percentage_labels(tmp_path) -> None:
    results_dir = tmp_path / "results"
    run_baseline_experiment(
        BaselineConfig(
            shape=(8, 8),
            seed=37,
            results_dir=results_dir,
            plots_dir=tmp_path / "plots",
            save_plots=False,
        )
    )
    run_outlier_experiment(
        OutlierExperimentConfig(
            shape=(8, 8),
            seed=37,
            outlier_fractions=(0.0, 0.125),
            outlier_scales=(4.0,),
            results_dir=results_dir,
            plots_dir=tmp_path / "plots",
            save_plots=False,
        )
    )
    outlier_records = run_results_analysis(
        ResultsAnalysisConfig(
            results_dir=results_dir,
            output_dir=results_dir,
            plots_dir=tmp_path / "analysis_plots",
            write_csv=False,
            save_plots=False,
        )
    ).outlier

    bar_fig = plot_baseline_analysis_bars(
        analyze_baseline_metrics(results_dir / "baseline_metrics.csv"),
        output_path=tmp_path / "baseline.png",
    )
    heatmap_fig = plot_outlier_zero_delta_heatmap(
        outlier_records,
        output_path=tmp_path / "outlier_zero.png",
    )

    assert bar_fig.axes[1].get_ylabel() == "Percentage points (%)"
    assert heatmap_fig.axes[1].get_ylabel() == "Percentage points (%)"
    plt.close(bar_fig)
    plt.close(heatmap_fig)


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

"""Analyze baseline and outlier quantization experiment CSV outputs."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class ResultsAnalysisConfig:
    """Configuration for analyzing generated experiment CSVs."""

    results_dir: Path = Path("results")
    output_dir: Path = Path("results")
    plots_dir: Path = Path("plots")
    write_csv: bool = True
    save_plots: bool = True


@dataclass(frozen=True)
class QuantizerComparisonRecord:
    """Comparison between INT4 and INT8 for one experiment condition."""

    experiment: str
    condition: str
    matrix_kind: str
    outlier_fraction: float | None
    outlier_scale: float | None
    int8_mse: float
    int4_mse: float
    mse_ratio_int4_over_int8: float
    int8_relative_frobenius_error: float
    int4_relative_frobenius_error: float
    relative_frobenius_ratio_int4_over_int8: float
    int8_snr_db: float
    int4_snr_db: float
    snr_db_delta_int4_minus_int8: float
    int8_zero_fraction: float
    int4_zero_fraction: float
    zero_fraction_delta_int4_minus_int8: float
    int8_saturation_fraction: float
    int4_saturation_fraction: float
    saturation_fraction_delta_int4_minus_int8: float


@dataclass(frozen=True)
class ResultsAnalysisReport:
    """Combined analysis output for implemented Milestone 1 experiments."""

    baseline: list[QuantizerComparisonRecord]
    outlier: list[QuantizerComparisonRecord]

    @property
    def records(self) -> list[QuantizerComparisonRecord]:
        """Return all comparison records in report order."""

        return [*self.baseline, *self.outlier]


def run_results_analysis(
    config: ResultsAnalysisConfig = ResultsAnalysisConfig(),
) -> ResultsAnalysisReport:
    """Analyze baseline and outlier CSVs, optionally writing comparison CSVs."""

    baseline_path = config.results_dir / "baseline_metrics.csv"
    outlier_path = config.results_dir / "outlier_metrics.csv"

    baseline = analyze_baseline_metrics(baseline_path)
    outlier = analyze_outlier_metrics(outlier_path)
    report = ResultsAnalysisReport(baseline=baseline, outlier=outlier)

    if config.write_csv:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(config.output_dir / "baseline_analysis.csv", baseline)
        _write_csv(config.output_dir / "outlier_analysis.csv", outlier)

    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)
        figure = plot_results_analysis_dashboard(
            report,
            output_path=config.plots_dir / "analysis_dashboard.png",
        )
        plt.close(figure)

    return report


def analyze_baseline_metrics(path: Path) -> list[QuantizerComparisonRecord]:
    """Compare INT4 and INT8 rows in a baseline metrics CSV."""

    rows = _read_rows(path)
    return _compare_groups(
        rows,
        experiment="baseline",
        key_fields=("matrix_kind",),
    )


def analyze_outlier_metrics(path: Path) -> list[QuantizerComparisonRecord]:
    """Compare INT4 and INT8 rows in an outlier metrics CSV."""

    rows = _read_rows(path)
    return _compare_groups(
        rows,
        experiment="outlier",
        key_fields=("matrix_kind", "outlier_fraction", "outlier_scale"),
    )


def print_summary(report: ResultsAnalysisReport) -> None:
    """Print a compact analysis summary."""

    print("Quantization results analysis")
    print("experiment, condition, mse_ratio_int4_over_int8, zero_delta, snr_delta_db")
    for record in report.records:
        print(
            f"{record.experiment}, {record.condition}, "
            f"{record.mse_ratio_int4_over_int8:.4f}, "
            f"{record.zero_fraction_delta_int4_minus_int8:.6f}, "
            f"{record.snr_db_delta_int4_minus_int8:.4f}"
        )

    worst = worst_by_mse_ratio(report.records)
    if worst is not None:
        print(
            "Worst INT4/INT8 MSE ratio: "
            f"{worst.experiment} {worst.condition} "
            f"({worst.mse_ratio_int4_over_int8:.4f})"
        )


def worst_by_mse_ratio(
    records: list[QuantizerComparisonRecord],
) -> QuantizerComparisonRecord | None:
    """Return the condition with the largest INT4-over-INT8 MSE ratio."""

    if not records:
        return None
    return max(records, key=lambda record: record.mse_ratio_int4_over_int8)


def plot_results_analysis_dashboard(
    report: ResultsAnalysisReport,
    *,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (15.0, 10.0),
) -> Figure:
    """Plot a single dashboard for baseline and outlier analysis metrics."""

    if not report.baseline:
        raise ValueError("baseline records must contain at least one row")
    if not report.outlier:
        raise ValueError("outlier records must contain at least one row")

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.9, 1.1))
    fig.suptitle("Quantization Analysis Dashboard", fontsize=15)

    baseline_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    _plot_baseline_analysis_bars_on_axes(report.baseline, baseline_axes)

    mse_axis = fig.add_subplot(grid[1, 0:2])
    zero_axis = fig.add_subplot(grid[1, 2])
    _plot_outlier_metric_heatmap_on_axis(
        report.outlier,
        metric_name="mse_ratio_int4_over_int8",
        title="Outlier Sweep INT4 / INT8 MSE Ratio",
        colorbar_label="MSE ratio",
        ax=mse_axis,
        cmap="magma",
    )
    _plot_outlier_metric_heatmap_on_axis(
        report.outlier,
        metric_name="zero_fraction_delta_int4_minus_int8",
        title="INT4 Zero-Fraction Increase",
        colorbar_label="Percentage points (%)",
        ax=zero_axis,
        cmap="viridis",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save_figure(fig, output_path)
    return fig


def plot_baseline_analysis_bars(
    records: list[QuantizerComparisonRecord],
    *,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (13.0, 8.0),
) -> Figure:
    """Plot benchmark-style baseline comparison bars."""

    if not records:
        raise ValueError("records must contain at least one row")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Baseline INT4 vs INT8 Analysis", fontsize=14)
    _plot_baseline_analysis_bars_on_axes(records, list(axes))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save_figure(fig, output_path)
    return fig


def plot_outlier_mse_ratio_heatmap(
    records: list[QuantizerComparisonRecord],
    *,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
) -> Figure:
    """Plot outlier-sweep heatmap colored by INT4-over-INT8 MSE ratio."""

    return plot_outlier_metric_heatmap(
        records,
        metric_name="mse_ratio_int4_over_int8",
        title="Outlier Sweep INT4 / INT8 MSE Ratio",
        colorbar_label="MSE ratio",
        output_path=output_path,
        figsize=figsize,
        cmap="magma",
    )


def plot_outlier_zero_delta_heatmap(
    records: list[QuantizerComparisonRecord],
    *,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
) -> Figure:
    """Plot outlier-sweep heatmap colored by INT4 zero-fraction increase."""

    return plot_outlier_metric_heatmap(
        records,
        metric_name="zero_fraction_delta_int4_minus_int8",
        title="Outlier Sweep INT4 Zero-Fraction Increase",
        colorbar_label="Percentage points (%)",
        output_path=output_path,
        figsize=figsize,
        cmap="viridis",
    )


def plot_outlier_metric_heatmap(
    records: list[QuantizerComparisonRecord],
    *,
    metric_name: str,
    title: str,
    colorbar_label: str,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
    cmap: str = "viridis",
) -> Figure:
    """Plot a fraction-by-scale heatmap for an outlier analysis metric."""

    if not records:
        raise ValueError("records must contain at least one row")
    if not all(
        record.outlier_fraction is not None and record.outlier_scale is not None
        for record in records
    ):
        raise ValueError("all records must include outlier_fraction and outlier_scale")
    if not hasattr(records[0], metric_name):
        raise ValueError(f"unknown metric: {metric_name}")

    fig, ax = plt.subplots(figsize=figsize)
    _plot_outlier_metric_heatmap_on_axis(
        records,
        metric_name=metric_name,
        title=title,
        colorbar_label=colorbar_label,
        ax=ax,
        cmap=cmap,
    )
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def main() -> None:
    """Run the default results analysis."""

    report = run_results_analysis()
    print_summary(report)
    print("Saved analysis to results/baseline_analysis.csv")
    print("Saved analysis to results/outlier_analysis.csv")
    print("Saved plot to plots/analysis_dashboard.png")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"metrics CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"metrics CSV contains no rows: {path}")
    return rows


def _compare_groups(
    rows: list[dict[str, str]],
    *,
    experiment: str,
    key_fields: tuple[str, ...],
) -> list[QuantizerComparisonRecord]:
    groups: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, {})[row["quantizer"]] = row

    records = []
    for key in sorted(groups, key=_group_sort_key):
        quantizer_rows = groups[key]
        if "int8" not in quantizer_rows or "int4" not in quantizer_rows:
            raise ValueError(f"condition {key} must include both int8 and int4 rows")
        records.append(
            _make_comparison_record(
                experiment=experiment,
                condition=_condition_label(key_fields, key),
                int8_row=quantizer_rows["int8"],
                int4_row=quantizer_rows["int4"],
            )
        )
    return records


def _make_comparison_record(
    *,
    experiment: str,
    condition: str,
    int8_row: dict[str, str],
    int4_row: dict[str, str],
) -> QuantizerComparisonRecord:
    int8_mse = _float_field(int8_row, "mse")
    int4_mse = _float_field(int4_row, "mse")
    int8_rel_frob = _float_field(int8_row, "relative_frobenius_error")
    int4_rel_frob = _float_field(int4_row, "relative_frobenius_error")
    int8_snr = _float_field(int8_row, "snr_db")
    int4_snr = _float_field(int4_row, "snr_db")
    int8_zero = _float_field(int8_row, "zero_fraction")
    int4_zero = _float_field(int4_row, "zero_fraction")
    int8_saturation = _float_field(int8_row, "saturation_fraction")
    int4_saturation = _float_field(int4_row, "saturation_fraction")

    return QuantizerComparisonRecord(
        experiment=experiment,
        condition=condition,
        matrix_kind=int8_row["matrix_kind"],
        outlier_fraction=_optional_float_field(int8_row, "outlier_fraction"),
        outlier_scale=_optional_float_field(int8_row, "outlier_scale"),
        int8_mse=int8_mse,
        int4_mse=int4_mse,
        mse_ratio_int4_over_int8=_ratio(int4_mse, int8_mse),
        int8_relative_frobenius_error=int8_rel_frob,
        int4_relative_frobenius_error=int4_rel_frob,
        relative_frobenius_ratio_int4_over_int8=_ratio(int4_rel_frob, int8_rel_frob),
        int8_snr_db=int8_snr,
        int4_snr_db=int4_snr,
        snr_db_delta_int4_minus_int8=int4_snr - int8_snr,
        int8_zero_fraction=int8_zero,
        int4_zero_fraction=int4_zero,
        zero_fraction_delta_int4_minus_int8=int4_zero - int8_zero,
        int8_saturation_fraction=int8_saturation,
        int4_saturation_fraction=int4_saturation,
        saturation_fraction_delta_int4_minus_int8=int4_saturation - int8_saturation,
    )


def _write_csv(path: Path, records: list[QuantizerComparisonRecord]) -> None:
    if not records:
        raise ValueError("records must contain at least one row")

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _plot_baseline_analysis_bars_on_axes(
    records: list[QuantizerComparisonRecord],
    axes: list[plt.Axes],
) -> None:
    labels = [record.condition for record in records]
    x_positions = np.arange(len(records))
    zero_fraction_percent = [
        record.zero_fraction_delta_int4_minus_int8 * 100.0 for record in records
    ]
    metrics = [
        (
            "INT4 / INT8 MSE Ratio",
            [record.mse_ratio_int4_over_int8 for record in records],
            "Ratio",
            "tab:blue",
        ),
        (
            "INT4 Zero-Fraction Increase",
            zero_fraction_percent,
            "Percentage points (%)",
            "tab:orange",
        ),
        (
            "INT4 SNR Drop",
            [-record.snr_db_delta_int4_minus_int8 for record in records],
            "dB drop",
            "tab:red",
        ),
    ]

    for ax, (title, values, ylabel, color) in zip(axes, metrics, strict=True):
        ax.bar(x_positions, values, color=color, alpha=0.82)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)


def _plot_outlier_metric_heatmap_on_axis(
    records: list[QuantizerComparisonRecord],
    *,
    metric_name: str,
    title: str,
    colorbar_label: str,
    ax: plt.Axes,
    cmap: str,
) -> None:
    if not records:
        raise ValueError("records must contain at least one row")
    if not all(
        record.outlier_fraction is not None and record.outlier_scale is not None
        for record in records
    ):
        raise ValueError("all records must include outlier_fraction and outlier_scale")
    if not hasattr(records[0], metric_name):
        raise ValueError(f"unknown metric: {metric_name}")

    fractions = sorted(
        {record.outlier_fraction for record in records if record.outlier_fraction is not None}
    )
    scales = sorted({record.outlier_scale for record in records if record.outlier_scale is not None})
    values = np.full((len(fractions), len(scales)), np.nan, dtype=np.float64)

    fraction_index = {fraction: index for index, fraction in enumerate(fractions)}
    scale_index = {scale: index for index, scale in enumerate(scales)}
    for record in records:
        value = float(getattr(record, metric_name))
        if metric_name == "zero_fraction_delta_int4_minus_int8":
            value *= 100.0
        values[
            fraction_index[record.outlier_fraction],
            scale_index[record.outlier_scale],
        ] = value

    image = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Outlier scale")
    ax.set_ylabel("Outlier fraction")
    ax.set_xticks(np.arange(len(scales)))
    ax.set_xticklabels([f"{scale:g}" for scale in scales])
    ax.set_yticks(np.arange(len(fractions)))
    ax.set_yticklabels([f"{fraction:g}" for fraction in fractions])
    ax.figure.colorbar(image, ax=ax, label=colorbar_label)

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isfinite(value):
                ax.text(
                    col_index,
                    row_index,
                    _format_heatmap_value(value),
                    ha="center",
                    va="center",
                    color=_annotation_color(image.cmap(image.norm(value))),
                    fontsize=8,
                )


def _save_figure(fig: Figure, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")


def _format_heatmap_value(value: float) -> str:
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _annotation_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _ = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.55 else "white"


def _condition_label(key_fields: tuple[str, ...], key: tuple[str, ...]) -> str:
    if key_fields == ("matrix_kind",):
        return key[0]
    return ", ".join(
        f"{field.removeprefix('outlier_')}={value}"
        for field, value in zip(key_fields[1:], key[1:], strict=True)
    )


def _group_sort_key(key: tuple[str, ...]) -> tuple[object, ...]:
    return tuple(_numeric_if_possible(value) for value in key)


def _numeric_if_possible(value: str) -> object:
    try:
        return float(value)
    except ValueError:
        return value


def _float_field(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _optional_float_field(row: dict[str, str], field: str) -> float | None:
    value = row.get(field)
    if value is None or value == "":
        return None
    return float(value)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


if __name__ == "__main__":
    main()

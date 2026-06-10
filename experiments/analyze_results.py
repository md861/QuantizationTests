"""Analyze baseline and outlier quantization experiment CSV outputs."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class ResultsAnalysisConfig:
    """Configuration for analyzing generated experiment CSVs."""

    results_dir: Path = Path("results")
    output_dir: Path = Path("results")
    write_csv: bool = True


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


def main() -> None:
    """Run the default results analysis."""

    report = run_results_analysis()
    print_summary(report)
    print("Saved analysis to results/baseline_analysis.csv")
    print("Saved analysis to results/outlier_analysis.csv")


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

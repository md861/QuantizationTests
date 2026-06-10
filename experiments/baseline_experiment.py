"""Baseline INT8 and INT4 quantization comparisons."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant.matrix_factory import gaussian_matrix, heavy_tailed_matrix, outlier_matrix
from quant.metrics import QuantizationMetrics, compute_quantization_metrics
from quant.quantizer import QuantizationResult, quantize_int4, quantize_int8
from quant.visualize import plot_quantization_comparison


MatrixFactory = Callable[[], np.ndarray]
Quantizer = Callable[[np.ndarray], QuantizationResult]


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the baseline quantization experiment."""

    shape: tuple[int, int] = (64, 64)
    seed: int = 42
    results_dir: Path = Path("results")
    plots_dir: Path = Path("plots")
    save_plots: bool = True


@dataclass(frozen=True)
class BaselineRecord:
    """One row of baseline experiment output."""

    matrix_kind: str
    quantizer: str
    shape: str
    seed: int
    scale: float
    quantizer_bitwidth: int
    qmin: int
    qmax: int
    original_dtype: str
    original_bits: int
    quantized_dtype: str
    quantized_storage_bits: int
    mse: float
    mae: float
    relative_frobenius_error: float
    cosine_similarity: float
    snr_db: float
    max_abs_error: float
    mean_error: float
    error_std: float
    spectrum_l2_error: float
    relative_spectrum_l2_error: float
    reference_rank: float
    candidate_rank: float
    reference_stable_rank: float
    candidate_stable_rank: float
    stable_rank_change: float
    saturation_fraction: float | None
    zero_fraction: float | None
    plot_path: str


def run_baseline_experiment(config: BaselineConfig = BaselineConfig()) -> list[BaselineRecord]:
    """Run INT8 and INT4 quantization on the baseline matrix families."""

    config.results_dir.mkdir(parents=True, exist_ok=True)
    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)

    records: list[BaselineRecord] = []
    for matrix_kind, matrix_factory in _matrix_factories(config).items():
        matrix = matrix_factory()
        quantization_results: dict[str, QuantizationResult] = {}
        metric_results: dict[str, QuantizationMetrics] = {}
        for quantizer_name, quantizer in _quantizers().items():
            result = quantizer(matrix)
            metrics = compute_quantization_metrics(
                matrix,
                result.dequantized,
                quantized=result.quantized,
                qmin=result.qmin,
                qmax=result.qmax,
            )
            quantization_results[quantizer_name] = result
            metric_results[quantizer_name] = metrics

        plot_path = ""
        if config.save_plots:
            plot_path = str(config.plots_dir / f"baseline_{matrix_kind}_comparison.png")
            fig = plot_quantization_comparison(
                matrix,
                quantization_results,
                output_path=plot_path,
                title=f"Baseline {matrix_kind} quantization comparison",
                metrics=metric_results,
            )
            plt.close(fig)

        for quantizer_name, result in quantization_results.items():
            records.append(
                _make_record(
                    matrix_kind=matrix_kind,
                    quantizer_name=quantizer_name,
                    matrix=matrix,
                    result=result,
                    metrics=metric_results[quantizer_name],
                    config=config,
                    plot_path=plot_path,
                )
            )

    _write_csv(config.results_dir / "baseline_metrics.csv", records)
    return records


def print_summary(records: list[BaselineRecord]) -> None:
    """Print a compact experiment summary."""

    print("Baseline quantization experiment")
    print("matrix_kind, quantizer, mse, rel_frob, cosine, snr_db, zero_frac, sat_frac")
    for record in records:
        print(
            f"{record.matrix_kind}, {record.quantizer}, "
            f"{record.mse:.8f}, "
            f"{record.relative_frobenius_error:.8f}, "
            f"{record.cosine_similarity:.8f}, "
            f"{record.snr_db:.4f}, "
            f"{_format_optional(record.zero_fraction)}, "
            f"{_format_optional(record.saturation_fraction)}"
        )


def main() -> None:
    """Run the default baseline experiment."""

    records = run_baseline_experiment()
    print_summary(records)
    print("Saved metrics to results/baseline_metrics.csv")
    print("Saved plots to plots/baseline_<matrix_kind>_comparison.png")


def _matrix_factories(config: BaselineConfig) -> dict[str, MatrixFactory]:
    return {
        "gaussian": lambda: gaussian_matrix(config.shape, seed=config.seed),
        "heavy_tailed": lambda: heavy_tailed_matrix(config.shape, df=2.0, seed=config.seed),
        "outlier": lambda: outlier_matrix(
            config.shape,
            outlier_fraction=0.01,
            outlier_scale=10.0,
            seed=config.seed,
        ),
    }


def _quantizers() -> dict[str, Quantizer]:
    return {
        "int8": quantize_int8,
        "int4": quantize_int4,
    }


def _make_record(
    *,
    matrix_kind: str,
    quantizer_name: str,
    matrix: np.ndarray,
    result: QuantizationResult,
    metrics: QuantizationMetrics,
    config: BaselineConfig,
    plot_path: str,
) -> BaselineRecord:
    metric_values = asdict(metrics)
    return BaselineRecord(
        matrix_kind=matrix_kind,
        quantizer=quantizer_name,
        shape=f"{matrix.shape[0]}x{matrix.shape[1]}",
        seed=config.seed,
        scale=result.scale,
        quantizer_bitwidth=result.bitwidth,
        qmin=result.qmin,
        qmax=result.qmax,
        original_dtype=str(matrix.dtype),
        original_bits=matrix.dtype.itemsize * 8,
        quantized_dtype=str(result.quantized.dtype),
        quantized_storage_bits=result.quantized.dtype.itemsize * 8,
        mse=metric_values["mse"],
        mae=metric_values["mae"],
        relative_frobenius_error=metric_values["relative_frobenius_error"],
        cosine_similarity=metric_values["cosine_similarity"],
        snr_db=metric_values["snr_db"],
        max_abs_error=metric_values["max_abs_error"],
        mean_error=metric_values["mean_error"],
        error_std=metric_values["error_std"],
        spectrum_l2_error=metric_values["spectrum_l2_error"],
        relative_spectrum_l2_error=metric_values["relative_spectrum_l2_error"],
        reference_rank=metric_values["reference_rank"],
        candidate_rank=metric_values["candidate_rank"],
        reference_stable_rank=metric_values["reference_stable_rank"],
        candidate_stable_rank=metric_values["candidate_stable_rank"],
        stable_rank_change=metric_values["stable_rank_change"],
        saturation_fraction=metric_values["saturation_fraction"],
        zero_fraction=metric_values["zero_fraction"],
        plot_path=plot_path,
    )


def _write_csv(path: Path, records: list[BaselineRecord]) -> None:
    if not records:
        raise ValueError("records must contain at least one row")

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()

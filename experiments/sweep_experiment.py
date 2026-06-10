"""Milestone 2 sweep: all quantization paths across outlier conditions."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant.matrix_factory import outlier_matrix
from quant.metrics import compute_quantization_metrics
from quant.quantizer import (
    QuantizationResult,
    quantize_int4,
    quantize_int4_grouped,
    quantize_int4_row_grouped,
)
from quant.rotations import apply_rotation, rotate_channel_pair
from quant.scaling import balance_channel_max_abs, invert_channel_scaling


@dataclass
class SweepConfig:
    """Grid parameters and output settings for the sweep experiment."""

    shape: tuple[int, int] = (32, 32)
    seeds: Sequence[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    outlier_fractions: Sequence[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.10]
    )
    outlier_scales: Sequence[float] = field(
        default_factory=lambda: [5.0, 10.0, 20.0]
    )
    row_group_sizes: Sequence[int] = field(default_factory=lambda: [4, 8, 16])
    col_group_sizes: Sequence[int] = field(default_factory=lambda: [4, 8])
    results_dir: Path = field(default_factory=lambda: Path("results"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    save_plots: bool = True


@dataclass(frozen=True)
class SweepRecord:
    """Metrics for one quantization method on one matrix condition."""

    seed: int
    outlier_fraction: float
    outlier_scale: float
    method: str
    mse: float
    relative_frobenius_error: float
    snr_db: float
    zero_fraction: float
    saturation_fraction: float | None


def run_sweep_experiment(
    config: SweepConfig = SweepConfig(),
) -> list[SweepRecord]:
    """Run all quantization paths across the configured condition grid.

    Iterates over every (seed, outlier_fraction, outlier_scale) combination,
    applies every quantization path, and records reconstruction metrics.

    Writes ``results/sweep_metrics.csv`` and optionally
    ``plots/sweep_dashboard.png``.
    """
    records: list[SweepRecord] = []

    for seed in config.seeds:
        for frac in config.outlier_fractions:
            for scale in config.outlier_scales:
                matrix = outlier_matrix(
                    config.shape,
                    outlier_fraction=float(frac),
                    outlier_scale=float(scale),
                    seed=seed,
                )
                for method, deq, result in _quantize_all_methods(matrix, config):
                    m = compute_quantization_metrics(
                        matrix,
                        deq,
                        quantized=result.quantized,
                        qmin=result.qmin,
                        qmax=result.qmax,
                    )
                    records.append(
                        SweepRecord(
                            seed=seed,
                            outlier_fraction=float(frac),
                            outlier_scale=float(scale),
                            method=method,
                            mse=m.mse,
                            relative_frobenius_error=m.relative_frobenius_error,
                            snr_db=m.snr_db,
                            zero_fraction=(
                                m.zero_fraction if m.zero_fraction is not None else 0.0
                            ),
                            saturation_fraction=m.saturation_fraction,
                        )
                    )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.results_dir / "sweep_metrics.csv", records)

    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)
        fig = _plot_dashboard(records, config)
        _save_figure(fig, config.plots_dir / "sweep_dashboard.png")

    return records


def methods_in_config(config: SweepConfig) -> list[str]:
    """Return the ordered list of method names produced by a given config."""
    names = ["global"]
    names += [f"col_grouped_g{g}" for g in config.col_group_sizes]
    names += [f"row_grouped_g{g}" for g in config.row_group_sizes]
    names += ["scale_global", "rotate_global", "rotate_scale_global"]
    names += [f"rotate_scale_row_g{g}" for g in config.row_group_sizes]
    return names


def print_summary(records: list[SweepRecord]) -> None:
    """Print a compact per-method summary averaged over all conditions."""
    ratios = _mse_ratios(records)
    zf: dict[str, list[float]] = {}
    for r in records:
        zf.setdefault(r.method, []).append(r.zero_fraction)

    methods = sorted(ratios, key=lambda m: float(np.mean(ratios[m])))
    print(f"\n{'Method':<35}  {'MSE ratio':>10}  {'Zero frac':>10}")
    print("-" * 60)
    for m in methods:
        print(
            f"{m:<35}  {float(np.mean(ratios[m])):>10.4f}  "
            f"{float(np.mean(zf.get(m, [0]))):>10.4f}"
        )


# ── private helpers ───────────────────────────────────────────────────────────

def _quantize_all_methods(
    matrix: np.ndarray,
    config: SweepConfig,
) -> list[tuple[str, np.ndarray, QuantizationResult]]:
    out: list[tuple[str, np.ndarray, QuantizationResult]] = []

    # Global INT4
    r = quantize_int4(matrix)
    out.append(("global", r.dequantized, r))

    # Column-grouped INT4
    for g in config.col_group_sizes:
        r = quantize_int4_grouped(matrix, group_size=g)
        out.append((f"col_grouped_g{g}", r.dequantized, r))

    # Row-grouped INT4
    for g in config.row_group_sizes:
        r = quantize_int4_row_grouped(matrix, row_group_size=g)
        out.append((f"row_grouped_g{g}", r.dequantized, r))

    # Rotation metadata shared across all rotation paths
    col_maxabs = np.max(np.abs(matrix), axis=0)
    top2 = np.argsort(col_maxabs)[-2:]
    rot_i, rot_j = int(top2[0]), int(top2[1])
    rotated, theta = rotate_channel_pair(
        matrix.astype(np.float64), rot_i, rot_j
    )
    rotated = rotated.astype(matrix.dtype)

    # Scaling + global INT4
    scaled, scaling = balance_channel_max_abs(matrix)
    r = quantize_int4(scaled)
    deq = invert_channel_scaling(r.dequantized, scaling)
    out.append(("scale_global", deq, r))

    # Rotation + global INT4
    r = quantize_int4(rotated)
    deq = apply_rotation(
        r.dequantized.astype(np.float64), rot_i, rot_j, -theta
    ).astype(matrix.dtype)
    out.append(("rotate_global", deq, r))

    # Rotation + scaling + global INT4
    rot_scaled, rot_scaling = balance_channel_max_abs(rotated)
    r = quantize_int4(rot_scaled)
    deq = invert_channel_scaling(r.dequantized, rot_scaling)
    deq = apply_rotation(deq.astype(np.float64), rot_i, rot_j, -theta).astype(
        matrix.dtype
    )
    out.append(("rotate_scale_global", deq, r))

    # Rotation + scaling + row-grouped INT4
    for g in config.row_group_sizes:
        r = quantize_int4_row_grouped(rot_scaled, row_group_size=g)
        deq = invert_channel_scaling(r.dequantized, rot_scaling)
        deq = apply_rotation(
            deq.astype(np.float64), rot_i, rot_j, -theta
        ).astype(matrix.dtype)
        out.append((f"rotate_scale_row_g{g}", deq, r))

    return out


def _mse_ratios(records: list[SweepRecord]) -> dict[str, list[float]]:
    global_mse: dict[tuple, float] = {
        (r.seed, r.outlier_fraction, r.outlier_scale): r.mse
        for r in records
        if r.method == "global"
    }
    ratios: dict[str, list[float]] = {}
    for r in records:
        key = (r.seed, r.outlier_fraction, r.outlier_scale)
        base = global_mse.get(key, 1.0)
        ratio = r.mse / base if base > 0 else 1.0
        ratios.setdefault(r.method, []).append(ratio)
    return ratios


def _plot_dashboard(
    records: list[SweepRecord],
    config: SweepConfig,
) -> plt.Figure:
    n_conditions = len({(r.seed, r.outlier_fraction, r.outlier_scale) for r in records})
    n_methods = len({r.method for r in records})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Quantization path sweep  —  {n_conditions} conditions × {n_methods} methods",
        fontsize=12,
    )

    ratios = _mse_ratios(records)
    sorted_methods = sorted(ratios, key=lambda m: float(np.mean(ratios[m])))

    # Panel 1: mean MSE ratio per method
    ax = axes[0, 0]
    means = [float(np.mean(ratios[m])) for m in sorted_methods]
    colors = ["tab:green" if v < 1.0 else "tab:red" for v in means]
    ax.barh(sorted_methods, means, color=colors)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("MSE / Global INT4 MSE  (< 1 = improvement)")
    ax.set_title("Mean MSE ratio vs global INT4")

    # Panel 2: mean zero fraction per method
    ax = axes[0, 1]
    zf: dict[str, list[float]] = {}
    for r in records:
        zf.setdefault(r.method, []).append(r.zero_fraction)
    zf_means = [float(np.mean(zf.get(m, [0.0]))) for m in sorted_methods]
    ax.barh(sorted_methods, zf_means, color="tab:blue")
    ax.set_xlabel("Mean zero fraction")
    ax.set_title("Mean zero fraction by method")

    # Panel 3: MSE ratio vs outlier_scale for key methods
    ax = axes[1, 0]
    global_by_cond: dict[tuple, float] = {
        (r.seed, r.outlier_fraction, r.outlier_scale): r.mse
        for r in records
        if r.method == "global"
    }
    key_methods = ["global", "scale_global", "rotate_scale_global"]
    if config.row_group_sizes:
        key_methods.append(f"row_grouped_g{config.row_group_sizes[0]}")
        key_methods.append(f"rotate_scale_row_g{config.row_group_sizes[0]}")
    key_methods = [m for m in key_methods if m in ratios]

    for method in key_methods:
        by_scale: dict[float, list[float]] = {}
        for r in records:
            if r.method != method:
                continue
            base = global_by_cond.get(
                (r.seed, r.outlier_fraction, r.outlier_scale), 1.0
            )
            ratio = r.mse / base if base > 0 else 1.0
            by_scale.setdefault(r.outlier_scale, []).append(ratio)
        xs = sorted(by_scale)
        ys = [float(np.median(by_scale[s])) for s in xs]
        ax.plot(xs, ys, marker="o", label=method)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Outlier scale")
    ax.set_ylabel("Median MSE ratio vs global INT4")
    ax.set_title("MSE ratio vs outlier severity")
    ax.legend(fontsize=7)

    # Panel 4: MSE ratio vs row_group_size
    ax = axes[1, 1]
    for prefix, label in [
        ("row_grouped_g", "Row-grouped"),
        ("rotate_scale_row_g", "Rot+Scale+Row-grouped"),
    ]:
        matched = {m: ratios[m] for m in ratios if m.startswith(prefix)}
        if not matched:
            continue
        g_vals = sorted(int(m.split("_g")[-1]) for m in matched)
        ys = [float(np.mean(matched[f"{prefix}{g}"])) for g in g_vals]
        ax.plot(g_vals, ys, marker="o", label=label)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Row group size  (smaller = more scales)")
    ax.set_ylabel("Mean MSE ratio vs global INT4")
    ax.set_title("Effect of row group size")
    ax.legend()

    fig.tight_layout()
    return fig


def _write_csv(path: Path, records: list[SweepRecord]) -> None:
    fieldnames = [
        "seed",
        "outlier_fraction",
        "outlier_scale",
        "method",
        "mse",
        "relative_frobenius_error",
        "snr_db",
        "zero_fraction",
        "saturation_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "seed": r.seed,
                    "outlier_fraction": r.outlier_fraction,
                    "outlier_scale": r.outlier_scale,
                    "method": r.method,
                    "mse": r.mse,
                    "relative_frobenius_error": r.relative_frobenius_error,
                    "snr_db": r.snr_db,
                    "zero_fraction": r.zero_fraction,
                    "saturation_fraction": (
                        "" if r.saturation_fraction is None else r.saturation_fraction
                    ),
                }
            )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = SweepConfig()
    records = run_sweep_experiment(config)
    print_summary(records)


if __name__ == "__main__":
    main()

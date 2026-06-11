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
from quant.rotations import (
    GivensRotation,
    apply_rotation,
    apply_sequential_rotations,
    rotate_channel_pair,
    rotate_top_width_pairs,
)
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
    top_width_pair_fractions: Sequence[float] = field(default_factory=list)
    rotation_search_steps: int = 360
    results_dir: Path = field(default_factory=lambda: Path("results"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    save_plots: bool = True
    csv_name: str = "sweep_metrics.csv"
    plot_name: str = "sweep_dashboard.png"


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
    rotation_count: int
    rotation_pair_fraction: float
    rotation_candidate_fraction: float


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
                for method_result in _quantize_all_methods(matrix, config):
                    method, deq, result = (
                        method_result.method,
                        method_result.dequantized,
                        method_result.quantization,
                    )
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
                            rotation_count=method_result.rotation_count,
                            rotation_pair_fraction=method_result.rotation_pair_fraction,
                            rotation_candidate_fraction=(
                                method_result.rotation_candidate_fraction
                            ),
                        )
                    )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.results_dir / config.csv_name, records)

    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)
        fig = _plot_dashboard(records, config)
        _save_figure(fig, config.plots_dir / config.plot_name)

    return records


def methods_in_config(config: SweepConfig) -> list[str]:
    """Return the ordered list of method names produced by a given config."""
    names = ["global"]
    names += [f"col_grouped_g{g}" for g in config.col_group_sizes]
    names += [f"row_grouped_g{g}" for g in config.row_group_sizes]
    names += ["scale_global", "rotate_global", "rotate_scale_global"]
    names += [f"rotate_scale_row_g{g}" for g in config.row_group_sizes]
    for p in config.top_width_pair_fractions:
        tag = _fraction_tag(float(p))
        names += [
            f"top_width_rotate_{tag}_global",
            f"top_width_rotate_scale_{tag}_global",
        ]
        names += [
            f"top_width_rotate_scale_{tag}_row_g{g}"
            for g in config.row_group_sizes
        ]
    return names


def print_summary(records: list[SweepRecord]) -> None:
    """Print a compact per-method summary over all sweep conditions."""
    ratios = _mse_ratios(records)
    zf: dict[str, list[float]] = {}
    for r in records:
        zf.setdefault(r.method, []).append(r.zero_fraction)

    methods = sorted(ratios, key=lambda m: float(np.mean(ratios[m])))
    print(
        f"\n{'Method':<35}  {'MSE mean':>10}  {'MSE std':>9}  "
        f"{'Zero mean':>10}  {'Zero std':>9}"
    )
    print("-" * 82)
    for m in methods:
        zero_values = zf.get(m, [0.0])
        print(
            f"{m:<35}  {float(np.mean(ratios[m])):>10.4f}  "
            f"{float(np.std(ratios[m])):>9.4f}  "
            f"{float(np.mean(zero_values)):>10.4f}  "
            f"{float(np.std(zero_values)):>9.4f}"
        )


# ── private helpers ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _MethodOutput:
    method: str
    dequantized: np.ndarray
    quantization: QuantizationResult
    rotation_count: int = 0
    rotation_pair_fraction: float = 0.0
    rotation_candidate_fraction: float = 0.0


def _quantize_all_methods(
    matrix: np.ndarray,
    config: SweepConfig,
) -> list[_MethodOutput]:
    out: list[_MethodOutput] = []

    # Global INT4
    r = quantize_int4(matrix)
    out.append(_MethodOutput("global", r.dequantized, r))

    # Column-grouped INT4
    for g in config.col_group_sizes:
        r = quantize_int4_grouped(matrix, group_size=g)
        out.append(_MethodOutput(f"col_grouped_g{g}", r.dequantized, r))

    # Row-grouped INT4
    for g in config.row_group_sizes:
        r = quantize_int4_row_grouped(matrix, row_group_size=g)
        out.append(_MethodOutput(f"row_grouped_g{g}", r.dequantized, r))

    # Rotation metadata shared across all rotation paths
    col_maxabs = np.max(np.abs(matrix), axis=0)
    top2 = np.argsort(col_maxabs)[-2:]
    rot_i, rot_j = int(top2[0]), int(top2[1])
    rotated, theta = rotate_channel_pair(
        matrix.astype(np.float64), rot_i, rot_j, n_search=config.rotation_search_steps
    )
    rotated = rotated.astype(matrix.dtype)

    # Scaling + global INT4
    scaled, scaling = balance_channel_max_abs(matrix)
    r = quantize_int4(scaled)
    deq = invert_channel_scaling(r.dequantized, scaling)
    out.append(_MethodOutput("scale_global", deq, r))

    # Rotation + global INT4
    r = quantize_int4(rotated)
    deq = apply_rotation(
        r.dequantized.astype(np.float64), rot_i, rot_j, -theta
    ).astype(matrix.dtype)
    out.append(
        _MethodOutput(
            "rotate_global",
            deq,
            r,
            rotation_count=1,
            rotation_pair_fraction=_pair_fraction_from_count(matrix, 1),
        )
    )

    # Rotation + scaling + global INT4
    rot_scaled, rot_scaling = balance_channel_max_abs(rotated)
    r = quantize_int4(rot_scaled)
    deq = invert_channel_scaling(r.dequantized, rot_scaling)
    deq = apply_rotation(deq.astype(np.float64), rot_i, rot_j, -theta).astype(
        matrix.dtype
    )
    out.append(
        _MethodOutput(
            "rotate_scale_global",
            deq,
            r,
            rotation_count=1,
            rotation_pair_fraction=_pair_fraction_from_count(matrix, 1),
        )
    )

    # Rotation + scaling + row-grouped INT4
    for g in config.row_group_sizes:
        r = quantize_int4_row_grouped(rot_scaled, row_group_size=g)
        deq = invert_channel_scaling(r.dequantized, rot_scaling)
        deq = apply_rotation(
            deq.astype(np.float64), rot_i, rot_j, -theta
        ).astype(matrix.dtype)
        out.append(
            _MethodOutput(
                f"rotate_scale_row_g{g}",
                deq,
                r,
                rotation_count=1,
                rotation_pair_fraction=_pair_fraction_from_count(matrix, 1),
            )
        )

    # Top-width-difference independent rotations + quantization paths
    for p in config.top_width_pair_fractions:
        tag = _fraction_tag(float(p))
        top_rotated, rotations = rotate_top_width_pairs(
            matrix.astype(np.float64),
            top_fraction=float(p),
            independent=True,
            n_search=config.rotation_search_steps,
        )
        top_rotated = top_rotated.astype(matrix.dtype)
        rotation_count = len(rotations)
        inverse_rotations = [
            GivensRotation(i=r.i, j=r.j, theta=-r.theta)
            for r in reversed(rotations)
        ]

        r = quantize_int4(top_rotated)
        deq = apply_sequential_rotations(
            r.dequantized.astype(np.float64), inverse_rotations
        ).astype(matrix.dtype)
        out.append(
            _MethodOutput(
                f"top_width_rotate_{tag}_global",
                deq,
                r,
                rotation_count=rotation_count,
                rotation_pair_fraction=_pair_fraction_from_count(
                    matrix, rotation_count
                ),
                rotation_candidate_fraction=float(p),
            )
        )

        top_rot_scaled, top_rot_scaling = balance_channel_max_abs(top_rotated)
        r = quantize_int4(top_rot_scaled)
        deq = invert_channel_scaling(r.dequantized, top_rot_scaling)
        deq = apply_sequential_rotations(
            deq.astype(np.float64), inverse_rotations
        ).astype(matrix.dtype)
        out.append(
            _MethodOutput(
                f"top_width_rotate_scale_{tag}_global",
                deq,
                r,
                rotation_count=rotation_count,
                rotation_pair_fraction=_pair_fraction_from_count(
                    matrix, rotation_count
                ),
                rotation_candidate_fraction=float(p),
            )
        )

        for g in config.row_group_sizes:
            r = quantize_int4_row_grouped(top_rot_scaled, row_group_size=g)
            deq = invert_channel_scaling(r.dequantized, top_rot_scaling)
            deq = apply_sequential_rotations(
                deq.astype(np.float64), inverse_rotations
            ).astype(matrix.dtype)
            out.append(
                _MethodOutput(
                    f"top_width_rotate_scale_{tag}_row_g{g}",
                    deq,
                    r,
                    rotation_count=rotation_count,
                    rotation_pair_fraction=_pair_fraction_from_count(
                        matrix, rotation_count
                    ),
                    rotation_candidate_fraction=float(p),
                )
            )

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
    stds = [float(np.std(ratios[m])) for m in sorted_methods]
    colors = ["tab:green" if v < 1.0 else "tab:red" for v in means]
    ax.barh(sorted_methods, means, xerr=stds, color=colors, alpha=0.88)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("MSE / Global INT4 MSE  (< 1 = improvement)")
    ax.set_title("Mean MSE ratio vs global INT4 (error bars = std)")

    # Panel 2: mean zero fraction per method
    ax = axes[0, 1]
    zf: dict[str, list[float]] = {}
    for r in records:
        zf.setdefault(r.method, []).append(r.zero_fraction)
    zf_means = [float(np.mean(zf.get(m, [0.0]))) for m in sorted_methods]
    zf_stds = [float(np.std(zf.get(m, [0.0]))) for m in sorted_methods]
    ax.barh(sorted_methods, zf_means, xerr=zf_stds, color="tab:blue", alpha=0.88)
    ax.set_xlabel("Mean zero fraction")
    ax.set_title("Mean zero fraction by method (error bars = std)")

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
        "rotation_count",
        "rotation_pair_fraction",
        "rotation_candidate_fraction",
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
                    "rotation_count": r.rotation_count,
                    "rotation_pair_fraction": r.rotation_pair_fraction,
                    "rotation_candidate_fraction": r.rotation_candidate_fraction,
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


def _fraction_tag(value: float) -> str:
    percent = value * 100.0
    if float(percent).is_integer():
        return f"p{int(percent)}"
    return f"p{str(round(percent, 4)).replace('.', '_')}"


def _pair_fraction_from_count(matrix: np.ndarray, rotation_count: int) -> float:
    n_cols = matrix.shape[1]
    total_pairs = n_cols * (n_cols - 1) // 2
    if total_pairs == 0:
        return 0.0
    return float(rotation_count / total_pairs)


def main() -> None:
    config = SweepConfig()
    records = run_sweep_experiment(config)
    print_summary(records)


if __name__ == "__main__":
    main()

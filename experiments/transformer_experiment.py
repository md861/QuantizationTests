"""Milestone 3: tiny transformer quantization harness.

Loads a small HuggingFace causal LM, quantizes its linear layer weights with
multiple quantization paths and bitwidths, and measures:
  - weight reconstruction per layer
  - activation drift per layer (captured inputs, no second forward pass)
  - full-model logit similarity, next-token loss, and perplexity

Config knobs:
  bitwidths              — [4, 8] by default; runs every path at each bitwidth
  row_group_sizes        — fixed group sizes (e.g. [4])
  row_group_fractions    — additional sizes computed as n_rows × fraction
                           (e.g. [0.5, 0.25, 0.0625] → n/2, n/4, n/16);
                           merged with fixed sizes and deduplicated per layer
  top_width_pair_fractions — fractions of channel pairs for sparse rotations
                             (e.g. [0.05, 0.10, 0.20])
  single_layer_name      — quantize one named layer only; None = all layers
  delete_hf_cache_after  — evict model from HF cache after the run (one model
                           at a time on constrained hardware)
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant.metrics import compute_quantization_metrics
from quant.quantizer import (
    quantize_int4,
    quantize_int4_row_grouped,
    quantize_int8,
    quantize_int8_row_grouped,
)
from quant.rotations import (
    GivensRotation,
    apply_sequential_rotations,
    rotate_top_width_pairs,
)
from quant.scaling import balance_channel_max_abs, invert_channel_scaling


CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Quantization reduces the precision of neural network weights to lower bitwidths.",
    "Language models learn statistical patterns from large text corpora.",
]

_SKIP_LAYER_NAMES = {"lm_head"}


@dataclass
class TransformerConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    calibration_texts: list[str] = field(
        default_factory=lambda: list(CALIBRATION_TEXTS)
    )
    single_layer_name: Optional[str] = "transformer.h.0.mlp.c_fc"
    bitwidths: list[int] = field(default_factory=lambda: [4, 8])
    row_group_sizes: list[int] = field(default_factory=lambda: [4])
    row_group_fractions: list[float] = field(
        default_factory=lambda: [0.5, 0.25, 0.0625]
    )
    top_width_pair_fractions: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20]
    )
    max_rotation_pairs: int = 1000
    rotation_search_steps: int = 360
    results_dir: Path = field(default_factory=lambda: Path("results"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    save_plots: bool = True
    delete_hf_cache_after: bool = False


@dataclass(frozen=True)
class WeightRecord:
    model_name: str
    layer_name: str
    weight_shape: str
    method: str
    bitwidth: int
    rotation_count: int
    rotation_pair_fraction: float
    rotation_candidate_fraction: float
    mse: float
    relative_frobenius_error: float
    cosine_similarity: float
    snr_db: float
    zero_fraction: float
    saturation_fraction: float


@dataclass(frozen=True)
class ActivationRecord:
    model_name: str
    layer_name: str
    method: str
    bitwidth: int
    activation_mse: float
    activation_cosine_similarity: float
    activation_relative_error: float


@dataclass(frozen=True)
class LogitRecord:
    model_name: str
    scope: str
    method: str
    bitwidth: int
    logit_mse: float
    logit_cosine_similarity: float
    top5_token_overlap: float
    loss: float
    original_loss: float
    loss_delta: float
    perplexity: float
    original_perplexity: float
    perplexity_ratio: float


def run_transformer_experiment(
    config: TransformerConfig = TransformerConfig(),
) -> tuple[list[WeightRecord], list[ActivationRecord], list[LogitRecord]]:
    """Run weight, activation, and logit experiments for one model.

    Returns (weight_records, activation_records, logit_records) and writes
    three CSVs plus an optional dashboard plot.

    ``method_deqs`` passed between helpers is keyed by (method_str, bitwidth_int)
    tuples to avoid collisions when the same method runs at multiple bitwidths.
    """
    print(f"Loading {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_inputs = [
        tokenizer(t, return_tensors="pt")["input_ids"]
        for t in config.calibration_texts
    ]

    if config.single_layer_name is not None:
        layers = {
            config.single_layer_name: _get_module(model, config.single_layer_name)
        }
    else:
        layers = _get_linear_layers(model)

    print(f"Processing {len(layers)} layer(s): {list(layers.keys())}")

    weight_records: list[WeightRecord] = []
    activation_records: list[ActivationRecord] = []
    # all_method_deqs: layer_name -> {(method, bitwidth): dequantized_weight}
    all_method_deqs: dict[str, dict[tuple[str, int], np.ndarray]] = {}

    for layer_name, module in layers.items():
        w = _extract_weight(module)
        print(f"  {layer_name}: shape={w.shape}")
        wr, method_deqs = _run_weight_experiment(config, layer_name, w)
        weight_records.extend(wr)
        all_method_deqs[layer_name] = method_deqs
        ar = _run_activation_experiment(
            model, token_inputs, layer_name, module, w, method_deqs, config.model_name
        )
        activation_records.extend(ar)

    scope = (
        f"single_layer:{config.single_layer_name}"
        if config.single_layer_name is not None
        else "all_layers"
    )
    logit_records = _run_logit_experiment(
        config, model, token_inputs, layers, all_method_deqs, scope
    )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    _write_weight_csv(
        config.results_dir / "transformer_weight_metrics.csv", weight_records
    )
    _write_activation_csv(
        config.results_dir / "transformer_activation_metrics.csv", activation_records
    )
    _write_logit_csv(
        config.results_dir / "transformer_logit_metrics.csv", logit_records
    )

    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)
        fig = _plot_dashboard(weight_records, activation_records, logit_records)
        fig.savefig(
            config.plots_dir / "transformer_dashboard.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)
        for bitwidth in sorted({r.bitwidth for r in weight_records}):
            bw_weight_records = [
                r for r in weight_records if r.bitwidth == bitwidth
            ]
            bw_activation_records = [
                r for r in activation_records if r.bitwidth == bitwidth
            ]
            bw_logit_records = [
                r for r in logit_records if r.bitwidth == bitwidth
            ]
            fig = _plot_dashboard(
                bw_weight_records,
                bw_activation_records,
                bw_logit_records,
                title=f"Transformer quantization — INT{bitwidth}",
            )
            fig.savefig(
                config.plots_dir / f"transformer_dashboard_int{bitwidth}.png",
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(fig)

    if config.delete_hf_cache_after:
        _clear_hf_cache(config.model_name)

    return weight_records, activation_records, logit_records


# ── weight experiment ─────────────────────────────────────────────────────────


def _run_weight_experiment(
    config: TransformerConfig,
    layer_name: str,
    weight: np.ndarray,
) -> tuple[list[WeightRecord], dict[tuple[str, int], np.ndarray]]:
    """Quantize weight with all configured paths and bitwidths."""
    records: list[WeightRecord] = []
    method_deqs: dict[tuple[str, int], np.ndarray] = {}
    n_rows, n_cols = weight.shape
    total_pairs = n_cols * (n_cols - 1) // 2
    all_group_sizes = _resolve_group_sizes(
        config.row_group_sizes, config.row_group_fractions, n_rows
    )

    def _add(
        method: str,
        bitwidth: int,
        deq: np.ndarray,
        qr,
        rot_count: int = 0,
        rot_frac: float = 0.0,
        rot_cand: float = 0.0,
    ) -> None:
        m = compute_quantization_metrics(
            weight, deq, quantized=qr.quantized, qmin=qr.qmin, qmax=qr.qmax
        )
        records.append(
            WeightRecord(
                model_name=config.model_name,
                layer_name=layer_name,
                weight_shape=str(weight.shape),
                method=method,
                bitwidth=bitwidth,
                rotation_count=rot_count,
                rotation_pair_fraction=rot_frac,
                rotation_candidate_fraction=rot_cand,
                mse=m.mse,
                relative_frobenius_error=m.relative_frobenius_error,
                cosine_similarity=m.cosine_similarity,
                snr_db=m.snr_db,
                zero_fraction=m.zero_fraction or 0.0,
                saturation_fraction=m.saturation_fraction or 0.0,
            )
        )
        method_deqs[(method, bitwidth)] = deq

    for bw in config.bitwidths:
        # Global
        r = _q_global(weight, bw)
        _add("global", bw, r.dequantized, r)

        # Row-grouped
        for g in all_group_sizes:
            r = _q_row_grouped(weight, g, bw)
            _add(f"row_grouped_g{g}", bw, r.dequantized, r)

        # Scale + row-grouped
        scaled, scaling = balance_channel_max_abs(weight)
        for g in all_group_sizes:
            r = _q_row_grouped(scaled, g, bw)
            deq = invert_channel_scaling(r.dequantized, scaling)
            _add(f"scale_row_g{g}", bw, deq, r)

        # Top-width rotate + scale + row-grouped
        if n_cols >= 2:
            for p in config.top_width_pair_fractions:
                n_pairs = max(1, round(total_pairs * float(p)))
                if n_pairs > config.max_rotation_pairs:
                    print(
                        f"    Skipping top-width rotation for {layer_name} "
                        f"(pairs={n_pairs} > max={config.max_rotation_pairs})"
                    )
                    continue
                tag = _fraction_tag(float(p))
                top_rotated, rotations = rotate_top_width_pairs(
                    weight.astype(np.float64),
                    top_fraction=float(p),
                    independent=True,
                    n_search=config.rotation_search_steps,
                )
                top_rotated = top_rotated.astype(weight.dtype)
                n_rot = len(rotations)
                inv_rots = [
                    GivensRotation(i=rot.i, j=rot.j, theta=-rot.theta)
                    for rot in reversed(rotations)
                ]
                rot_frac = float(n_rot / total_pairs) if total_pairs > 0 else 0.0
                top_rot_scaled, top_rot_scaling = balance_channel_max_abs(top_rotated)

                for g in all_group_sizes:
                    r = _q_row_grouped(top_rot_scaled, g, bw)
                    deq = invert_channel_scaling(r.dequantized, top_rot_scaling)
                    deq = apply_sequential_rotations(
                        deq.astype(np.float64), inv_rots
                    ).astype(weight.dtype)
                    _add(
                        f"top_width_rotate_{tag}_scale_row_g{g}",
                        bw,
                        deq,
                        r,
                        rot_count=n_rot,
                        rot_frac=rot_frac,
                        rot_cand=float(p),
                    )

    return records, method_deqs


# ── activation experiment ─────────────────────────────────────────────────────


def _run_activation_experiment(
    model,
    token_inputs: list,
    layer_name: str,
    module,
    weight: np.ndarray,
    method_deqs: dict[tuple[str, int], np.ndarray],
    model_name: str,
) -> list[ActivationRecord]:
    """Capture layer inputs once, then analytically compute drift per method."""
    captured: list[torch.Tensor] = []

    def _hook(mod, inp, out):
        captured.append(inp[0].detach().cpu())

    handle = module.register_forward_hook(_hook)
    with torch.no_grad():
        for ids in token_inputs:
            model(ids)
    handle.remove()

    bias = (
        module.bias.detach().float().numpy() if module.bias is not None else None
    )

    records: list[ActivationRecord] = []
    for (method, bitwidth), w_deq in method_deqs.items():
        mses, cos_sims, rel_errs = [], [], []
        for x in captured:
            x_flat = x.float().numpy().reshape(-1, x.shape[-1])
            # Both weight and w_deq are (in, out) — same layout for Conv1D and Linear
            out_orig = x_flat @ weight
            out_q = x_flat @ w_deq
            if bias is not None:
                out_orig = out_orig + bias
                out_q = out_q + bias
            o1 = out_orig.flatten()
            o2 = out_q.flatten()
            diff = o1 - o2
            norm1 = np.linalg.norm(o1)
            norm2 = np.linalg.norm(o2)
            mses.append(float(np.mean(diff**2)))
            cos_sims.append(float(np.dot(o1, o2) / (norm1 * norm2 + 1e-8)))
            rel_errs.append(float(np.linalg.norm(diff) / (norm1 + 1e-8)))

        records.append(
            ActivationRecord(
                model_name=model_name,
                layer_name=layer_name,
                method=method,
                bitwidth=bitwidth,
                activation_mse=float(np.mean(mses)),
                activation_cosine_similarity=float(np.mean(cos_sims)),
                activation_relative_error=float(np.mean(rel_errs)),
            )
        )

    return records


# ── logit/loss experiment ─────────────────────────────────────────────────────


def _run_logit_experiment(
    config: TransformerConfig,
    model,
    token_inputs: list,
    layers: dict,
    all_method_deqs: dict[str, dict[tuple[str, int], np.ndarray]],
    scope: str,
) -> list[LogitRecord]:
    """Run full-model forward passes: original once, then once per (method, bitwidth)."""
    orig_logits, orig_loss = _forward_pass(model, token_inputs)
    orig_perplexity = math.exp(orig_loss)

    method_keys = list(next(iter(all_method_deqs.values())).keys())
    records: list[LogitRecord] = []

    for (method, bitwidth) in method_keys:
        saved: dict[str, np.ndarray] = {}
        for lname, module in layers.items():
            saved[lname] = _extract_weight(module).copy()
            _set_weight(module, all_method_deqs[lname][(method, bitwidth)])

        q_logits, q_loss = _forward_pass(model, token_inputs)

        for lname, module in layers.items():
            _set_weight(module, saved[lname])

        o_flat = np.concatenate([l.flatten() for l in orig_logits])
        q_flat = np.concatenate([l.flatten() for l in q_logits])
        logit_mse = float(np.mean((o_flat - q_flat) ** 2))
        norm_o = np.linalg.norm(o_flat)
        norm_q = np.linalg.norm(q_flat)
        logit_cos = float(np.dot(o_flat, q_flat) / (norm_o * norm_q + 1e-8))
        top5 = _top5_overlap(orig_logits, q_logits)
        q_perplexity = math.exp(q_loss)

        records.append(
            LogitRecord(
                model_name=config.model_name,
                scope=scope,
                method=method,
                bitwidth=bitwidth,
                logit_mse=logit_mse,
                logit_cosine_similarity=logit_cos,
                top5_token_overlap=top5,
                loss=q_loss,
                original_loss=orig_loss,
                loss_delta=q_loss - orig_loss,
                perplexity=q_perplexity,
                original_perplexity=orig_perplexity,
                perplexity_ratio=q_perplexity / orig_perplexity,
            )
        )

    return records


def _forward_pass(
    model, token_inputs: list
) -> tuple[list[np.ndarray], float]:
    logits_list = []
    total_loss = 0.0
    with torch.no_grad():
        for ids in token_inputs:
            out = model(ids, labels=ids)
            logits_list.append(out.logits.squeeze(0).cpu().float().numpy())
            total_loss += float(out.loss.item())
    avg_loss = total_loss / len(token_inputs) if token_inputs else 0.0
    return logits_list, avg_loss


def _top5_overlap(
    orig_list: list[np.ndarray],
    q_list: list[np.ndarray],
    k: int = 5,
) -> float:
    overlaps = []
    for orig, q in zip(orig_list, q_list):
        for t in range(orig.shape[0]):
            top_o = set(np.argsort(orig[t])[-k:])
            top_q = set(np.argsort(q[t])[-k:])
            overlaps.append(len(top_o & top_q) / k)
    return float(np.mean(overlaps)) if overlaps else 1.0


# ── model utilities ───────────────────────────────────────────────────────────


def _get_module(model, name: str):
    obj = model
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _get_linear_layers(model) -> dict:
    """Return all Conv1D and nn.Linear layers, excluding embeddings and lm_head."""
    layers = {}
    for name, module in model.named_modules():
        if name.split(".")[-1] in _SKIP_LAYER_NAMES:
            continue
        if type(module).__name__ == "Conv1D":
            layers[name] = module
        elif isinstance(module, torch.nn.Linear) and not isinstance(
            module, torch.nn.Embedding
        ):
            layers[name] = module
    return layers


def _extract_weight(module) -> np.ndarray:
    """Return weight as a contiguous float32 copy in (in_features, out_features) layout.

    Always returns a copy so callers are not inadvertently holding a view into the
    module's parameter storage (which would change if copy_() is called later).
    """
    w = module.weight.detach().float().numpy()
    if isinstance(module, torch.nn.Linear):
        w = w.T  # (out, in) → (in, out)
    return w.copy()  # always a fresh contiguous array


def _set_weight(module, w: np.ndarray) -> None:
    """Set weight from (in_features, out_features) numpy array."""
    # Transpose the numpy array before tensor creation so the resulting tensor
    # is contiguous — transposing a tensor produces a non-contiguous view that
    # copy_() does not always apply correctly.
    w_store = w.T.copy() if isinstance(module, torch.nn.Linear) else w
    with torch.no_grad():
        module.weight.copy_(torch.tensor(w_store, dtype=module.weight.dtype))


def _clear_hf_cache(model_name: str) -> None:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    target = cache_dir / slug
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        print(f"  Cleared HF cache: {target}")


# ── quantization helpers ──────────────────────────────────────────────────────


def _q_global(weight: np.ndarray, bitwidth: int):
    return quantize_int4(weight) if bitwidth == 4 else quantize_int8(weight)


def _q_row_grouped(weight: np.ndarray, group_size: int, bitwidth: int):
    if bitwidth == 4:
        return quantize_int4_row_grouped(weight, row_group_size=group_size)
    return quantize_int8_row_grouped(weight, row_group_size=group_size)


def _resolve_group_sizes(
    fixed: list[int], fractions: list[float], n_rows: int
) -> list[int]:
    """Combine fixed group sizes with fraction-derived sizes, deduplicated."""
    dynamic = [max(1, round(n_rows * f)) for f in fractions]
    return list(dict.fromkeys(fixed + dynamic))


# ── CSV writers ───────────────────────────────────────────────────────────────


def _write_weight_csv(path: Path, records: list[WeightRecord]) -> None:
    if not records:
        return
    fields = [
        "model_name", "layer_name", "weight_shape", "method", "bitwidth",
        "rotation_count", "rotation_pair_fraction", "rotation_candidate_fraction",
        "mse", "relative_frobenius_error", "cosine_similarity",
        "snr_db", "zero_fraction", "saturation_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})


def _write_activation_csv(path: Path, records: list[ActivationRecord]) -> None:
    if not records:
        return
    fields = [
        "model_name", "layer_name", "method", "bitwidth",
        "activation_mse", "activation_cosine_similarity", "activation_relative_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})


def _write_logit_csv(path: Path, records: list[LogitRecord]) -> None:
    if not records:
        return
    fields = [
        "model_name", "scope", "method", "bitwidth",
        "logit_mse", "logit_cosine_similarity", "top5_token_overlap",
        "loss", "original_loss", "loss_delta",
        "perplexity", "original_perplexity", "perplexity_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})


# ── dashboard ─────────────────────────────────────────────────────────────────


def _plot_dashboard(
    weight_records: list[WeightRecord],
    activation_records: list[ActivationRecord],
    logit_records: list[LogitRecord],
    title: str = "Transformer quantization — Milestone 3 dashboard",
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=12)

    # Log-scale floor: lossless methods (g1) have MSE=0; clip to this value so
    # they appear as a minimal bar at the left edge rather than breaking log scale.
    _LOG_FLOOR = 1e-20

    def _label(method: str, bitwidth: int) -> str:
        return f"{method} (INT{bitwidth})"

    bitwidths = sorted({r.bitwidth for r in weight_records})
    baseline_bitwidth = bitwidths[0] if len(bitwidths) == 1 else 4

    # Panel 1: weight MSE ratio vs global baseline (log scale)
    ax = axes[0, 0]
    global_baseline_mse = float(
        np.mean(
            [
                r.mse
                for r in weight_records
                if r.method == "global" and r.bitwidth == baseline_bitwidth
            ]
            or [1.0]
        )
    )
    seen: dict[tuple[str, int], float] = {}
    for r in weight_records:
        key = (r.method, r.bitwidth)
        seen[key] = float(np.mean([r2.mse for r2 in weight_records if r2.method == r.method and r2.bitwidth == r.bitwidth]))
    sorted_keys = sorted(seen, key=lambda k: (k[1], seen[k]))
    labels = [_label(m, bw) for m, bw in sorted_keys]
    ratios = [
        seen[k] / global_baseline_mse if global_baseline_mse > 0 else 1.0
        for k in sorted_keys
    ]
    colors = ["tab:green" if v < 1.0 else "tab:red" for v in ratios]
    ax.barh(labels, [max(v, _LOG_FLOOR) for v in ratios], color=colors, alpha=0.85)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(f"MSE / Global INT{baseline_bitwidth} MSE  (log scale, < 1 = improvement)")
    ax.set_title("Weight reconstruction MSE ratio")

    # Panel 2: activation drift (MSE) per method × bitwidth (log scale)
    ax = axes[0, 1]
    if activation_records:
        act_seen: dict[tuple[str, int], float] = {}
        for r in activation_records:
            key = (r.method, r.bitwidth)
            vals = [r2.activation_mse for r2 in activation_records if r2.method == r.method and r2.bitwidth == r.bitwidth]
            act_seen[key] = float(np.mean(vals))
        sorted_act = sorted(act_seen, key=lambda k: (k[1], act_seen[k]))
        ax.barh(
            [_label(m, bw) for m, bw in sorted_act],
            [max(act_seen[k], _LOG_FLOOR) for k in sorted_act],
            color="tab:blue",
            alpha=0.85,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Mean activation MSE (log scale)")
    ax.set_title("Activation drift (MSE)")

    # Panel 3: full-model logit MSE per method × bitwidth (log scale)
    ax = axes[1, 0]
    if logit_records:
        sorted_logit = sorted(
            logit_records, key=lambda r: (r.bitwidth, r.logit_mse)
        )
        ax.barh(
            [_label(r.method, r.bitwidth) for r in sorted_logit],
            [max(r.logit_mse, _LOG_FLOOR) for r in sorted_logit],
            color="tab:orange",
            alpha=0.85,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Logit MSE vs original (log scale)")
    ax.set_title("Full-model logit drift (MSE)")

    # Panel 4: next-token loss delta per method × bitwidth
    ax = axes[1, 1]
    if logit_records:
        colors = [
            "tab:green" if r.loss_delta <= 0 else "tab:red"
            for r in sorted_logit
        ]
        ax.barh(
            [_label(r.method, r.bitwidth) for r in sorted_logit],
            [r.loss_delta for r in sorted_logit],
            color=colors,
            alpha=0.85,
        )
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Loss delta (quantized − original)")
    ax.set_title("Next-token loss delta")

    fig.tight_layout()
    return fig


# ── helpers ───────────────────────────────────────────────────────────────────


def _fraction_tag(value: float) -> str:
    percent = value * 100.0
    if float(percent).is_integer():
        return f"p{int(percent)}"
    return f"p{str(round(percent, 4)).replace('.', '_')}"


def print_summary(
    weight_records: list[WeightRecord],
    activation_records: list[ActivationRecord],
    logit_records: list[LogitRecord],
) -> None:
    print(
        f"\n{'Layer':<30} {'Method':<38} {'BW':>4} "
        f"{'MSE':>12} {'CosS':>8} {'SNR(dB)':>9}"
    )
    print("-" * 106)
    for r in sorted(weight_records, key=lambda r: (r.layer_name, r.bitwidth, r.method)):
        print(
            f"{r.layer_name:<30} {r.method:<38} {r.bitwidth:>4} "
            f"{r.mse:>12.6f} {r.cosine_similarity:>8.4f} {r.snr_db:>9.2f}"
        )

    if activation_records:
        print(
            f"\n{'Layer':<30} {'Method':<38} {'BW':>4} "
            f"{'Act MSE':>12} {'Act Cos':>9}"
        )
        print("-" * 97)
        for r in sorted(
            activation_records, key=lambda r: (r.layer_name, r.bitwidth, r.method)
        ):
            print(
                f"{r.layer_name:<30} {r.method:<38} {r.bitwidth:>4} "
                f"{r.activation_mse:>12.6f} {r.activation_cosine_similarity:>9.4f}"
            )

    if logit_records:
        print(
            f"\n{'Method':<38} {'BW':>4} {'Logit MSE':>12} "
            f"{'Top5':>8} {'Loss':>10} {'ΔLoss':>10} {'PPL':>10} {'PPLx':>8}"
        )
        print("-" * 106)
        for r in sorted(logit_records, key=lambda r: (r.bitwidth, r.method)):
            print(
                f"{r.method:<38} {r.bitwidth:>4} {r.logit_mse:>12.6f} "
                f"{r.top5_token_overlap:>8.4f} {r.loss:>10.4f} "
                f"{r.loss_delta:>10.4f} {r.perplexity:>10.4f} "
                f"{r.perplexity_ratio:>8.4f}"
            )


def main() -> None:
    config = TransformerConfig()
    wr, ar, lr = run_transformer_experiment(config)
    print_summary(wr, ar, lr)


if __name__ == "__main__":
    main()

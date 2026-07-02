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
  local_files_only       — load Hugging Face artifacts from local cache only
  incremental_results    — append CSV records during the run for partial recovery
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
DEFAULT_TEXT_SOURCE = "built-in calibration texts"

_SKIP_LAYER_NAMES = {"lm_head"}


@dataclass
class TransformerConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    calibration_texts: list[str] = field(
        default_factory=lambda: list(CALIBRATION_TEXTS)
    )
    calibration_text_source: str = DEFAULT_TEXT_SOURCE
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
    local_files_only: bool = False
    incremental_results: bool = False
    delete_hf_cache_after: bool = False
    device: str = "auto"
    # Optional progress callback: on_progress(phase, done, total)
    # phase is "layer" or "logit"; done and total are int counts.
    on_progress: object = field(default=None, repr=False, compare=False)


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
    calibration_text_source: str
    calibration_text_count: int
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
    device = _resolve_torch_device(config.device)
    print(f"Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not config.calibration_texts:
        raise ValueError("TransformerConfig.calibration_texts must not be empty")

    token_inputs = [
        tokenizer(t, return_tensors="pt")["input_ids"].to(device)
        for t in config.calibration_texts
    ]

    if config.single_layer_name is not None:
        layers = {
            config.single_layer_name: _get_module(model, config.single_layer_name)
        }
    else:
        layers = _get_linear_layers(model)

    print(f"Processing {len(layers)} layer(s): {list(layers.keys())}")
    max_total_pairs = _max_total_channel_pairs(layers)
    top_width_pair_fractions = _effective_top_width_pair_fractions(
        config.top_width_pair_fractions,
        max_total_pairs=max_total_pairs,
        max_rotation_pairs=config.max_rotation_pairs,
    )

    weight_records: list[WeightRecord] = []
    activation_records: list[ActivationRecord] = []
    config.results_dir.mkdir(parents=True, exist_ok=True)
    weight_csv_path = config.results_dir / "transformer_weight_metrics.csv"
    activation_csv_path = config.results_dir / "transformer_activation_metrics.csv"
    logit_csv_path = config.results_dir / "transformer_logit_metrics.csv"
    if config.incremental_results:
        _reset_incremental_csvs(weight_csv_path, activation_csv_path, logit_csv_path)

    method_keys_by_layer: dict[str, dict[tuple[str, int], None]] = {}
    n_layers = len(layers)

    for layer_idx, (layer_name, module) in enumerate(layers.items()):
        w = _extract_weight(module)
        print(f"  {layer_name}: shape={w.shape}")
        wr, method_deqs = _run_weight_experiment(
            config,
            layer_name,
            w,
            top_width_pair_fractions=top_width_pair_fractions,
        )
        weight_records.extend(wr)
        ar = _run_activation_experiment(
            model,
            token_inputs,
            layer_name,
            module,
            w,
            method_deqs,
            config.model_name,
        )
        activation_records.extend(ar)
        if config.incremental_results:
            _append_weight_csv(weight_csv_path, wr)
            _append_activation_csv(activation_csv_path, ar)
        method_keys_by_layer[layer_name] = dict.fromkeys(method_deqs)
        del method_deqs
        if config.on_progress is not None:
            config.on_progress("layer", layer_idx + 1, n_layers)

    scope = (
        f"single_layer:{config.single_layer_name}"
        if config.single_layer_name is not None
        else "all_layers"
    )
    logit_records = _run_logit_experiment(
        config,
        model,
        token_inputs,
        layers,
        method_keys_by_layer,
        scope,
        top_width_pair_fractions=top_width_pair_fractions,
        incremental_logit_path=logit_csv_path if config.incremental_results else None,
        on_progress=config.on_progress,
    )

    if not config.incremental_results:
        _write_weight_csv(weight_csv_path, weight_records)
        _write_activation_csv(activation_csv_path, activation_records)
        _write_logit_csv(logit_csv_path, logit_records)

    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)
        fig = _plot_dashboard(weight_records, activation_records, logit_records)
        fig.savefig(
            config.plots_dir / "transformer_dashboard.png",
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
    top_width_pair_fractions: Optional[list[float]] = None,
) -> tuple[list[WeightRecord], dict[tuple[str, int], np.ndarray]]:
    """Quantize weight with all configured paths and bitwidths."""
    records: list[WeightRecord] = []
    method_deqs: dict[tuple[str, int], np.ndarray] = {}
    n_rows, n_cols = weight.shape
    total_pairs = n_cols * (n_cols - 1) // 2
    all_group_sizes = _resolve_group_sizes(
        config.row_group_sizes, config.row_group_fractions, n_rows
    )
    if top_width_pair_fractions is None:
        top_width_pair_fractions = config.top_width_pair_fractions

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

    rotation_preps = []
    if n_cols >= 2:
        for p in top_width_pair_fractions:
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
            inv_rots = [
                GivensRotation(i=rot.i, j=rot.j, theta=-rot.theta)
                for rot in reversed(rotations)
            ]
            rot_frac = float(len(rotations) / total_pairs) if total_pairs > 0 else 0.0
            top_rot_scaled, top_rot_scaling = balance_channel_max_abs(top_rotated)
            rotation_preps.append(
                (
                    tag,
                    top_rot_scaled,
                    top_rot_scaling,
                    inv_rots,
                    len(rotations),
                    rot_frac,
                    float(p),
                )
            )

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
        for (
            tag,
            top_rot_scaled,
            top_rot_scaling,
            inv_rots,
            n_rot,
            rot_frac,
            p,
        ) in rotation_preps:
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
                    rot_cand=p,
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
    method_keys_by_layer: dict[str, dict[tuple[str, int], object]],
    scope: str,
    top_width_pair_fractions: Optional[list[float]] = None,
    incremental_logit_path: Optional[Path] = None,
    on_progress=None,
) -> list[LogitRecord]:
    """Run full-model passes while regenerating one method's weights at a time."""
    orig_logits, orig_loss = _forward_pass(model, token_inputs)
    orig_perplexity = math.exp(orig_loss)

    method_keys = _common_method_keys(method_keys_by_layer)
    n_methods = len(method_keys)
    original_weights = {
        lname: _extract_weight(module).copy() for lname, module in layers.items()
    }
    records: list[LogitRecord] = []

    for method_idx, (method, bitwidth) in enumerate(method_keys):
        try:
            for lname, module in layers.items():
                deq = _dequantize_method(
                    config,
                    original_weights[lname],
                    method,
                    bitwidth,
                    top_width_pair_fractions=top_width_pair_fractions,
                )
                _set_weight(module, deq)

            q_logits, q_loss = _forward_pass(model, token_inputs)
        finally:
            for lname, module in layers.items():
                _set_weight(module, original_weights[lname])

        o_flat = np.concatenate([l.flatten() for l in orig_logits])
        q_flat = np.concatenate([l.flatten() for l in q_logits])
        logit_mse = float(np.mean((o_flat - q_flat) ** 2))
        norm_o = np.linalg.norm(o_flat)
        norm_q = np.linalg.norm(q_flat)
        logit_cos = float(np.dot(o_flat, q_flat) / (norm_o * norm_q + 1e-8))
        top5 = _top5_overlap(orig_logits, q_logits)
        q_perplexity = math.exp(q_loss)

        record = LogitRecord(
            model_name=config.model_name,
            scope=scope,
            method=method,
            bitwidth=bitwidth,
            calibration_text_source=config.calibration_text_source,
            calibration_text_count=len(config.calibration_texts),
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
        records.append(record)
        if incremental_logit_path is not None:
            _append_logit_csv(incremental_logit_path, [record])
        if on_progress is not None:
            on_progress("logit", method_idx + 1, n_methods)

    return records


def load_text_batch(path: Path, max_texts: Optional[int] = None) -> list[str]:
    """Load paragraph-separated evaluation texts from a UTF-8 resource file.

    Lines beginning with ``#`` are metadata/comments and are ignored. Blank lines
    separate examples, matching the tracked research-resource files.
    """
    if max_texts is not None and max_texts < 1:
        raise ValueError("max_texts must be at least 1 when provided")

    raw = path.read_text(encoding="utf-8")
    texts: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            text = " ".join(part.strip() for part in current).strip()
            if text:
                texts.append(text)
            current.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_current()
            if max_texts is not None and len(texts) >= max_texts:
                break
            continue
        if stripped.startswith("#"):
            continue
        current.append(stripped)

    if max_texts is None or len(texts) < max_texts:
        flush_current()

    if max_texts is not None:
        texts = texts[:max_texts]
    if not texts:
        raise ValueError(f"No evaluation texts found in {path}")
    return texts


def _common_method_keys(
    all_method_deqs: dict[str, dict[tuple[str, int], object]],
) -> list[tuple[str, int]]:
    """Return method keys available for every layer, preserving first-layer order."""
    if not all_method_deqs:
        return []

    layer_methods = list(all_method_deqs.values())
    common = set(layer_methods[0])
    for methods in layer_methods[1:]:
        common.intersection_update(methods)

    return [key for key in layer_methods[0] if key in common]


def _dequantize_method(
    config: TransformerConfig,
    weight: np.ndarray,
    method: str,
    bitwidth: int,
    top_width_pair_fractions: Optional[list[float]] = None,
) -> np.ndarray:
    """Regenerate one dequantized method for one layer."""
    if method == "global":
        return _q_global(weight, bitwidth).dequantized

    if method.startswith("row_grouped_g"):
        group_size = _parse_group_size(method, prefix="row_grouped_g")
        return _q_row_grouped(weight, group_size, bitwidth).dequantized

    if method.startswith("scale_row_g"):
        group_size = _parse_group_size(method, prefix="scale_row_g")
        scaled, scaling = balance_channel_max_abs(weight)
        r = _q_row_grouped(scaled, group_size, bitwidth)
        return invert_channel_scaling(r.dequantized, scaling)

    if method.startswith("top_width_rotate_") and "_scale_row_g" in method:
        tag, group_size = _parse_top_width_method(method)
        if top_width_pair_fractions is None:
            top_width_pair_fractions = config.top_width_pair_fractions
        fraction_by_tag = {
            _fraction_tag(float(fraction)): float(fraction)
            for fraction in top_width_pair_fractions
        }
        if tag not in fraction_by_tag:
            raise ValueError(f"unknown top-width rotation tag: {tag}")

        top_rotated, rotations = rotate_top_width_pairs(
            weight.astype(np.float64),
            top_fraction=fraction_by_tag[tag],
            independent=True,
            n_search=config.rotation_search_steps,
        )
        top_rotated = top_rotated.astype(weight.dtype)
        inv_rots = [
            GivensRotation(i=rot.i, j=rot.j, theta=-rot.theta)
            for rot in reversed(rotations)
        ]
        top_rot_scaled, top_rot_scaling = balance_channel_max_abs(top_rotated)
        r = _q_row_grouped(top_rot_scaled, group_size, bitwidth)
        deq = invert_channel_scaling(r.dequantized, top_rot_scaling)
        return apply_sequential_rotations(deq.astype(np.float64), inv_rots).astype(
            weight.dtype
        )

    raise ValueError(f"unknown quantization method: {method}")


def _parse_group_size(method: str, prefix: str) -> int:
    try:
        group_size = int(method.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"invalid group size in method: {method}") from exc
    if group_size < 1:
        raise ValueError(f"group size must be positive in method: {method}")
    return group_size


def _parse_top_width_method(method: str) -> tuple[str, int]:
    prefix = "top_width_rotate_"
    suffix = "_scale_row_g"
    body = method.removeprefix(prefix)
    tag, sep, group_text = body.partition(suffix)
    if not sep or not tag:
        raise ValueError(f"invalid top-width method: {method}")
    try:
        group_size = int(group_text)
    except ValueError as exc:
        raise ValueError(f"invalid group size in method: {method}") from exc
    if group_size < 1:
        raise ValueError(f"group size must be positive in method: {method}")
    return tag, group_size


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


def _resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cpu, cuda")


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


def _max_total_channel_pairs(layers: dict) -> int:
    """Return the largest channel-pair count among selected quantized layers."""
    max_pairs = 0
    for module in layers.values():
        _, n_cols = _extract_weight(module).shape
        total_pairs = n_cols * (n_cols - 1) // 2
        max_pairs = max(max_pairs, total_pairs)
    return max_pairs


def _extract_weight(module) -> np.ndarray:
    """Return weight as a contiguous float32 copy in (in_features, out_features) layout.

    Always returns a copy so callers are not inadvertently holding a view into the
    module's parameter storage (which would change if copy_() is called later).
    """
    w = module.weight.detach().float().cpu().numpy()
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
        module.weight.copy_(torch.as_tensor(w_store, dtype=module.weight.dtype, device=module.weight.device))


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


def _effective_top_width_pair_fractions(
    configured_fractions: list[float],
    max_total_pairs: int,
    max_rotation_pairs: int,
) -> list[float]:
    """Cap top-width fractions so every selected layer can run the same paths."""
    if max_total_pairs <= 0 or max_rotation_pairs <= 0:
        return []

    cap_fraction = max_rotation_pairs / max_total_pairs
    effective = [
        min(float(fraction), cap_fraction)
        for fraction in configured_fractions
        if float(fraction) > 0.0
    ]
    return list(dict.fromkeys(effective))


# ── CSV writers ───────────────────────────────────────────────────────────────

_WEIGHT_CSV_FIELDS = [
    "model_name", "layer_name", "weight_shape", "method", "bitwidth",
    "rotation_count", "rotation_pair_fraction", "rotation_candidate_fraction",
    "mse", "relative_frobenius_error", "cosine_similarity",
    "snr_db", "zero_fraction", "saturation_fraction",
]

_ACTIVATION_CSV_FIELDS = [
    "model_name", "layer_name", "method", "bitwidth",
    "activation_mse", "activation_cosine_similarity", "activation_relative_error",
]

_LOGIT_CSV_FIELDS = [
    "model_name", "scope", "method", "bitwidth",
    "calibration_text_source", "calibration_text_count",
    "logit_mse", "logit_cosine_similarity", "top5_token_overlap",
    "loss", "original_loss", "loss_delta",
    "perplexity", "original_perplexity", "perplexity_ratio",
]


def _write_weight_csv(path: Path, records: list[WeightRecord]) -> None:
    _write_records_csv(path, _WEIGHT_CSV_FIELDS, records)


def _write_activation_csv(path: Path, records: list[ActivationRecord]) -> None:
    _write_records_csv(path, _ACTIVATION_CSV_FIELDS, records)


def _write_logit_csv(path: Path, records: list[LogitRecord]) -> None:
    _write_records_csv(path, _LOGIT_CSV_FIELDS, records)


def _append_weight_csv(path: Path, records: list[WeightRecord]) -> None:
    _append_records_csv(path, _WEIGHT_CSV_FIELDS, records)


def _append_activation_csv(path: Path, records: list[ActivationRecord]) -> None:
    _append_records_csv(path, _ACTIVATION_CSV_FIELDS, records)


def _append_logit_csv(path: Path, records: list[LogitRecord]) -> None:
    _append_records_csv(path, _LOGIT_CSV_FIELDS, records)


def _write_records_csv(path: Path, fields: list[str], records: list[object]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})


def _append_records_csv(path: Path, fields: list[str], records: list[object]) -> None:
    if not records:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})


def _reset_incremental_csvs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


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
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Loss delta (quantized − original, symmetric log scale)")
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

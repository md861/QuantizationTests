"""Optional bitsandbytes external baseline for Milestone 4.

This runner compares a Transformers bitsandbytes NF4 model against the original
model on the same text batch. It intentionally reports only shared end-to-end
metrics (logits, loss, perplexity), because bitsandbytes uses quantized runtime
modules rather than dequantized NumPy weights from the project quantizers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.transformer_experiment import (
    CALIBRATION_TEXTS,
    DEFAULT_TEXT_SOURCE,
    LogitRecord,
    _forward_pass,
    _resolve_torch_device,
    _top5_overlap,
    load_text_batch,
)


_LOGIT_CSV_FIELDS = [
    "model_name",
    "scope",
    "method",
    "bitwidth",
    "calibration_text_source",
    "calibration_text_count",
    "logit_mse",
    "logit_cosine_similarity",
    "top5_token_overlap",
    "loss",
    "original_loss",
    "loss_delta",
    "perplexity",
    "original_perplexity",
    "perplexity_ratio",
]


@dataclass
class BitsAndBytesBaselineConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    evaluation_texts: list[str] = field(default_factory=lambda: list(CALIBRATION_TEXTS))
    evaluation_text_source: str = DEFAULT_TEXT_SOURCE
    results_dir: Path = field(
        default_factory=lambda: Path("results/bitsandbytes_tinyllama_nf4_baseline")
    )
    local_files_only: bool = False
    device: str = "auto"
    compute_dtype: str = "float16"
    quant_type: str = "nf4"
    double_quant: bool = False


def run_bitsandbytes_baseline(config: BitsAndBytesBaselineConfig) -> LogitRecord:
    if not config.evaluation_texts:
        raise ValueError("BitsAndBytesBaselineConfig.evaluation_texts must not be empty")

    device = _resolve_torch_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("bitsandbytes NF4 baseline requires CUDA for this project run")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_inputs = [
        tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        for text in config.evaluation_texts
    ]

    original_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
    )
    original_model.to(device)
    original_model.eval()
    original_logits, original_loss = _forward_pass(original_model, token_inputs)
    del original_model
    torch.cuda.empty_cache()

    quantization_config = _make_bitsandbytes_config(config)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        quantization_config=quantization_config,
        device_map="auto",
    )
    quantized_model.eval()
    quantized_device = _first_parameter_device(quantized_model) or device
    quantized_inputs = [ids.to(quantized_device) for ids in token_inputs]
    quantized_logits, quantized_loss = _forward_pass(quantized_model, quantized_inputs)

    record = _make_logit_record(
        config=config,
        original_logits=original_logits,
        original_loss=original_loss,
        quantized_logits=quantized_logits,
        quantized_loss=quantized_loss,
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    _write_logit_csv(config.results_dir / "bitsandbytes_logit_metrics.csv", [record])
    return record


def _make_bitsandbytes_config(config: BitsAndBytesBaselineConfig, config_cls=None):
    if config_cls is None:
        try:
            from transformers import BitsAndBytesConfig as config_cls
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes baseline requires a Transformers install with "
                "BitsAndBytesConfig plus the optional bitsandbytes package"
            ) from exc

    return config_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_compute_dtype=_resolve_compute_dtype(config.compute_dtype),
        bnb_4bit_use_double_quant=config.double_quant,
    )


def _resolve_compute_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as exc:
        raise ValueError("compute dtype must be one of: float16, bfloat16, float32") from exc


def _first_parameter_device(model) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _make_logit_record(
    *,
    config: BitsAndBytesBaselineConfig,
    original_logits: list[np.ndarray],
    original_loss: float,
    quantized_logits: list[np.ndarray],
    quantized_loss: float,
) -> LogitRecord:
    original_flat = np.concatenate([logits.flatten() for logits in original_logits])
    quantized_flat = np.concatenate([logits.flatten() for logits in quantized_logits])
    logit_mse = float(np.mean((original_flat - quantized_flat) ** 2))
    norm_original = np.linalg.norm(original_flat)
    norm_quantized = np.linalg.norm(quantized_flat)
    logit_cos = float(
        np.dot(original_flat, quantized_flat) / (norm_original * norm_quantized + 1e-8)
    )
    original_perplexity = math.exp(original_loss)
    quantized_perplexity = math.exp(quantized_loss)

    return LogitRecord(
        model_name=config.model_name,
        scope="external_baseline",
        method=f"external_bitsandbytes_{config.quant_type}_{config.compute_dtype}",
        bitwidth=4,
        calibration_text_source=config.evaluation_text_source,
        calibration_text_count=len(config.evaluation_texts),
        logit_mse=logit_mse,
        logit_cosine_similarity=logit_cos,
        top5_token_overlap=_top5_overlap(original_logits, quantized_logits),
        loss=quantized_loss,
        original_loss=original_loss,
        loss_delta=quantized_loss - original_loss,
        perplexity=quantized_perplexity,
        original_perplexity=original_perplexity,
        perplexity_ratio=quantized_perplexity / original_perplexity,
    )


def _write_logit_csv(path: Path, records: list[LogitRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOGIT_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in _LOGIT_CSV_FIELDS})


def _git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _cuda_memory_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024 * 1024), 3)


def _collect_metadata(
    *,
    config: BitsAndBytesBaselineConfig,
    elapsed_seconds: float,
    record: LogitRecord,
) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    vram_total_mb = None
    peak_allocated_mb = None
    peak_reserved_mb = None
    if cuda_available:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        gpu_name = props.name
        vram_total_mb = _cuda_memory_mb(props.total_memory)
        peak_allocated_mb = _cuda_memory_mb(torch.cuda.max_memory_allocated(device_index))
        peak_reserved_mb = _cuda_memory_mb(torch.cuda.max_memory_reserved(device_index))

    return {
        "baseline": "bitsandbytes",
        "method": record.method,
        "model_name": config.model_name,
        "device_request": config.device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb,
        "cuda_peak_memory_allocated_mb": peak_allocated_mb,
        "cuda_peak_memory_reserved_mb": peak_reserved_mb,
        "evaluation_text_source": config.evaluation_text_source,
        "evaluation_text_count": len(config.evaluation_texts),
        "quant_type": config.quant_type,
        "compute_dtype": config.compute_dtype,
        "double_quant": config.double_quant,
        "commit_hash": _git_commit_hash(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "counts": {"logit": 1},
    }


def _write_metadata(results_dir: Path, metadata: dict[str, object]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "bitsandbytes_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the optional bitsandbytes NF4 external baseline.",
    )
    parser.add_argument(
        "--model-name",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face causal LM to evaluate.",
    )
    parser.add_argument(
        "--eval-text-file",
        type=Path,
        default=None,
        help="Paragraph-separated UTF-8 evaluation text file.",
    )
    parser.add_argument(
        "--max-eval-texts",
        type=int,
        default=None,
        help="Use at most this many texts from --eval-text-file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/bitsandbytes_tinyllama_nf4_baseline"),
        help="Directory for bnb CSV and metadata.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda"],
        default="auto",
        help="Device request; NF4 baseline is intended for CUDA.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="bnb 4-bit compute dtype.",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        help="Enable nested/double quantization in bitsandbytes.",
    )
    return parser


def build_config(args: argparse.Namespace) -> BitsAndBytesBaselineConfig:
    evaluation_texts = list(CALIBRATION_TEXTS)
    evaluation_text_source = DEFAULT_TEXT_SOURCE
    if args.eval_text_file is not None:
        evaluation_texts = load_text_batch(args.eval_text_file, args.max_eval_texts)
        evaluation_text_source = str(args.eval_text_file)

    return BitsAndBytesBaselineConfig(
        model_name=args.model_name,
        evaluation_texts=evaluation_texts,
        evaluation_text_source=evaluation_text_source,
        results_dir=args.results_dir,
        local_files_only=bool(args.local_files_only),
        device=args.device,
        compute_dtype=args.compute_dtype,
        double_quant=bool(args.double_quant),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)

    print(f"bitsandbytes baseline model: {config.model_name}")
    print(f"results: {config.results_dir}")
    print(f"evaluation texts: {len(config.evaluation_texts)} from {config.evaluation_text_source}")
    print(f"compute dtype: {config.compute_dtype}")
    print(f"double quant: {config.double_quant}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    record = run_bitsandbytes_baseline(config)
    elapsed = time.time() - t0
    metadata = _collect_metadata(config=config, elapsed_seconds=elapsed, record=record)
    metadata_path = _write_metadata(config.results_dir, metadata)
    print(
        "logit_mse={:.6g} top5={:.3f} loss_delta={:.6g} ppl_ratio={:.6g}".format(
            record.logit_mse,
            record.top5_token_overlap,
            record.loss_delta,
            record.perplexity_ratio,
        )
    )
    print(f"metadata: {metadata_path}")
    print(f"elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Optional GPTQ external baseline for Milestone 4.

This runner compares a pre-quantized GPTQ Transformers checkpoint against the
original model on the same text batch. It reports shared end-to-end metrics
only, keeping GPTQ's packed checkpoint/runtime path separate from the project's
dequantized NumPy weight artifacts.
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
    _count_input_tokens,
    _forward_pass,
    _resolve_torch_device,
    _throughput_metrics,
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
    "method_elapsed_seconds",
    "method_cuda_peak_allocated_mb",
    "method_cuda_peak_reserved_mb",
    "total_input_tokens",
    "method_tokens_per_second",
    "method_ms_per_token",
    "reference_weight_bytes",
    "estimated_packed_weight_bytes",
    "estimated_scale_metadata_bytes",
    "estimated_total_artifact_bytes",
]


@dataclass
class GptqBaselineConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    gptq_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    evaluation_texts: list[str] = field(default_factory=lambda: list(CALIBRATION_TEXTS))
    evaluation_text_source: str = DEFAULT_TEXT_SOURCE
    results_dir: Path = field(
        default_factory=lambda: Path("results/gptq_tinyllama_baseline")
    )
    local_files_only: bool = False
    device: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True


def run_gptq_baseline(config: GptqBaselineConfig) -> LogitRecord:
    if not config.evaluation_texts:
        raise ValueError("GptqBaselineConfig.evaluation_texts must not be empty")

    device = _resolve_torch_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("GPTQ baseline requires CUDA for this project run")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_inputs = [
        tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        for text in config.evaluation_texts
    ]
    total_input_tokens = _count_input_tokens(token_inputs)

    original_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    original_model.to(device)
    original_model.eval()
    original_logits, original_loss = _forward_pass(original_model, token_inputs)
    del original_model
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    method_t0 = time.time()
    gptq_model = AutoModelForCausalLM.from_pretrained(
        config.gptq_model_name,
        local_files_only=config.local_files_only,
        torch_dtype=_resolve_torch_dtype(config.torch_dtype),
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=config.low_cpu_mem_usage,
    )
    gptq_model.eval()
    gptq_device = _first_parameter_device(gptq_model) or device
    gptq_inputs = [ids.to(gptq_device) for ids in token_inputs]
    gptq_logits, gptq_loss = _forward_pass(gptq_model, gptq_inputs)
    method_elapsed = time.time() - method_t0

    method_peak_allocated_mb = None
    method_peak_reserved_mb = None
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        method_peak_allocated_mb = _cuda_memory_mb(
            torch.cuda.max_memory_allocated(device_index)
        )
        method_peak_reserved_mb = _cuda_memory_mb(
            torch.cuda.max_memory_reserved(device_index)
        )
    tokens_per_second, ms_per_token = _throughput_metrics(
        total_input_tokens, method_elapsed
    )

    record = _make_logit_record(
        config=config,
        original_logits=original_logits,
        original_loss=original_loss,
        gptq_logits=gptq_logits,
        gptq_loss=gptq_loss,
        method_elapsed_seconds=round(method_elapsed, 3),
        method_cuda_peak_allocated_mb=method_peak_allocated_mb,
        method_cuda_peak_reserved_mb=method_peak_reserved_mb,
        total_input_tokens=total_input_tokens,
        method_tokens_per_second=tokens_per_second,
        method_ms_per_token=ms_per_token,
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    _write_logit_csv(config.results_dir / "gptq_logit_metrics.csv", [record])
    return record


def _resolve_torch_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as exc:
        raise ValueError("torch dtype must be one of: float16, bfloat16, float32") from exc


def _first_parameter_device(model) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _make_logit_record(
    *,
    config: GptqBaselineConfig,
    original_logits: list[np.ndarray],
    original_loss: float,
    gptq_logits: list[np.ndarray],
    gptq_loss: float,
    method_elapsed_seconds: Optional[float] = None,
    method_cuda_peak_allocated_mb: Optional[float] = None,
    method_cuda_peak_reserved_mb: Optional[float] = None,
    total_input_tokens: Optional[int] = None,
    method_tokens_per_second: Optional[float] = None,
    method_ms_per_token: Optional[float] = None,
) -> LogitRecord:
    original_flat = np.concatenate([logits.flatten() for logits in original_logits])
    gptq_flat = np.concatenate([logits.flatten() for logits in gptq_logits])
    logit_mse = float(np.mean((original_flat - gptq_flat) ** 2))
    norm_original = np.linalg.norm(original_flat)
    norm_gptq = np.linalg.norm(gptq_flat)
    logit_cos = float(
        np.dot(original_flat, gptq_flat) / (norm_original * norm_gptq + 1e-8)
    )
    original_perplexity = math.exp(original_loss)
    gptq_perplexity = math.exp(gptq_loss)

    return LogitRecord(
        model_name=config.model_name,
        scope="external_baseline",
        method="external_gptq_w4",
        bitwidth=4,
        calibration_text_source=config.evaluation_text_source,
        calibration_text_count=len(config.evaluation_texts),
        logit_mse=logit_mse,
        logit_cosine_similarity=logit_cos,
        top5_token_overlap=_top5_overlap(original_logits, gptq_logits),
        loss=gptq_loss,
        original_loss=original_loss,
        loss_delta=gptq_loss - original_loss,
        perplexity=gptq_perplexity,
        original_perplexity=original_perplexity,
        perplexity_ratio=gptq_perplexity / original_perplexity,
        method_elapsed_seconds=method_elapsed_seconds,
        method_cuda_peak_allocated_mb=method_cuda_peak_allocated_mb,
        method_cuda_peak_reserved_mb=method_cuda_peak_reserved_mb,
        total_input_tokens=total_input_tokens,
        method_tokens_per_second=method_tokens_per_second,
        method_ms_per_token=method_ms_per_token,
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
    config: GptqBaselineConfig,
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
        "baseline": "gptq",
        "method": record.method,
        "model_name": config.model_name,
        "gptq_model_name": config.gptq_model_name,
        "device_request": config.device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb,
        "cuda_peak_memory_allocated_mb": peak_allocated_mb,
        "cuda_peak_memory_reserved_mb": peak_reserved_mb,
        "evaluation_text_source": config.evaluation_text_source,
        "evaluation_text_count": len(config.evaluation_texts),
        "torch_dtype": config.torch_dtype,
        "trust_remote_code": config.trust_remote_code,
        "commit_hash": _git_commit_hash(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "counts": {"logit": 1},
    }


def _write_metadata(results_dir: Path, metadata: dict[str, object]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "gptq_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an optional GPTQ external baseline from a pre-quantized checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Original Hugging Face causal LM used as the reference.",
    )
    parser.add_argument(
        "--gptq-model-name",
        required=True,
        help="Pre-quantized GPTQ causal LM checkpoint to evaluate.",
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
        default=Path("results/gptq_tinyllama_baseline"),
        help="Directory for GPTQ CSV and metadata.",
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
        help="Device request; GPTQ baseline is intended for CUDA.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="dtype passed when loading the GPTQ checkpoint.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote model code if the selected GPTQ checkpoint requires it.",
    )
    return parser


def build_config(args: argparse.Namespace) -> GptqBaselineConfig:
    evaluation_texts = list(CALIBRATION_TEXTS)
    evaluation_text_source = DEFAULT_TEXT_SOURCE
    if args.eval_text_file is not None:
        evaluation_texts = load_text_batch(args.eval_text_file, args.max_eval_texts)
        evaluation_text_source = str(args.eval_text_file)

    return GptqBaselineConfig(
        model_name=args.model_name,
        gptq_model_name=args.gptq_model_name,
        evaluation_texts=evaluation_texts,
        evaluation_text_source=evaluation_text_source,
        results_dir=args.results_dir,
        local_files_only=bool(args.local_files_only),
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)

    print(f"reference model: {config.model_name}")
    print(f"GPTQ model: {config.gptq_model_name}")
    print(f"results: {config.results_dir}")
    print(f"evaluation texts: {len(config.evaluation_texts)} from {config.evaluation_text_source}")
    print(f"torch dtype: {config.torch_dtype}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    record = run_gptq_baseline(config)
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

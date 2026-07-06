"""Safer entry point for long Milestone 3 transformer benchmarks.

The presets here are intentionally conservative. They separate model download
from quantization, support local-cache-only runs, and let heavy runs throttle
Torch CPU threads so editor extension processes have room to breathe.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class BenchmarkPreset:
    model_name: str
    bitwidths: list[int]
    top_width_pair_fractions: list[float]
    results_dir: Path
    plots_dir: Path
    row_group_sizes: Optional[list[int]] = None
    row_group_fractions: Optional[list[float]] = None
    save_plots: bool = False
    incremental_results: bool = True
    evaluation_text_file: Optional[Path] = None
    max_eval_texts: Optional[int] = None
    single_layer_name: Optional[str] = None
    calibration_texts: Optional[list[str]] = None
    torch_dtype: Optional[str] = None


PRESETS: dict[str, BenchmarkPreset] = {
    "tiny-gpt2-smoke": BenchmarkPreset(
        model_name="sshleifer/tiny-gpt2",
        bitwidths=[8],
        top_width_pair_fractions=[],
        results_dir=Path("results/smoke_tiny_gpt2_runner"),
        plots_dir=Path("plots/smoke_tiny_gpt2_runner"),
    ),
    "tinyllama-1.1b-int4-smoke": BenchmarkPreset(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        bitwidths=[4],
        top_width_pair_fractions=[],
        results_dir=Path("results/transformer_tinyllama_1_1b_int4_smoke"),
        plots_dir=Path("plots/transformer_tinyllama_1_1b_int4_smoke"),
        single_layer_name="model.layers.0.self_attn.q_proj",
        calibration_texts=["Quantization smoke test."],
    ),
    "tinyllama-1.1b-int4-matrix": BenchmarkPreset(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        bitwidths=[4],
        top_width_pair_fractions=[],
        row_group_sizes=[4, 8],
        row_group_fractions=[],
        results_dir=Path("results/transformer_tinyllama_1_1b_int4_matrix"),
        plots_dir=Path("plots/transformer_tinyllama_1_1b_int4_matrix"),
        evaluation_text_file=Path(
            "docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt"
        ),
        max_eval_texts=256,
        single_layer_name=None,
    ),
    "qwen2.5-3b-int4-smoke": BenchmarkPreset(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        bitwidths=[4],
        top_width_pair_fractions=[],
        row_group_sizes=[4],
        row_group_fractions=[],
        results_dir=Path("results/transformer_qwen2_5_3b_int4_smoke"),
        plots_dir=Path("plots/transformer_qwen2_5_3b_int4_smoke"),
        single_layer_name="model.layers.0.self_attn.q_proj",
        calibration_texts=["Quantization smoke test."],
    ),
    "qwen2.5-3b-int4-scale-row-g4": BenchmarkPreset(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        bitwidths=[4],
        top_width_pair_fractions=[],
        row_group_sizes=[4],
        row_group_fractions=[],
        results_dir=Path("results/transformer_qwen2_5_3b_int4_scale_row_g4"),
        plots_dir=Path("plots/transformer_qwen2_5_3b_int4_scale_row_g4"),
        evaluation_text_file=Path(
            "docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt"
        ),
        max_eval_texts=256,
        single_layer_name=None,
    ),
    "opt-2.7b-int4-smoke": BenchmarkPreset(
        model_name="facebook/opt-2.7b",
        bitwidths=[4],
        top_width_pair_fractions=[],
        row_group_sizes=[4],
        row_group_fractions=[],
        results_dir=Path("results/transformer_opt_2_7b_int4_smoke"),
        plots_dir=Path("plots/transformer_opt_2_7b_int4_smoke"),
        single_layer_name="model.decoder.layers.0.self_attn.q_proj",
        calibration_texts=["Quantization smoke test."],
    ),
    "opt-2.7b-int4-scale-row-g4": BenchmarkPreset(
        model_name="facebook/opt-2.7b",
        bitwidths=[4],
        top_width_pair_fractions=[],
        row_group_sizes=[4],
        row_group_fractions=[],
        results_dir=Path("results/transformer_opt_2_7b_int4_scale_row_g4"),
        plots_dir=Path("plots/transformer_opt_2_7b_int4_scale_row_g4"),
        evaluation_text_file=Path(
            "docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt"
        ),
        max_eval_texts=256,
        single_layer_name=None,
    ),
    "mistral-7b-v0.2-int4-smoke": BenchmarkPreset(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        bitwidths=[4],
        top_width_pair_fractions=[],
        row_group_sizes=[4],
        row_group_fractions=[],
        results_dir=Path("results/transformer_mistral_7b_v0_2_int4_smoke"),
        plots_dir=Path("plots/transformer_mistral_7b_v0_2_int4_smoke"),
        single_layer_name="model.layers.0.self_attn.q_proj",
        calibration_texts=["Quantization smoke test."],
        torch_dtype="float16",
    ),
    "mistral-7b-v0.2-int4-scale-row-g4": BenchmarkPreset(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        bitwidths=[4],
        top_width_pair_fractions=[],
        row_group_sizes=[4],
        row_group_fractions=[],
        results_dir=Path("results/transformer_mistral_7b_v0_2_int4_scale_row_g4"),
        plots_dir=Path("plots/transformer_mistral_7b_v0_2_int4_scale_row_g4"),
        evaluation_text_file=Path(
            "docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt"
        ),
        max_eval_texts=256,
        single_layer_name=None,
        torch_dtype="float16",
    ),
    "pythia-14m-int8-baseline": BenchmarkPreset(
        model_name="EleutherAI/pythia-14m",
        bitwidths=[8],
        top_width_pair_fractions=[],
        results_dir=Path("results/transformer_pythia_14m_int8_baseline"),
        plots_dir=Path("plots/transformer_pythia_14m_int8_baseline"),
    ),
    "pythia-14m-int4-baseline": BenchmarkPreset(
        model_name="EleutherAI/pythia-14m",
        bitwidths=[4],
        top_width_pair_fractions=[],
        results_dir=Path("results/transformer_pythia_14m_int4_baseline"),
        plots_dir=Path("plots/transformer_pythia_14m_int4_baseline"),
    ),
    "pythia-70m-int8-baseline": BenchmarkPreset(
        model_name="EleutherAI/pythia-70m",
        bitwidths=[8],
        top_width_pair_fractions=[],
        results_dir=Path("results/transformer_pythia_70m_int8_baseline"),
        plots_dir=Path("plots/transformer_pythia_70m_int8_baseline"),
    ),
    "pythia-70m-int4-baseline": BenchmarkPreset(
        model_name="EleutherAI/pythia-70m",
        bitwidths=[4],
        top_width_pair_fractions=[],
        results_dir=Path("results/transformer_pythia_70m_int4_baseline"),
        plots_dir=Path("plots/transformer_pythia_70m_int4_baseline"),
    ),
    "pythia-14m-int4-rotation": BenchmarkPreset(
        model_name="EleutherAI/pythia-14m",
        bitwidths=[4],
        top_width_pair_fractions=[0.05, 0.10, 0.20],
        results_dir=Path("results/transformer_pythia_14m_int4_rotation"),
        plots_dir=Path("plots/transformer_pythia_14m_int4_rotation"),
    ),
    "pythia-70m-int4-rotation": BenchmarkPreset(
        model_name="EleutherAI/pythia-70m",
        bitwidths=[4],
        top_width_pair_fractions=[0.05, 0.10, 0.20],
        results_dir=Path("results/transformer_pythia_70m_int4_rotation"),
        plots_dir=Path("plots/transformer_pythia_70m_int4_rotation"),
    ),
    "distilgpt2-int4-rotation": BenchmarkPreset(
        model_name="distilgpt2",
        bitwidths=[4],
        top_width_pair_fractions=[0.05, 0.10, 0.20],
        results_dir=Path("results/transformer_distilgpt2_int4_rotation"),
        plots_dir=Path("plots/transformer_distilgpt2_int4_rotation"),
    ),
    "distilgpt2-int8-baseline": BenchmarkPreset(
        model_name="distilgpt2",
        bitwidths=[8],
        top_width_pair_fractions=[],
        results_dir=Path("results/transformer_distilgpt2_int8_baseline"),
        plots_dir=Path("plots/transformer_distilgpt2_int8_baseline"),
    ),
    "distilgpt2-int4-baseline": BenchmarkPreset(
        model_name="distilgpt2",
        bitwidths=[4],
        top_width_pair_fractions=[],
        results_dir=Path("results/transformer_distilgpt2_int4_baseline"),
        plots_dir=Path("plots/transformer_distilgpt2_int4_baseline"),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run safer Milestone 3 transformer benchmark presets.",
    )
    parser.add_argument(
        "preset",
        nargs="?",
        default="pythia-14m-int8-baseline",
        choices=sorted(PRESETS),
        help="Benchmark preset to run.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available presets and exit.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/load model and tokenizer, then exit before quantization.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for model execution; auto uses CUDA when available.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Set torch CPU thread count for this run.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override preset results directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Override preset plots directory.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Write dashboard plots for this run.",
    )
    parser.add_argument(
        "--delete-hf-cache-after",
        action="store_true",
        help="Delete this model from Hugging Face cache after a successful run.",
    )
    parser.add_argument(
        "--no-incremental-results",
        action="store_true",
        help="Write CSVs only at the end instead of appending during the run.",
    )
    parser.add_argument(
        "--eval-text-file",
        type=Path,
        default=None,
        help=(
            "Load paragraph-separated evaluation texts from this UTF-8 file "
            "instead of the built-in three-sentence batch."
        ),
    )
    parser.add_argument(
        "--max-eval-texts",
        type=int,
        default=None,
        help="Use at most this many texts from --eval-text-file.",
    )
    parser.add_argument(
        "--logit-only",
        action="store_true",
        help="Skip per-layer weight/activation metrics and run only end-to-end logit/loss metrics.",
    )
    parser.add_argument(
        "--logit-methods",
        default=None,
        help="Comma-separated method names to evaluate in --logit-only mode.",
    )
    return parser


def build_config(args: argparse.Namespace) -> TransformerConfig:
    from experiments.transformer_experiment import TransformerConfig, load_text_batch

    preset = PRESETS[args.preset]
    eval_text_file = args.eval_text_file or preset.evaluation_text_file
    max_eval_texts = (
        args.max_eval_texts
        if args.max_eval_texts is not None
        else preset.max_eval_texts
    )
    calibration_texts = None
    calibration_text_source = None
    if eval_text_file is not None:
        calibration_texts = load_text_batch(eval_text_file, max_texts=max_eval_texts)
        calibration_text_source = str(eval_text_file)
    return TransformerConfig(
        model_name=preset.model_name,
        calibration_texts=calibration_texts
        if calibration_texts is not None
        else preset.calibration_texts
        if preset.calibration_texts is not None
        else TransformerConfig().calibration_texts,
        calibration_text_source=calibration_text_source
        if calibration_text_source is not None
        else TransformerConfig().calibration_text_source,
        single_layer_name=preset.single_layer_name,
        bitwidths=list(preset.bitwidths),
        row_group_sizes=list(preset.row_group_sizes)
        if preset.row_group_sizes is not None
        else TransformerConfig().row_group_sizes,
        row_group_fractions=list(preset.row_group_fractions)
        if preset.row_group_fractions is not None
        else TransformerConfig().row_group_fractions,
        top_width_pair_fractions=list(preset.top_width_pair_fractions),
        results_dir=args.results_dir or preset.results_dir,
        plots_dir=args.plots_dir or preset.plots_dir,
        save_plots=bool(args.save_plots or preset.save_plots),
        local_files_only=bool(args.local_files_only),
        incremental_results=not bool(args.no_incremental_results)
        and preset.incremental_results,
        delete_hf_cache_after=bool(args.delete_hf_cache_after),
        device=args.device,
        logit_only=bool(args.logit_only),
        logit_method_names=_parse_logit_methods(args.logit_methods),
        torch_dtype=preset.torch_dtype,
    )


def _parse_logit_methods(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    methods = [part.strip() for part in value.split(",") if part.strip()]
    return methods or None


def configure_threads(torch_threads: Optional[int]) -> None:
    if torch_threads is None:
        return
    if torch_threads < 1:
        raise ValueError("--torch-threads must be at least 1")
    torch.set_num_threads(torch_threads)
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["TORCH_NUM_THREADS"] = str(torch_threads)
    print(f"Using torch CPU threads: {torch_threads}")


def download_artifacts(model_name: str, local_files_only: bool) -> None:
    print(f"Preparing Hugging Face artifacts for {model_name}...")
    AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    print("Model and tokenizer are available.")


def print_presets() -> None:
    for name in sorted(PRESETS):
        preset = PRESETS[name]
        bitwidths = ",".join(str(bw) for bw in preset.bitwidths)
        rotations = preset.top_width_pair_fractions or "off"
        print(
            f"{name}: model={preset.model_name} bitwidths={bitwidths} "
            f"rotations={rotations} results={preset.results_dir}"
        )




def _git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _resolved_device_label(device_request: str) -> str:
    if device_request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_request


def _cuda_memory_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024 * 1024), 3)


def _collect_benchmark_metadata(
    *,
    args: argparse.Namespace,
    config,
    elapsed_seconds: Optional[float] = None,
    counts: Optional[dict[str, int]] = None,
) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
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
        "preset": args.preset,
        "model_name": config.model_name,
        "single_layer_name": config.single_layer_name,
        "bitwidths": list(config.bitwidths),
        "row_group_sizes": list(config.row_group_sizes),
        "row_group_fractions": list(config.row_group_fractions),
        "top_width_pair_fractions": list(config.top_width_pair_fractions),
        "logit_only": bool(config.logit_only),
        "logit_method_names": list(config.logit_method_names or []),
        "device_request": args.device,
        "resolved_device": _resolved_device_label(args.device),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb,
        "cuda_peak_memory_allocated_mb": peak_allocated_mb,
        "cuda_peak_memory_reserved_mb": peak_reserved_mb,
        "commit_hash": _git_commit_hash(),
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "counts": counts or {},
    }


def _write_benchmark_metadata(results_dir: Path, metadata: dict[str, object]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "benchmark_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

def _make_progress_callback(label: str):
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        return None

    _bars: dict = {}
    _prev_phase: list = [None]

    def _callback(phase: str, current: int, total: int) -> None:
        if phase not in _bars:
            prev = _prev_phase[0]
            if prev is not None and prev in _bars:
                _bars[prev].close()
            _bars[phase] = _tqdm(
                total=total,
                desc=f"{label} {phase}s",
                unit=phase,
                dynamic_ncols=True,
            )
        _bars[phase].update(1)
        _prev_phase[0] = phase

    return _callback


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        print_presets()
        return 0

    configure_threads(args.torch_threads)
    config = build_config(args)

    print(f"Preset: {args.preset}")
    print(f"Model: {config.model_name}")
    print(f"Results: {config.results_dir}")
    print(f"Plots: {config.plots_dir}")
    print(f"Bitwidths: {config.bitwidths}")
    print(f"Row group sizes: {config.row_group_sizes}")
    print(f"Row group fractions: {config.row_group_fractions or 'off'}")
    print(f"Rotations: {config.top_width_pair_fractions or 'off'}")
    print(f"Logit only: {config.logit_only}")
    print(f"Logit methods: {config.logit_method_names or 'all'}")
    print(f"Single layer: {config.single_layer_name or 'all'}")
    print(f"Device request: {args.device}")
    print(f"Resolved device: {_resolved_device_label(args.device)}")
    print(
        f"Evaluation texts: {len(config.calibration_texts)} "
        f"from {config.calibration_text_source}"
    )
    print(f"Local files only: {config.local_files_only}")
    print(f"Incremental CSVs: {config.incremental_results}")

    if args.download_only:
        t0 = time.time()
        download_artifacts(config.model_name, local_files_only=config.local_files_only)
        elapsed = time.time() - t0
        metadata = _collect_benchmark_metadata(
            args=args,
            config=config,
            elapsed_seconds=elapsed,
            counts={"download_only": 1},
        )
        metadata_path = _write_benchmark_metadata(config.results_dir, metadata)
        print(f"metadata: {metadata_path}")
        print(f"download elapsed: {elapsed:.1f}s")
        return 0

    from experiments.transformer_experiment import print_summary, run_transformer_experiment

    config.on_progress = _make_progress_callback(args.preset)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    wr, ar, lr = run_transformer_experiment(config)
    elapsed = time.time() - t0
    print_summary(wr, ar, lr)
    counts = {"weight": len(wr), "activation": len(ar), "logit": len(lr)}
    metadata = _collect_benchmark_metadata(
        args=args,
        config=config,
        elapsed_seconds=elapsed,
        counts=counts,
    )
    metadata_path = _write_benchmark_metadata(config.results_dir, metadata)
    print(f"counts weight={len(wr)} activation={len(ar)} logit={len(lr)}")
    print(f"metadata: {metadata_path}")
    print(f"elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

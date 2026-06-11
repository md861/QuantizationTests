"""Safer entry point for long Milestone 3 transformer benchmarks.

The presets here are intentionally conservative. They separate model download
from quantization, support local-cache-only runs, and let heavy runs throttle
Torch CPU threads so editor extension processes have room to breathe.
"""

from __future__ import annotations

import argparse
import os
import sys
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
    save_plots: bool = False
    incremental_results: bool = True


PRESETS: dict[str, BenchmarkPreset] = {
    "tiny-gpt2-smoke": BenchmarkPreset(
        model_name="sshleifer/tiny-gpt2",
        bitwidths=[8],
        top_width_pair_fractions=[],
        results_dir=Path("results/smoke_tiny_gpt2_runner"),
        plots_dir=Path("plots/smoke_tiny_gpt2_runner"),
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
    return parser


def build_config(args: argparse.Namespace) -> TransformerConfig:
    from experiments.transformer_experiment import TransformerConfig

    preset = PRESETS[args.preset]
    return TransformerConfig(
        model_name=preset.model_name,
        single_layer_name=None,
        bitwidths=list(preset.bitwidths),
        top_width_pair_fractions=list(preset.top_width_pair_fractions),
        results_dir=args.results_dir or preset.results_dir,
        plots_dir=args.plots_dir or preset.plots_dir,
        save_plots=bool(args.save_plots or preset.save_plots),
        local_files_only=bool(args.local_files_only),
        incremental_results=not bool(args.no_incremental_results)
        and preset.incremental_results,
        delete_hf_cache_after=bool(args.delete_hf_cache_after),
    )


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
    print(f"Rotations: {config.top_width_pair_fractions or 'off'}")
    print(f"Local files only: {config.local_files_only}")
    print(f"Incremental CSVs: {config.incremental_results}")

    if args.download_only:
        download_artifacts(config.model_name, local_files_only=config.local_files_only)
        return 0

    from experiments.transformer_experiment import print_summary, run_transformer_experiment

    wr, ar, lr = run_transformer_experiment(config)
    print_summary(wr, ar, lr)
    print(f"counts weight={len(wr)} activation={len(ar)} logit={len(lr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

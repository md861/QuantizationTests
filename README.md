# Quantization Lab

![ParoQuant project journey](docs/assets/paroquant-journey.png)

Quantization Lab is a research-oriented educational sandbox for seeing how low-bit
quantization changes matrices, spectra, and reconstruction error.

The project starts at matrix level before scaling toward transformer models.
Milestone 1 (quantization sandbox), Milestone 2 (ParoQuant core), and Milestone
3 (tiny transformer integration) are complete. Milestone 3 implemented the
transformer harness (`experiments/transformer_experiment.py`), measured weight
reconstruction, activation drift, logit/loss, and perplexity across INT4 and
INT8 paths, and completed benchmark runs on `sshleifer/tiny-gpt2`,
`roneneldan/TinyStories-1M`, `EleutherAI/pythia-14m`,
`EleutherAI/pythia-70m`, and `distilgpt2`.

## Project Roadmap

| Milestone | Focus | Status |
| --- | --- | --- |
| 4. Real LLM Benchmarking | Scale to larger open-source LLMs and compare against GPTQ, AWQ, and bitsandbytes | Next |
| 3. Tiny Transformer Integration | Apply INT4/INT8 quantizer to `sshleifer/tiny-gpt2` ✓, `roneneldan/TinyStories-1M` ✓, `EleutherAI/pythia-14m` ✓, `EleutherAI/pythia-70m` ✓, `distilgpt2` ✓; INT4 rotation presets ✓; WikiText-2 validation ✓ | Complete |
| 2. ParoQuant Core | Givens rotations, channel scaling, grouped quantization, and outlier suppression | Complete |
| 1. Quantization Sandbox | Matrix generation, INT8/INT4 quantization, metrics, spectra, and visual diagnostics | Complete |

## Progress

| Area | Status |
| --- | --- |
| Project scaffold and environment | Complete |
| Synthetic matrix generation | Complete |
| Symmetric INT8 and INT4 quantization | Complete |
| Reconstruction and spectrum metrics | Complete |
| Baseline experiment | Complete |
| Outlier-severity sweep | Complete |
| Consolidated comparison plots | Complete |
| Integration and hygiene tests | Complete |
| Histogram visualizations | Complete |
| Results analysis helper | Complete |
| Pairwise Givens rotation module | Complete |
| Top-width sparse rotation selection | Complete |
| Per-channel scaling | Complete |
| Grouped quantization | Complete |
| Rotation/scaling experiment | Complete |
| Comparative sweep experiment | Complete |
| Transformer harness (weight + activation + logit metrics) | Complete |
| Milestone 4 GPU-aware runner metadata | Complete |
| TinyLlama 1.1B smoke preset | Complete |
| bitsandbytes NF4 external baseline runner and 256-text baseline | Complete |
| AWQ external baseline runner and 256-text baseline | Complete |
| GPTQ external baseline runner and 256-text baseline | Complete |
| Qwen2.5-3B project smoke/focused presets | Complete |
| Qwen2.5-3B RunPod command plan | Complete |
| Qwen2.5-3B smoke/cache readiness | Partial: project smoke passed; AWQ/GPTQ external smokes blocked |
| OPT-2.7B larger-model comparison target | Selected |
| OPT-2.7B project smoke/focused presets | Complete |
| OPT-2.7B RunPod command plan | Complete |
| WikiText-2 256-record evaluation resource | Complete |
| RunPod persistent Hugging Face cache policy | Complete |
| TinyLlama bitsandbytes NF4 one-record smoke | Complete |
| TinyLlama INT4 full-matrix preset | Complete |
| Row-grouped quantization vectorized implementation | Complete |
| TinyLlama project logit-only matrix path | Complete |
| Per-method logit runtime and CUDA peak telemetry | Complete |
| Logit throughput and theoretical artifact-size telemetry | Complete |

## Next Milestone 4

Milestone 4 should proceed in small, hardware-aware steps. The project has
access to RunPod GPUs, but RunPod is reserved for benchmark execution only;
code generation, local dry runs, analysis, plotting, and documentation stay on
the local machine unless a GPU-only failure must be debugged remotely. Raw
RunPod SSH details, keys, account identifiers, and Pod-specific connection
strings must not be committed.

1. TinyLlama external-baseline set is complete for bitsandbytes NF4, AWQ, and GPTQ on the same tracked 256-record WikiText-2 raw validation resource and shared logit/loss/runtime/memory fields.
2. Qwen2.5-3B was tried as the first modern scale-up target. The project `scale_row_g4` smoke passed, but Qwen AWQ/GPTQ external smokes failed with exit `132` after selecting Marlin-family kernels. Treat this as a backend-compatibility detour, not a research result.
3. The active larger-model comparison target is now `facebook/opt-2.7b`, chosen as a simpler Transformers model for a boring external-compatible scale-up path, with bitsandbytes NF4 as the first external baseline.
4. Estimate expected RunPod runtime and cost before each GPU run from the timing table and usage ledger; choose GPU class by cost per useful benchmark, not raw theoretical speed.
5. Run another single-layer or small-subset smoke before a full benchmark whenever the model, comparison matrix, evaluation text, dependencies, or GPU class changes.
6. Run full-model benchmarks only from detached tmux, writing logs/results under persistent /workspace on RunPod and recording elapsed time, GPU type, VRAM, peak memory, method telemetry, commit hash, hourly rate, and estimated spend in the bookkeeping docs.
7. Stop the RunPod Pod as soon as benchmark execution finishes unless another GPU benchmark is already queued to start within about 30 minutes; otherwise pull CSVs/logs/results back locally for analysis and documentation.
8. Keep total RunPod benchmark spend under the project budget ceiling of about GBP 200; update the RunPod usage ledger, usage dashboard, and Benchmark Run Timings table after every Pod segment.
9. Compare quality, runtime, memory pressure, and artifact size across the project method and external baselines, while keeping run/provenance details out of the research draft.
10. Update the research draft, README, project summary, and lab book after each completed model.

The external baseline scaffolds are experiments/bitsandbytes_baseline.py,
experiments/awq_baseline.py, and experiments/gptq_baseline.py. They are
intentionally separate from the project quantizer harness: these baselines load
quantized Transformers runtime modules or pre-quantized checkpoints, so the
fair shared comparison is logit/loss/perplexity plus runtime and memory
metadata, not project weight or activation reconstruction tables. OPT-2.7B and
Qwen2.5-3B project presets now live in
`experiments/run_transformer_benchmark.py`; the active OPT RunPod command plan
is `docs/runpod/opt_2_7b_plan.md`. Keep these baselines optional; normal local
tests do not require their optional packages or CUDA. The AWQ and GPTQ runners
require explicit `--awq-model-name` and `--gptq-model-name` arguments so a
reference checkpoint is not accidentally reported as an external quantized
baseline.

The first controlled TinyLlama matrix is locked: original Hugging Face
reference, project INT4 global, project INT4 row_grouped_g4/g8, project INT4
scale_row_g4/g8, and bitsandbytes NF4 float16. Do not include rotations in this
first matrix; use the tracked 256-record WikiText-2 resource and compare bnb to
the project methods only on shared end-to-end fields. Use `--logit-only` for
Milestone 4 project-method runs when weight/activation reconstruction tables are
not part of the comparison.

Current RunPod setup notes:

- Detailed RunPod technical operations live in `docs/runpod/operations.md`.
- SSH access is local-only via alias `runpod-pq`; raw connection details, keys,
  account identifiers, Pod IDs, ports, and hostnames must not be committed.
- The selected baseline worker class is RTX 4000 Ada with about 20 GB VRAM,
  50 GB RAM, and 9 vCPU on the current Pod. This is enough for TinyLlama-era
  smoke tests and small controlled baselines; reassess before larger models or
  memory-heavy external baselines.
- A persistent `/workspace` network volume is used instead of a Pod-local volume
  disk because it survives Pod replacement, can be reattached across compatible
  Pods, and was cheaper per GB in the RunPod pricing table checked during setup.
  The recommended minimum for this phase is 100 GB to leave room for the repo,
  venv, Hugging Face cache, logs, and benchmark artifacts.
- The Pod repo lives at `/workspace/PQ_project`; its clean self-contained venv is
  `/workspace/PQ_project/.venv` with PyTorch 2.6.0+cu124 and Transformers 5.12.1.
  Full repo verification on the Pod passed: `212 passed, 1 warning in 349.22s`.
  Hugging Face cache should be kept under `/workspace/hf_cache` so model files
  survive Pod replacement.
- First TinyLlama single-layer INT4 smoke on the Pod passed at commit `c15113a`:
  `228.3s (3.8 min)`, peak CUDA allocated `2124 MB`, peak reserved `2224 MB`.
- First TinyLlama bitsandbytes NF4 one-record smoke passed at commit `4b5d5d0`:
  `44.2s` runner elapsed, peak CUDA allocated `2173 MB`, logit MSE `0.311986`,
  top-5 overlap `0.865`, loss delta `+0.044535`, and perplexity ratio `1.04554`.
  Treat this as a readiness smoke, not the final research comparison.
- First TinyLlama project INT4 logit-only 256-text matrix passed at commit
  `ceddbaf`: `1004.4s (16.7 min)` runner elapsed, `19m20s` command wall,
  peak CUDA allocated `2274 MB`, with `scale_row_g4` giving PPL ratio `0.9860`
  and top-5 overlap `0.9019` on the tracked WikiText-2 resource.
  The result makes `global` INT4 the failure/control row and shows g4 row grouping
  is essentially loss-neutral on this bounded TinyLlama validation subset.
- TinyLlama project INT4 per-method telemetry rerun passed at commit `049d42a`:
  `1208.7s` runner elapsed, `23m43s` command wall, and peak CUDA
  `2274 MB` allocated / `2658 MB` reserved. Isolated project method times were
  `global` `79.527s`, `row_grouped_g4` `34.542s`, `row_grouped_g8` `30.810s`,
  `scale_row_g4` `38.282s`, and `scale_row_g8` `34.937s`.
- First TinyLlama bitsandbytes NF4 256-text run passed at commit `92b4f5e`:
  `231.4s` runner elapsed, `6m17s` command wall, peak CUDA allocated `2274 MB`,
  logit MSE `0.253299`, top-5 overlap `0.857917`, loss delta `+0.023453`, and
  perplexity ratio `1.023730`. On this bounded comparison, project
  `scale_row_g4` has better quality, while bnb is faster because it evaluates a
  single external method rather than the five project rows.
- TinyLlama bitsandbytes NF4 256-text telemetry rerun passed at commit
  `d8c7d09`: `191.5s` runner elapsed, `6m24s` command wall, isolated bnb method
  loop `24.577s`, `1354.168 tokens/s`, `0.738 ms/token`, peak CUDA
  `962.886 MB` allocated / `1322 MB` reserved, logit MSE `0.253722`,
  top-5 overlap `0.857737`, loss delta `+0.023356`, and PPL ratio `1.023631`.
- TinyLlama AWQ 4-bit 256-text external baseline passed at commit `97bc484`:
  `238.2s` runner elapsed, isolated AWQ method loop `39.409s`,
  `844.535 tokens/s`, `1.184 ms/token`, peak CUDA `904.183 MB` allocated /
  `1240 MB` reserved, logit MSE `0.252777`, top-5 overlap `0.854252`, loss
  delta `+0.040232`, and PPL ratio `1.041052`.
- TinyLlama GPTQ 4-bit 256-text external baseline passed at commit `97bc484`:
  `262.2s` runner elapsed, isolated GPTQ method loop `58.086s`,
  `572.980 tokens/s`, `1.745 ms/token`, peak CUDA `903.581 MB` allocated /
  `1242 MB` reserved, logit MSE `0.349270`, top-5 overlap `0.837882`, loss
  delta `+0.021532`, and PPL ratio `1.021766`.

## Completed Milestone 3

Milestone 3 applies the ParoQuant INT4/INT8 pipeline to real transformer weights.
The harness (`experiments/transformer_experiment.py`) supports single-layer and
all-layer modes, comparing global, row-grouped, scale+row-grouped, and top-width
rotate+scale+row-grouped paths across both bitwidths on any HuggingFace causal LM.
It measures weight reconstruction (MSE, cosine similarity, SNR), activation drift
(MSE, cosine similarity, relative error), and full-model logit/loss quality
(logit MSE, top-5 token overlap, next-token loss delta, perplexity, and
perplexity ratio).

Completed all-layer runs:

- `sshleifer/tiny-gpt2`: eight compatible transformer layers quantized; all
  tested paths preserved top-5 token overlap on the built-in calibration batch,
  and perplexity ratios stayed within about six parts per million of 1.0. This
  is treated as harness validation because the model's linear layers are
  extremely small.
- `roneneldan/TinyStories-1M`: 48 compatible transformer layers quantized. INT4
  global quantization damaged full-model behavior on the built-in calibration
  batch (perplexity ratio 16.1x), while INT4 row-grouped g4 reduced the hit to
  1.21x and capped top-width rotation+scale+row g4 reduced it further to 1.14x.
  INT8 paths stayed close to the original model, with logit MSE far below INT4.
- `EleutherAI/pythia-14m`: 25 compatible transformer layers quantized (INT8 and
  INT4 baselines). First model where INT8 global is not lossless (perplexity
  ratio 1.24); INT8 row-grouped g4 restores losslessness (0.994). INT4 global is
  catastrophic (perplexity ratio 15,074x); INT4 row-grouped g4 gives 1.33x.
  Group size 4 vs 32 is a >2x quality difference at INT4. The first capped
  top-width rotation run completed in 240.1s and improved the best INT4 g4 path
  slightly, from scale_row_g4 PPLx 1.318 to rotate+scale_row_g4 PPLx 1.302.
- `EleutherAI/pythia-70m`: 25 compatible transformer layers quantized (INT8 and
  INT4 baselines, ~13 min each). INT8 global degrades further (PPL ratio 1.44);
  INT8 g4 remains lossless (0.971). INT4 global catastrophic (~501 trillion PPLx);
  INT4 row-grouped g4 gives 7.52x — a qualitative jump from 14m's 1.33x.
  Group size effect at INT4: g4 vs g128 is a 478x quality gap. INT8 and INT4
  take identical wall-clock time (~13 min), confirming runtime is dominated by
  weight passes not bitwidth arithmetic. The capped INT4 rotation run took
  1174.6s and worsened g4 PPLx from 7.52 to 8.05.

- `distilgpt2`: 24 compatible transformer layers quantized (INT8 and INT4
  baselines, ~11–12 min each). INT8 g4 fully lossless (PPLx 0.999, top-5 1.000).
  INT4 g4 gives **1.058x** — best INT4 result of any real model in this study.
  INT4 global: 48.85x (catastrophic but less so than Pythia models). Key finding:
  architecture and distillation training dominate over parameter count for INT4
  quality; distilgpt2 quantizes 7x better than Pythia-70m at INT4 g4 despite
  being larger. The capped INT4 rotation run took 1024.9s; it slightly worsened
  g4 PPLx to 1.062 but slightly improved the coarse g192 path to 1.201.

All planned baseline models and INT4 rotation presets are complete. The
cross-model rotation synthesis is documented in `docs/research_draft.md`: on the
tracked WikiText-2 validation sample, sparse uncalibrated rotations worsen or
fail to improve the best INT4 g4 path on Pythia-14M, Pythia-70M, and distilgpt2.
Next: estimate and approve the OPT-2.7B smoke/cache readiness segment for
original/reference, project `scale_row_g4`, and bitsandbytes NF4 before any
full 256-record comparison.

## Completed Milestone 2

Milestone 2 implemented the ParoQuant core: transformations applied before
quantization to reduce outlier pressure:

- **Pairwise Givens rotations** (`quant/rotations.py`): rotate any column pair
  by an angle that minimises the joint max-abs, redistributing outlier energy
  across channels before INT4/INT8 quantization.  Key functions:
  `rotation_matrix`, `apply_rotation`, `optimal_angle`, `rotate_channel_pair`,
  `channel_widths`, `top_width_channel_pairs`, `rotate_top_width_pairs`,
  `apply_sequential_rotations`.
- Verified via entry-zeroing (arctan2 angle analytically zeros a target entry),
  full Givens QR decomposition (Q@R=A cross-checked against `numpy.linalg.qr`),
  and column orthogonalisation (Jacobi angle drives inner product to machine zero).
- **Per-channel scaling** (`quant/scaling.py`): compute one reversible positive
  factor per column so nonzero channel max-abs values share a target before
  quantization. Key functions: `column_max_abs`, `compute_channel_scaling`,
  `apply_channel_scaling`, `invert_channel_scaling`, `balance_channel_max_abs`.
- **Grouped quantization** (`quant/quantizer.py`): two strategies — (1) column-grouped:
  contiguous column blocks share one scale per block; (2) row-grouped: each column
  is split into row-groups each with their own scale (the industry-standard GPTQ/AWQ
  approach, giving tighter precision when outliers are localised within a column).
  Key functions: `grouped_symmetric_quantize`, `row_grouped_symmetric_quantize`,
  `quantize_int8_grouped`, `quantize_int4_grouped`,
  `quantize_int8_row_grouped`, `quantize_int4_row_grouped`.
- **Rotation/scaling experiment** (`experiments/rotation_experiment.py`):
  compares baseline INT4, rotation-only INT4, scaling-only INT4, and
  rotation+scaling INT4 on one controlled outlier-heavy matrix.
- **Comparative sweep experiment** (`experiments/sweep_experiment.py`):
  sweeps 12 quantization paths (global, col-grouped, row-grouped, scale+global,
  rotate+global, rotate+scale+global, rotate+scale+row-grouped) across a grid
  of seeds, outlier fractions, and outlier scales. Outputs `results/sweep_metrics.csv`
  and a 4-panel `plots/sweep_dashboard.png`. Key finding: row-grouped and
  rotate+scale+row-grouped are the only paths that consistently beat global INT4
  for row-localised outliers; column-grouped gives no improvement at any group size.
  Optional `top_width_pair_fractions` add ParoQuant-style sparse rotation paths
  such as `top_width_rotate_p10_global` and
  `top_width_rotate_scale_p10_row_g4`, selecting independent channel pairs from
  the top percentage of max-abs width differences.
  Sweep CSV rows include `rotation_count`, `rotation_pair_fraction`, and
  `rotation_candidate_fraction` so rotation-heavy comparisons remain interpretable.
  Summary tables report condition-wise mean and standard deviation for MSE ratio
  and zero fraction; dashboard error bars show the same cross-condition spread.

## Completed Milestone 1

Milestone 1 built the quantization sandbox:

- matrix generation for Gaussian, heavy-tailed, and outlier-heavy data
- symmetric INT8 and INT4 quantization
- reconstruction metrics: MSE, MAE, cosine similarity, relative Frobenius error, SNR
- singular-value spectrum diagnostics
- value, residual, and quantized-code histogram diagnostics
- comparison plots showing original matrices, residuals, spectra, and metrics

Use the safer benchmark runner (`experiments/run_transformer_benchmark.py`) for
future transformer benchmark runs. Always launch from a detached tmux session with
`; tmux kill-session -t bench` appended so the session self-destructs on
completion. Pre-download each model with `--download-only` before the heavy run:

```bash
tmux new-session -d -s bench && tmux send-keys -t bench \
  "MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/run_transformer_benchmark.py \
  pythia-70m-int8-baseline --download-only 2>&1 | tee /tmp/pythia70m_download.log \
  ; tmux kill-session -t bench" Enter
```

This repository prioritizes clarity, modularity, reproducibility, and
mathematical transparency over production inference performance.

Milestone 4 GPU benchmarks use [RunPod](https://www.runpod.io/) compute as the
project's benchmark-worker infrastructure. RunPod is used only for GPU-bound
benchmark execution; local development, analysis, plotting, and documentation
remain local.

This project is being developed with the help of AI-assisted coding, with the
lab book and project summary used to keep the research process inspectable.

## Run Tests

Use the project virtual environment:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Current expected test state:

```text
225 passed, 1 warning
```

## Reproduce Artifacts

Generated experiment outputs are intentionally ignored by Git. Recreate them
locally with:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/rotation_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/sweep_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/transformer_experiment.py
```

These commands write CSV files under `results/` and comparison figures under
`plots/`.

For transformer loss/perplexity reruns on the tracked 256-record WikiText-2
validation resource, pass the research resource explicitly:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/run_transformer_benchmark.py \
  tiny-gpt2-smoke --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt
```

The analysis helper also writes a collated benchmark-style visual:

- `plots/analysis_dashboard.png`

The rotation/scaling experiment writes:

- `results/rotation_metrics.csv`
- `plots/rotation_scaling_comparison.png`

The sweep experiment writes:

- `results/sweep_metrics.csv` and `plots/sweep_dashboard.png` (32×32, seeds 0–4)
- `results/sweep_metrics_320x320.csv` and `plots/sweep_dashboard_320x320.png` (320×320, seeds 5–9)
- `results/sweep_metrics_top_width_32x32.csv` and `plots/sweep_dashboard_top_width_32x32.png`
- `results/sweep_metrics_top_width_320x320.csv` and `plots/sweep_dashboard_top_width_320x320.png`

The top-width sparse-rotation sweeps use `top_width_pair_fractions=[0.05, 0.10, 0.20]`.

The transformer harness writes:

- `results/transformer_weight_metrics.csv`
- `results/transformer_activation_metrics.csv`
- `results/transformer_logit_metrics.csv`
- `plots/transformer_dashboard.png`

To reproduce the large-matrix sweep:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python - << 'EOF'
from pathlib import Path
from experiments.sweep_experiment import SweepConfig, run_sweep_experiment, print_summary
config = SweepConfig(
    shape=(320, 320), seeds=[5,6,7,8,9],
    outlier_fractions=[0.02, 0.07, 0.15], outlier_scales=[7.5, 15.0, 30.0],
    row_group_sizes=[4, 8, 16, 32], col_group_sizes=[4, 8, 16],
    csv_name="sweep_metrics_320x320.csv", plot_name="sweep_dashboard_320x320.png",
)
records = run_sweep_experiment(config)
print_summary(records)
EOF
```

Milestone 2 development visuals include:

- `plots/channel_scaling_dashboard.png`

## Project Notes

- `project_summary.md` is the compact handoff for resuming work.
- `lab_book/project_journey.md` is the chronological development record.
- `docs/research_draft.md` is the living paper-style draft of findings,
  examples, figures, and current claims.
- `docs/figures/` contains tracked figures used by the research draft.
- When a committed change affects the research story, update the draft and
  commit any required `docs/figures/` resources with it.
- `plots/` and `results/` are local generated artifacts, not tracked source.

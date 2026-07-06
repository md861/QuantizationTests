# Quantization Lab Project Summary

This is the compact handoff document for resuming work on the Quantization Lab research sandbox. For the full chronological history, see `lab_book/project_journey.md`.

## Current State

Milestone 1, Milestone 2, and Milestone 3 are complete. Milestone 2
(ParoQuant Core) includes pairwise Givens rotations, per-channel scaling,
top-width channel-pair selection for sparse rotations, column-grouped
quantization, row-grouped quantization, the rotation/scaling experiment, and
comparative sweeps across the implemented quantization paths. Milestone 3
(tiny transformer integration) implemented and tested the transformer harness:
it loads HuggingFace causal LMs, runs INT4 and INT8 paths on each linear layer,
and measures weight reconstruction, activation drift, logit/loss quality, and
perplexity. Milestone 3 runs are complete on tiny-gpt2, TinyStories-1M,
Pythia-14M, Pythia-70M, and distilgpt2, including INT4 rotation presets and
WikiText-2 validation reruns.

Implemented so far:

- synthetic matrix generation
- symmetric INT8 and INT4 quantization
- reconstruction and spectrum metrics
- heatmap, spectrum, and quantization-summary visualizations
- baseline experiment comparing INT8 and INT4 across matrix families
- outlier-severity sweep comparing INT8 and INT4 across controlled outlier fractions and scales
- histogram visualizations for values, residuals, and quantized codes
- results-analysis helper comparing INT4 against INT8 from generated CSVs, including a collated benchmark-style dashboard
- integration and repository-hygiene tests
- pairwise Givens rotation utilities (`quant/rotations.py`)
- ParoQuant-style top-width-difference channel-pair selection for independent sparse rotations
- per-channel scaling utilities (`quant/scaling.py`)
- grouped symmetric quantization utilities, including column-grouped and row-grouped paths
- rotation/scaling experiment comparing four INT4 preprocessing paths
- comparative sweep experiment comparing global, grouped, scaled, rotated, and combined INT4 paths
- living research-paper draft in `docs/research_draft.md`
- tracked paper figures in `docs/figures/`
- tests for all implemented modules
- Milestone 3 transformer harness (`experiments/transformer_experiment.py`): loads
  any HuggingFace causal LM, quantizes linear layers at INT4 and INT8 across
  global, row-grouped, scale+row-grouped, and top-width rotate+scale+row-grouped
  paths; measures weight reconstruction, activation drift, full-model logit/loss,
  and perplexity; writes three CSVs and a 4-panel dashboard with log-scale MSE
  panels and symmetric-log loss deltas;
  supports single-layer and all-layer modes;
  `delete_hf_cache_after=True` evicts the model after each run
- first all-layer `sshleifer/tiny-gpt2` transformer run: eight compatible
  layers, 196 weight records, 196 activation records, and 22 logit/loss records;
  top-5 overlap stayed 1.0 and perplexity ratios stayed within about six parts
  per million of 1.0 on the built-in calibration batch
- first all-layer `roneneldan/TinyStories-1M` transformer run: 48 compatible
  layers, 1008 weight records, 1008 activation records, and 14 all-layer
  logit/loss records; INT4 global had a 16.1x perplexity ratio on the built-in
  calibration batch, INT4 row-grouped g4 reduced the ratio to 1.21x, and capped
  top-width rotate+scale+row g4 reduced it to 1.14x; INT8 paths stayed close to
  the original model
- `EleutherAI/pythia-14m` INT8 and INT4 baseline runs complete (25 compatible
  layers; 225 weight records, 225 activation records, 5 logit/loss records each):
  INT8 global is
  NOT lossless here (PPL ratio 1.24, top-5 overlap 0.672), unlike the smaller
  models; INT8 row_grouped_g4 restores losslessness (PPL ratio 0.994); INT4
  global is catastrophic (PPL ratio 15,074); INT4 row_grouped_g4 gives 1.33x
  and scale_row_g4 gives 1.32x; g32 is 2.52x worse than g4 for INT4
- `EleutherAI/pythia-14m` INT4 capped top-width rotation run complete (25 layers,
  325 weight records, 325 activation records, 7 logit/loss records): elapsed
  240.1s (4.0 min); the effective model-wide rotation candidate fraction was
  p0.0001% because `embed_out` has 50,304 output channels; rotate+scale_row_g4
  modestly improved PPLx to 1.302 vs scale_row_g4 1.318 and row_grouped_g4 1.330,
  while rotate+scale_row_g32 worsened to 2.598 vs row_grouped_g32 2.518
- `EleutherAI/pythia-70m` INT8 and INT4 baseline runs complete (225 weight
  records, 225 activation records, 5 logit/loss records each): INT8 global
  worse than 14m (PPL ratio 1.44 vs 1.24); INT8 g4 lossless (0.971); INT4
  global catastrophic (~501 trillion PPLx); INT4 g4 gives 7.52x — much
  worse than 14m's 1.33x; g4 vs g128 at INT4 is a 478x quality gap;
  scale_row provides negligible benefit over raw row_grouped at this scale;
  timing: INT8 ~798s, INT4 780s (both ≈ 13 min, confirming bitwidth does not
  affect wall-clock runtime)
- `EleutherAI/pythia-70m` INT4 capped top-width rotation run complete (25 layers,
  325 weight records, 325 activation records, 7 logit/loss records): elapsed
  1174.6s (19.6 min); rotation worsened g4 PPLx to 8.049 vs row_grouped_g4
  7.520 and scale_row_g4 7.673; coarse g128 stayed catastrophic (3,583 PPLx)
- `distilgpt2` INT8 and INT4 baseline runs complete (216 weight records, 216
  activation records, 5 logit/loss records each, 24 compatible layers): INT8
  global shows logit MSE 8.2 and top-5 0.844 despite sub-1 PPL ratio (noise);
  INT8 g4 fully lossless (PPLx 0.999, top-5 1.000); INT4 global PPLx 48.85;
  INT4 g4 gives **1.058x** — best INT4 result of any real model in this study,
  better than Pythia-14m (1.33x) and dramatically better than Pythia-70m (7.52x)
  despite 82M params; key finding: architecture and training regime dominate over
  parameter count for INT4 quality; timing: INT8 705s, INT4 679s (~11–12 min each)
- `distilgpt2` INT4 capped top-width rotation run complete (24 layers, 312 weight
  records, 312 activation records, 7 logit/loss records): elapsed 1024.9s
  (17.1 min); rotation slightly worsened g4 PPLx to 1.062 vs row_grouped_g4
  1.058, but modestly improved the coarser g192 path to 1.201 vs 1.208
- tqdm progress bars wired into benchmark runner: two sequential bars
  (layers phase, then logit phase) driven by `on_progress` callback in
  `TransformerConfig`; falls back to no-op if tqdm not installed
- safer benchmark runner at `experiments/run_transformer_benchmark.py` with
  disconnect-safe presets, `--local-files-only`, `--torch-threads`, `--download-only`,
  and incremental CSV writes; run all heavy benchmarks from this runner in a
  detached tmux session, not from the VSCode integrated terminal
- optional bitsandbytes NF4 external baseline runner at experiments/bitsandbytes_baseline.py; reports logit/loss/perplexity plus runtime and memory metadata only, keeping bitsandbytes separate from project weight/activation reconstruction metrics and optional for local tests
- logit metrics CSVs now include per-method operational telemetry when measured:
  method elapsed seconds, CUDA peak allocated/reserved MB, total input tokens,
  tokens/sec, ms/token, and theoretical project artifact-size estimates
  (reference weight bytes, packed weight bytes, scale/scaling metadata bytes,
  and total estimated artifact bytes). Treat project runtime/memory as harness
  metrics until a real packed low-bit runtime exists.
- top-width rotation fractions are capped model-wide for transformer runs:
  requested p5/p10/p20 paths are lowered when needed so the widest selected layer
  stays within `max_rotation_pairs=1000`; for TinyStories this produced one
  common p3.0637% rotation path instead of skipping the 256-output MLP expansion
  layers

Resume reminder: `quant/rotations.py`, `quant/scaling.py`, grouped quantization (both column-grouped and row-grouped in `quant/quantizer.py`), `experiments/rotation_experiment.py`, and `experiments/sweep_experiment.py` are all complete. The sweep experiment compares 12 baseline quantization paths (global, col-grouped, row-grouped, scale, rotate, rotate+scale, rotate+scale+row-grouped) across a grid of seeds, outlier fractions, and outlier scales, writing `results/sweep_metrics.csv` and `plots/sweep_dashboard.png`. It can also opt into top-width sparse-rotation paths via `SweepConfig.top_width_pair_fractions`, e.g. `top_width_rotate_p10_global` and `top_width_rotate_scale_p10_row_g4`. Key findings from the historical sweeps: 32×32 sweep (45 cond, 12 methods) — row_grouped_g4 MSE ratio 0.112 (~9×); scale_global 0.531; rotation alone 0.902. 320×320 sweep (45 cond, 15 methods, new seeds/conditions) — row_grouped_g4 MSE ratio 0.143 (~7×); rotation adds zero measurable benefit over row-grouped at this scale; scale_global collapses to 0.845 (random scatter means every column has outliers); column-grouped converges toward global. New top-width p5/p10/p20 sweeps show sparse rotations improve global rotation paths, especially 320×320 rotate+scale_global (best p20 ratio 0.820 vs single-pair 0.844), but do not beat row-grouped quantization; row_grouped_g4 remains 0.112 on 32×32 and 0.143 on 320×320. Group size remains the dominant variable across both scales. Milestone 3 is complete: all planned transformer baselines, INT4 rotation presets, WikiText-2 validation reruns, and the rotation synthesis are documented. Next step: begin Milestone 4 planning and larger-model benchmarking.

## Environment

Repo root:

```bash
/home/mynk/PQ_project
```

Use the project virtual environment:

```bash
.venv/bin/python
```

Run tests with:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

For the current passing count and test-scope note, see the canonical Tests section below.

Matplotlib note: use `MPLCONFIGDIR=/tmp/paroquant-mpl` because the default home config path may be read-only.

Git note: this folder is now a valid Git repo on branch `main`, tracking the public GitHub repository:

```text
https://github.com/md861/QuantizationTests
```

Generated artifacts under `plots/` and `results/`, the virtual environment, caches, and local tools are ignored by Git.

## Working Protocol For Future Agents

If another coding agent resumes this project, the safest order is:

1. Read this file first, then skim the latest entry in `lab_book/project_journey.md`.
2. Use `.venv/bin/python` for project commands and set `MPLCONFIGDIR=/tmp/paroquant-mpl` for Matplotlib-backed tests or scripts.
3. Treat `plots/` and `results/` as disposable generated artifacts unless the task explicitly says otherwise.
4. Keep the current branch clean before making unrelated changes; check `git status --short --branch` first.
5. Do not commit or push unless the user explicitly asks, or it is the agreed end-of-day checkpoint.
6. When commit or push is requested, use the existing public remote `origin` for `main` in `QuantizationTests`.
7. Keep docs in sync with code changes, especially this summary and the lab book, so handoff remains easy.
8. **Always update `README.md` before every commit and push.** The README is the public-facing entry point on GitHub and must never be stale. At minimum check: milestone statuses, progress table rows, current-milestone description, and the expected test count. Treat a stale README as a broken handoff.
9. **Keep the research draft current before every commit and push when the work changes the research story.** Update `docs/research_draft.md` with new findings, examples, caveats, and figure references. Copy any paper-ready figure resources into `docs/figures/` and commit them with the draft. Do not rely on ignored `plots/` artifacts for GitHub-visible paper figures.
10. **Always record rotation metadata for rotation experiments.** Any experiment, CSV, table, figure caption, or research-draft claim involving rotations must state the number of pair rotations applied per matrix/layer and the actual percentage/fraction of possible channel pairs used. For non-rotation baselines, record `rotation_count=0` and `rotation_pair_fraction=0.0`. For top-width sparse rotations, also record the configured candidate percentage as `rotation_candidate_fraction` and distinguish it from the actual independent rotation count/fraction.
11. **Handover Diagnostic shorthand.** When the user says "handover diagnostic" or after a VS Code/Codex restart, do this checklist: read `project_summary.md`, skim the latest `lab_book/project_journey.md` entry, check `git status --short --branch`, inspect recent commits with `git log --oneline -12`, verify the test suite, search docs for stale milestone/test-count/output references, **check the Benchmark Run Timings table below and flag any missing entries**, **if `docs/runpod/usage_ledger.md` changed or RunPod was used, reconcile `docs/runpod/README.md` Usage Dashboard totals against the ledger**, run the post-restart autonomy checks (`git --version` from PowerShell if available, `./scripts/codex.cmd status` from PowerShell if the local helper exists, or `make status` from WSL if the local helper exists), then report what changed since the last session, current stale states, tool/autonomy health, and the next recommended step.
12. **Always record wall-clock timing for every benchmark run.** When launching a run, state the estimated duration upfront based on the timings table below and, when available, `tools/estimate_benchmark_runtime.py` over prior result CSVs. When the run completes, copy the `elapsed: Xs (Ymin)` line from the runner log into both the lab book session entry and the Benchmark Run Timings table. Future logit metrics CSVs must include per-row method timing and CUDA peak fields when the runner can measure them. If timing was not captured, write "timing not captured" explicitly — never leave the table entry blank.
13. **Prompt before GPU benchmark runs.** Before launching any nontrivial RunPod benchmark, report the predicted duration/cost, the evidence used for the prediction, the exact command/preset, target commit, output paths, and whether the estimate is job-level or per-method. Wait for explicit user approval before starting the run.
14. **GPU choice value rule.** Do not choose a RunPod GPU solely because it is faster on paper. Choose the cheapest GPU that can answer the benchmark question within the user's time window, and justify any GPU-class change with expected wall time, method throughput when available, VRAM headroom, hourly rate, setup/model-load overhead, and estimated cost per completed benchmark. Use prior rows in the Benchmark Run Timings table and `docs/runpod/usage_ledger.md`; record the estimate and approval before launch.
15. **RunPod credit guardrails.** Treat RunPod as a benchmark worker only, not as the default development environment. Follow `docs/runpod/operations.md` for Pod setup, storage, stop-window, security policy, GPU-choice value checks, and `docs/runpod/usage_ledger.md` for RunPod time accounting. Keep code generation, ordinary tests, data analysis, plotting, README/research-draft/lab-book updates, and report writing local unless debugging a GPU-only failure. Before any RunPod run: make the smallest local dry run pass, estimate cost and duration, run a single-layer or small-subset smoke benchmark first, launch long jobs only in detached tmux, write logs and results under persistent /workspace, record commit hash, GPU, VRAM, peak memory, elapsed time, method telemetry when available, and hourly rate, update the RunPod ledger, dashboard, and Benchmark Run Timings table before handover, and pull back only artifacts needed for analysis. Keep Hugging Face cache under `/workspace/hf_cache` via `HF_HOME` and `HUGGINGFACE_HUB_CACHE`; rerun online cache prep after Pod replacement before using `--local-files-only`. Stop the Pod after each benchmark unless another GPU benchmark is already queued to start within about 30 minutes.
16. **Track implementation time for substantial work.** For multi-step coding/research sessions, update `docs/implementation_time_log.md` with active implementation time separately from user-wait and hardware/tool-wait time. This log is for future prediction calibration, not research evidence.

17. **Shell and editing discipline.** Prefer WSL-native commands for repo work and use local helpers when present: make status, make test, make collect-tests, or scripts/codex.cmd status. Avoid complex PowerShell-mediated one-liner edits against WSL files because PowerShell, wsl.exe, Bash, Perl, and sed can each interpret dollar signs, pipes, backticks, quotes, and escapes differently. Keep pipelines inside bash -lc when a pipeline is needed. Prefer apply_patch for manual edits when it can access the file; otherwise use small, inspectable WSL line-based edits. Avoid Perl/Python one-liners with shell-sensitive variables or heredocs from PowerShell. Temporary helper scripts for complex edits must live under WSL /tmp, not in the repo root, unless the helper itself is intended to be tracked. Remove any helper before status/commit. If a WSL command fails with Access is denied in the managed environment, treat it as a sandbox/launch-context boundary and rerun the same necessary command through the approved/escalated path rather than debugging it as a project failure.

18. **Single source of truth for tests.** Keep the canonical test command, scope, and latest passing count only in the Tests section below. Do not duplicate the passing count elsewhere. When tests are added/removed or the count changes, update that section and run a stale-doc search before commit:

```bash
rg "passed, [0-9]+ warnings|tests/test_" project_summary.md README.md docs/research_draft.md
```

## Benchmark Run Timings

This table is the authoritative record of wall-clock runtimes for benchmark
runs. Update it every time a run completes. Use it to give the user upfront
duration estimates before launching any new run.

Hardware context: WSL2 on Windows, 32 GB RAM, CPU-only (no GPU), 2 Torch
threads (`--torch-threads 2`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`).

| Model | Params | Layers | Bitwidth | Rotations | Elapsed | Notes |
|---|---:|---:|---|---|---:|---|
| sshleifer/tiny-gpt2 | ~0.1M | 8 | INT4+INT8 | p5/p10/p20 | timing not captured | run predates timer |
| roneneldan/TinyStories-1M | 1M | 48 | INT4+INT8 | p3.0637% | timing not captured | run predates timer |
| EleutherAI/pythia-14m | 14M | 25 | INT8 | none | ~3 min | file-timestamp estimate |
| EleutherAI/pythia-14m | 14M | 25 | INT4 | none | ~3.5 min | file-timestamp estimate |
| EleutherAI/pythia-14m | 14M | 25 | INT4 | p0.0001% effective | failed before fix | old selector enumerated ~1.26B `embed_out` pairs and stalled at 24/25 layers |
| EleutherAI/pythia-14m | 14M | 25 | INT4 | p0.0001% effective | 240.1s (4.0 min) | elapsed from runner log after wide-layer selector fix |
| EleutherAI/pythia-14m | 14M | 25 | INT4 | p0.0001% effective | 288.8s (4.8 min) | WikiText-2 validation sample, 7 texts |
| EleutherAI/pythia-70m | 70M | 25 | INT8 | none | ~798s (13.3 min) | file-timestamp estimate; elapsed line missed (run predates timer fix) |
| EleutherAI/pythia-70m | 70M | 25 | INT4 | none | 780s (13.0 min) | elapsed from runner log |
| EleutherAI/pythia-70m | 70M | 25 | INT4 | p0.0001% effective | 1174.6s (19.6 min) | elapsed from runner log |
| EleutherAI/pythia-70m | 70M | 25 | INT4 | p0.0001% effective | 1284.5s (21.4 min) | WikiText-2 validation sample, 7 texts |
| distilgpt2 | 82M | 24 | INT8 | none | 705s (11.8 min) | elapsed from runner log |
| distilgpt2 | 82M | 24 | INT4 | none | 679s (11.3 min) | elapsed from runner log |
| distilgpt2 | 82M | 24 | INT4 | p0.0212% effective | 1024.9s (17.1 min) | elapsed from runner log |
| distilgpt2 | 82M | 24 | INT4 | p0.0212% effective | 1137.2s (19.0 min) | WikiText-2 validation sample, 7 texts |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 1 | INT4 | none | 228.3s (3.8 min) | RunPod RTX 4000 Ada smoke, single `q_proj` layer, 1 calibration text, peak CUDA allocated 2124 MB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 154 | INT4 logit-only matrix | none | 1004.4s (16.7 min) | RunPod RTX 4000 Ada, 256 WikiText-2 records, 5 project methods, peak CUDA allocated 2274 MB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 154 | INT4 logit-only matrix with per-method telemetry | none | 1208.7s (20.1 min) | RunPod RTX 4000 Ada, 256 WikiText-2 records, 5 project methods, wall 23m43s; isolated method seconds: global 79.527, row_g4 34.542, row_g8 30.810, scale_g4 38.282, scale_g8 34.937; peak CUDA allocated 2274 MB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | external | bitsandbytes NF4 float16 | none | 231.4s (3.9 min) | RunPod RTX 4000 Ada, 256 WikiText-2 records, peak CUDA allocated 2274 MB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | external | bitsandbytes NF4 float16 telemetry rerun | none | 191.5s (3.2 min) | RunPod RTX 4090, 256 WikiText-2 records, wall 6m24s; isolated method loop 24.577s, 1354.168 tokens/s, 0.738 ms/token, peak CUDA allocated 963 MB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | external | AWQ 4-bit | none | 238.2s (4.0 min) | RunPod RTX 4090, 256 WikiText-2 records; isolated method loop 39.409s, 844.535 tokens/s, 1.184 ms/token, peak CUDA allocated 904 MB; first Pod pass required gptqmodel/ninja dependency setup and Marlin JIT compile |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | external | GPTQ 4-bit | none | 262.2s (4.4 min) | RunPod RTX 4090, 256 WikiText-2 records; isolated method loop 58.086s, 572.980 tokens/s, 1.745 ms/token, peak CUDA allocated 904 MB; dependency setup shared with AWQ |
| Qwen/Qwen2.5-3B-Instruct | 3B | download/cache | reference cache prep | none | 56.5s runner; 283s wall | RunPod RTX 4090, model/tokenizer cache prep, peak CUDA 0 MB, HF cache grew to about 7.9 GB |
| Qwen/Qwen2.5-3B-Instruct | 3B | 1 layer | INT4 `scale_row_g4` logit-only smoke | none | 73.8s runner; 306s wall | RunPod RTX 4090, single `model.layers.0.self_attn.q_proj`, 1 text, logit MSE 0.002384, PPL ratio 1.0534, peak CUDA allocated 6011 MB |
| Qwen/Qwen2.5-3B-Instruct-AWQ | 3B | external | AWQ 4-bit smoke | none | failed after 357s wall | RunPod RTX 4090, exit 132 after selecting `AwqMarlinLinear`; no metrics written |
| Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 | 3B | external | GPTQ 4-bit smoke | none | failed after 374s wall | RunPod RTX 4090, exit 132 after selecting `MarlinLinear`; no metrics written |
| facebook/opt-2.7b | 2.7B | download/cache | reference cache prep | none | 75.7s runner; 288s wall | RunPod RTX 4090, model/tokenizer cache prep, peak CUDA 0 MB, HF cache grew to about 23 GB |
| facebook/opt-2.7b | 2.7B | 1 layer | INT4 `scale_row_g4` logit-only smoke | none | 37.4s runner; 214s wall | RunPod RTX 4090, single `model.decoder.layers.0.self_attn.q_proj`, 1 text, logit MSE 0.000161, PPL ratio 1.0012, peak CUDA allocated 5079 MB |
| facebook/opt-2.7b | 2.7B | external | bitsandbytes NF4 float16 smoke | none | 54.3s runner; 244s wall | RunPod RTX 4090, 1 WikiText-2 record, logit MSE 0.167516, PPL ratio 1.0598, peak CUDA allocated 1984 MB |

**Prediction rule (update as more data arrives):** Pythia-14m baselines ~3 min
(25 layers), Pythia-14m rotation ~4 min after the wide-layer selector fix,
Pythia-70m baselines ~13 min and rotation ~20 min (25 layers), distilgpt2
baselines ~11-12 min and rotation ~17 min (24 layers). Runtime
scales with layer count, not parameter count — distilgpt2 is faster than Pythia-70m
despite being larger, with a similar compatible-layer count (24 vs 25).
INT8 and INT4 take essentially identical wall-clock time at all model sizes,
confirming runtime is dominated by weight passes not bitwidth arithmetic.

Typical publish flow:

```bash
git status --short --branch
# update README.md, project_summary.md, lab_book/project_journey.md as needed
# update docs/research_draft.md and docs/figures/ when findings or paper figures change
git add <files>
git commit -m "<short message>"
git push origin main
```

## Implemented Modules

### `quant/matrix_factory.py`

Creates reproducible synthetic matrices for controlled quantization tests. These matrices are the input data used by the quantizer, metrics, and experiments.

- `gaussian_matrix(...)`
  - Creates independent Gaussian entries.
  - Formula: $x_{ij} \sim \mathcal{N}(\mu, \sigma^2)$
- `heavy_tailed_matrix(...)`
  - Creates Student-t entries for outlier-prone, heavy-tailed data.
  - Formula: $x_{ij} = s \cdot t_{\nu}$, where $\nu$ is the degrees of freedom and $s$ is a scale factor.
- `outlier_matrix(...)`
  - Creates a Gaussian base matrix, then replaces a controlled fraction of entries with large signed outliers.
  - Approximate rule: choose $k = \mathrm{round}(f \cdot mn)$ entries and set them near $\mu \pm |\mathcal{N}(\alpha\sigma, \sigma^2)|$, where $f$ is `outlier_fraction` and $\alpha$ is `outlier_scale`.
- `make_matrix(...)`
  - Dispatches by `MatrixKind` or string name for experiment-friendly matrix creation.
- `MatrixKind`

All generators support `shape`, `seed`, and `dtype`.

### `quant/quantizer.py`

Implements symmetric full-matrix quantization and stores both integer codes and dequantized reconstructions.

- `QuantizationResult`
  - Stores `quantized`, `dequantized`, `scale`, `bitwidth`, `qmin`, and `qmax`.
- `symmetric_quantize(matrix, bitwidth=...)`
  - Computes a single full-matrix scale from the maximum absolute value.
  - Scale: $s = \max(|W|) / (2^{b-1} - 1)$
  - Quantize: $Q = \mathrm{clip}(\mathrm{round}(W / s), q_{\min}, q_{\max})$
  - Dequantize: $\hat{W} = sQ$
- `quantize_int8(matrix)`
- `quantize_int4(matrix)`
- `grouped_symmetric_quantize(matrix, bitwidth=..., group_size=...)`
  - Quantizes contiguous **column** groups with one symmetric scale per group.
  - For group $g$, scale is $s_g = \max(|W_g|) / (2^{b-1}-1)$.
  - Dequantization uses $\hat{W}_g = s_g Q_g$.
- `quantize_int8_grouped(matrix, group_size=...)`
- `quantize_int4_grouped(matrix, group_size=...)`
- `row_grouped_symmetric_quantize(matrix, bitwidth=..., row_group_size=...)`
  - Quantizes contiguous **row** groups within each column independently (the GPTQ/AWQ approach).
  - Gives $n_{\mathrm{cols}} \times \lceil n_{\mathrm{rows}} / g \rceil$ scales in total.
  - Outliers in one row-group only inflate that group's scale; all other groups keep tight precision.
  - Implementation is vectorized by row group across all columns to keep TinyLlama-scale INT4 runs feasible.
- `quantize_int8_row_grouped(matrix, row_group_size=...)`
- `quantize_int4_row_grouped(matrix, row_group_size=...)`

Ranges:

- INT8: `[-127, 127]`
- INT4: `[-7, 7]`, stored as NumPy `int8`

Zero matrices use `scale=1.0` and reconstruct exactly to zeros.

Column-grouped results store scales in `QuantizationResult.scales` (shape: `(n_col_groups,)`) and `group_size`.
Row-grouped results store scales in `QuantizationResult.scales` (shape: `(n_cols, n_row_groups)`) and `row_group_size`.
The scalar `scale` field is the mean of all group scales for summary compatibility.

### `quant/metrics.py`

Computes reconstruction, similarity, spectrum, and integer-code diagnostics comparing an original matrix $W$ with a dequantized reconstruction $\hat{W}$. Let $E = \hat{W} - W$.

- MSE: $\mathrm{mean}(E^2)$
- MAE: $\mathrm{mean}(|E|)$
- relative Frobenius error: $\|W - \hat{W}\|_F / \|W\|_F$
- cosine similarity: $\langle W, \hat{W} \rangle / (\|W\|_2\|\hat{W}\|_2)$ after flattening
- SNR in dB: $10\log_{10}(\sum W^2 / \sum (W-\hat{W})^2)$
- max absolute error: $\max(|E|)$
- mean signed error: $\mathrm{mean}(E)$
- error standard deviation: $\mathrm{std}(E)$
- spectrum L2 error: $\|\sigma(W) - \sigma(\hat{W})\|_2$
- relative spectrum L2 error: $\|\sigma(W) - \sigma(\hat{W})\|_2 / \|\sigma(W)\|_2$
- rank and stable-rank diagnostics
- optional saturation fraction: fraction of codes equal to $q_{\min}$ or $q_{\max}$
- optional zero fraction: fraction of codes equal to 0

Main API:

- `QuantizationMetrics`
- `compute_quantization_metrics(...)`

### `quant/spectrum.py`

Computes singular-value analysis for understanding how quantization changes matrix geometry. For a matrix $W$, singular values are $\sigma_1 \ge \sigma_2 \ge \dots$.

- `SpectrumStats`
  - Stores singular values, rank, spectral/nuclear/Frobenius norms, condition number, stable rank, and explained energy.
- `singular_values(...)`
  - Computes $\sigma(W)$ using SVD.
- `explained_energy(...)`
  - Computes cumulative squared singular-value energy.
  - Formula: $\mathrm{energy}_k = \sum_{i=1}^{k}\sigma_i^2 / \sum_i \sigma_i^2$
- `analyze_spectrum(...)`
  - Computes rank, norms, condition number, stable rank, and explained energy.
  - Stable rank: $\|W\|_F^2 / \|W\|_2^2 = \sum_i\sigma_i^2 / \sigma_1^2$
- `compare_spectra(...)`
  - Compares singular-value spectra and stable-rank changes between reference and candidate matrices.

### `quant/rotations.py`

Implements pairwise Givens channel rotations for outlier redistribution before quantization.

- `GivensRotation`
  - Frozen dataclass storing `i`, `j`, and `theta` for one rotation.
- `rotation_matrix(n, i, j, theta)`
  - Returns an $n \times n$ identity matrix with the $(i, j)$ subblock replaced by the Givens rotation:
    $R[i,i] = R[j,j] = \cos\theta$, $R[i,j] = -\sin\theta$, $R[j,i] = \sin\theta$.
  - Right-multiplying: $W' = W R$ rotates columns $i$ and $j$ of $W$.
- `apply_rotation(matrix, i, j, theta)`
  - Applies the rotation directly to columns $i$ and $j$ without constructing the full $n \times n$ matrix:
    $w_i' = \cos\theta \cdot w_i + \sin\theta \cdot w_j$,
    $w_j' = -\sin\theta \cdot w_i + \cos\theta \cdot w_j$.
  - Preserves input dtype; computes in float64 internally.
- `optimal_angle(matrix, i, j, *, n_search=360)`
  - Grid-searches $\theta \in [0, \pi)$ for the angle minimising $\max(|w_i'|_\infty, |w_j'|_\infty)$.
  - The objective is $\pi$-periodic, so $[0, \pi)$ covers the full search space.
- `rotate_channel_pair(matrix, i, j, *, n_search=360)`
  - Convenience wrapper returning `(rotated_matrix, theta)`.
- `apply_sequential_rotations(matrix, rotations)`
  - Applies a list of `GivensRotation` objects in order.

Key invariant: Givens rotations are orthogonal, so $\|W'\|_F = \|W\|_F$ exactly.

### `quant/scaling.py`

Implements reversible per-channel scaling for balancing column magnitudes before quantization.

- `ChannelScaling`
  - Frozen dataclass storing `factors` and `target_max_abs`.
- `column_max_abs(matrix)`
  - Returns $\max_i |W_{ij}|$ for each column $j$.
- `compute_channel_scaling(matrix, *, target_max_abs=None)`
  - Computes one positive factor per column.
  - Default target: mean of nonzero column max-abs values.
  - Formula for nonzero columns: $d_j = \tau / \max_i |W_{ij}|$, where $\tau$ is the target max-abs.
  - Zero columns receive factor 1.0.
- `apply_channel_scaling(matrix, scaling)`
  - Applies $W' = W D$, where $D = \mathrm{diag}(d_1,\dots,d_n)$.
- `invert_channel_scaling(matrix, scaling)`
  - Applies $W = W' D^{-1}$.
- `balance_channel_max_abs(matrix, *, target_max_abs=None)`
  - Convenience wrapper returning `(scaled_matrix, scaling)`.

Key invariant: scaling is exactly reversible up to floating-point rounding when the same `ChannelScaling` metadata is used for inversion.

### `quant/visualize.py`

Provides Matplotlib visualizations:

- `plot_matrix_heatmap(...)`
- `plot_matrix_grid(...)`
- `plot_singular_values(...)`
- `plot_value_histogram(...)`
- `plot_residual_histogram(...)`
- `plot_quantized_code_histogram(...)`
- `plot_spectrum_comparison(...)`
- `plot_quantization_summary(...)`
- `plot_quantization_comparison(...)`
- `plot_quantization_histograms(...)`
- `plot_channel_scaling_quantization_dashboard(...)`

Spectrum styling convention:

- `Original`: thicker dotted line
- quantized/dequantized curves: thinner solid lines

`plot_quantization_summary(...)` includes:

- original heatmap
- quantized-code heatmap
- dequantized heatmap
- residual heatmap
- spectrum comparison
- metrics panel with error and spectrum diagnostics
- data representation fields:
  - original dtype and bitwidth
  - quantized storage dtype and storage bitwidth

Default experiment plots use `plot_quantization_comparison(...)` to reduce plot count. Each comparison figure includes:

- original matrix
- one residual heatmap per quantization method
- singular-value spectra comparing original and dequantized data per method
- side-by-side metric summaries per method

`plot_channel_scaling_quantization_dashboard(...)` compares global quantization with per-channel scaled quantization. It includes:

- original matrix, global residual, and channel-scaled residual heatmaps
- original and scaled column max-abs bars
- residual max-abs per column
- singular-value spectra for original, global INT4, and channel-scaled INT4
- per-column MSE bars
- summary metrics for global and channel-scaled paths

### `experiments/baseline_experiment.py`

Runs the first end-to-end Milestone 1 experiment:

- generates Gaussian, heavy-tailed, and outlier matrices
- quantizes each with INT8 and INT4
- computes metrics
- writes `results/baseline_metrics.csv`
- writes three quantization comparison plots:
  - `plots/baseline_gaussian_comparison.png`
  - `plots/baseline_heavy_tailed_comparison.png`
  - `plots/baseline_outlier_comparison.png`

### `experiments/outlier_experiment.py`

Runs an outlier-severity sweep:

- sweeps controlled outlier fractions and outlier scales
- quantizes each matrix with INT8 and INT4
- computes the same reconstruction, spectrum, zero-fraction, and saturation diagnostics as baseline
- writes `results/outlier_metrics.csv`
- optionally writes one quantization comparison plot per fraction/scale condition:
  - `plots/outlier_fraction_<fraction>_scale_<scale>_comparison.png`

### `experiments/analyze_results.py`

Analyzes generated baseline and outlier CSVs:

- compares INT4 against INT8 for each experiment condition
- computes MSE ratios, relative-Frobenius ratios, SNR deltas, zero-fraction deltas, and saturation deltas
- prints a compact summary
- optionally writes:
  - `results/baseline_analysis.csv`
  - `results/outlier_analysis.csv`
- optionally writes a collated benchmark-style dashboard:
  - `plots/analysis_dashboard.png`

### `experiments/sweep_experiment.py`

Comparative sweep across all quantization paths and outlier conditions (Milestone 2).

- **Paths compared** (for a 32×32 matrix by default):
  - `global`: standard full-matrix INT4
  - `col_grouped_g{g}`: column-grouped INT4 for each configured group size
  - `row_grouped_g{g}`: row-grouped INT4 for each configured row group size
  - `scale_global`: per-channel scale then global INT4
  - `rotate_global`: pairwise Givens rotation then global INT4
  - `rotate_scale_global`: rotation + scaling then global INT4
  - `rotate_scale_row_g{g}`: rotation + scaling then row-grouped INT4
  - `top_width_rotate_p{pct}_global`: opt-in sparse independent rotations
    selected from the top percentage of channel width differences, then global INT4
  - `top_width_rotate_scale_p{pct}_global`: top-width rotations + scaling + global INT4
  - `top_width_rotate_scale_p{pct}_row_g{g}`: top-width rotations + scaling + row-grouped INT4
- **Condition grid**: seeds × outlier_fractions × outlier_scales (configurable via `SweepConfig`)
- **Outputs**: `results/sweep_metrics.csv` and `plots/sweep_dashboard.png`
- **Rotation metadata**: every CSV row includes `rotation_count`, actual `rotation_pair_fraction`, and configured `rotation_candidate_fraction`; non-rotation baselines use 0, 0.0, and 0.0
- **Aggregation**: summary tables use condition-wise MSE ratios (`method_mse / global_mse` on the same matrix), then report mean and standard deviation across sweep conditions. Standard deviation is spread across seeds/outlier severities, not a confidence interval.
- **Dashboard**: 4 panels — mean MSE ratio per method with std error bars, mean zero fraction per method with std error bars, MSE ratio vs outlier severity, effect of row group size
- **Sweep results — 32×32** (45 cond, 12 methods): row_grouped_g4=0.112, rotate_scale_row_g4=0.111, scale_global=0.531, col_grouped_g4=0.766, rotate_global=0.902. Column-grouped gives no improvement for row-localised outliers. Rotation adds small benefit through scaling.
- **Sweep results — 320×320** (45 cond, 15 methods, new seeds/conditions/group sizes): row_grouped_g4=0.143, rotate_scale_row_g4=0.143, scale_global=0.845, col_grouped_g4=0.898, rotate_global=0.965. At large scale with random scatter: rotation adds zero measurable benefit; per-channel scaling loses effectiveness because random scatter means every column has outliers; only row-grouped remains effective.
- **Top-width sparse sweep results**: p5/p10/p20 top-width rotations improve rotation-only global paths and 320×320 rotate+scale_global, but they do not improve the best row-grouped results. On 32×32, p20 rotation-only global improves to 0.811 vs single-pair rotate_global 0.902, while rotate_scale_row_g4 remains best at 0.111. On 320×320, p20 rotate+scale_global improves to 0.820 vs single-pair rotate_scale_global 0.844, but row_grouped_g4 and rotate_scale_row_g4 remain best at 0.143.

API:

- `SweepConfig` — dataclass with grid parameters and output settings; `csv_name` and `plot_name` fields allow multiple sweeps to coexist without overwriting; `top_width_pair_fractions` enables opt-in top-width sparse-rotation paths
- `SweepRecord` — frozen dataclass with metrics for one (condition, method) pair
- `run_sweep_experiment(config) -> list[SweepRecord]`
- `methods_in_config(config) -> list[str]`
- `print_summary(records) -> None`

Two sweeps have been run:
- 32×32, seeds 0–4, fractions [0.01,0.05,0.10], scales [5,10,20] → `results/sweep_metrics.csv`, `docs/figures/sweep_dashboard.png`
- 320×320, seeds 5–9, fractions [0.02,0.07,0.15], scales [7.5,15,30] → `results/sweep_metrics_320x320.csv`, `docs/figures/sweep_dashboard_320x320.png`
- 32×32 with top-width p5/p10/p20 rotations → `results/sweep_metrics_top_width_32x32.csv`, `docs/figures/sweep_dashboard_top_width_32x32.png`
- 320×320 with top-width p5/p10/p20 rotations → `results/sweep_metrics_top_width_320x320.csv`, `docs/figures/sweep_dashboard_top_width_320x320.png`

### `experiments/transformer_experiment.py`

Milestone 3 transformer quantization harness.

- **Config**: `TransformerConfig` — `model_name`, `calibration_texts`,
  `calibration_text_source`,
  `single_layer_name` (None = all layers), `bitwidths` ([4, 8] by default),
  `row_group_sizes` (fixed sizes applied to every layer, e.g. [4]),
  `row_group_fractions` (sizes relative to each layer's n_rows: 0.5 → n/2,
  0.25 → n/4, 0.0625 → n/16; computed as `max(1, round(n_rows × f))` then
  merged with fixed sizes and deduplicated, so a 64-row weight produces group
  sizes [4, 32, 16] while a 256-row weight produces [4, 128, 64, 16]),
  `top_width_pair_fractions` ([0.05, 0.10, 0.20]),
  `max_rotation_pairs` (safety cap for large models; transformer runs lower
  configured top-width fractions model-wide when needed so every selected layer
  can run the same capped rotation paths),
  `local_files_only` (load Hugging Face model/tokenizer artifacts from cache
  only),
  `incremental_results` (append CSV rows during the run so partial long runs
  preserve completed records),
  `delete_hf_cache_after` (evict HF model cache after run).
- **Records**: `WeightRecord`, `ActivationRecord`, `LogitRecord` — each carries
  a `bitwidth` field (4 or 8) alongside `method`.
- **Weight experiment**: quantizes each linear layer weight at each configured
  bitwidth with four path families — global, row-grouped (all resolved group
  sizes), scale+row-grouped, top-width-rotate+scale+row-grouped — and computes
  MSE, relative Frobenius error, cosine similarity, SNR, zero fraction,
  saturation fraction, and rotation metadata per layer per (method, bitwidth).
- **Top-width cap policy**: before an all-layer run, the harness computes the
  widest selected layer's possible channel-pair count and lowers requested
  `top_width_pair_fractions` as needed so `round(total_pairs * fraction)` never
  exceeds `max_rotation_pairs`. Duplicate effective fractions are deduplicated.
  This keeps rotation method names common across layers for full-model logit/loss
  comparisons.
- **Activation experiment**: registers a forward hook to capture layer inputs once,
  then analytically computes the output drift for each quantized weight without
  re-running the full model. Measures activation MSE, cosine similarity, and
  relative error.
- **Logit/loss experiment**: temporarily swaps all selected layer weights per
  method, but regenerates one method's dequantized weights just-in-time instead
  of storing every layer/method reconstruction. This reduces temp disk pressure
  for larger Milestone 3 models such as Pythia while preserving the same output
  metrics: logit MSE, cosine similarity, top-5 token overlap, next-token loss
  delta, perplexity, original perplexity, perplexity ratio, calibration text
  source, and calibration text count.
- **Held-out text resource**:
  `docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt`
  contains the primary tracked Milestone 4 WikiText-2 raw validation resource:
  256 paragraph-separated records, 23,742 words, and 125,852 UTF-8 bytes,
  with attribution, license notes, dataset revision, and extraction recipe in
  the file header. Use `--eval-text-file` on
  `experiments/run_transformer_benchmark.py` or
  `experiments/bitsandbytes_baseline.py` to evaluate on it. The older
  `wikitext2_raw_validation_sample.txt` remains as a tiny historical sample.
- **Outputs**: `results/transformer_weight_metrics.csv`,
  `results/transformer_activation_metrics.csv`,
  `results/transformer_logit_metrics.csv`, and `plots/transformer_dashboard.png`.
- **HF Conv1D note**: GPT-2-style Conv1D stores weights as (in, out); nn.Linear
  stores (out, in). The harness normalises both to (in, out) internally via
  `_extract_weight` / `_set_weight`.
- **Hardware note**: `lm_head` is excluded from quantization (embedding-tied,
  not a typical quantization target). Set `max_rotation_pairs` (default 1000) to
  avoid slow top-width rotation on large weight matrices in bigger models.

### `experiments/run_transformer_benchmark.py`

Safer runner for transformer benchmark work. It wraps
`run_transformer_experiment` with conservative named presets, explicit
Hugging Face cache modes, CPU-thread throttling, progress bars, elapsed-time
logging, optional held-out evaluation text, and incremental CSV output.

Current presets:

- `tiny-gpt2-smoke`
- `pythia-14m-int8-baseline`
- `pythia-14m-int4-baseline`
- `pythia-14m-int4-rotation`
- `pythia-70m-int8-baseline`
- `pythia-70m-int4-baseline`
- `pythia-70m-int4-rotation`
- `distilgpt2-int8-baseline`
- `distilgpt2-int4-baseline`
- `distilgpt2-int4-rotation`

Important options:

- `--download-only`: prepare Hugging Face model/tokenizer artifacts and exit.
- `--local-files-only`: require cached Hugging Face files during the run.
- `--torch-threads N`: call `torch.set_num_threads(N)` and set
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `TORCH_NUM_THREADS` for the process.
- `--no-incremental-results`: opt out of the default incremental CSV append.

Example cached-run flow:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/run_transformer_benchmark.py pythia-14m-int8-baseline --download-only
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/run_transformer_benchmark.py pythia-14m-int8-baseline --local-files-only --torch-threads 2
```

### `experiments/rotation_experiment.py`

Runs the first Milestone 2 transformation experiment on one controlled outlier-heavy matrix.

Compared INT4 paths:

- `baseline`: $W \rightarrow \mathrm{INT4} \rightarrow \hat{W}$
- `rotation_only`: $W \rightarrow W R \rightarrow \mathrm{INT4} \rightarrow \widehat{WR} \rightarrow \widehat{W} = \widehat{WR} R^{-1}$
- `scaling_only`: $W \rightarrow W D \rightarrow \mathrm{INT4} \rightarrow \widehat{WD} \rightarrow \widehat{W} = \widehat{WD}D^{-1}$
- `rotation_scaling`: $W \rightarrow W R \rightarrow W R D \rightarrow \mathrm{INT4} \rightarrow \widehat{WRD} \rightarrow \widehat{W} = \widehat{WRD}D^{-1}R^{-1}$

The rotation pair is selected from the two columns with largest max-abs values, then `rotate_channel_pair(...)` chooses the angle that minimises the joint max-abs over that pair.

Default outputs:

- `results/rotation_metrics.csv`
- `plots/rotation_scaling_comparison.png`

The dashboard includes:

- transformed matrix heatmaps
- final residual heatmaps against the original matrix
- transformed column max-abs bars
- per-column MSE bars
- singular-value spectra against the original matrix
- summary metrics for all four paths

## Tests

Test scope: all `tests/test_*.py` files in [`tests/`](tests/).
Do not manually maintain a long test-file list here; use the folder glob as the
source of truth.

Run all tests:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
```

Current known passing test state:

```text
225 passed, 1 warning
```

## Design Conventions

- Prioritize clarity, reproducibility, and mathematical transparency over performance.
- Keep numerical logic separate from plotting:
  - generation: `matrix_factory.py`
  - quantization: `quantizer.py`
  - reconstruction metrics: `metrics.py`
  - singular-value analysis: `spectrum.py`
  - plotting: `visualize.py`
- Expose intermediate tensors and metadata for debugging.
- Use seeded NumPy generators for reproducibility.
- Keep APIs small and experiment-friendly.
- Update both this file and `lab_book/project_journey.md` as the project evolves.
- Update `docs/research_draft.md` when a result becomes part of the research story.
- Copy paper-ready figure resources into `docs/figures/` and commit them with the draft.

## Research Draft

The living paper-style draft is:

```text
docs/research_draft.md
```

It currently summarizes the matrix-level sandbox, INT8/INT4 examples, metrics, outlier failure modes, result-analysis dashboards, rotation/scaling tests, row-grouped quantization, top-width sparse rotation selection, Milestone 2 sweep findings, Milestone 3 transformer benchmarks, WikiText-2 validation reruns, and the negative current conclusion on uncalibrated sparse rotations. Tracked paper figures live under `docs/figures/`. Keep claims cautious unless they are supported by sweeps or repeated evidence.

## Current Baseline Result

Run:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
```

## Current Outlier Sweep

Run:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
```

Key observation from the first run:

- INT8 preserves Gaussian, heavy-tailed, and outlier matrices well.
- INT4 degrades much more on heavy-tailed and outlier-heavy matrices.
- Heavy-tailed INT4 produced a very high zero fraction, matching the expected low-bit outlier-pressure failure mode.

## Next Recommended Step

Milestone 3 has completed the planned baseline and INT4 rotation preset runs on
tiny-gpt2, TinyStories-1M, Pythia-14M, Pythia-70M, and distilgpt2. Do not start
a new better-rotation-strategy branch yet.

Milestone 4 hardware/cache audit is complete: this machine has 31 GiB RAM, about
226 GiB free on the project filesystem, a 535 MiB Hugging Face cache, and no
usable CUDA device in the current environment. The user has access to RunPod
GPUs, so larger-model GPU benchmarks should run there under strict credit
guardrails while development, analysis, plotting, and documentation remain local.

RunPod benchmark-worker setup is active for controlled smoke benchmarks. SSH
access is configured locally as alias `runpod-pq`; raw connection details,
keys, account identifiers, Pod IDs, ports, and hostnames are intentionally not
committed. The repo is cloned at `/workspace/PQ_project` on a persistent
`/workspace` network volume. The current replacement Pod observed on
2026-07-03 is NVIDIA RTX 4000 Ada Generation with 20475 MiB VRAM and driver
550.127.05, synced to commit `4b5d5d0`. The project venv on the Pod is a clean
self-contained `/workspace/PQ_project/.venv` with PyTorch 2.6.0+cu124,
Transformers 5.12.1, Accelerate 1.14.0, bitsandbytes 0.49.2, CUDA available,
and the full test suite previously verified on this worker class:

```text
212 passed, 1 warning in 349.22s (0:05:49)
```

Hugging Face cache must live on the network volume via `HF_HOME=/workspace/hf_cache`
and `HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub`; TinyLlama is currently
cached there at about 2.1 GB. Detailed RunPod technical operations live in
`docs/runpod/operations.md`.

Locked first TinyLlama comparison matrix:

| Row | Method | Scope | Eval texts | Purpose |
| --- | --- | --- | ---: | --- |
| 1 | Original HF model | Reference logits/loss only | 256 | Anchor loss/PPL/logits |
| 2 | Project INT4 `global` | All compatible linear layers, excluding `lm_head` | 256 | Negative/control row |
| 3 | Project INT4 `row_grouped_g4` | All compatible linear layers | 256 | Main project row |
| 4 | Project INT4 `row_grouped_g8` | All compatible linear layers | 256 | Conservative row-group comparison |
| 5 | Project INT4 `scale_row_g4` | All compatible linear layers | 256 | Check whether scaling helps |
| 6 | Project INT4 `scale_row_g8` | All compatible linear layers | 256 | Scaling check for g8 |
| 7 | bitsandbytes NF4 `float16` | External runtime baseline | 256 | First external baseline |

Do not include rotations in the first TinyLlama matrix. Use
`docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt` for
research-grade comparisons and reserve one-text/built-in text batches for
smoke checks. Compare bitsandbytes only on shared end-to-end fields: logit
MSE/cosine, top-5 overlap, loss delta, PPL/PPL ratio, elapsed time, peak CUDA
memory, and artifact size.


Project INT4 logit-only 256-text TinyLlama result at commit `ceddbaf`:

| Method | Logit MSE | Logit cosine | Top-5 | Loss delta | PPL ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global` | 21.9267 | 0.2739 | 0.0002 | +8.5388 | 5109.2560 |
| `row_grouped_g4` | 0.1123 | 0.9988 | 0.9019 | -0.0127 | 0.9874 |
| `row_grouped_g8` | 0.1747 | 0.9978 | 0.8819 | +0.0027 | 1.0027 |
| `scale_row_g4` | 0.1122 | 0.9988 | 0.9019 | -0.0141 | 0.9860 |
| `scale_row_g8` | 0.1745 | 0.9978 | 0.8819 | +0.0035 | 1.0035 |

Interpretation: `global` INT4 is the expected failure row, while row grouping makes
TinyLlama essentially loss-neutral on this bounded 256-record validation subset.
Group size 4 is stronger than group size 8 on logit MSE/top-5, and scaling is
neutral-to-slightly positive at g4. This is a project logit-only result; do not
compare it against bitsandbytes on weight/activation reconstruction fields.

Per-method telemetry rerun at commit `049d42a`: the same five project rows were
rerun on the same 256-record resource. Whole-job elapsed was 1208.7s runner /
23m43s wall. Isolated method timings were `global` 79.527s, `row_grouped_g4`
34.542s, `row_grouped_g8` 30.810s, `scale_row_g4` 38.282s, and `scale_row_g8`
34.937s. All rows reported the same peak CUDA value, 2273.896 MB allocated /
2658 MB reserved, so treat this as peak harness telemetry rather than sustained
per-method memory. This rerun predates the later throughput/artifact-size
columns, which are now available for future runs.

Completed external-baseline comparison: bitsandbytes NF4 float16 on the same
256-record WikiText-2 resource produced logit MSE 0.253299, top-5 0.857917,
loss delta +0.023453, and PPL ratio 1.023730. Runtime was 231.4s runner elapsed
and 6m17s command wall, with peak CUDA allocated 2274 MB. On this bounded
TinyLlama subset, the best project row (`scale_row_g4`) is higher quality
(logit MSE 0.112199, top-5 0.901881, PPL ratio 0.986014), while bnb is faster
because it evaluates one external method rather than the five-row project
matrix.

bnb telemetry rerun at commit `d8c7d09`: the same 256-record NF4 float16 row
produced logit MSE 0.253722, top-5 0.857737, loss delta +0.023356, PPL ratio
1.023631, isolated method runtime 24.577s, throughput 1354.168 tokens/s,
0.738 ms/token, and peak CUDA 962.886 MB allocated / 1322 MB reserved. Whole
runner elapsed was 191.5s and command wall was 6m24s on RTX 4090. The primary
comparison is now: project `scale_row_g4` is better quality, while bnb NF4 is
faster and lower-memory in method-level telemetry.

Milestone 4 roadmap from this checkpoint:

Successor-model rule: a larger-than-TinyLlama model is not accepted as the main
Milestone 4 successor unless project `scale_row_g4`, bitsandbytes NF4, AWQ, and
GPTQ all have a viable checkpoint/runtime plan and pass smoke on the same
evaluation resource. If one path fails, record the model as a partial probe or
backend detour and keep the details in the lab book/RunPod ledger rather than
the research draft.

| Step | Goal | Repo implementation estimate | Run estimate | Notes |
| --- | --- | ---: | ---: | --- |
| 4A | TinyLlama AWQ external baseline | Complete | 238.2s runner after setup | Added optional pre-quantized AWQ checkpoint runner and completed the 256-record RTX 4090 run. First Pod pass required gptqmodel/ninja/helper dependency setup and Marlin JIT compilation. |
| 4B | TinyLlama GPTQ external baseline | Complete | 262.2s runner after setup | Added optional pre-quantized GPTQ checkpoint runner and completed the 256-record RTX 4090 run. Shared the dependency stack established for AWQ. |
| 4C | Distill TinyLlama external-baseline comparison | Complete | no GPU run | Research draft now contains the distilled project/bnb/AWQ/GPTQ comparison; run-history details live in lab book and RunPod ledger. |
| 4D | Try Qwen2.5-3B as first scale-up target | Complete/blocked for external comparison | no further Qwen run approved | Qwen reference cache and project one-layer `scale_row_g4` smoke passed, but AWQ/GPTQ smokes failed with exit 132 after selecting Marlin-family kernels. Keep this as bookkeeping/backend-compatibility evidence, not a research-draft result. |
| 4E | Probe OPT-2.7B as a partial scale-up target | Complete/partial | 75.7s cache runner, 37.4s project smoke runner, 54.3s bnb smoke runner | Project `scale_row_g4` one-layer smoke and bitsandbytes NF4 one-record smoke passed on RTX 4090. OPT is not the main full-comparison successor unless AWQ/GPTQ support is validated. |
| 4F | Select full-comparison larger-model candidate | Complete | no GPU run | Recommend `mistralai/Mistral-7B-Instruct-v0.2` because it is Apache-2.0, widely used, LLaMA/Mistral-family, public Transformers-compatible, and has established AWQ/GPTQ checkpoints via TheBloke. |
| 4G | Mistral-7B local prep and four-path smoke plan | Complete | no GPU run | Added Mistral project smoke/focused presets, tests, and command-safe cache/project/bnb/AWQ/GPTQ smoke commands. Local runner tests and preset listing passed. |
| 4H | Mistral-7B smoke/readiness segment | Pending approval | likely 45-120 min wall for cache plus project/bnb/AWQ/GPTQ smokes | Promote to full benchmark only if project `scale_row_g4`, bnb NF4, AWQ, and GPTQ all pass smoke. |
| 4I | Mistral-7B full 256-record comparison | Pending smoke success | likely several hours wall; refresh estimate from smoke telemetry | Run original/reference, project `scale_row_g4`, bnb NF4, AWQ, and GPTQ on the same 256-record resource, then distill only the comparison knowledge into the research draft. |

Fresh resume roadmap:

1. Start from `docs/runpod/mistral_7b_plan.md`.
2. Confirm the Mistral local prep commit is present and the repo is clean.
3. Estimate or refresh RTX 4090 wall time, cost, and VRAM risk, then ask the user for
   explicit approval before any Pod command.
4. Run only the smoke/readiness segment first. Promote to full 256-record
   benchmarking only if all planned paths pass.
5. After any GPU segment, update the RunPod ledger/dashboard, lab book,
   project summary, README, and, only where scientifically distilled,
   `docs/research_draft.md`.

Run estimates are intentionally ranges. They use the current timing table:
TinyLlama bnb NF4 needed 191.5s runner / 6m24s wall on RTX 4090, while the
project five-row TinyLlama telemetry run needed 1208.7s runner / 23m43s wall on
RTX 4000 Ada. AWQ/GPTQ first runs should budget extra setup time for optional
dependencies, model-format surprises, and one-time Marlin JIT compilation.
After the dependency stack is warm on the same Pod, AWQ/GPTQ 256-record runner
times were about 4.0-4.4 minutes. Before each GPU segment, follow the
RunPod GPU value rule: estimate wall time, cost, VRAM headroom, output paths,
target commit, and whether the estimate is job-level or method-level, then wait
for user approval.

Current handover state after AWQ/GPTQ integration:

1. TinyLlama external baselines are complete for bitsandbytes NF4, AWQ, and
   GPTQ on the tracked 256-record WikiText-2 raw validation resource.
2. The research draft has been distilled: section 19 keeps the method metrics,
   the project/bnb/AWQ/GPTQ comparison, and the concise runtime/memory caveat.
   Whole-job and dependency-resolution details live in the lab book and RunPod
   ledger.
3. The latest TinyLlama conclusion is: project `scale_row_g4` remains the
   strongest quality row; bnb NF4 is the fastest external method loop; AWQ is
   close to bnb on logit MSE; GPTQ has the best external PPL ratio but weaker
   logit MSE/top-5 overlap.
4. Qwen smoke/readiness result: reference cache prep and project one-layer
   `scale_row_g4` smoke passed, but AWQ/GPTQ external smokes failed at
   Marlin-family backend selection with exit 132. Keep this out of the research
   draft except as a very brief backend limitation if needed.
5. Next Milestone 4 step: prepare the Mistral-7B full-comparison smoke plan.
   OPT is useful partial evidence, but it is not the main successor until
   AWQ/GPTQ are included.
6. OPT smoke/readiness result: reference cache prep, project one-layer
   `scale_row_g4` smoke, and bitsandbytes NF4 one-record smoke all passed on
   RTX 4090.
7. Before the next GPU segment, estimate runtime/cost from the benchmark timing
   table, ask for approval, run `tools/runpod_bootstrap.sh` on any new or
   migrated Pod, and launch long jobs inside detached `tmux`.
8. Continue updating RunPod ledger, lab book, research draft, README, and
   project summary after each GPU segment.
9. The Qwen 3B RunPod command plan remains in `docs/runpod/qwen2_5_3b_plan.md`
   as historical/backend-debug context.
10. The OPT RunPod command plan remains in `docs/runpod/opt_2_7b_plan.md` as
    partial-probe context.
11. Resume from the Mistral-7B smoke approval gate. RunPod is not needed until
    the user approves the 45-120 minute RTX 4090 smoke/readiness estimate and
    provides fresh Pod details.

Stale-state audit on 2026-07-06 00:31:51 BST confirmed the TinyLlama
AWQ/GPTQ data are represented in the research draft, README, project summary,
RunPod usage dashboard/ledger, lab book, operations notes, and implementation
time log. The only provenance gap is exact package-install substep timing during
AWQ/GPTQ setup; available smoke and benchmark tmux start timestamps are now in
the RunPod ledger and lab book.

Regression and artifact acceptance checks:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/rotation_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/sweep_experiment.py
```

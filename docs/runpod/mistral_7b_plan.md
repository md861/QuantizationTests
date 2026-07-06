# Mistral-7B Full-Comparison Candidate Plan

This file is commit-safe operational bookkeeping for the recommended
larger-than-TinyLlama Milestone 4 successor candidate. Do not add raw SSH hosts,
ports, usernames, Pod IDs, private key paths, API keys, account identifiers, or
one-off connection strings.

## Successor Gate

This model can be promoted to the main post-TinyLlama comparison only if all of
the following paths pass smoke on the same evaluation resource:

- Original/reference logits and loss from `mistralai/Mistral-7B-Instruct-v0.2`
- Project `scale_row_g4`
- bitsandbytes NF4 `float16`
- AWQ via `MaziyarPanahi/Mistral-7B-Instruct-v0.2-AWQ`
- GPTQ via `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`

If any path fails, record the failure as a partial probe/backend note in the lab
book and RunPod ledger. Do not present it as a completed research comparison.

## Candidate Rationale

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- License: Apache-2.0
- Architecture family: Mistral/LLaMA-style decoder-only model with standard
  attention projection modules expected to be compatible with the project
  layer-targeting pattern.
- External baseline availability: public AWQ and GPTQ checkpoints exist. The
  successful Mistral AWQ path uses
  `MaziyarPanahi/Mistral-7B-Instruct-v0.2-AWQ`; the initial TheBloke AWQ smoke
  failed in the current AutoAWQ/Triton stack and is kept as an operational
  backend note rather than a research result.
- Main risk: 7B is a larger jump than the earlier 3B probes, so RTX 4090 VRAM
  and wall-time must be checked with smoke/readiness before any 256-record run.

## Required Approval Point

RunPod is first needed after local preset/command prep is committed and pushed.
Before running any command on a Pod, report the target commit, GPU class/hourly
rate, expected runtime/cost, output paths, and whether the estimate is for
smoke or full 256-record benchmarking. Wait for explicit user approval.

## Next Local Prep

Local prep is complete:

1. Added `mistral-7b-v0.2-int4-smoke`, targeting
   `model.layers.0.self_attn.q_proj`.
2. Added `mistral-7b-v0.2-int4-scale-row-g4`, a focused all-layer project
   preset for `--logit-only --logit-methods scale_row_g4`.
3. Prepared cache/project/bnb/AWQ/GPTQ smoke commands using the tracked
   `docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt`
   resource.
4. Local config checks passed with
   `.venv/bin/python -m pytest tests/test_run_transformer_benchmark.py` and
   `.venv/bin/python -m experiments.run_transformer_benchmark --list-presets`.

Do not launch GPU work until the user approves the refreshed runtime/cost
estimate.

## Completed Smoke/Full-Run Outcome

Mistral-7B passed the successor gate on 2026-07-06 on an RTX 4090 Pod at commit
`41a5945` after local half-precision reference loading and a scratch local venv
for the external-baseline stack.

Smoke/readiness results:

- Base/reference cache prep passed: runner `39.1s`.
- Project one-layer `scale_row_g4` smoke passed: runner `40.6s`.
- bitsandbytes NF4 smoke passed: runner `27.4s`.
- GPTQ smoke passed with `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`: runner
  `125.7s`.
- AWQ smoke passed with `MaziyarPanahi/Mistral-7B-Instruct-v0.2-AWQ`: runner
  `97.4s`. The initial TheBloke AWQ checkpoint failed in this stack with an
  AutoAWQ/Triton GEMM issue.

Full 256-record results:

- Project `scale_row_g4`: runner `1183.713s`, isolated method loop `816.141s`,
  logit MSE `0.042644`, top-5 overlap `0.927161`, PPL ratio `1.003481`, peak
  CUDA `14057.727 MB` allocated / `14164 MB` reserved.
- bitsandbytes NF4: runner `137.108s`, isolated method loop `22.772s`, logit
  MSE `0.102315`, top-5 overlap `0.893112`, PPL ratio `1.023186`, peak CUDA
  `4630.234 MB` allocated / `4724 MB` reserved.
- GPTQ: runner `141.423s`, isolated method loop `29.500s`, logit MSE
  `0.129736`, top-5 overlap `0.881939`, PPL ratio `1.016313`, peak CUDA
  `4187.336 MB` allocated / `4256 MB` reserved.
- AWQ: runner `153.382s`, isolated method loop `30.876s`, logit MSE
  `0.107433`, top-5 overlap `0.891707`, PPL ratio `1.022042`, peak CUDA
  `4203.726 MB` allocated / `4234 MB` reserved.

The current Mistral row is complete. Keep the commands below as reproducibility
templates, but do not treat them as the next active step.

## Historical Smoke Estimate

For an RTX 4090-class Pod, budget about 45-120 minutes wall time for the first
Mistral-7B smoke/readiness segment:

- Reference/cache prep: about 10-30 minutes, depending on network/cache state.
- Project one-layer `scale_row_g4` smoke: about 5-15 minutes.
- bitsandbytes NF4 one-record smoke: about 5-15 minutes.
- AWQ one-record smoke: about 10-30 minutes if extra loader/JIT work appears.
- GPTQ one-record smoke: about 10-30 minutes if extra loader/JIT work appears.

At $0.69/hr, this is roughly $0.52-$1.38 of compute before storage/cache
overhead. This estimate is intentionally conservative because Mistral-7B is a
larger jump than the Qwen/OPT probes and may expose backend or VRAM surprises.

## Smoke/Readiness Commands

Run these only after local prep is committed, pushed, estimated, and approved.
Use a fresh, bootstrapped, synced Pod checkout at the approved commit.

Reference/cache prep:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark mistral-7b-v0.2-int4-smoke --download-only --device cuda --results-dir results/transformer_mistral_7b_v0_2_download
```

Project `scale_row_g4` one-layer smoke from local cache:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark mistral-7b-v0.2-int4-smoke --device cuda --local-files-only --logit-only --logit-methods scale_row_g4
```

bitsandbytes NF4 one-record smoke:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.bitsandbytes_baseline --model-name mistralai/Mistral-7B-Instruct-v0.2 --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 1 --results-dir results/bitsandbytes_mistral_7b_v0_2_nf4_smoke --local-files-only --device cuda --compute-dtype float16
```

AWQ one-record smoke. On the first Mistral run, omit `--local-files-only` so
the AWQ checkpoint can populate `/workspace/hf_cache`; add it only for repeat
runs after the checkpoint is confirmed cached. Prefer the MaziyarPanahi
checkpoint below for this repo state.

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.awq_baseline --model-name mistralai/Mistral-7B-Instruct-v0.2 --awq-model-name MaziyarPanahi/Mistral-7B-Instruct-v0.2-AWQ --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 1 --results-dir results/awq_mistral_7b_v0_2_smoke --device cuda --torch-dtype float16
```

GPTQ one-record smoke. On the first Mistral run, omit `--local-files-only` so
the GPTQ checkpoint can populate `/workspace/hf_cache`; add it only for repeat
runs after the checkpoint is confirmed cached.

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.gptq_baseline --model-name mistralai/Mistral-7B-Instruct-v0.2 --gptq-model-name TheBloke/Mistral-7B-Instruct-v0.2-GPTQ --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 1 --results-dir results/gptq_mistral_7b_v0_2_smoke --device cuda --torch-dtype float16
```

## Full 256-Record Commands

Do not launch these until every smoke path above passes and the user approves
the refreshed full-run estimate.

Project focused comparison:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark mistral-7b-v0.2-int4-scale-row-g4 --device cuda --local-files-only --logit-only --logit-methods scale_row_g4
```

bitsandbytes NF4:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.bitsandbytes_baseline --model-name mistralai/Mistral-7B-Instruct-v0.2 --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 256 --results-dir results/bitsandbytes_mistral_7b_v0_2_nf4_256 --local-files-only --device cuda --compute-dtype float16
```

AWQ:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.awq_baseline --model-name mistralai/Mistral-7B-Instruct-v0.2 --awq-model-name MaziyarPanahi/Mistral-7B-Instruct-v0.2-AWQ --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 256 --results-dir results/awq_mistral_7b_v0_2_256 --local-files-only --device cuda --torch-dtype float16
```

GPTQ:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.gptq_baseline --model-name mistralai/Mistral-7B-Instruct-v0.2 --gptq-model-name TheBloke/Mistral-7B-Instruct-v0.2-GPTQ --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 256 --results-dir results/gptq_mistral_7b_v0_2_256 --local-files-only --device cuda --torch-dtype float16
```

## Resume Checklist

1. Confirm the repo is clean and current with `main`.
2. Confirm Mistral local prep commit is present.
3. Produce or refresh the RunPod estimate for cache plus five smoke commands:
   reference/cache, project, bitsandbytes NF4, AWQ, and GPTQ.
4. Wait for user approval and fresh Pod details.
5. Run smoke in detached `tmux`, pull artifacts, and update bookkeeping.

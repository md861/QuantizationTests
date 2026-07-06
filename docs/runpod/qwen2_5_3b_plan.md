# Qwen2.5-3B Run Plan

This file is commit-safe operational bookkeeping for the next Milestone 4 GPU
segment. Do not add raw SSH hosts, ports, usernames, Pod IDs, private key
paths, API keys, or one-off connection strings.

## Target

- Reference model: `Qwen/Qwen2.5-3B-Instruct`
- Project method: `scale_row_g4`
- External baseline order: AWQ first, then GPTQ if AWQ smoke is stable
- Evaluation resource:
  `docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt`

## Current Estimates

- Local implementation prep: about 1-2 active hours.
- First RunPod smoke/cache/readiness segment: about 30-90 minutes wall time.
- Focused 256-record comparison after smoke passes: about 1.25-3.5 hours wall
  time for original + project `scale_row_g4` + AWQ/GPTQ.
- Rough RTX 4090 first-pass compute budget: about $1.50-$4.00 plus small
  storage/cache overhead, assuming no prolonged dependency debugging.

## Required Approval Point

RunPod is first needed after local prep is committed and pushed. Before running
any command below on a Pod, report the current target commit, GPU class/hourly
rate, expected runtime/cost, output paths, and whether the estimate is for a
smoke or full 256-record run. Wait for explicit user approval.

## Smoke/Readiness Commands

Run cache prep first on a freshly bootstrapped and synced Pod:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark qwen2.5-3b-int4-smoke --download-only --device cuda --results-dir results/transformer_qwen2_5_3b_download
```

Then run the project smoke from local cache:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark qwen2.5-3b-int4-smoke --device cuda --local-files-only --logit-only --logit-methods scale_row_g4
```

Run AWQ one-record smoke:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.awq_baseline --model-name Qwen/Qwen2.5-3B-Instruct --awq-model-name Qwen/Qwen2.5-3B-Instruct-AWQ --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 1 --results-dir results/awq_qwen2_5_3b_smoke --local-files-only --device cuda --torch-dtype float16
```

Run GPTQ one-record smoke only after AWQ is stable:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.gptq_baseline --model-name Qwen/Qwen2.5-3B-Instruct --gptq-model-name Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 1 --results-dir results/gptq_qwen2_5_3b_smoke --local-files-only --device cuda --torch-dtype float16
```

Add `--trust-remote-code` only if the loader requires it and record that fact in
the lab book and RunPod ledger.

## Focused 256-Record Commands

Do not launch these until the smoke/readiness segment passes and the user
approves the updated runtime/cost estimate.

Project focused comparison:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark qwen2.5-3b-int4-scale-row-g4 --device cuda --local-files-only --logit-only --logit-methods scale_row_g4
```

AWQ 256-record baseline:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.awq_baseline --model-name Qwen/Qwen2.5-3B-Instruct --awq-model-name Qwen/Qwen2.5-3B-Instruct-AWQ --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 256 --results-dir results/awq_qwen2_5_3b_256 --local-files-only --device cuda --torch-dtype float16
```

GPTQ 256-record baseline:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.gptq_baseline --model-name Qwen/Qwen2.5-3B-Instruct --gptq-model-name Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 256 --results-dir results/gptq_qwen2_5_3b_256 --local-files-only --device cuda --torch-dtype float16
```

## Expected Artifacts

- `results/transformer_qwen2_5_3b_int4_smoke/benchmark_metadata.json`
- `results/transformer_qwen2_5_3b_int4_scale_row_g4/benchmark_metadata.json`
- `results/transformer_qwen2_5_3b_int4_scale_row_g4/transformer_logit_metrics.csv`
- `results/awq_qwen2_5_3b_smoke/awq_metadata.json`
- `results/awq_qwen2_5_3b_256/awq_logit_metrics.csv`
- `results/gptq_qwen2_5_3b_smoke/gptq_metadata.json`
- `results/gptq_qwen2_5_3b_256/gptq_logit_metrics.csv`

After every Pod segment, update the RunPod usage ledger/dashboard, benchmark
timing table, lab book, and implementation time log before handover.

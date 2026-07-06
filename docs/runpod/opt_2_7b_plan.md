# OPT-2.7B Run Plan

This file is commit-safe operational bookkeeping for the next Milestone 4 GPU
segment. Do not add raw SSH hosts, ports, usernames, Pod IDs, private key
paths, API keys, account identifiers, or one-off connection strings.

Status after roadmap clarification: this is a partial-probe plan, not the main
larger-model successor plan. The main post-TinyLlama Milestone 4 successor must
support project `scale_row_g4`, bitsandbytes NF4, AWQ, and GPTQ. OPT-2.7B has
only passed project and bitsandbytes smokes so far; do not promote it to a full
research comparison unless AWQ/GPTQ support is explicitly added and smoke
passes.

## Target

- Reference model: `facebook/opt-2.7b`
- Project method: `scale_row_g4`
- First external baseline: bitsandbytes NF4 `float16`
- Evaluation resource:
  `docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt`

## Current Estimates

- Local implementation prep: about 45-90 active minutes.
- First RunPod smoke/cache/readiness segment: about 30-90 minutes wall time.
- Focused 256-record comparison after smoke passes: about 1-3 hours wall time
  for original/project `scale_row_g4` plus bitsandbytes NF4.
- Rough RTX 4090 first-pass compute budget: about $1.00-$3.50 plus small
  storage/cache overhead, assuming no prolonged loader/debug issue.

## Required Approval Point

RunPod is first needed after local prep is committed and pushed. Before running
any command below on a Pod, report the current target commit, GPU class/hourly
rate, expected runtime/cost, output paths, and whether the estimate is for a
smoke or full 256-record run. Wait for explicit user approval.

## Smoke/Readiness Commands

Run cache prep first on a freshly bootstrapped and synced Pod:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark opt-2.7b-int4-smoke --download-only --device cuda --results-dir results/transformer_opt_2_7b_download
```

Then run the project smoke from local cache:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark opt-2.7b-int4-smoke --device cuda --local-files-only --logit-only --logit-methods scale_row_g4
```

Run bitsandbytes NF4 one-record smoke:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.bitsandbytes_baseline --model-name facebook/opt-2.7b --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 1 --results-dir results/bitsandbytes_opt_2_7b_nf4_smoke --local-files-only --device cuda --compute-dtype float16
```

## Focused 256-Record Commands

Do not launch these until the smoke/readiness segment passes and the user
approves the updated runtime/cost estimate.

Project focused comparison:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.run_transformer_benchmark opt-2.7b-int4-scale-row-g4 --device cuda --local-files-only --logit-only --logit-methods scale_row_g4
```

bitsandbytes NF4 256-record baseline:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m experiments.bitsandbytes_baseline --model-name facebook/opt-2.7b --eval-text-file docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt --max-eval-texts 256 --results-dir results/bitsandbytes_opt_2_7b_nf4_256 --local-files-only --device cuda --compute-dtype float16
```

## Expected Artifacts

- `results/transformer_opt_2_7b_download/benchmark_metadata.json`
- `results/transformer_opt_2_7b_int4_smoke/benchmark_metadata.json`
- `results/transformer_opt_2_7b_int4_smoke/transformer_logit_metrics.csv`
- `results/transformer_opt_2_7b_int4_scale_row_g4/benchmark_metadata.json`
- `results/transformer_opt_2_7b_int4_scale_row_g4/transformer_logit_metrics.csv`
- `results/bitsandbytes_opt_2_7b_nf4_smoke/bitsandbytes_metadata.json`
- `results/bitsandbytes_opt_2_7b_nf4_256/bitsandbytes_logit_metrics.csv`

After every Pod segment, update the RunPod usage ledger/dashboard, benchmark
timing table, lab book, and implementation time log before handover.

# RunPod Usage Ledger

This ledger tracks where RunPod time is spent. It complements, but does not
replace, the benchmark timing table in `project_summary.md`.

Keep this file commit-safe: do not record raw SSH hosts, ports, usernames,
private key paths, Pod IDs, API keys, account identifiers, or one-off connection
strings.

## Purpose

Track two different time costs separately:

1. Benchmark compute time: time spent running actual GPU/benchmark workloads.
   These runs should also be recorded in the benchmark timing table in
   `project_summary.md` because they inform future runtime prediction and
   research comparisons.
2. Pod operations time: setup, dependency installation, downloads, cache work,
   environment verification, debugging, idle windows, and cleanup. These entries
   explain where RunPod credits go outside benchmark computation.

For cost reasoning, record elapsed wall-clock time even when GPU memory/use is
low, because a running GPU Pod can still consume credits. If the RunPod account
is shared or has other active resources, account-level billing may be
contaminated by unrelated usage. Prefer per-Pod billing line items when
available; otherwise estimate this project's Pod cost as:

```text
estimated Pod cost = elapsed_hours * recorded hourly Pod rate
```

Label rate-based values as estimates.

## Budget Guardrail

Project budget ceiling: about GBP 200 total RunPod benchmark spend. Treat this as a hard planning limit for benchmark work. Record estimated compute and storage costs after every Pod segment, keep the dashboard current, and pause before any GPU-class change or benchmark expansion that could materially alter expected spend.

GPU selection rule: pick the cheapest GPU that can answer the current research
question within the user's time window. Before changing GPU class, estimate cost
per useful benchmark from prior timing rows, expected wall time, method
throughput when available, VRAM needs, hourly rate, and setup/model-load
overhead. Record the estimate and user approval in the relevant lab-book or
handover entry.

## Entry Categories

Use one of these categories in the ledger:

- `setup`: first-time Pod setup, repo clone, venv creation, dependency installs
- `verification`: tests, import checks, CUDA checks, smoke readiness checks
- `download_cache`: model or dataset download/cache population
- `benchmark_smoke`: single-layer or small-subset benchmark runs
- `benchmark_full`: full benchmark runs that should also enter
  `project_summary.md` benchmark timings
- `debug`: GPU-only debugging or failed-run investigation
- `idle_admin`: intentional short idle window between queued GPU jobs
- `cleanup_sync`: pulling artifacts, git sync, stop-prep checks

## Required Fields

For every RunPod session segment, record:

- Date
- Category
- Start time and end time, or elapsed time if exact clock times were not captured
- GPU class and VRAM
- Commit hash, if tied to repo work
- Whether GPU compute was meaningfully used
- Log/output path, if available
- Hourly Pod rate used for estimates, if exact per-Pod billing is unavailable
- Estimated credits or USD spent, preferably exact per-Pod billing; otherwise
  `elapsed_hours * hourly_rate`, labeled as an estimate
- Notes and next action

## Current Known Entries

| Date | Category | Task | Elapsed | Compute Rate | Est. Compute Cost | GPU | Commit | GPU used? | Evidence / output | Notes |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| 2026-07-02 | setup | SSH alias, repo clone, tmux install, initial dependency attempts | timing not fully captured | $0.26/hr | not captured | RTX 4000 Ada, ~20 GB | pre-093d41c | No meaningful GPU use observed | setup shell history and lab-book notes | Includes the discarded mixed `--system-site-packages` venv path. Future sessions should avoid this pattern. |
| 2026-07-02 | setup | Clean self-contained `.venv` creation and package install | 35.0 min | $0.26/hr | $0.15 | RTX 4000 Ada, ~20 GB | pre-093d41c | No, GPU idle at about 2 MiB | `/workspace/pq_clean_venv.log`, `/workspace/pq_clean_venv.exit` | Network-volume small-file writes made installation slow. |
| 2026-07-02 | verification | CUDA/import checks and GPT-2/Transformers import verification | timing partially captured | $0.26/hr | not captured | RTX 4000 Ada, ~20 GB | pre-093d41c | No meaningful GPU use observed | shell output, lab-book notes | First Transformers/GPT-2 imports were slow on network-volume venv but completed. |
| 2026-07-02 | verification | Full repo test suite on Pod | 5.8 min | $0.26/hr | $0.03 | RTX 4000 Ada, ~20 GB | pre-093d41c | No benchmark GPU use; GPU idle after run | `/workspace/pq_pytest_clean.log` | `212 passed, 1 warning`; environment deployment-ready. |
| 2026-07-02 | idle_admin | Telemetry uptime not covered by logged setup/verification segments | 264.2 min | $0.26/hr | $1.15 | RTX 4000 Ada, ~20 GB | pre-ea17632 through ea17632 | No benchmark GPU use known | RunPod telemetry showed 5h 5m total uptime; logged work time was ~40.8 min | Treat as unlogged admin/idle overhead. Stop Pod outside short queued benchmark windows. |
| 2026-07-02 | cleanup_sync | Sync RunPod checkout after docs commits | timing not captured | $0.26/hr | not captured | RTX 4000 Ada, ~20 GB | 093d41c, 40a5de3, 9c94a51, 07b0530, c7d5d8a, 95a1461, ea17632 | No | git pull output | Documentation sync only. |
| 2026-07-02 | storage | Container + network volume storage | 5h 5m telemetry window | n/a | $0.07 | n/a | n/a | n/a | RunPod pricing docs and Pod details panel | Storage is accumulated once in the dashboard header, not distributed across compute rows. |
| 2026-07-02 | debug | Fixed RunPod tmux wrapper after `/usr/bin/time` was unavailable and PowerShell expanded shell snippets too early | 1.5 min | $0.26/hr | $0.01 | RTX 4000 Ada, ~20 GB | c15113a | No benchmark GPU use | `/workspace/pq_tinyllama_download.log` first failed wrapper, shell output | Command-wrapper overhead only; no model benchmark ran in this segment. |
| 2026-07-02 | download_cache | TinyLlama 1.1B Hugging Face download/cache prep | 29.2s | $0.26/hr | $0.00 | RTX 4000 Ada, ~20 GB | c15113a | No, GPU idle at about 2 MiB | `/workspace/pq_tinyllama_download.log`, `results/transformer_tinyllama_1_1b_int4_smoke/benchmark_metadata.json` | Exit 0; cache about 2.1 GB. Unauthenticated HF warning noted; public model download succeeded. |
| 2026-07-02 | benchmark_smoke | TinyLlama 1.1B single-layer INT4 smoke, `model.layers.0.self_attn.q_proj` | 228.3s | $0.26/hr | $0.02 | RTX 4000 Ada, ~20 GB | c15113a | Yes | `/workspace/pq_tinyllama_smoke.log`, `results/transformer_tinyllama_1_1b_int4_smoke/benchmark_metadata.json` | Exit 0; counts weight=9 activation=9 logit=9; peak CUDA allocated 2124 MB, reserved 2224 MB; adequate smoke headroom. |
| 2026-07-03 | setup | Replacement Pod verification, repo sync, optional bnb stack install, and tmux install | ~1.0 min install + admin checks | $0.26/hr | ~$0.00 | RTX 4000 Ada, 20475 MiB | 4b5d5d0 | Minimal GPU use only | shell output; local-only inventory `.local/runpod_inventory.md` | Replacement Pod attached the preserved network volume; observed GPU NVIDIA RTX 4000 Ada Generation, driver 550.127.05; installed accelerate 1.14.0, bitsandbytes 0.49.2, psutil 7.2.2, and tmux. |
| 2026-07-03 | debug | Failed bnb NF4 smoke with `--local-files-only` before persistent HF cache was populated | timing not captured | $0.26/hr | not captured | RTX 4000 Ada, 20475 MiB | 4b5d5d0 | No meaningful GPU use observed | `/workspace/pq_bnb_smoke.log`, `/workspace/pq_bnb_smoke.exit` | Failed because TinyLlama files were not in cache and outgoing traffic was disabled. Lesson: use `/workspace/hf_cache` and run online cache prep before `--local-files-only`. |
| 2026-07-03 | download_cache | TinyLlama 1.1B Hugging Face cache prep into persistent `/workspace/hf_cache` | 27.6s runner elapsed; 182.6s wall command | $0.26/hr | ~$0.01 | RTX 4000 Ada, 20475 MiB | 4b5d5d0 | No, GPU idle during download/cache prep | `results/transformer_tinyllama_1_1b_int4_smoke/benchmark_metadata.json`, `/workspace/hf_cache` | Exit 0; cache now about 2.1 GB under network volume. Future runs should export `HF_HOME=/workspace/hf_cache` and `HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub`. |
| 2026-07-03 | benchmark_smoke | TinyLlama 1.1B bitsandbytes NF4 external baseline smoke, 1 WikiText-2 record | 44.2s runner elapsed; 262.2s wall command | $0.26/hr | ~$0.02 | RTX 4000 Ada, 20475 MiB | 4b5d5d0 | Yes | `results/bitsandbytes_tinyllama_nf4_smoke/bitsandbytes_metadata.json`, `results/bitsandbytes_tinyllama_nf4_smoke/bitsandbytes_logit_metrics.csv` | Exit 0; method `external_bitsandbytes_nf4_float16`; logit MSE 0.311986; top-5 overlap 0.865; loss delta 0.044535; PPL ratio 1.04554; peak CUDA allocated 2173.082 MB, reserved 2268 MB. |
| 2026-07-03 | benchmark_smoke | Aborted full-harness TinyLlama project matrix smoke, 1 WikiText-2 record | ~60 min before manual stop | $0.26/hr | ~$0.26 | RTX 4000 Ada, 20475 MiB | 9473a51 | Partial GPU/CPU use | `results/transformer_tinyllama_1_1b_int4_matrix_smoke/` | Stopped after only 95 weight and 95 activation rows; showed full Milestone 3 harness was too slow for the first Milestone 4 shared comparison. |
| 2026-07-03 | benchmark_smoke | Aborted vectorized full-harness TinyLlama project matrix smoke, 1 WikiText-2 record | ~60 min before manual stop | $0.26/hr | ~$0.26 | RTX 4000 Ada, 20475 MiB | fdf14d0 | Partial GPU/CPU use | `results/transformer_tinyllama_1_1b_int4_matrix_smoke_v2/` | Vectorization improved throughput but full harness still spent too much time on weight/activation bookkeeping; led to logit-only path. |
| 2026-07-03 | benchmark_smoke | TinyLlama project INT4 logit-only matrix smoke, 1 WikiText-2 record | 264.1s runner elapsed; 7m8.5s wall command | $0.26/hr | ~$0.03 | RTX 4000 Ada, 20475 MiB | ceddbaf | Yes | `results/transformer_tinyllama_1_1b_int4_matrix_logit_smoke/benchmark_metadata.json` | Exit 0; counts weight=0 activation=0 logit=5; peak CUDA allocated not recorded here; established logit-only path feasibility. |
| 2026-07-03 | benchmark | TinyLlama project INT4 logit-only matrix, 256 WikiText-2 records | 1004.4s runner elapsed; 19m20s wall command | $0.26/hr | ~$0.08 | RTX 4000 Ada, 20475 MiB | ceddbaf | Yes | `results/transformer_tinyllama_1_1b_int4_matrix_logit_256/benchmark_metadata.json`, `results/transformer_tinyllama_1_1b_int4_matrix_logit_256/transformer_logit_metrics.csv` | Exit 0; counts weight=0 activation=0 logit=5; peak CUDA allocated 2273.896 MB, reserved 2658 MB; best PPL ratio `scale_row_g4` 0.9860, top-5 0.9019. |
| 2026-07-04 | benchmark | TinyLlama bitsandbytes NF4 external baseline, 256 WikiText-2 records | 231.4s runner elapsed; 6m17s wall command | $0.26/hr | ~$0.03 | RTX 4000 Ada, 20475 MiB | 92b4f5e | Yes | `results/bitsandbytes_tinyllama_nf4_256/bitsandbytes_metadata.json`, `results/bitsandbytes_tinyllama_nf4_256/bitsandbytes_logit_metrics.csv`, `/workspace/pq_bnb_nf4_256.log` | Exit 0; method `external_bitsandbytes_nf4_float16`; logit MSE 0.253299; top-5 overlap 0.857917; loss delta +0.023453; PPL ratio 1.023730; peak CUDA allocated 2273.896 MB, reserved 2680 MB. |
| 2026-07-05 | benchmark | TinyLlama project INT4 logit-only matrix rerun with per-method telemetry, 256 WikiText-2 records | 1208.7s runner elapsed; 23m43s wall command | $0.26/hr | ~$0.10 | RTX 4000 Ada, 20475 MiB | 049d42a | Yes | `results/transformer_tinyllama_1_1b_int4_matrix_logit_256_per_method/benchmark_metadata.json`, `results/transformer_tinyllama_1_1b_int4_matrix_logit_256_per_method/transformer_logit_metrics.csv`, `results/transformer_tinyllama_1_1b_int4_matrix_logit_256_per_method/run_manifest.md` | Exit 0; counts weight=0 activation=0 logit=5; isolated method seconds: global 79.527, row_g4 34.542, row_g8 30.810, scale_g4 38.282, scale_g8 34.937; peak CUDA allocated 2273.896 MB, reserved 2658 MB. |
| 2026-07-05 | setup | Fresh RTX 4090 Pod sync and bootstrap before bnb telemetry rerun | ~0.6 min | $0.69/hr | ~$0.01 | RTX 4090, 24564 MiB | d8c7d09 | Minimal GPU use only | `tools/runpod_bootstrap.sh` output | Synced checkout from `049d42a` to `d8c7d09`; bootstrap installed ephemeral `tmux` and `rsync`; persistent HF cache was present at about 2.1 GB. |
| 2026-07-05 | benchmark | TinyLlama bitsandbytes NF4 external baseline telemetry rerun, 256 WikiText-2 records | 191.5s runner elapsed; 6m24s wall command | $0.69/hr | ~$0.07 | RTX 4090, 24564 MiB | d8c7d09 | Yes | `results/bitsandbytes_tinyllama_nf4_256_telemetry/bitsandbytes_metadata.json`, `results/bitsandbytes_tinyllama_nf4_256_telemetry/bitsandbytes_logit_metrics.csv`, `/workspace/pq_bnb_nf4_256_telemetry.log` | Exit 0; method `external_bitsandbytes_nf4_float16`; logit MSE 0.253722; top-5 overlap 0.857737; loss delta +0.023356; PPL ratio 1.023631; method loop 24.577s; throughput 1354.168 tokens/s; peak CUDA allocated 962.886 MB, reserved 1322 MB. |
| 2026-07-05 | idle_admin | RTX 4090 Pod uptime before AWQ/GPTQ segment not covered by itemized ledger rows | ~371 min residual from 6h18m details-panel uptime after subtracting prior logged RTX 4090 setup/bnb work | $0.69/hr | ~$4.27 | RTX 4090, 24564 MiB | d8c7d09 through 97bc484 | No benchmark use known during residual | RunPod details panel screenshot; prior ledger rows | Treat as unlogged admin/idle overhead. This is an estimate from observed uptime, not a billing export. |
| 2026-07-05 | setup | AWQ/GPTQ dependency resolution, smoke probes, and Marlin JIT readiness on RTX 4090 Pod | ~90 min wall wait/active mix | $0.69/hr | ~$1.04 | RTX 4090, 24564 MiB | 97bc484 | Partial GPU use during smokes/JIT | `/workspace/pq_awq_smoke.log`, `/workspace/pq_gptq_smoke.log`, shell install output | Installed `gptqmodel==7.1.0` without upgrading torch, plus helper packages, `optimum`, `logbar==0.4.3`, and `ninja`; AWQ smoke tmux start 2026-07-05 22:13:38 Europe/London; GPTQ smoke tmux start 2026-07-05 22:39:43 Europe/London; exact package-install substep timestamps were not captured. |
| 2026-07-05 | benchmark | TinyLlama AWQ 4-bit external baseline, 256 WikiText-2 records | 238.2s runner elapsed | $0.69/hr | ~$0.05 | RTX 4090, 24564 MiB | 97bc484 | Yes | `results/awq_tinyllama_1_1b_256/awq_metadata.json`, `results/awq_tinyllama_1_1b_256/awq_logit_metrics.csv`, `/workspace/pq_awq_256.log` | Exit 0; tmux start 2026-07-05 22:24:34 Europe/London; method `external_awq_w4`; logit MSE 0.252777; top-5 overlap 0.854252; loss delta +0.040232; PPL ratio 1.041052; method loop 39.409s; throughput 844.535 tokens/s; peak CUDA allocated 904.183 MB, reserved 1240 MB. |
| 2026-07-05 | benchmark | TinyLlama GPTQ 4-bit external baseline, 256 WikiText-2 records | 262.2s runner elapsed | $0.69/hr | ~$0.05 | RTX 4090, 24564 MiB | 97bc484 | Yes | `results/gptq_tinyllama_1_1b_256/gptq_metadata.json`, `results/gptq_tinyllama_1_1b_256/gptq_logit_metrics.csv`, `/workspace/pq_gptq_256.log` | Exit 0; tmux start 2026-07-05 22:46:05 Europe/London; method `external_gptq_w4`; logit MSE 0.349270; top-5 overlap 0.837882; loss delta +0.021532; PPL ratio 1.021766; method loop 58.086s; throughput 572.980 tokens/s; peak CUDA allocated 903.581 MB, reserved 1242 MB. |
| 2026-07-06 | setup | Migrated RTX 4090 Pod readiness, repo sync, and bootstrap before Qwen smoke | ~3.5 min observed setup window | $0.69/hr | ~$0.04 | RTX 4090, 24564 MiB | 4a85f75 | Minimal GPU use only | `tools/runpod_bootstrap.sh` output; local pulled logs under `.local/runpod_logs/qwen2_5_3b_smoke/` | Pod reachable at 2026-07-06 10:20:58 UTC; synced checkout from `97bc484` to `4a85f75`; bootstrap installed ephemeral `tmux` and `rsync`; torch 2.6.0+cu124 and Transformers 5.12.1 verified. |
| 2026-07-06 | download_cache | Qwen2.5-3B reference cache/readiness, `Qwen/Qwen2.5-3B-Instruct` | 56.5s runner elapsed; 283s wall command | $0.69/hr | ~$0.05 | RTX 4090, 24564 MiB | 4a85f75 | No meaningful GPU use after metadata showed 0 MB peak | `results/transformer_qwen2_5_3b_download/benchmark_metadata.json`, `/workspace/pq_qwen_download.log` | Exit 0; HF cache grew from about 2.1 GB to about 7.9 GB; unauthenticated HF warning observed. |
| 2026-07-06 | benchmark_smoke | Qwen2.5-3B project `scale_row_g4` one-layer smoke, `model.layers.0.self_attn.q_proj` | 73.8s runner elapsed; 306s wall command | $0.69/hr | ~$0.06 | RTX 4090, 24564 MiB | 4a85f75 | Yes | `results/transformer_qwen2_5_3b_int4_smoke/benchmark_metadata.json`, `results/transformer_qwen2_5_3b_int4_smoke/transformer_logit_metrics.csv`, `/workspace/pq_qwen_project_smoke.log` | Exit 0; logit MSE 0.002384; logit cosine 0.999844; top-5 overlap 1.0; loss delta +0.052055; PPL ratio 1.053434; method loop 0.312s; peak CUDA allocated 6010.587 MB, reserved 6172 MB. |
| 2026-07-06 | benchmark_smoke | Qwen2.5-3B AWQ one-record smoke/cache, `Qwen/Qwen2.5-3B-Instruct-AWQ` | failed after 357s wall command | $0.69/hr | ~$0.07 | RTX 4090, 24564 MiB | 4a85f75 | Partial GPU/backend use before crash | `/workspace/pq_qwen_awq_smoke.log`, `/workspace/pq_qwen_awq_smoke.exit` | Exit 132 after selecting `AwqMarlinLinear`; no AWQ metrics or metadata written. Treat as external-backend smoke failure, not a quality result. |
| 2026-07-06 | benchmark_smoke | Qwen2.5-3B GPTQ one-record smoke/cache, `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` | failed after 374s wall command | $0.69/hr | ~$0.07 | RTX 4090, 24564 MiB | 4a85f75 | Partial GPU/backend use before crash | `/workspace/pq_qwen_gptq_smoke.log`, `/workspace/pq_qwen_gptq_smoke.exit` | Exit 132 after selecting `MarlinLinear`; no GPTQ metrics or metadata written. Treat as external-backend smoke failure, not a quality result. |
| 2026-07-06 | debug | Qwen smoke polling, Marlin-family failure inspection, artifact pullback, and local metric extraction | ~15.2 min observed residual in 10:20:58-11:01 UTC Pod window after itemized commands | $0.69/hr | ~$0.17 | RTX 4090, 24564 MiB | 4a85f75 | No benchmark use known during residual | local `.local/runpod_logs/qwen2_5_3b_smoke/`; pulled ignored `results/transformer_qwen2_5_3b_*` artifacts | Used to reconcile observed Pod window. Raw logs/artifacts are local/ignored; distilled facts are recorded in commit-safe docs. |
| 2026-07-06 | setup | Migrated RTX 4090 Pod readiness, repo sync, and bootstrap before OPT smoke | ~5.2 min observed setup window | $0.69/hr | ~$0.06 | RTX 4090, 24564 MiB | a368b8f | Minimal GPU use only | `tools/runpod_bootstrap.sh` output; local pulled logs under `.local/runpod_logs/opt_2_7b_smoke/` | Pod reachable at 2026-07-06 11:42:12 UTC; synced checkout from `4a85f75` to `a368b8f`; bootstrap installed ephemeral `tmux` and `rsync`; torch 2.6.0+cu124, Transformers 5.12.1, and bitsandbytes 0.49.2 verified. |
| 2026-07-06 | download_cache | OPT-2.7B reference cache/readiness, `facebook/opt-2.7b` | 75.7s runner elapsed; 288s wall command | $0.69/hr | ~$0.06 | RTX 4090, 24564 MiB | a368b8f | No meaningful GPU use after metadata showed 0 MB peak | `results/transformer_opt_2_7b_download/benchmark_metadata.json`, `/workspace/pq_opt_download.log` | Exit 0; HF cache grew from about 13 GB to about 23 GB; unauthenticated HF warning observed. |
| 2026-07-06 | benchmark_smoke | OPT-2.7B project `scale_row_g4` one-layer smoke, `model.decoder.layers.0.self_attn.q_proj` | 37.4s runner elapsed; 214s wall command | $0.69/hr | ~$0.04 | RTX 4090, 24564 MiB | a368b8f | Yes | `results/transformer_opt_2_7b_int4_smoke/benchmark_metadata.json`, `results/transformer_opt_2_7b_int4_smoke/transformer_logit_metrics.csv`, `/workspace/pq_opt_project_smoke.log` | Exit 0; logit MSE 0.000161; logit cosine 0.999986; top-5 overlap 1.0; loss delta +0.001195; PPL ratio 1.001196; method loop 0.114s; peak CUDA allocated 5078.676 MB, reserved 5308 MB. |
| 2026-07-06 | benchmark_smoke | OPT-2.7B bitsandbytes NF4 one-record smoke, `facebook/opt-2.7b` | 54.3s runner elapsed; 244s wall command | $0.69/hr | ~$0.05 | RTX 4090, 24564 MiB | a368b8f | Yes | `results/bitsandbytes_opt_2_7b_nf4_smoke/bitsandbytes_metadata.json`, `results/bitsandbytes_opt_2_7b_nf4_smoke/bitsandbytes_logit_metrics.csv`, `/workspace/pq_opt_bnb_smoke.log` | Exit 0; method `external_bitsandbytes_nf4_float16`; logit MSE 0.167516; logit cosine 0.986857; top-5 overlap 0.889744; loss delta +0.058090; PPL ratio 1.059811; method loop 13.332s; throughput 11.702 tokens/s; peak CUDA allocated 1984.248 MB, reserved 2112 MB. |
| 2026-07-06 | debug | OPT smoke polling, artifact pullback, and local metric extraction | ~5.4 min observed residual in 11:42:12-12:05 UTC Pod window after itemized commands | $0.69/hr | ~$0.06 | RTX 4090, 24564 MiB | a368b8f | No benchmark use known during residual | local `.local/runpod_logs/opt_2_7b_smoke/`; pulled ignored `results/transformer_opt_2_7b_*` and `results/bitsandbytes_opt_2_7b_*` artifacts | Used to reconcile observed Pod window. Raw logs/artifacts are local/ignored; distilled facts are recorded in commit-safe docs. |
| 2026-07-06 | setup | Mistral-7B Pod sync, bootstrap, scratch venv, half-precision reference fix, and external-baseline stack compatibility work | ~65 min active/wait mix | $0.69/hr | ~$0.75 | RTX 4090, 24564 MiB | fa2488f through 41a5945 | Partial GPU use during stack smoke attempts | local `.local/runpod_logs/mistral_7b_full_256/`; scratch venv and wrapper notes in lab book | Installed ephemeral `tmux`/`rsync`, used a local scratch venv on container disk, copied the torch 2.6 CUDA stack, added half-precision reference loading, and used a pytree compatibility wrapper for Transformers/torchao/GPTQModel stack differences. |
| 2026-07-06 | benchmark_smoke | Mistral-7B smoke/readiness segment: reference cache, project `scale_row_g4`, bnb NF4, GPTQ, and AWQ | ~25 min observed smoke/backend window | $0.69/hr | ~$0.29 | RTX 4090, 24564 MiB | 41a5945 | Yes | smoke logs under `/workspace/pq_mistral_*`; distilled in `docs/runpod/mistral_7b_plan.md` | All required paths passed smoke after stack fixes. Runner times: reference cache 39.1s, project 40.6s, bnb 27.4s, GPTQ 125.7s, AWQ 97.4s. Initial TheBloke AWQ smoke failed; successful AWQ path used `MaziyarPanahi/Mistral-7B-Instruct-v0.2-AWQ`. |
| 2026-07-06 | benchmark | Mistral-7B project `scale_row_g4`, 256 WikiText-2 records | 1183.7s runner elapsed; 19m49s wall command | $0.69/hr | ~$0.23 | RTX 4090, 24564 MiB | 41a5945 | Yes | `results/transformer_mistral_7b_v0_2_int4_scale_row_g4/benchmark_metadata.json`, `results/transformer_mistral_7b_v0_2_int4_scale_row_g4/transformer_logit_metrics.csv`, `/workspace/pq_mistral_full_project_scale_row_g4.log` | Exit 0; method loop 816.141s; logit MSE 0.042644; top-5 overlap 0.927161; loss delta +0.003475; PPL ratio 1.003481; throughput 39.938 tokens/s; peak CUDA allocated 14057.727 MB, reserved 14164 MB. |
| 2026-07-06 | benchmark | Mistral-7B bitsandbytes NF4 external baseline, 256 WikiText-2 records | 137.1s runner elapsed; 2m23s wall command | $0.69/hr | ~$0.03 | RTX 4090, 24564 MiB | 41a5945 | Yes | `results/bitsandbytes_mistral_7b_v0_2_nf4_256/bitsandbytes_metadata.json`, `results/bitsandbytes_mistral_7b_v0_2_nf4_256/bitsandbytes_logit_metrics.csv`, `/workspace/pq_mistral_full_bnb_nf4.log` | Exit 0; method loop 22.772s; logit MSE 0.102315; top-5 overlap 0.893112; loss delta +0.022921; PPL ratio 1.023186; throughput 1431.369 tokens/s; peak CUDA allocated 4630.234 MB, reserved 4724 MB. |
| 2026-07-06 | benchmark | Mistral-7B GPTQ 4-bit external baseline, 256 WikiText-2 records | 141.4s runner elapsed; 2m27s wall command | $0.69/hr | ~$0.03 | RTX 4090, 24564 MiB | 41a5945 | Yes | `results/gptq_mistral_7b_v0_2_256/gptq_metadata.json`, `results/gptq_mistral_7b_v0_2_256/gptq_logit_metrics.csv`, `/workspace/pq_mistral_full_gptq.log` | Exit 0; checkpoint `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`; method loop 29.500s; logit MSE 0.129736; top-5 overlap 0.881939; loss delta +0.016181; PPL ratio 1.016313; throughput 1104.924 tokens/s; peak CUDA allocated 4187.336 MB, reserved 4256 MB. |
| 2026-07-06 | benchmark | Mistral-7B AWQ 4-bit external baseline, 256 WikiText-2 records | 153.4s runner elapsed; 2m38s wall wrapper rerun | $0.69/hr | ~$0.03 | RTX 4090, 24564 MiB | 41a5945 | Yes | `results/awq_mistral_7b_v0_2_256/awq_metadata.json`, `results/awq_mistral_7b_v0_2_256/awq_logit_metrics.csv`, `/workspace/pq_mistral_awq_256_wrapper.log` | Exit 0 after wrapper rerun; checkpoint `MaziyarPanahi/Mistral-7B-Instruct-v0.2-AWQ`; method loop 30.876s; logit MSE 0.107433; top-5 overlap 0.891707; loss delta +0.021802; PPL ratio 1.022042; throughput 1055.681 tokens/s; peak CUDA allocated 4203.726 MB, reserved 4234 MB. Initial full-script AWQ command exited 1 without metrics because it missed the pytree wrapper. |

## Template

Copy this row for future entries:

```markdown
| YYYY-MM-DD | category | task | elapsed | compute rate | est. compute cost | GPU class / VRAM | commit | GPU used? | evidence / output | notes |
```

## Maintenance Rules

- Update this ledger whenever RunPod is started, used, left running between jobs,
  or stopped after project work. In the same handover, reconcile
  `docs/runpod/README.md` Usage Dashboard totals against this ledger.
- Do not count local analysis, plotting, report writing, or ordinary local tests
  as RunPod usage.
- For actual benchmark runs, update both this ledger and the Benchmark Run
  Timings table in `project_summary.md`.
- If timing or cost was not captured, write `timing not captured` or
  `not captured` explicitly and improve instrumentation before the next run.
- Prefer exact per-Pod credits/USD from the RunPod console or billing export.
  If only account-level billing is available and the account has unrelated
  activity, do not treat it as project-specific truth.
- If using a rate multiplied by elapsed time, record the hourly rate and label
  the value as an estimate. Keep row-level costs compute-only; accumulate
  storage separately in the dashboard/header.
- When possible, use `time`, runner elapsed logs, or tmux log timestamps rather
  than estimating from memory.
- Reconcile `docs/runpod/README.md` every time ledger rows are added or changed:
  logged work time, bucket totals, compute/storage cost, shares, and
  interpretation bullets must stay current.
- Benchmark rows must also update `project_summary.md` Benchmark Run Timings so
  future agents can estimate runtime/cost and GPU choice from prior data.

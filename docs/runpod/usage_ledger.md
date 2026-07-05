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
- When possible, use `time`, runner elapsed logs, or tmux log timestamps instead
  of memory.

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

## Template

Copy this row for future entries:

```markdown
| YYYY-MM-DD | category | task | elapsed | compute rate | est. compute cost | GPU class / VRAM | commit | GPU used? | evidence / output | notes |
```

## Maintenance Rules

- Update this ledger whenever RunPod is started, used, left running between jobs,
  or stopped after project work.
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

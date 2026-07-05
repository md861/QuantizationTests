# Implementation Time Log

This log separates active implementation time from waiting time so future
duration estimates can be calibrated more honestly. It is operational
bookkeeping, not research evidence.

Guidelines:

- Active implementation includes reading code, editing, local reasoning, local
  tests, and documentation updates.
- User-wait includes time paused for user approval, clarification, or decisions.
- Hardware/tool-wait includes GPU benchmarks, long installs, remote jobs, and
  long-running tests where no active implementation work is happening.
- For concurrent work, record the main-agent active work separately from
  delegated run-agent or hardware wait time.

## Entries

| Date | Work item | Active implementation | User-wait | Hardware/tool-wait | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-07-05 | Per-method telemetry, throughput, artifact-size estimates, and metric-scope docs | ~9 min | 0 min | ~1 min local pytest wait; RunPod run-agent benchmark running separately and not counted | Added throughput fields, theoretical project artifact-size estimates, metric-scope docs, README/project-summary notes, and tests. Full local suite: 225 passed, 1 warning. |
| 2026-07-05 | Integrate TinyLlama per-method rerun results into docs | ~18 min | 0 min | RunPod benchmark already complete; no new GPU wait counted | Updated research draft, README, project summary, RunPod ledger/dashboard, and lab book with per-method runtimes, peak CUDA caveats, and comparison limits. |
| 2026-07-05 | Launch and integrate bnb NF4 telemetry rerun | ~20 min active | 0 min | ~6.4 min benchmark wall; bootstrap/install wait mostly remote tool wait | Synced fresh RTX 4090 Pod, reran bnb NF4 256-text with latest telemetry schema, then updated research draft, README, project summary, RunPod docs, and lab book. |

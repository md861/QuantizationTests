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

# RunPod Documentation

This folder keeps RunPod-specific technical documentation separate from the
main research narrative.

Keep every file here commit-safe: do not record raw SSH hosts, ports, usernames,
private key paths, Pod IDs, API keys, account identifiers, or one-off connection
strings.

## Usage Dashboard

Known captured RunPod time so far:

| Bucket | Captured Time | Est. Credits / Cost | Share Of Captured Time | Visual |
|---|---:|---:|---:|---|
| Setup / install | ~35.0 min | not captured | ~86% | `#################---` |
| Verification / readiness | ~5.8 min | not captured | ~14% | `###-----------------` |
| Benchmark compute | 0.0 min | 0 | 0% | `--------------------` |
| Cleanup / sync | timing not captured | not captured | n/a | `?` |

Interpretation:

- Current captured time is setup-heavy because this Pod has only been prepared
  and verified; no real GPU benchmark has run yet.
- The full Pod test suite took `349.22s (0:05:49)` and is categorized as
  verification, not benchmark compute.
- Several setup/sync/admin windows were not precisely timed. They are recorded
  as `timing not captured` in the ledger so future agents can improve tracking
  instead of silently dropping overhead.
- Credits/cost were not captured for the initial setup session. Going forward,
  copy the credit or USD amount from the RunPod console/billing view whenever it
  is available, and mark uncertain values as estimates.
- Actual benchmark runs must update both the usage ledger and the Benchmark Run
  Timings table in `project_summary.md`.

## Files

- `operations.md`: Pod setup, GPU/storage selection, credit guardrails,
  stop-window policy, and command discipline.
- `usage_ledger.md`: running ledger of setup, verification, idle/admin, cleanup,
  and benchmark time spent on RunPod.

## Maintenance Rule

When RunPod is started, stopped, used, left idle between queued jobs, or used for
benchmarking, update `usage_ledger.md`. When usage patterns change materially,
refresh the dashboard above so the first screen continues to show where credits
are going.

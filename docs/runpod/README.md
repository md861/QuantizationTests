# RunPod Documentation

This folder keeps RunPod-specific technical documentation separate from the
main research narrative.

Keep every file here commit-safe: do not record raw SSH hosts, ports, usernames,
private key paths, Pod IDs, API keys, account identifiers, or one-off connection
strings.

## Usage Dashboard

Known captured RunPod time so far:

<table>
  <tr>
    <th colspan="2">Logged work time: ~40.8 min</th>
    <th colspan="3">RunPod telemetry uptime: 5h 5m (~5.08 h)</th>
  </tr>
  <tr>
    <th colspan="5">Estimated active Pod cost so far: ~$1.32 compute, plus ~$0.02 container storage and ~$0.05 network-volume storage</th>
  </tr>
  <tr>
    <th>Bucket</th>
    <th>Captured Time</th>
    <th>Est. Credits / Cost</th>
    <th>Share Of Captured Time</th>
    <th>Visual</th>
  </tr>
  <tr>
    <td>Setup / install</td>
    <td>~35.0 min</td>
    <td>~$0.15 compute + ~$0.002 container storage</td>
    <td>~86%</td>
    <td><code>#################---</code></td>
  </tr>
  <tr>
    <td>Verification / readiness</td>
    <td>~5.8 min</td>
    <td>~$0.03 compute + &lt;$0.001 container storage</td>
    <td>~14%</td>
    <td><code>###-----------------</code></td>
  </tr>
  <tr>
    <td>Benchmark compute</td>
    <td>0.0 min</td>
    <td>$0.00</td>
    <td>0%</td>
    <td><code>--------------------</code></td>
  </tr>
  <tr>
    <td>Unlogged uptime / admin / idle</td>
    <td>~4h 24m</td>
    <td>~$1.15 compute + ~$0.01 container storage</td>
    <td>not part of captured work-time share</td>
    <td><code>????????????????????</code></td>
  </tr>
  <tr>
    <td>Network volume storage</td>
    <td>5h 5m elapsed</td>
    <td>~$0.05 at 100 GB * $0.07/GB-month</td>
    <td>n/a</td>
    <td><code>storage</code></td>
  </tr>
</table>

Interpretation:

- Current captured work time is setup-heavy because this Pod has only been
  prepared and verified; no real GPU benchmark has run yet.
- RunPod telemetry shows higher wall-clock uptime than our logged work time. The
  gap is tracked as unlogged uptime/admin/idle and should be reduced going
  forward by stopping the Pod outside short queued benchmark windows.
- The full Pod test suite took `349.22s (0:05:49)` and is categorized as
  verification, not benchmark compute.
- Several setup/sync/admin windows were not precisely timed. They are recorded
  as `timing not captured` in the ledger so future agents can improve tracking
  instead of silently dropping overhead.
- Account-level billing can be contaminated by other Pods or users on the same
  RunPod account. Prefer per-Pod billing line items when the console provides
  them. Otherwise estimate project Pod cost as `elapsed_hours * recorded hourly
  Pod rate`, and label it as an estimate.
- Current observed rates from the RunPod details panel: compute `$0.26/hr`,
  container storage `$0.003/hr`, displayed total `$0.26/hr` before separately
  estimating network-volume storage.
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

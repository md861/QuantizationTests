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
    <th colspan="2">Logged work time: 46.6 min</th>
    <th colspan="3">Telemetry uptime last observed: 5h 5m</th>
  </tr>
  <tr>
    <th colspan="5">Compute: $1.35 | Storage: $0.07 | Total: $1.42 | Budget ceiling: about GBP 200</th>
  </tr>
  <tr>
    <th>Bucket</th>
    <th>Time</th>
    <th>Compute Cost</th>
    <th>Share</th>
    <th>Visual</th>
  </tr>
  <tr>
    <td>Setup / install</td>
    <td>35.0 min</td>
    <td>$0.15</td>
    <td>75%</td>
    <td><code>###############-----</code></td>
  </tr>
  <tr>
    <td>Verification / readiness</td>
    <td>5.8 min</td>
    <td>$0.03</td>
    <td>12%</td>
    <td><code>##------------------</code></td>
  </tr>
  <tr>
    <td>Benchmark compute</td>
    <td>3.8 min</td>
    <td>$0.02</td>
    <td>8%</td>
    <td><code>##------------------</code></td>
  </tr>
  <tr>
    <td>Download / cache</td>
    <td>0.5 min</td>
    <td>$0.00</td>
    <td>1%</td>
    <td><code>--------------------</code></td>
  </tr>
  <tr>
    <td>RunPod debug / command wrapper</td>
    <td>1.5 min</td>
    <td>$0.01</td>
    <td>3%</td>
    <td><code>#-------------------</code></td>
  </tr>
  <tr>
    <td>Unlogged uptime / admin / idle</td>
    <td>264.2 min</td>
    <td>$1.15</td>
    <td>n/a</td>
    <td><code>????????????????????</code></td>
  </tr>
</table>

Interpretation:

- Current captured work time is still setup-heavy, but the first TinyLlama
  single-layer GPU smoke has now completed successfully.
- RunPod telemetry has shown higher wall-clock uptime than our logged work time. The
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
  container storage `$0.003/hr`, and network volume estimated from 100 GB at
  `$0.07/GB-month`. The dashboard keeps row-level costs compute-only and shows
  accumulated storage once in the header.
- Actual benchmark runs must update both the usage ledger and the Benchmark Run
  Timings table in `project_summary.md`. Keep cumulative RunPod benchmark spend
  under the project budget ceiling of about GBP 200; pause and ask the user
  before any plan that could materially change the expected spend profile.

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

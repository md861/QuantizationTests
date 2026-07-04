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
    <th colspan="2">Logged work time: 207.8 min</th>
    <th colspan="3">Telemetry uptime last manually reconciled: 5h 5m on 2026-07-02</th>
  </tr>
  <tr>
    <th colspan="5">Compute: ~$2.05 | Storage: ~$0.07 | Total: ~$2.12 | Budget ceiling: about GBP 200</th>
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
    <td>36.0 min</td>
    <td>~$0.15</td>
    <td>17%</td>
    <td><code>####----------------</code></td>
  </tr>
  <tr>
    <td>Verification / readiness</td>
    <td>5.8 min</td>
    <td>~$0.03</td>
    <td>3%</td>
    <td><code>#-------------------</code></td>
  </tr>
  <tr>
    <td>Benchmark compute / smoke</td>
    <td>161.0 min</td>
    <td>~$0.70</td>
    <td>77%</td>
    <td><code>###############-----</code></td>
  </tr>
  <tr>
    <td>Download / cache</td>
    <td>3.5 min</td>
    <td>~$0.01</td>
    <td>2%</td>
    <td><code>--------------------</code></td>
  </tr>
  <tr>
    <td>RunPod debug / command wrapper</td>
    <td>1.5 min</td>
    <td>~$0.01</td>
    <td>1%</td>
    <td><code>--------------------</code></td>
  </tr>
  <tr>
    <td>Unlogged uptime / admin / idle</td>
    <td>264.2 min</td>
    <td>~$1.15</td>
    <td>n/a</td>
    <td><code>????????????????????</code></td>
  </tr>
</table>

Interpretation:

- The dashboard now reflects the completed TinyLlama project INT4 logit-only
  256-text matrix, the completed bitsandbytes NF4 256-text external baseline,
  and the two intentionally aborted full-harness attempts that motivated the
  logit-only Milestone 4 path.
- Benchmark compute is now the dominant captured RunPod work bucket. This is
  expected after Step 3, but the two one-hour aborted harness attempts are a
  visible cost lesson: use the logit-only path for shared bnb comparisons unless
  weight/activation reconstruction is explicitly required.
- RunPod telemetry has shown higher wall-clock uptime than our logged work time.
  The gap is tracked as unlogged uptime/admin/idle and should be reduced by
  stopping the Pod outside short queued benchmark windows.
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
benchmarking, update `usage_ledger.md` and reconcile the Usage Dashboard above in
the same handover. Do this even when the ledger change feels small; the dashboard
is a manual summary and will otherwise drift. At minimum update logged work time,
compute/storage totals, bucket times/costs/shares, and interpretation bullets when
new benchmark, debug, idle, setup, or cache rows are added.

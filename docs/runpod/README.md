# RunPod Documentation

This folder keeps RunPod-specific technical documentation separate from the
main research narrative.

Keep every file here commit-safe: do not record raw SSH hosts, ports, usernames,
private key paths, Pod IDs, API keys, account identifiers, or one-off connection
strings.

## Usage Dashboard

Known captured/reconciled RunPod time so far:

<table>
  <tr>
    <th colspan="2">Logged work time: 377.5 min</th>
    <th colspan="3">Captured/reconciled runtime incl. unlogged/admin: 1012.6 min (16.9 h)</th>
  </tr>
  <tr>
    <th colspan="5">Compute: ~$8.11 | Storage: ~$0.15 | Total: ~$8.26 | Budget ceiling: about GBP 200</th>
  </tr>
  <tr>
    <th>Bucket</th>
    <th>Time</th>
    <th>Compute Cost</th>
    <th>Share of Observed Time</th>
    <th>Visual</th>
  </tr>
  <tr>
    <td>Setup / install</td>
    <td>130.1 min</td>
    <td>~$1.24</td>
    <td>13%</td>
    <td><code>###-----------------</code></td>
  </tr>
  <tr>
    <td>Verification / readiness</td>
    <td>5.8 min</td>
    <td>~$0.03</td>
    <td>1%</td>
    <td><code>--------------------</code></td>
  </tr>
  <tr>
    <td>Benchmark compute / smoke</td>
    <td>216.6 min</td>
    <td>~$1.17</td>
    <td>21%</td>
    <td><code>####----------------</code></td>
  </tr>
  <tr>
    <td>Download / cache</td>
    <td>8.2 min</td>
    <td>~$0.06</td>
    <td>1%</td>
    <td><code>--------------------</code></td>
  </tr>
  <tr>
    <td>RunPod debug / command wrapper</td>
    <td>16.7 min</td>
    <td>~$0.18</td>
    <td>2%</td>
    <td><code>--------------------</code></td>
  </tr>
  <tr>
    <td>Unlogged uptime / admin / idle</td>
    <td>635.2 min</td>
    <td>~$5.42</td>
    <td>63%</td>
    <td><code>#############-------</code></td>
  </tr>
</table>

Interpretation:

- The dashboard now reflects the completed TinyLlama project INT4 logit-only
  256-text matrix, the completed bitsandbytes NF4/AWQ/GPTQ 256-text external
  baselines, the TinyLlama project INT4 per-method telemetry rerun, and the two
  intentionally aborted full-harness attempts that motivated the logit-only
  Milestone 4 path. It also includes the Qwen2.5-3B smoke/readiness segment:
  reference cache prep and project one-layer `scale_row_g4` smoke passed, while
  Qwen AWQ/GPTQ external smokes failed at Marlin-family backend selection.
- The AWQ/GPTQ segment added a useful cost lesson: first-time external baseline
  setup can dominate the actual benchmark. The successful AWQ/GPTQ 256-record
  runner times were about 4.0-4.4 min each, but dependency probing, package
  compatibility checks, and Marlin JIT compilation consumed about 90 min wall
  time on the RTX 4090 Pod.
- AWQ/GPTQ ledger timestamps use local Europe/London tmux creation times from
  the remote session output. Treat them as operational provenance, not a
  RunPod billing export.
- Unlogged/admin/idle time is now the dominant captured cost bucket because the
  RTX 4090 Pod details panel showed a substantial uptime window before the
  AWQ/GPTQ segment began. Treat this as an estimate from observed uptime, not a
  billing export; stop Pods aggressively outside short queued benchmark windows.
- RunPod telemetry has shown higher wall-clock uptime than our logged work time.
  The gap is tracked as unlogged uptime/admin/idle and should be reduced by
  stopping the Pod outside short queued benchmark windows. The share column now
  uses observed total time for every row, so the buckets are comparable even
  though the unlogged row is not attributable to a specific project task.
- The captured/reconciled runtime total combines itemized ledger rows with one
  historical unlogged/admin residual from an earlier telemetry reconciliation.
  It excludes ledger rows explicitly marked `timing not captured`, so it is a
  best current accounting total rather than a complete billing export.
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
- GPU selection should optimize cost per useful benchmark, not raw theoretical
  speed. Before switching GPU class, compare expected wall time, method
  throughput, VRAM headroom, hourly rate, and setup/model-load overhead against
  prior ledger and benchmark-timing rows. Record the estimate and user approval.

## Files

- `operations.md`: Pod setup, GPU/storage selection, credit guardrails,
  stop-window policy, and command discipline.
- `qwen2_5_3b_plan.md`: commit-safe smoke and focused-run command plan for the
  next larger-than-TinyLlama Milestone 4 model.
- `usage_ledger.md`: running ledger of setup, verification, idle/admin, cleanup,
  and benchmark time spent on RunPod.

## Maintenance Rule

When RunPod is started, stopped, used, left idle between queued jobs, or used for
benchmarking, update `usage_ledger.md` and reconcile the Usage Dashboard above in
the same handover. Do this even when the ledger change feels small; the dashboard
is a manual summary and will otherwise drift. At minimum update logged work time,
compute/storage totals, bucket times/costs/shares, and interpretation bullets when
new benchmark, debug, idle, setup, or cache rows are added. For benchmark rows,
also update the Benchmark Run Timings table in `project_summary.md` with GPU
class, wall/runner time, method telemetry when available, peak memory, and any
cost-per-run notes needed for future GPU selection.

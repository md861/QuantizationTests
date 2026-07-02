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
    <th colspan="5">Total captured Pod uptime: ~40.8 min, plus uncaptured setup/sync windows</th>
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
    <td>rate not captured; estimate as 0.583 h * hourly rate</td>
    <td>~86%</td>
    <td><code>#################---</code></td>
  </tr>
  <tr>
    <td>Verification / readiness</td>
    <td>~5.8 min</td>
    <td>rate not captured; estimate as 0.097 h * hourly rate</td>
    <td>~14%</td>
    <td><code>###-----------------</code></td>
  </tr>
  <tr>
    <td>Benchmark compute</td>
    <td>0.0 min</td>
    <td>0</td>
    <td>0%</td>
    <td><code>--------------------</code></td>
  </tr>
  <tr>
    <td>Cleanup / sync</td>
    <td>timing not captured</td>
    <td>not captured</td>
    <td>n/a</td>
    <td><code>?</code></td>
  </tr>
</table>

Interpretation:

- Current captured time is setup-heavy because this Pod has only been prepared
  and verified; no real GPU benchmark has run yet.
- The full Pod test suite took `349.22s (0:05:49)` and is categorized as
  verification, not benchmark compute.
- Several setup/sync/admin windows were not precisely timed. They are recorded
  as `timing not captured` in the ledger so future agents can improve tracking
  instead of silently dropping overhead.
- Account-level billing can be contaminated by other Pods or users on the same
  RunPod account. Prefer per-Pod billing line items when the console provides
  them. Otherwise estimate project Pod cost as `elapsed_hours * recorded hourly
  Pod rate`, and label it as an estimate.
- The hourly Pod rate was not captured for the initial setup session, so the
  dashboard shows formulas instead of retroactive dollar/credit values.
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

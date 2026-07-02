# RunPod Operations

This note captures the technical operating policy for using RunPod as the
project's GPU benchmark worker. The folder index at `README.md` contains the
current usage dashboard. Keep this file commit-safe: do not record raw SSH
hosts, ports, usernames, private key paths, Pod IDs, API keys, account
identifiers, or one-off connection strings.

## Current Role

RunPod is reserved for GPU-bound benchmark execution. Keep code generation,
ordinary local tests, data analysis, plotting, research writing, README updates,
project-summary updates, and lab-book updates local unless debugging a GPU-only
failure requires direct remote work.

The local SSH alias is:

```text
runpod-pq
```

The alias may point to the active Pod through local SSH config. Do not commit
the raw connection details behind the alias.

## Baseline Worker

Current baseline Pod class:

- GPU: NVIDIA RTX 4000 Ada Generation class
- VRAM: about 20 GB
- System memory / CPU class: 50 GB RAM / 9 vCPU observed on current Pod
- Persistent workspace: `/workspace` on a 100 GB network volume
- Repo path on Pod: `/workspace/PQ_project`
- Project venv on Pod: `/workspace/PQ_project/.venv`
- Region observed during setup: EU-RO-1 secure cloud
- Container disk observed during setup: 20 GB
- Template/image family observed during setup: RunPod PyTorch CUDA 12.4 / torch v240 family
- Verified Pod stack: PyTorch 2.6.0+cu124, Transformers 5.12.1, CUDA available
- Verified Pod test state: `212 passed, 1 warning in 349.22s`

This worker is enough for TinyLlama-era smoke tests and narrow controlled
baselines. Current observed active-Pod rates are `$0.26/hr` compute and
`$0.003/hr` container storage, with displayed total `$0.26/hr`; estimate
network-volume storage separately. Reassess before larger models, longer contexts, broader evaluation
sets, or memory-heavy external baselines.

## Selection Obstacles Encountered

GPU and storage selection had a few practical constraints that future agents
should remember:

- GPU availability is region-dependent and can be low for popular cards. Prefer
  the smallest GPU that can answer the current research question instead of
  waiting for a larger card by default.
- The RTX 4000 Ada class was chosen as a low-cost starting point because about
  20 GB VRAM is enough for TinyLlama-era smoke tests and narrow baselines. It is
  not a blanket approval for larger LLMs, longer contexts, larger eval batches,
  or memory-heavy GPTQ/AWQ/bitsandbytes workflows.
- If a benchmark plan appears likely to exceed 20 GB VRAM, pause and ask the
  user before switching Pod class. Do not burn time trying repeated failing runs
  on an under-sized GPU.
- RunPod may require choosing either a network volume or a volume disk before a
  GPU Pod can be deployed. Use a network volume for this project unless the user
  explicitly chooses an ephemeral throwaway run.
- Data center choice matters because a network volume must be compatible with
  where the Pod runs. Prefer a Europe-region volume for the user's current
  workflow when compatible GPUs are available; fall back to another region only
  for availability or price reasons.
- Network volumes reduce rebuild friction but can make virtualenv installs slow
  because they write many small files. Run dependency installs in `tmux`, log to
  `/workspace`, and avoid unnecessary package upgrades.
- Volume disk can be simpler for one-off disposable Pods, but it is less useful
  here because we want cache, repo, venv, logs, and artifacts to survive Pod
  replacement.

## Storage Policy

Use a network volume for `/workspace`.

Reasons:

- It survives Pod stop/termination.
- It can be reattached across compatible Pods.
- It avoids rebuilding the repo, venv, Hugging Face cache, and benchmark
  artifacts after every Pod replacement.
- It was cheaper per GB than volume disk in the RunPod storage table checked
  during setup.

Recommended minimum for the current phase: 100 GB. This leaves room for the
repo, virtual environment, Hugging Face cache, logs, CSVs, and benchmark plots.

## Usage Accounting

Maintain `docs/runpod/usage_ledger.md` for RunPod time accounting. This ledger
is separate from the benchmark timing table in `project_summary.md`:

- benchmark timing table: actual benchmark compute runtimes used for research
  comparison and future runtime prediction
- RunPod usage ledger: setup, dependency installs, downloads, verification,
  debugging, idle windows, cleanup/sync, plus benchmark runs

Every RunPod segment should record category, elapsed time, GPU class, commit
hash when relevant, whether meaningful GPU compute was used, log/output path, and
notes. If exact timing was missed, write `timing not captured` explicitly and
improve instrumentation before the next run.

## Credit Guardrails

Project budget ceiling: keep total RunPod benchmark spend under about GBP 200. Track estimated compute and storage spend in `docs/runpod/usage_ledger.md` after every Pod segment. Pause and ask the user before changing Pod class, launching a broader benchmark matrix, or continuing after a failed run pattern that could materially increase spend.

Before spending RunPod credits:

1. Make the smallest relevant local dry run pass.
2. State the expected duration and cost risk to the user.
3. Confirm the target commit hash.
4. Run a single-layer or small-subset smoke benchmark first.
5. Launch long jobs only inside detached `tmux`.
6. Write logs and results under persistent `/workspace`.
7. Record GPU type, VRAM, peak memory, commit hash, elapsed time, and output
   paths in the bookkeeping docs.
8. Pull back only artifacts needed for local analysis and reporting.

## Stop-Window Policy

Default: stop the Pod after each benchmark finishes.

Exception: if another GPU benchmark is already queued and ready to start within
about 30 minutes, the Pod may stay running between short back-to-back GPU jobs.
For a planned same-day batch, keeping the Pod running is acceptable only when the
next command, expected runtime, and stop point are explicit.

Never leave a GPU Pod running during local editing, analysis, plotting,
documentation, long discussion, breaks, or overnight work.

## Command Discipline

- Use detached `tmux` for long installs, downloads, and benchmarks.
- Prefer small smoke runs before full-model runs.
- Keep Hugging Face/model caches under `/workspace`.
- Keep logs under `/workspace` with descriptive names.
- Do not rely on VSCode or an interactive SSH session staying connected.
- Do not run full benchmarks until GPU-aware runner logging records device mode,
  CUDA availability, GPU name, VRAM, peak memory, commit hash, and elapsed time.

## Known Setup Lesson

Do not create the project virtualenv with `--system-site-packages` on this Pod.
The first mixed venv reused image packages but produced unstable
Transformers/GPT-2 imports. Use the clean self-contained
`/workspace/PQ_project/.venv`.

# Mistral-7B Full-Comparison Candidate Plan

This file is commit-safe operational bookkeeping for the recommended
larger-than-TinyLlama Milestone 4 successor candidate. Do not add raw SSH hosts,
ports, usernames, Pod IDs, private key paths, API keys, account identifiers, or
one-off connection strings.

## Successor Gate

This model can be promoted to the main post-TinyLlama comparison only if all of
the following paths pass smoke on the same evaluation resource:

- Original/reference logits and loss from `mistralai/Mistral-7B-Instruct-v0.2`
- Project `scale_row_g4`
- bitsandbytes NF4 `float16`
- AWQ via `TheBloke/Mistral-7B-Instruct-v0.2-AWQ`
- GPTQ via `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`

If any path fails, record the failure as a partial probe/backend note in the lab
book and RunPod ledger. Do not present it as a completed research comparison.

## Candidate Rationale

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- License: Apache-2.0
- Architecture family: Mistral/LLaMA-style decoder-only model with standard
  attention projection modules expected to be compatible with the project
  layer-targeting pattern.
- External baseline availability: public AWQ and GPTQ checkpoints exist under
  TheBloke with matching base model metadata.
- Main risk: 7B is a larger jump than the earlier 3B probes, so RTX 4090 VRAM
  and wall-time must be checked with smoke/readiness before any 256-record run.

## Required Approval Point

RunPod is first needed after local preset/command prep is committed and pushed.
Before running any command on a Pod, report the target commit, GPU class/hourly
rate, expected runtime/cost, output paths, and whether the estimate is for
smoke or full 256-record benchmarking. Wait for explicit user approval.

## Next Local Prep

1. Add or verify a Mistral project smoke preset targeting a single attention
   projection such as `model.layers.0.self_attn.q_proj`.
2. Add a focused project preset for all compatible layers with
   `--logit-only --logit-methods scale_row_g4`.
3. Prepare cache/project/bnb/AWQ/GPTQ smoke commands using the tracked
   `docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt`
   resource.
4. Run local config/tests only. Do not launch GPU work until the user approves
   the refreshed runtime/cost estimate.

## Resume Checklist

1. Confirm the repo is clean and current with `main`.
2. Implement Mistral local presets/tests.
3. Commit and push local prep.
4. Produce a fresh RunPod estimate for cache plus five smoke commands:
   reference/cache, project, bitsandbytes NF4, AWQ, and GPTQ.
5. Wait for user approval and fresh Pod details.
6. Run smoke in detached `tmux`, pull artifacts, and update bookkeeping.

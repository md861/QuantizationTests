# Side Research Note: Coherence Check — Memory / Speed / Quality

Date: 2026-07-03

This note captures a research-coherence review of the current milestone plan against
the project's three stated evaluation axes: **memory usage**, **speed**, and
**model quality/performance**. It was produced by reading the existing repo state
(README.md, project_summary.md, docs/research_draft.md, lab_book/project_journey.md,
context.md, and the experiment harness code) — no code or docs were modified as part
of this review.

## Headline finding

The project's own quantizer harness (`experiments/transformer_experiment.py`) performs
**fake (simulated) quantization**: a weight matrix is quantized and then immediately
dequantized back to float, and that float tensor is written back into the model
(`experiments/transformer_experiment.py:506`, `:770-775`). There is no packed INT4
storage and no real low-bit runtime kernel anywhere in `quant/quantizer.py` or the
harness — every "quantized" forward pass actually executes at full precision.

This is a legitimate and common technique for studying reconstruction/quality impact,
but it structurally cannot produce real memory or speed numbers for the project's own
method, no matter how many additional models are run through the current harness.

## Axis 1 — Memory

- **Measured so far**: only one aggregate number exists — total process CUDA peak
  memory (2124 MB allocated / 2224 MB reserved) from the single TinyLlama smoke run
  (README.md:96, project_summary.md:194). This reflects whole-harness memory (model +
  numpy quantize/dequantize buffers), not memory saved by INT4 vs FP32.
- **Harness capability**: `run_transformer_benchmark.py` and `bitsandbytes_baseline.py`
  both capture `torch.cuda.max_memory_allocated`/`max_memory_reserved`, but only as one
  aggregate figure per whole run, not per method/path. Zero memory columns exist in any
  per-record CSV (`WeightRecord`, `ActivationRecord`, `LogitRecord`).
- **Plan says**: Milestone 4 next-steps mention recording "VRAM, peak memory" and
  comparing "memory pressure and artifact size" across methods (README.md:66,69) —
  but nowhere is it explained how a real compressed-memory number would be obtained
  for ParoQuant's own method, given it never produces a reduced-precision artifact.
- **Docs gap**: this omission is not mentioned in the Limitations section
  (docs/research_draft.md §18) at all — it's a silent gap, not an acknowledged caveat.

## Axis 2 — Speed / Latency / Throughput

- **Measured so far**: only whole-experiment wall-clock time (e.g., "INT8 ≈ 798s,
  INT4 = 780s" — docs/research_draft.md:964-966). This is harness execution time
  (looping over layers/methods in Python/numpy), not inference latency or throughput
  of a deployed quantized model.
- **Harness capability**: a single `time.time()` wrap around the entire experiment run
  (`run_transformer_benchmark.py:428-430`, `bitsandbytes_baseline.py:358-360`). No
  per-forward-pass or tokens/sec measurement exists anywhere.
- **Plan says**: same "record elapsed time" language as the memory axis, with the same
  ambiguity about what it will actually represent once TinyLlama-scale runs happen.
- **Docs gap**: README.md:219-220 states the repo "prioritizes clarity, modularity,
  reproducibility, and mathematical transparency over production inference
  performance" — the closest thing to an honest caveat here, but it's a
  design-philosophy statement rather than a flagged research limitation.

## Axis 3 — Model Quality / Performance (well covered)

- **Measured so far**: real, per-method measurement — weight reconstruction (MSE,
  cosine similarity, SNR, zero/saturation fraction), activation drift, and full-model
  logit/loss/perplexity across five real models (tiny-gpt2, TinyStories-1M,
  Pythia-14M/70M, distilgpt2), with multiple bitwidths, group sizes, and rotation
  presets. Extensively documented in docs/research_draft.md §§12-17 and
  README.md:109-150.
- **Harness capability**: fully captures this axis end to end — this is what the
  harness was built to measure.
- **Plan says**: quality comparison against GPTQ/AWQ/bitsandbytes on a tracked
  256-record WikiText-2 validation set is central to the Milestone 4 plan.
- **Docs**: existing limitations are already honestly written down (synthetic-matrix
  realism, rotation-pair selection not calibrated on transformer data, tiny-gpt2 being
  "harness-validation evidence, not a realistic compression result").

## Overall verdict

The Milestone 4 plan is **coherent for quality** but has an **unacknowledged
structural gap for memory and speed**. As currently scoped, those two axes will most
likely end up measured only for the external baselines (bitsandbytes/GPTQ/AWQ, which
do use real low-bit runtimes) — not for ParoQuant's own method — unless one of the
following happens:

1. The project's quantizer is extended to produce a real packed-INT4 artifact or is
   loaded through a genuine low-bit kernel (e.g., bitsandbytes' `Linear4bit`) so that
   real memory/speed numbers can be measured for ParoQuant itself, not just quality.
2. The research claim is explicitly scoped down to quality/accuracy for ParoQuant,
   with memory/speed framed as properties of the baseline methods being compared
   against — not of ParoQuant itself — and this scoping is stated plainly in the
   Limitations section.

This is a scoping decision, not a bug in the existing code — but it is currently
invisible in the written limitations and worth resolving before investing in
further TinyLlama-scale (or larger) benchmark runs.

---

# Side Research Note: Effective Bits-per-Weight — Is `scale_row_g4` vs AWQ/GPTQ a Fair Fight?

Date: 2026-07-06

## Question

> Your `scale_row_g4` method beat AWQ on quality — at what effective bits per weight, and is that a fair fight?

## Headline finding

**No, not at matched compression.** The project's own artifact-size telemetry
(already implemented in `experiments/transformer_experiment.py:809-856`) shows
that `scale_row_g4` costs roughly **~12 bits/weight** as currently coded (or
~8 bits/weight under an optimistic fp16-repacked estimate) once its per-4-row
scale metadata is counted — not the 4 bits its name implies. The real AWQ and
GPTQ checkpoints this project benchmarks against use `group_size=128` and land
at **~4.1–4.2 bits/weight**. That is a **~2–3x storage-budget advantage** for
the project method, hidden behind a shared "4-bit" label. The Milestone 4
comparison tables (`docs/research_draft.md` §19.1–19.3) report logit MSE,
cosine similarity, top-5 overlap, PPL ratio, runtime, and CUDA memory — but
never bits/weight or artifact size — so this gap is currently invisible in the
written comparison.

## The actual bits/weight, from real data

**Project `scale_row_g4` on Mistral-7B**, from
`results/transformer_mistral_7b_v0_2_int4_scale_row_g4/transformer_logit_metrics.csv`:

| | bytes |
|---|---:|
| `estimated_packed_weight_bytes` (INT4 payload) | 3,489,660,928 |
| `estimated_scale_metadata_bytes` (per-4-row scale + per-channel factor) | 6,984,826,880 |
| `estimated_total_artifact_bytes` | 10,474,487,808 |

The packed payload alone is exactly 4 bits/weight. But the scale metadata — one
`float32` scale per group of 4 weights, computed in
`transformer_experiment.py:833-847` — adds roughly another 8 bits/weight,
because a 4-byte (32-bit) scale is shared across only 4 weights. Total:
**~12.0 bits/weight**. Even assuming a real implementation repacked those
scales as `fp16` (halving the overhead), it would still be **~8 bits/weight**.

**AWQ and GPTQ** — checked directly against the actual pre-quantized
checkpoints this project loads (`MaziyarPanahi/Mistral-7B-Instruct-v0.2-AWQ`
and `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`) via their `config.json` on
Hugging Face:

- AWQ: `bits=4, group_size=128, zero_point=true` → one fp16 scale + one 4-bit
  zero point per 128 weights → **≈4.16 bits/weight**
- GPTQ: `bits=4, group_size=128, sym=true` → one fp16 scale per 128 weights,
  no separate zero point → **≈4.13 bits/weight**

## Comparison table

| Method | Effective bits/weight |
|---|---:|
| project `scale_row_g4` (as coded, fp32 scales) | ~12.0 |
| project `scale_row_g4` (optimistic fp16-repack) | ~8.0 |
| AWQ (real checkpoint, group_size=128) | ~4.16 |
| GPTQ (real checkpoint, group_size=128) | ~4.13 |

## Reasoning

- Grouping/scaling granularity is a direct storage/quality trade: a smaller
  group size (`g4`) gives every 4 weights their own scale, which is why
  `scale_row_g4` reconstructs so well — but that precision is bought with
  metadata, not free.
- AWQ and GPTQ use `group_size=128` in their real, shipped checkpoints — 32x
  coarser grouping than the project's `g4` preset — which is why their
  overhead per weight is close to negligible (~0.13–0.16 bits/weight) while
  the project's is ~4–8 bits/weight.
- Because the label "4-bit"/"INT4" only describes the payload width and not
  the total artifact size, comparing `scale_row_g4` against AWQ/GPTQ under
  that label alone conflates "same nominal bitwidth" with "same compression
  ratio." They are not the same thing here.
- It is unsurprising, in hindsight, that a method spending ~2–3x the bits per
  weight of a true ~4.1-bit format achieves lower reconstruction error — this
  is close to comparing an INT4 method against a de facto mixed 8–12 bit
  method.

## Conclusion / recommended fix

The "beats AWQ on quality" claim, as currently documented, is more precisely
"beats AWQ on quality at roughly 2–3x AWQ's storage cost per weight" — a real
and useful data point, but weaker than the current phrasing implies. A fair,
matched-bit-budget comparison does not require new kernel work: rerunning the
project method with `scale_row_g128` (or `row_grouped_g128`) instead of `g4`
would land at roughly **~4.25 bits/weight** (fp32 scales) — genuinely
comparable to AWQ's ~4.16 and GPTQ's ~4.13 — and would answer the real
question: does the project's rotation/scaling scheme still win once its extra
~2–3x metadata budget is taken away? Recommended before the "beats AWQ" claim
is repeated further: (1) add a bits/weight or artifact-size column to the
Milestone 4 comparison tables, and (2) rerun at a matched group size as the
real head-to-head.

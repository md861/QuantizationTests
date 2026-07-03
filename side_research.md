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

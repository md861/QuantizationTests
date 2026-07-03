# ParoQuant Research Project — Codex Context

## Project Overview

This repository is a research-oriented educational implementation of the ParoQuant quantization method for large language models (LLMs).

The goal is to:

1. understand quantization deeply,
2. reproduce core ParoQuant ideas,
3. benchmark quantization error,
4. experiment with modifications,
5. and eventually scale to real open-source LLMs.

The implementation should prioritize:

- clarity,
- modularity,
- experimentation,
- reproducibility,
- mathematical transparency.

This is NOT initially intended to be a production inference engine.

---

# Core Research Goals

The project will progress through several stages.

## Stage 1 — Quantization Sandbox

Build a matrix-level quantization playground:

- generate matrices,
- quantize/dequantize,
- measure reconstruction error,
- visualize distortions,
- compare quantization schemes.

---

## Stage 2 — ParoQuant Core

Implement:

- pairwise Givens rotations,
- channel scaling,
- grouped quantization,
- outlier suppression experiments.

---

## Stage 3 — Tiny Transformer Integration

Apply the quantizer to:

- tiny-gpt2,
- DistilGPT2.

Measure:

- perplexity,
- output similarity,
- activation drift.

---

## Stage 4 — Real LLM Benchmarking

Scale to:

- TinyLlama 1.1B,
- Gemma 2B,
- Mistral 7B,
- Llama 3 8B.

Compare against:

- GPTQ,
- AWQ,
- bitsandbytes.

---

# Key ParoQuant Insight

ParoQuant reduces quantization error by:

1. applying pairwise channel rotations,
2. redistributing outliers,
3. smoothing activation/weight distributions before quantization.

The central operation is a Givens rotation between channel pairs.

For channels i and j:

```text
R(theta) = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
```

Applied as:

```text
[x_i']
[x_j'] = R(theta) [x_i]
                    [x_j]
```

The goal is to reduce outlier concentration and improve low-bit quantization quality.

---

# Current Milestone

Milestone 1 (Quantization Sandbox), Milestone 2 (ParoQuant Core), and
Milestone 3 (Tiny Transformer Integration) are complete.

The project has now begun:

# Milestone 4 — Real LLM Benchmarking

Planned steps:

1. Local hardware/cache audit is complete; RunPod is configured as the GPU benchmark worker.
2. The first larger target is TinyLlama 1.1B; its single-layer INT4 RunPod smoke passed and is a readiness check only.
3. Define the controlled TinyLlama benchmark matrix before any full run: original model, project row-grouped INT4 g4/g8-style paths where feasible, and the lightest feasible external baseline from GPTQ, AWQ, or bitsandbytes.
4. Establish a reproducible evaluation text source larger than the current tiny smoke/WikiText-style samples.
5. Estimate RunPod runtime/cost before each GPU run and run another narrow smoke whenever the matrix, evaluation text, dependencies, or GPU class changes.
6. Run full benchmarks only from detached tmux under persistent /workspace, recording elapsed time, GPU, VRAM, peak memory, commit hash, result counts, and estimated spend.
7. Compare quality, runtime, memory pressure, and artifact size across the project method and external baselines.
8. Update the research draft, README, project summary, lab book, and RunPod usage ledger after each model or GPU segment.

---

# Recommended Project Structure

```text
paroquant-lab/
│
├── quant/
│   ├── quantizer.py
│   ├── rotations.py
│   ├── scaling.py
│   ├── metrics.py
│   ├── matrix_factory.py
│   └── visualize.py
│
├── experiments/
│   ├── baseline_experiment.py
│   ├── outlier_experiment.py
│   └── rotation_experiment.py
│
├── notebooks/
│   ├── sandbox.ipynb
│   └── analysis.ipynb
│
├── plots/
├── results/
├── papers/
├── notes/
│
├── README.md
├── requirements.txt
└── main.py
```

---

# Quantization Equations

## Symmetric Quantization

```text
q = round(x / s)
```

## Scale

```text
s = max(abs(x)) / (2^(b-1) - 1)
```

## Dequantization

```text
x_hat = s * q
```

where:

- b = bitwidth.

---

# Metrics

## MSE

```text
MSE = mean((W - W_hat)^2)
```

## Relative Error

```text
||W - W_hat||_F / ||W||_F
```

## Cosine Similarity

```text
cos(theta) = dot(W, W_hat) / (||W|| ||W_hat||)
```

## SNR

```text
10 * log10( ||W||^2 / ||W - W_hat||^2 )
```

---

# Initial Experiment Plan

## Experiment 1

Compare:

- INT8
- INT4

on:

- Gaussian matrices
- outlier-heavy matrices

Measure:

- MSE
- cosine similarity
- SNR

Expected outcome:

- INT4 struggles with outliers,
- motivating ParoQuant rotations.

---

# Coding Priorities

The codebase should prioritize:

- readability,
- modular design,
- reproducibility,
- type hints where useful,
- clean experiment APIs,
- minimal dependencies.

Avoid premature optimization.

The first goal is understanding and experimentation.

---

# Immediate Tasks

Implement:

1. matrix_factory.py
2. quantizer.py
3. metrics.py
4. baseline_experiment.py
5. visualize.py

Start with matrix-level experiments before integrating any transformer models.

---

# Long-Term Research Directions

Potential future experiments:

- grouped quantization,
- per-channel scaling,
- adaptive rotations,
- learned scaling,
- Hessian-aware quantization,
- KV-cache quantization,
- mixed precision,
- sparse + quantized hybrids.

---

# Important Notes

This project is educational and research-oriented.

The implementation should:

- expose intermediate tensors,
- support debugging and visualization,
- make experimentation easy,
- and preserve mathematical clarity.

Do not optimize aggressively at the expense of interpretability.

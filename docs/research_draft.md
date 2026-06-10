# Quantization Lab: A Matrix-Level Study of Low-Bit Quantization, Outlier Pressure, and Preprocessing Transformations

Status: living draft. This document should be updated as experiments, figures, and conclusions mature.

## Abstract

This project studies low-bit quantization from a deliberately small and inspectable starting point: synthetic matrices. The central question is how symmetric low-bit quantization changes matrix values, reconstruction error, singular-value structure, and integer-code usage under different distributional conditions. We first build a matrix-level quantization sandbox with Gaussian, heavy-tailed, and outlier-injected matrices. We then compare INT8 and INT4 symmetric quantization using reconstruction metrics, spectrum diagnostics, residual plots, histograms, and benchmark-style dashboards. We then test ParoQuant-inspired preprocessing transformations — pairwise Givens rotations, reversible per-channel scaling — and two grouped quantization strategies: column-grouped (one scale per block of columns) and row-grouped (one scale per block of rows within each column, the approach used by GPTQ and AWQ).

The current evidence shows that INT4 is highly sensitive to outlier-dominated scales, producing elevated reconstruction error and high zero-code fractions. Per-channel scaling substantially reduces this failure mode. A key new finding is that column-grouped quantization offers no improvement over global INT4 when outliers are confined to a single row, because every column group still contains that row. Row-grouped quantization directly addresses this: in a controlled row-outlier example, row-grouping with group size 4 reduces MSE by 6× and zero fraction from 92% to 23% compared with global INT4, while column-grouped quantization at any group size leaves both metrics unchanged. We treat these as promising observations pending broader sweeps.

## 1. Research Motivation

Low-bit quantization replaces high-precision weights with a finite set of representable integer codes. This can reduce memory and compute cost, but it also introduces reconstruction error. The key problem studied here is **outlier pressure**: if a small number of unusually large values determine the quantization scale, ordinary values are represented coarsely and may collapse toward zero.

For a matrix $W \in \mathbb{R}^{m \times n}$, symmetric full-matrix quantization with bitwidth $b$ uses a single scale

$$
s = \frac{\max_{i,j}|W_{ij}|}{2^{b-1} - 1}.
$$

The integer code matrix is

$$
Q = \mathrm{clip}\left(\mathrm{round}(W / s), q_{\min}, q_{\max}\right),
$$

and the dequantized reconstruction is

$$
\hat{W} = sQ.
$$

For INT8, the integer range is $[-127, 127]$. For INT4, the range is $[-7, 7]$. INT4 has far fewer available codes, so it is much more vulnerable when the scale is set by extreme values.

This project therefore asks:

1. How does INT4 change matrix values compared with INT8?
2. Which metrics reveal different aspects of quantization error?
3. How do heavy tails and outliers change low-bit failure modes?
4. Can rotations and scaling reduce outlier pressure before quantization?
5. Which findings are robust enough to become claims, and which remain hypotheses?

## 2. Experimental Sandbox

The project currently uses three synthetic matrix families:

- **Gaussian matrices**: entries sampled independently from $x_{ij} \sim \mathcal{N}(\mu,\sigma^2)$.
- **Heavy-tailed matrices**: entries sampled from a scaled Student-t distribution, $x_{ij} = s \cdot t_{\nu}$.
- **Outlier matrices**: Gaussian base matrices with a controlled fraction of entries replaced by large signed outliers.

These families are not intended to model neural-network weights perfectly. They are controlled test cases for learning how quantization behaves as distributional assumptions become less friendly.

Figure:

![Synthetic matrix families](figures/research_matrix_families.png)

The Gaussian case is a relatively benign baseline. The heavy-tailed and outlier cases are stress tests for global scaling: a few large values can dominate the scale and reduce effective resolution elsewhere.

## 3. Quantization Error Metrics

Let $W$ be the original matrix and $\hat{W}$ the reconstructed matrix after quantization and dequantization. Let the residual be

$$
E = \hat{W} - W.
$$

The project currently reports the following metrics.

### Mean Squared Error

$$
\mathrm{MSE}(W,\hat{W}) = \frac{1}{mn}\sum_{i,j} E_{ij}^{2}.
$$

MSE measures average squared reconstruction error. It is sensitive to larger residuals.

### Mean Absolute Error

$$
\mathrm{MAE}(W,\hat{W}) = \frac{1}{mn}\sum_{i,j}|E_{ij}|.
$$

MAE is easier to interpret in the original value scale and is less dominated by large residuals than MSE.

### Relative Frobenius Error

$$
\frac{\|W-\hat{W}\|_F}{\|W\|_F}.
$$

This normalizes the reconstruction error by the energy of the original matrix, which makes comparisons across matrix scales more meaningful.

### Cosine Similarity

After flattening both matrices,

$$
\frac{\langle W, \hat{W}\rangle}{\|W\|_2\|\hat{W}\|_2}.
$$

Cosine similarity measures whether the reconstruction points in a similar direction to the original, even if its magnitude changes.

### Signal-to-Noise Ratio

The project computes SNR in decibels as

$$
\mathrm{SNR}_{dB} = 10\log_{10}\left(\frac{\sum_{i,j}W_{ij}^2}{\sum_{i,j}(W_{ij}-\hat{W}_{ij})^2}\right).
$$

Higher SNR means the signal energy dominates the reconstruction noise.

### Spectrum Error

Let $\sigma(W)$ denote the vector of singular values of $W$. The relative spectrum error is

$$
\frac{\|\sigma(W)-\sigma(\hat{W})\|_2}{\|\sigma(W)\|_2}.
$$

This measures how much quantization changes the singular-value geometry of the matrix.

### Zero Fraction and Saturation Fraction

For integer codes $Q$:

$$
\mathrm{zero\_fraction} = \frac{|\{(i,j): Q_{ij}=0\}|}{mn},
$$

and

$$
\mathrm{saturation\_fraction} = \frac{|\{(i,j): Q_{ij}=q_{\min} \text{ or } Q_{ij}=q_{\max}\}|}{mn}.
$$

High zero fraction can indicate collapse of ordinary values toward zero. High saturation fraction can indicate many values hitting the representable range boundary.

## 4. Example: INT8 vs INT4 on a Gaussian Matrix

We first quantize a Gaussian matrix with INT8 and INT4 using the same symmetric full-matrix quantization rule.

Figure:

![Gaussian INT8 vs INT4 comparison](figures/research_int4_gaussian_comparison.png)

For this example, INT8 reconstructs the matrix with very small error, while INT4 introduces visibly larger residuals and a much higher zero-code fraction.

| Quantizer | MSE | Rel. Frobenius | SNR dB | Zero frac | Sat. frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| INT8 | 0.000044 | 0.006713 | 43.462 | 0.007812 | 0.003906 |
| INT4 | 0.013845 | 0.118753 | 18.507 | 0.187500 | 0.003906 |

This example illustrates an important baseline learning: INT4 can still preserve broad structure on a benign Gaussian matrix, but the reduction from 255 signed levels to 15 signed levels produces a large jump in reconstruction error.

## 5. Example: Outlier-Driven Failure Modes

The outlier matrix case makes the scale problem sharper. A small number of large values increase $s$, which coarsens the representation of smaller entries.

Figure:

![Outlier matrix quantization histograms](figures/research_outlier_histograms.png)

In the outlier example below, INT4 has much higher MSE and a high zero fraction.

| Quantizer | MSE | Rel. Frobenius | SNR dB | Zero frac | Sat. frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| INT8 | 0.000892 | 0.012393 | 38.136 | 0.058594 | 0.003906 |
| INT4 | 0.297624 | 0.226335 | 12.905 | 0.664062 | 0.015625 |

This supports the central working hypothesis of the project: low-bit global quantization is not only a question of bitwidth, but also a question of distribution shape. Outliers can make an otherwise ordinary matrix difficult to represent with INT4.

## 6. Baseline and Outlier Sweep Analysis

The baseline experiment compares INT8 and INT4 across Gaussian, heavy-tailed, and outlier matrix families. The outlier experiment sweeps outlier fraction and outlier scale. The analysis helper converts these CSV outputs into a single dashboard.

Figure:

![Analysis dashboard](figures/analysis_dashboard.png)

Current findings from the analysis dashboard:

- INT4 error is consistently much larger than INT8 error in the tested matrix families.
- Heavy-tailed INT4 has a particularly high zero fraction.
- Increasing outlier scale tends to increase zero-code pressure in INT4.
- The visual dashboard is useful because MSE, SNR, zero fraction, and outlier severity tell complementary parts of the story.

At this stage, the dashboard is best interpreted as a diagnostic tool, not as a final benchmark. The synthetic examples are small and intentionally controlled.

## 7. Test Cases as Mathematical Evidence

The project uses tests not only as software checks, but also as compact mathematical examples.

### Givens Rotations

A pairwise Givens rotation between columns $i$ and $j$ uses

$$
R_{ij}(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}.
$$

Right-multiplying $W$ by the corresponding full identity-embedded matrix rotates two columns while preserving Frobenius norm:

$$
W' = WR, \qquad \|W'\|_F = \|W\|_F.
$$

Current tests verify:

- applying a rotation and its inverse reconstructs the original matrix;
- rotations preserve Frobenius norm;
- the analytic angle $\theta=\arctan2(b,a)$ can zero a selected entry;
- cascaded rotations can implement Givens QR;
- a Jacobi-style angle can orthogonalize a column pair.

These tests turn implementation details into reusable research facts.

### Per-Channel Scaling

Per-channel scaling computes one positive factor per column. Let

$$
m_j = \max_i |W_{ij}|.
$$

For a target max-absolute value $\tau$, nonzero columns receive

$$
d_j = \frac{\tau}{m_j}.
$$

The scaled matrix is

$$
W' = WD,
$$

where $D=\mathrm{diag}(d_1,\dots,d_n)$. The inverse transform is

$$
W = W'D^{-1}.
$$

Tests verify that scaling is reversible up to floating-point rounding, that zero columns receive identity factors, and that column max-absolute imbalance is reduced.

## 8. Global Scaling vs Channel-Wise Scaling

The channel-scaling dashboard compares:

1. global INT4 quantization;
2. per-channel scaled INT4 quantization, followed by inverse scaling.

Figure:

![Global vs channel-scaled quantization](figures/channel_scaling_dashboard.png)

This figure is designed to answer a specific question: does balancing column magnitudes before quantization reduce error in columns that would otherwise be poorly represented by a single global scale?

The current example shows that channel scaling can dramatically lower zero fraction and per-column error on an imbalanced matrix. However, this is still a controlled example. It motivates broader experiments rather than closing the question.

## 9. Rotation and Scaling Experiment

The first Milestone 2 experiment compares four INT4 paths on one controlled outlier-heavy matrix.

### Paths

Baseline:

$$
W \rightarrow \mathrm{INT4} \rightarrow \hat{W}.
$$

Rotation only:

$$
W \rightarrow WR \rightarrow \mathrm{INT4} \rightarrow \widehat{WR} \rightarrow \widehat{W}=\widehat{WR}R^{-1}.
$$

Scaling only:

$$
W \rightarrow WD \rightarrow \mathrm{INT4} \rightarrow \widehat{WD} \rightarrow \widehat{W}=\widehat{WD}D^{-1}.
$$

Rotation followed by scaling:

$$
W \rightarrow WR \rightarrow WRD \rightarrow \mathrm{INT4} \rightarrow \widehat{WRD} \rightarrow \widehat{W}=\widehat{WRD}D^{-1}R^{-1}.
$$

Figure:

![Rotation and scaling INT4 comparison](figures/rotation_scaling_comparison.png)

Current default-run metrics:

| Method | MSE | Rel. Frobenius | SNR dB | Zero frac | Sat. frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.3756 | 0.3157 | 10.014 | 0.703125 | 0.000488 |
| Rotation only | 0.3195 | 0.2912 | 10.717 | 0.668457 | 0.001953 |
| Scaling only | 0.1845 | 0.2213 | 13.101 | 0.479980 | 0.022217 |
| Rotation + scaling | 0.1786 | 0.2177 | 13.244 | 0.476318 | 0.022705 |

In this single run, rotation + scaling is best among the four paths on MSE, relative Frobenius error, SNR, and zero fraction. The interpretation should be cautious. The improvement over scaling alone is modest, while the improvement from baseline to scaling is large. The current evidence therefore supports the narrower claim:

> Per-channel scaling substantially improves this controlled INT4 example, and rotation + scaling gives a small additional improvement in this run.

It does not yet justify the broad claim that rotation + scaling is always the best strategy.

## 10. Grouped Quantization

### 10.1 Column-Grouped Quantization

The first grouped strategy splits the matrix into contiguous **column blocks**, each receiving its own scale. For a column-group $W_g$ and bitwidth $b$,

$$
s_g = \frac{\max |W_g|}{2^{b-1}-1},
$$

with codes

$$
Q_g = \mathrm{clip}\left(\mathrm{round}(W_g / s_g), q_{\min}, q_{\max}\right),
$$

and reconstruction $\hat{W}_g = s_g Q_g$. When one column contains an extreme value, only that group's scale is inflated; all other column groups retain tight scales.

On the column-outlier example from Section 5, column-grouped INT4 improves over global INT4, and per-column grouping improves substantially more:

| Method | MSE | Rel. Frobenius | SNR dB | Zero frac |
| --- | ---: | ---: | ---: | ---: |
| Global INT4 | 0.297624 | 0.226335 | 12.905 | 0.664062 |
| Column-grouped INT4 (group=4 cols) | 0.275787 | 0.217873 | 13.236 | 0.652344 |
| Column-grouped INT4 (group=1 col) | 0.089124 | 0.123855 | 18.142 | 0.339844 |

### 10.2 Row-Grouped Quantization

Column-grouped quantization has a blind spot: if an outlier appears in a **single row** that spans many columns, every column group is affected and the scale improvements vanish. The standard industry approach — used in GPTQ and AWQ — addresses this by grouping **rows within each column**. Each column is split independently into row groups of size $g$, giving one scale per group per column:

$$
s_{c,k} = \frac{\max_{i \in \text{group } k} |W_{i,c}|}{2^{b-1}-1}.
$$

This yields $n_{\mathrm{cols}} \times \lceil n_{\mathrm{rows}} / g \rceil$ scales in total. An outlier confined to a row group inflates only that group's scale; all other row groups in all columns are unaffected.

### 10.3 Column-Grouped vs Row-Grouped: A Critical Comparison

The distinction becomes concrete when a row-level outlier is present. The following example uses a $16 \times 16$ Gaussian matrix (seed 42) with the entire first row set to 30.0 (approximately 30× the background standard deviation).

| Method | MSE | Rel. Frobenius | SNR dB | Zero frac |
| --- | ---: | ---: | ---: | ---: |
| Global INT4 | 0.769182 | 0.116081 | 18.705 | 0.921875 |
| Column-grouped INT4 (group=4 cols) | 0.769182 | 0.116081 | 18.705 | 0.921875 |
| Column-grouped INT4 (group=1 col) | 0.769182 | 0.116081 | 18.705 | 0.921875 |
| Row-grouped INT4 (group=4 rows) | 0.109244 | 0.043747 | 27.181 | 0.226562 |
| Row-grouped INT4 (group=1 row) | 0.000000 | 0.000000 | ∞ | 0.000000 |

Column-grouped quantization provides **no improvement at any group size**: because the outlier row is present in every column group, every group's scale is dominated by it regardless of how the columns are partitioned. Row-grouped quantization at group size 4 reduces MSE by 7× and zero fraction from 92% to 23%. At group size 1 (one scale per element per column), reconstruction is exact up to floating-point rounding.

This demonstrates that the choice of grouping axis matters as much as the number of groups. When outliers are row-localised, only row-grouped quantization addresses the root cause.

### 10.4 Implications for the Comparison Landscape

Grouped quantization expands the set of quantization paths that should be compared:

$$
\text{global INT4},\quad \text{column-grouped INT4},\quad \text{row-grouped INT4},\quad
\text{rotation/scaling + any of the above}.
$$

## 11. Current Findings

The project has produced the following working findings.

1. INT4 is much more sensitive than INT8 to outliers under global symmetric quantization.
2. Zero fraction is an important diagnostic because it reveals collapse toward the zero code.
3. Spectrum plots reveal whether reconstruction preserves the global geometry of the matrix, not only entrywise values.
4. Givens rotations preserve matrix energy and can redistribute outlier pressure.
5. Per-channel scaling is reversible and directly reduces column magnitude imbalance.
6. In the first rotation/scaling experiment, scaling explains most of the observed improvement, while rotation + scaling is the best path but only by a small margin over scaling alone.
7. Column-grouped quantization improves over global INT4 when outliers are column-localised, but provides no benefit when outliers are row-localised, because the outlier row spans every column group regardless of group size.
8. Row-grouped quantization (one scale per row-group per column, the GPTQ/AWQ approach) directly addresses row-localised outliers. In a controlled row-outlier example, row-grouping with group size 4 reduces MSE by 7× and zero fraction from 92% to 23% compared with global INT4, while column-grouped quantization leaves both metrics unchanged at any group size.
9. The comparative sweep (5 seeds × 3 outlier fractions × 3 outlier scales = 45 conditions, 12 methods each) produces the following mean MSE ratios relative to global INT4, averaged across all conditions:

| Method | Mean MSE ratio | Mean zero fraction |
|---|---|---|
| rotate_scale_row_g4 | **0.111** | 0.137 |
| row_grouped_g4 | **0.112** | 0.136 |
| rotate_scale_row_g8 | 0.216 | 0.219 |
| row_grouped_g8 | 0.219 | 0.219 |
| rotate_scale_row_g16 | 0.353 | 0.311 |
| row_grouped_g16 | 0.362 | 0.312 |
| rotate_scale_global | 0.507 | 0.402 |
| scale_global | 0.531 | 0.410 |
| col_grouped_g4 | 0.766 | 0.519 |
| col_grouped_g8 | 0.875 | 0.560 |
| rotate_global | 0.902 | 0.570 |
| global | 1.000 | 0.593 |

Key observations from the sweep: (a) row_grouped_g4 achieves ~9× MSE reduction on average; (b) rotation adds only marginal benefit on top of row-grouped alone (0.111 vs 0.112 at g=4) — the value of rotation is realised through its combination with scaling; (c) scale_global (0.53×) outperforms column-grouped at any group size; (d) rotation alone (0.90×) barely moves the needle; (e) group size is the dominant variable for row-grouped — g=4 gives 9× improvement, g=16 gives only 3×.

## 12. Limitations

The current results are intentionally preliminary.

- Matrices are synthetic and small.
- Rotation-pair selection is simple: the two columns with largest max-abs values.
- Scaling currently balances full-column max-absolute values, not groups or learned activation-aware statistics.
- The sweep covers outlier fractions up to 10% and outlier scales up to 20×; extreme conditions may shift the relative ordering of methods.
- No transformer-layer or language-model benchmark has been run yet.

These limitations are useful: they define the next experiments rather than weakening the value of the sandbox.

## 13. Next Work

The next research steps move from matrix-level evidence to transformer-level evidence.

1. Apply the best-performing pipeline (rotation + scaling + row-grouped INT4) to weight matrices from a tiny transformer (tiny-gpt2 or DistilGPT2).
2. Measure perplexity and activation drift before and after quantization across methods.
3. Compare rotation-pair selection strategies (max-abs pair vs. Jacobi-sweep vs. learned).
4. Scale to larger open-source LLMs and compare against GPTQ and AWQ published results.

## Appendix A. Reproducing Current Figures

Core experiment commands:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/rotation_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/sweep_experiment.py
```

Current tracked figure references used in this draft:

- `docs/figures/research_matrix_families.png`
- `docs/figures/research_int4_gaussian_comparison.png`
- `docs/figures/research_outlier_histograms.png`
- `docs/figures/analysis_dashboard.png`
- `docs/figures/channel_scaling_dashboard.png`
- `docs/figures/rotation_scaling_comparison.png`

Generated experiment outputs under `plots/` and `results/` remain local ignored artifacts. Paper figures are copied into `docs/figures/` when they are ready to be referenced by the tracked draft.

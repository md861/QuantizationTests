# Quantization Lab: A Matrix-Level Study of Low-Bit Quantization, Outlier Pressure, and Preprocessing Transformations

Status: living draft. This document should be updated as experiments, figures, and conclusions mature.

## Abstract

This project studies low-bit quantization from a deliberately small and inspectable starting point: synthetic matrices. The central question is how symmetric low-bit quantization changes matrix values, reconstruction error, singular-value structure, and integer-code usage under different distributional conditions. We first build a matrix-level quantization sandbox with Gaussian, heavy-tailed, and outlier-injected matrices. We then compare INT8 and INT4 symmetric quantization using reconstruction metrics, spectrum diagnostics, residual plots, histograms, and benchmark-style dashboards. Finally, we begin testing ParoQuant-inspired preprocessing transformations, specifically pairwise Givens rotations and reversible per-channel scaling, to reduce outlier pressure before INT4 quantization.

The current evidence shows that INT4 quantization is highly sensitive to outlier-dominated scales, often producing elevated reconstruction error and high zero-code fractions. Per-channel scaling substantially reduces this failure mode in our first controlled examples. Pairwise rotation alone helps modestly in the current single-matrix experiment, while rotation followed by scaling gives the best result among the four tested paths on that example. We treat this as a promising observation, not yet a general claim, because broader sweeps over seeds, outlier strengths, and transformation strategies are still needed.

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

Grouped quantization is now implemented as an intermediate path between full-matrix global quantization and fully per-column scaling. Instead of one scale for the whole matrix, contiguous column groups receive separate symmetric scales.

For a group $W_g$ and bitwidth $b$,

$$
s_g = \frac{\max |W_g|}{2^{b-1}-1},
$$

with integer codes

$$
Q_g = \mathrm{clip}\left(\mathrm{round}(W_g / s_g), q_{\min}, q_{\max}\right),
$$

and reconstruction

$$
\hat{W}_g = s_g Q_g.
$$

This matters because grouped quantization is closer to practical low-bit quantization than a single full-matrix scale. It can reduce outlier pressure when outliers are localized to a small subset of columns.

On the same outlier example used earlier, grouped INT4 improves over global INT4, and per-column grouping improves much more:

| Method | MSE | Rel. Frobenius | SNR dB | Zero frac | Sat. frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global INT4 | 0.297624 | 0.226335 | 12.905 | 0.664062 | 0.015625 |
| Grouped INT4 (group=4) | 0.275787 | 0.217873 | 13.236 | 0.652344 | 0.023438 |
| Column INT4 (group=1) | 0.089124 | 0.123855 | 18.142 | 0.339844 | 0.078125 |

The saturation fraction rises as groups become smaller because each group can use its own local range more aggressively. This is not necessarily bad, but it means grouped quantization must be evaluated using multiple diagnostics rather than MSE alone.

Grouped quantization changes the next research question. The fairer comparison is no longer only:

$$
\text{global INT4} \quad \text{vs.} \quad \text{rotation/scaling + global INT4}.
$$

It should become:

$$
\text{global INT4},\quad \text{grouped INT4},\quad
\text{scaling + global/grouped INT4},\quad
\text{rotation + scaling + global/grouped INT4}.
$$

## 11. Current Findings

The project has produced the following working findings.

1. INT4 is much more sensitive than INT8 to outliers under global symmetric quantization.
2. Zero fraction is an important diagnostic because it reveals collapse toward the zero code.
3. Spectrum plots reveal whether reconstruction preserves the global geometry of the matrix, not only entrywise values.
4. Givens rotations preserve matrix energy and can redistribute outlier pressure.
5. Per-channel scaling is reversible and directly reduces column magnitude imbalance.
6. In the first rotation/scaling experiment, scaling explains most of the observed improvement, while rotation + scaling is the best path but only by a small margin over scaling alone.
7. Grouped quantization improves over global INT4 in the first outlier example, especially when groups are small, but it must be compared carefully because saturation behavior also changes.

## 12. Limitations

The current results are intentionally preliminary.

- Matrices are synthetic and small.
- Most examples use a single seed or a small number of conditions.
- Rotation-pair selection is simple: the two columns with largest max-abs values.
- Scaling currently balances full-column max-absolute values, not groups or learned activation-aware statistics.
- The current grouped quantizer uses simple contiguous column groups and has not yet been swept across group sizes or combined with all preprocessing paths.
- No transformer-layer or language-model benchmark has been run yet.

These limitations are useful: they define the next experiments rather than weakening the value of the sandbox.

## 13. Next Work

The next research steps should turn isolated examples into evidence.

1. Run rotation/scaling sweeps over seeds, outlier fractions, outlier scales, and matrix shapes.
2. Report average improvement and win rate for each method.
3. Sweep grouped quantization over group sizes and compare it against rotation/scaling paths.
4. Test whether rotation/scaling still improves grouped INT4.
5. Compare rotation-pair selection strategies.
6. Start applying the same metrics to small transformer weight matrices.

## Appendix A. Reproducing Current Figures

Core experiment commands:

```bash
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/analyze_results.py
MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/rotation_experiment.py
```

Current tracked figure references used in this draft:

- `docs/figures/research_matrix_families.png`
- `docs/figures/research_int4_gaussian_comparison.png`
- `docs/figures/research_outlier_histograms.png`
- `docs/figures/analysis_dashboard.png`
- `docs/figures/channel_scaling_dashboard.png`
- `docs/figures/rotation_scaling_comparison.png`

Generated experiment outputs under `plots/` and `results/` remain local ignored artifacts. Paper figures are copied into `docs/figures/` when they are ready to be referenced by the tracked draft.

# ParoQuant Lab

ParoQuant Lab is a research-oriented educational implementation of ParoQuant-style quantization ideas for large language models.

The project starts with a matrix-level sandbox for understanding quantization behavior before scaling toward transformer models. The current focus is Milestone 1: generating synthetic matrices, applying symmetric INT8 and INT4 quantization, measuring reconstruction error, and visualizing distortion.

## Current Milestone

Milestone 1 builds the quantization sandbox:

- matrix generation for Gaussian, heavy-tailed, and outlier-heavy data
- symmetric INT8 and INT4 quantization
- reconstruction metrics such as MSE, MAE, cosine similarity, relative Frobenius error, and SNR
- histograms and heatmaps for inspecting quantization effects

This repository prioritizes clarity, modularity, reproducibility, and mathematical transparency over production inference performance.

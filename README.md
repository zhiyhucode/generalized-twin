# Generalized Twin Image
Code to reproduce the results of the paper

> Z. Hu, M. Unser and J. Dong, "Explaining the Effects of Regularization on Phase Retrieval with the Generalized Twin Image," submitted to the IEEE International Symposium on Biomedical Imaging (ISBI) 2027.

Structured random models for phase retrieval measure $y = |A x|^2$ with $A = S F P O$, a cascade of Fourier transforms and random diagonal masks of depth $L$.
Below the sampling ratio $\alpha = 1$, every such model has an exact ambiguity, the generalized twin $x' = P^{-1} T(P x)$, where $T$ is the conjugate reflection.
The ratio of the total variation of the twin to that of the signal predicts how well an image prior recovers the signal at each depth.

## Installation
The environment is managed with [uv](https://docs.astral.sh/uv/).
`uv.lock` pins every package, including PyTorch 2.14 with CUDA 13.0 on Linux and the deepinv commit the paper used.

```bash
git clone https://github.com/zhiyhucode/generalized-twin.git
cd generalized-twin
uv sync
```

## Usage
Each result of the paper has one command:

```bash
uv run python run.py --table1
uv run python run.py --fig1
uv run python run.py --fig2
```

| Command | Computes | Output in `results/` | Hardware |
|---|---|---|---|
| `--table1` | twins of the 68 CBSD68 images at their original size, 10 masks each, depths 1 to 2.5 | `table1.md`, `table1_runs.csv` | CPU, 2 min on 8 cores |
| `--fig1` | 50 regularized reconstructions of the cameraman at $\alpha = 0.5$, depths 0.5 to 2.5 | `fig1.pdf`, `fig1.png` | one GPU, 20 min on a B300 |
| `--fig2` | 4200 pairs of reconstructions with and without regularization: 4 depths, 21 sampling ratios, 50 masks | `fig2.pdf`, `fig2.png`, `fig2_summary.csv` | GPUs, 30 GPU hours on B300 (26 s per pair) |

Every reconstruction is saved under `results/fig1` or `results/fig2` as soon as it finishes, and a rerun skips the saved ones.
A figure is drawn once all its reconstructions exist.
To split a computation over several GPUs, start one process per GPU with the same `--n-shards` and a different `--shard`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python run.py --fig2 --shard 0 --n-shards 4
CUDA_VISIBLE_DEVICES=1 uv run python run.py --fig2 --shard 1 --n-shards 4
```

Then run `uv run python run.py --fig2` once more to draw the figure.
Other options: `--device` (default `cuda` when available) and `--out` (default `results`).

## Method
- **Signal.** The cameraman of deepinv at 256 × 256, mapped from $[0, 1]$ to the phase $[0, \pi]$ of a unit-magnitude signal.
- **Forward model.** Unitary centered 2-D FFTs and Rademacher masks; centered zero-padding above $\alpha = 1$ and centered trimming below; noiseless measurements. The output side is `round(256 * sqrt(alpha))`.
- **Generalized twin.** $T(z)[k] = \overline{z}[-k-1]$ with $k$ counted from the zero frequency of the centered grid.
- **Reconstruction.** A spectral initialization (5000 power iterations, early stop at relative change $10^{-5}$), then 6000 iterations. With regularization, each iteration applies the pretrained DRUNet of deepinv to the real and imaginary parts mapped from $[-1, 1]$ to $[0, 1]$, at the noise level $\sigma_t = 0.99943^t$, with the eight flips and rotations cycled, followed by the exact measurement projection $x \leftarrow x + A^H(\sqrt{y}\, e^{j \angle A x} - A x)$. Without regularization, each iteration applies the projection only.
- **Metric.** The PSNR of the phase, $20 \log_{10}(\pi / \mathrm{RMSE})$, of the wrapped phase error after global phase alignment. A run above 100 dB is drawn as exact.

## Data
Everything is downloaded on first use:
- the cameraman image and the DRUNet weights, through deepinv;
- CBSD68, through deepinv from the Hugging Face hub, into `data/CBSD68`.

## Reproducibility
The masks and the random start of the spectral method are seeded per run.
GPU results depend on the GPU architecture and the CUDA libraries.
The paper's reconstructions ran with this environment on NVIDIA B300 GPUs.
On other GPUs, the PSNR of single runs can differ, mainly at $L = 1.5$, where the outcome of a run is bimodal.
Table 1 runs on the CPU.

## License
The code is released under the BSD 3-Clause License, see `LICENSE`.
The TeX Gyre Termes fonts in `assets/fonts` are distributed under the GUST Font License, see `assets/fonts/GUST-FONT-LICENSE.txt`.

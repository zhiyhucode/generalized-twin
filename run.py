import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from matplotlib.layout_engine import ConstrainedLayoutEngine

ROOT = Path(__file__).resolve().parent
IMG_SIZE = 256
COLUMN_WIDTH = 3.5
SIGMA0 = 1.0
SIGMA_MULT = 0.99943
ITERS = 6000
SPECTRAL_ITERS = 5000
FIG1_DEPTHS = (0.5, 1.0, 1.5, 2.0, 2.5)
FIG1_RATIO = 0.5
FIG1_SEEDS = 10
FIG2_DEPTHS = (0.5, 1.0, 1.5, 2.0)
FIG2_RATIOS = (0.1,) + tuple(round(0.2 * k, 1) for k in range(1, 21))
FIG2_SEEDS = 50
TABLE1_DEPTHS = (1.0, 1.5, 2.0, 2.5)
TABLE1_SEEDS = 10
RAYLEIGH = math.sqrt(math.pi) / 2
EXACT_DB = 100.0
EXACT_ROW = 44.5
PHASE = dict(cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="none")


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), norm="ortho"))


def ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x), norm="ortho"))


def pad(x, size):
    dh, dw = size[0] - x.shape[-2], size[1] - x.shape[-1]
    return torch.nn.functional.pad(x, (math.ceil(dw / 2), dw // 2, math.ceil(dh / 2), dh // 2))


def crop(x, size):
    top = math.ceil((x.shape[-2] - size[0]) / 2)
    left = math.ceil((x.shape[-1] - size[1]) / 2)
    return x[..., top:top + size[0], left:left + size[1]]


class StructuredRandomModel:
    def __init__(self, depth, img_shape, out_shape, device):
        self.n_diagonals = math.floor(depth)
        self.half = depth - self.n_diagonals == 0.5
        self.img_shape = tuple(img_shape)
        self.out_shape = tuple(out_shape)
        oversampled = self.out_shape[0] > self.img_shape[0] and self.out_shape[1] > self.img_shape[1]
        self.mid_shape = self.out_shape if oversampled else self.img_shape
        signs = torch.tensor([1, -1])
        self.diagonals = [
            signs[torch.randint(0, 2, (1, *self.mid_shape))].to(torch.complex64).to(device)
            for _ in range(self.n_diagonals)
        ]

    def kernel(self, z):
        if self.half:
            z = fft2c(z)
        for d in self.diagonals:
            z = fft2c(d * z)
        return z

    def kernel_adjoint(self, w):
        for d in reversed(self.diagonals):
            w = ifft2c(w) * d.conj()
        if self.half:
            w = ifft2c(w)
        return w

    def linear(self, x):
        return crop(self.kernel(pad(x, self.mid_shape)), self.out_shape)

    def adjoint(self, y):
        return crop(self.kernel_adjoint(pad(y, self.mid_shape)), self.img_shape)

    def __call__(self, x):
        return self.linear(x).abs().square()

    def before_last(self, z):
        if self.half and self.n_diagonals >= 1:
            z = fft2c(z)
        for i, d in enumerate(self.diagonals):
            z = d * z
            if i < self.n_diagonals - 1:
                z = fft2c(z)
        return z

    def before_last_adjoint(self, w):
        for i in reversed(range(self.n_diagonals)):
            if i < self.n_diagonals - 1:
                w = ifft2c(w)
            w = w * self.diagonals[i].conj()
        if self.half and self.n_diagonals >= 1:
            w = ifft2c(w)
        return w

    def twin(self, x):
        w = self.before_last(pad(x, self.mid_shape))
        t = torch.flip(w.conj(), dims=(-2, -1))
        shifts = tuple(-1 if n % 2 else 0 for n in w.shape[-2:])
        if any(shifts):
            t = torch.roll(t, shifts=shifts, dims=(-2, -1))
        x_twin = crop(self.before_last_adjoint(t), self.img_shape)
        error = ((self(x_twin) - self(x)).norm() / self(x).norm()).item()
        return x_twin, error


def side_for_ratio(ratio):
    return int(round(IMG_SIZE * math.sqrt(ratio)))


def make_model(depth, ratio, seed, device):
    side = side_for_ratio(ratio)
    torch.manual_seed(seed)
    return StructuredRandomModel(depth, (IMG_SIZE, IMG_SIZE), (side, side), device)


def load_cameraman(device):
    from deepinv.utils import load_example

    img = load_example("cameraman.png", img_size=IMG_SIZE, grayscale=True, resize_mode="resize", device=device,
                       dtype=torch.float32)
    return torch.exp(1j * math.pi * img)


def load_denoiser(device):
    from deepinv.models import DRUNet

    return DRUNet(in_channels=1, out_channels=1, device=device)


def spectral_initialization(y, model, x):
    norm = torch.sqrt(y.sum())
    y = y / torch.mean(y)
    weights = torch.max(1 - 1 / y, torch.tensor(-5.0)).to(torch.cfloat)
    for _ in range(SPECTRAL_ITERS):
        x_new = model.adjoint(weights * model.linear(x)) + 10.0 * x
        x_new = x_new / torch.linalg.norm(x_new)
        if torch.linalg.norm(x_new - x) / torch.linalg.norm(x) < 1e-5:
            break
        x = x_new
    return x * norm


def project(x, y, model):
    ax = model.linear(x)
    return model.adjoint(y.sqrt() * torch.exp(1j * ax.angle()) - ax) + x


def augment(x, t):
    k = t % 8
    if k == 1:
        return torch.flip(x, [2])
    if k == 2:
        return torch.flip(x, [3])
    if k == 3:
        return torch.rot90(x, 1, [2, 3])
    if k == 4:
        return torch.rot90(x, 2, [2, 3])
    if k == 5:
        return torch.rot90(x, 3, [2, 3])
    if k == 6:
        return torch.flip(torch.rot90(x, 1, [2, 3]), [2])
    if k == 7:
        return torch.flip(torch.rot90(x, 1, [2, 3]), [3])
    return x


def deaugment(x, t):
    k = t % 8
    if k == 1:
        return torch.flip(x, [2])
    if k == 2:
        return torch.flip(x, [3])
    if k == 3:
        return torch.rot90(x, 3, [2, 3])
    if k == 4:
        return torch.rot90(x, 2, [2, 3])
    if k == 5:
        return torch.rot90(x, 1, [2, 3])
    if k == 6:
        return torch.rot90(torch.flip(x, [2]), 3, [2, 3])
    if k == 7:
        return torch.rot90(torch.flip(x, [3]), 3, [2, 3])
    return x


def denoise(denoiser, x, sigma):
    parts = torch.cat([x.real / 2.0 + 0.5, x.imag / 2.0 + 0.5], dim=0).clamp(0.0, 1.0)
    d = denoiser(parts, sigma=sigma).clamp(0.0, 1.0)
    return (d[0:1] - 0.5) * 2.0 + 1j * ((d[1:2] - 0.5) * 2.0)


def reconstruct_regularized(x0, y, model, denoiser):
    x = x0.clone()
    sigma = SIGMA0
    for t in range(ITERS):
        x = deaugment(denoise(denoiser, augment(x, t), sigma), t)
        x = project(x, y, model)
        sigma = sigma * SIGMA_MULT
    return x


def reconstruct_unregularized(x0, y, model):
    x = x0.clone()
    for _ in range(ITERS):
        x = project(x, y, model)
    return x


def phase_psnr(x_true, x_rec):
    d = torch.exp(1j * (torch.angle(x_rec) - torch.angle(x_true)))
    theta = torch.angle(d.mean())
    err = torch.angle(d * torch.exp(-1j * theta))
    rmse = torch.sqrt((err ** 2).mean()).item()
    return 20 * math.log10(math.pi / rmse) if rmse > 0 else float("inf")


def spectral_start(x, depth, ratio, seed, device):
    model = make_model(depth, ratio, seed, device)
    y = model(x)
    torch.manual_seed(seed)
    x0 = torch.randn((1, 1, IMG_SIZE, IMG_SIZE), dtype=torch.complex64, device=device)
    return model, y, spectral_initialization(y, model, x0)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def shard_of(items, shard, n_shards):
    return [item for i, item in enumerate(items) if i % n_shards == shard]


def use_paper_style(font_size=8.0):
    files = sorted((ROOT / "assets/fonts").glob("texgyretermes-*.otf"))
    for f in files:
        font_manager.fontManager.addfont(str(f))
    matplotlib.rcParams.update({
        "font.family": "TeX Gyre Termes" if files else "STIXGeneral",
        "font.size": font_size,
        "axes.titlesize": "medium",
        "figure.labelsize": "medium",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
        "savefig.pad_inches": 0.02,
    })


def column_colorbar(fig, mappable, axes, margin_in, ticks):
    axs = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    w_fig = fig.get_size_inches()[0]
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        rect = list(engine.get().get("rect", (0, 0, 1, 1)))
        engine.set(rect=(rect[0], rect[1], min(rect[2], 1 - margin_in / w_fig), rect[3]))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    boxes = [ax.get_position() for ax in axs]
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    cax = fig.add_axes([x1 + 0.06 / w_fig, y0, 0.08 / w_fig, y1 - y0])
    return fig.colorbar(mappable, cax=cax, ticks=ticks)


def total_variation(v):
    return ((torch.roll(v, -1, -2) - v).abs().mean() + (torch.roll(v, -1, -1) - v).abs().mean()).item() / 2


def aligned_phase(x, x_true):
    d = torch.exp(1j * (torch.angle(x) - torch.angle(x_true)))
    return torch.angle(x * torch.exp(-1j * torch.angle(d.mean()))).squeeze().numpy()


def fig1_run_path(out, depth, seed):
    return out / "fig1" / f"L{depth:g}_s{seed:02d}.pt"


def fig1(args, out):
    jobs = [(depth, seed) for depth in FIG1_DEPTHS for seed in range(FIG1_SEEDS)]
    todo = [job for job in shard_of(jobs, args.shard, args.n_shards)
            if not fig1_run_path(out, *job).exists()]
    if todo:
        x = load_cameraman(args.device)
        denoiser = load_denoiser(args.device)
        for depth, seed in todo:
            model, y, x_sm = spectral_start(x, depth, FIG1_RATIO, seed, args.device)
            x_rec = reconstruct_regularized(x_sm, y, model, denoiser)
            path = fig1_run_path(out, depth, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(x_rec.cpu(), path)
            print(f"Fig. 1: L = {depth:g}, seed {seed}: {phase_psnr(x, x_rec):.2f} dB", flush=True)
    missing = [job for job in jobs if not fig1_run_path(out, *job).exists()]
    if missing:
        print(f"Fig. 1: {len(missing)} of {len(jobs)} runs missing, figure not drawn yet")
        return
    plot_fig1(out)


def plot_fig1(out):
    x = load_cameraman("cpu")
    tv_x = total_variation(x)
    shown = {}
    for depth in FIG1_DEPTHS:
        rows = []
        for seed in range(FIG1_SEEDS):
            x_rec = torch.load(fig1_run_path(out, depth, seed), weights_only=True)
            x_twin, error = make_model(depth, FIG1_RATIO, seed, "cpu").twin(x)
            rows.append(dict(seed=seed, psnr=phase_psnr(x, x_rec), x_rec=x_rec, x_twin=x_twin, error=error,
                             tv_ratio=total_variation(x_twin) / tv_x))
        psnr = np.array([r["psnr"] for r in rows])
        shown[depth] = rows[int(np.argmin(np.abs(psnr - np.median(psnr))))]
        ratios = [r["tv_ratio"] for r in rows]
        print(f"L = {depth:<4g} TV ratio {np.mean(ratios):5.2f} ({min(ratios):.2f} to {max(ratios):.2f}), "
              f"twin identity error {max(r['error'] for r in rows):.1e}, PSNR {psnr.mean():5.1f} dB "
              f"({psnr.min():.1f} to {psnr.max():.1f}), {int((psnr > 30).sum())} of {FIG1_SEEDS} above 30 dB, "
              f"shown seed {shown[depth]['seed']}")
    use_paper_style(font_size=7.0)
    n = len(FIG1_DEPTHS)
    fig = plt.figure(figsize=(COLUMN_WIDTH, 1.5), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.0, hspace=0.0, wspace=0.0)
    grid = fig.add_gridspec(4, n + 1)
    ax_signal = fig.add_subplot(grid[1:3, 0])
    axes = np.array([[fig.add_subplot(grid[2 * i:2 * i + 2, j + 1]) for j in range(n)] for i in range(2)])
    ax_signal.imshow(x.angle().squeeze().numpy(), **PHASE)
    ax_signal.set_title(r"Signal $\mathbf{x}$", pad=2)
    axes[0, 0].set_ylabel(r"Gen. twin $\mathbf{x}'$", labelpad=2)
    axes[1, 0].set_ylabel(r"Recon. $\hat{\mathbf{x}}$", labelpad=2)
    for j, depth in enumerate(FIG1_DEPTHS):
        s = shown[depth]
        im = axes[0, j].imshow(s["x_twin"].angle().squeeze().numpy(), **PHASE)
        axes[0, j].set_title(f"$L = {depth:g}$\n$\\rho_\\mathrm{{TV}}{{=}}{s['tv_ratio']:.1f}$", pad=2)
        axes[1, j].imshow(aligned_phase(s["x_rec"], x), **PHASE)
        axes[1, j].set_title(f"{s['psnr']:.1f} dB", pad=2)
    for ax in list(axes.flat) + [ax_signal]:
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = column_colorbar(fig, im, axes, margin_in=0.34, ticks=[-np.pi, 0, np.pi])
    cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    pdf = out / "fig1.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Fig. 1: {pdf}")


def fig2_run_path(out, depth, ratio, seed):
    return out / "fig2" / f"L{depth:g}" / f"a{ratio:.1f}_s{seed:02d}.json"


def fig2(args, out):
    jobs = [(depth, ratio, seed) for depth in FIG2_DEPTHS for ratio in FIG2_RATIOS for seed in range(FIG2_SEEDS)]
    todo = [job for job in shard_of(jobs, args.shard, args.n_shards) if not fig2_run_path(out, *job).exists()]
    if todo:
        x = load_cameraman(args.device)
        denoiser = load_denoiser(args.device)
        for depth, ratio, seed in todo:
            model, y, x_sm = spectral_start(x, depth, ratio, seed, args.device)
            psnr_noreg = phase_psnr(x, reconstruct_unregularized(x_sm, y, model))
            psnr_reg = phase_psnr(x, reconstruct_regularized(x_sm, y, model, denoiser))
            realized = (side_for_ratio(ratio) / IMG_SIZE) ** 2
            write_json(fig2_run_path(out, depth, ratio, seed),
                       dict(depth=depth, ratio=ratio, realized_ratio=realized, seed=seed, psnr_reg=psnr_reg,
                            psnr_noreg=psnr_noreg))
            print(f"Fig. 2: L = {depth:g}, ratio {ratio:g}, seed {seed}: {psnr_reg:.2f} dB with regularization, "
                  f"{psnr_noreg:.2f} dB without", flush=True)
    missing = [job for job in jobs if not fig2_run_path(out, *job).exists()]
    if missing:
        print(f"Fig. 2: {len(missing)} of {len(jobs)} runs missing, figure not drawn yet")
        return
    plot_fig2(out)


def plot_fig2(out):
    use_paper_style()
    color = {"reg": "C0", "noreg": "C3"}
    marker = {0.5: "s", 1.0: "o", 1.5: "d", 2.0: "^"}
    markersize = {0.5: 1.3, 1.0: 1.5, 1.5: 1.7, 2.0: 1.8}
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH, 3.0), sharex=True, sharey=True)
    for ax, depth in zip(axes.flat, FIG2_DEPTHS):
        runs = [[json.loads(fig2_run_path(out, depth, ratio, seed).read_text()) for seed in range(FIG2_SEEDS)]
                for ratio in FIG2_RATIOS]
        x = np.array([r[0]["realized_ratio"] for r in runs])
        for branch in ("reg", "noreg"):
            values = np.array([[run[f"psnr_{branch}"] for run in r] for r in runs])
            placed = np.where(values >= EXACT_DB, EXACT_ROW, values)
            line = np.percentile(placed, 50, axis=1, method="lower")
            ax.plot(x, line, linestyle="-", marker=marker[depth], markersize=markersize[depth], color=color[branch],
                    linewidth=0.7, zorder=2)
            keep = [i for i, col in enumerate(placed) if col.std() > 0.05]
            if keep:
                parts = ax.violinplot([placed[i] for i in keep], positions=x[keep], widths=0.16, showextrema=False,
                                      showmedians=False, points=200, bw_method=0.2)
                for body in parts["bodies"]:
                    body.set(facecolor=color[branch], edgecolor="none", alpha=0.3, zorder=1)
            rows += [dict(depth=depth, branch=branch, ratio=ratio, median=m, above_30=int((v >= 30).sum()),
                          exact=int((v >= EXACT_DB).sum())) for ratio, m, v in zip(FIG2_RATIOS, line, values)]
        ax.axhline(42.6, color="0.8", linewidth=0.5, zorder=1)
        blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for y0 in (42.1, 43.1):
            ax.plot([-0.012, 0.012], [y0 - 0.35, y0 + 0.35], transform=blend, color="black", linewidth=0.6,
                    clip_on=False)
        ax.text(0.04, 0.95, f"$L = {depth:g}$", transform=ax.transAxes, fontsize=8, va="top")
        ax.set_xlim(0, 4.1)
        ax.set_xticks(range(5))
        ax.set_ylim(0, 46)
        ax.set_yticks([0, 10, 20, 30, 40, EXACT_ROW], ["0", "10", "20", "30", "40", "exact"])
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"Sampling Ratio ($\alpha$)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("PSNR [dB]", fontsize=8)
    handles = [plt.Line2D([], [], color=color[b], linewidth=0.7, label=label)
               for label, b in (("with regularization", "reg"), ("without", "noreg"))]
    axes[0, 1].legend(handles=handles, fontsize=7, loc="center right", frameon=True, borderpad=0.4, handlelength=1.6)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.4)
    pdf = out / "fig2.pdf"
    fig.savefig(pdf, transparent=True, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    lines = ["depth,branch,ratio,median_psnr_db,runs_above_30_db,runs_exact"]
    lines += [f"{r['depth']:g},{r['branch']},{r['ratio']:g},{r['median']:.2f},{r['above_30']},{r['exact']}" for r in rows]
    (out / "fig2_summary.csv").write_text("\n".join(lines) + "\n")
    print(f"Fig. 2: {pdf}, summary in {out / 'fig2_summary.csv'}")


def difference_stats(x):
    d = torch.cat([(torch.roll(x, -1, 0) - x).flatten(), (torch.roll(x, -1, 1) - x).flatten()])
    return d.abs().mean().item(), d.abs().pow(2).mean().sqrt().item()


def cbsd68_signals(data):
    from deepinv.datasets import CBSD68

    for i, img in enumerate(CBSD68(root=str(data / "CBSD68"), download=True)):
        a = np.asarray(img.convert("L")).astype(np.float64)
        yield f"cbsd68_{i:02d}", torch.from_numpy((a - a.min()) / (a.max() - a.min()))


def table1(args, out):
    estimate = {1.0: lambda tv, g: 0.5 + 1 / tv, 1.5: lambda tv, g: RAYLEIGH * g / tv,
                2.0: lambda tv, g: RAYLEIGH * math.sqrt(2) / tv, 2.5: lambda tv, g: RAYLEIGH * math.sqrt(2) / tv}
    rows = []
    for name, img in cbsd68_signals(ROOT / "data"):
        x = torch.exp(1j * math.pi * img)
        tv, g = difference_stats(x)
        for depth in TABLE1_DEPTHS:
            for seed in range(TABLE1_SEEDS):
                torch.manual_seed(seed)
                model = StructuredRandomModel(depth, x.shape, x.shape, "cpu")
                x_twin, error = model.twin(x[None, None].to(torch.complex64))
                assert error < 1e-5, f"{name}, L = {depth:g}, seed {seed}: twin identity error {error:.1e}"
                measured = difference_stats(x_twin.squeeze().to(torch.complex128))[0] / tv
                predicted = estimate[depth](tv, g)
                rows.append(dict(image=name, depth=depth, seed=seed, measured=measured, estimate=predicted,
                                 error=(predicted - measured) / measured))
        print(f"Table 1: {name} done", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["image,depth,seed,measured_tv_ratio,estimate,relative_error"]
    lines += [f"{r['image']},{r['depth']:g},{r['seed']},{r['measured']:.6f},{r['estimate']:.6f},{r['error']:.6f}"
              for r in rows]
    (out / "table1_runs.csv").write_text("\n".join(lines) + "\n")
    table = ["| L | median | 90th percentile | max |", "|---|---|---|---|"]
    for depth in TABLE1_DEPTHS:
        e = np.abs([r["error"] for r in rows if r["depth"] == depth]) * 100
        table.append(f"| {depth:g} | {np.median(e):.1f} % | {np.quantile(e, 0.9):.1f} % | {e.max():.1f} % |")
    (out / "table1.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"Table 1: {out / 'table1.md'}, runs in {out / 'table1_runs.csv'}")


def main():
    p = argparse.ArgumentParser(description="Reproduce the figures and the table of the generalized twin paper.")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--fig1", action="store_true", help="Fig. 1: generalized twins and reconstructions")
    target.add_argument("--fig2", action="store_true", help="Fig. 2: PSNR against the sampling ratio")
    target.add_argument("--table1", action="store_true", help="Table 1: error of the closed-form TV ratios")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--shard", type=int, default=0, help="index of this process when the runs are split")
    p.add_argument("--n-shards", type=int, default=1, help="number of processes the runs are split over")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args()
    assert 0 <= args.shard < args.n_shards, "shard index outside the number of shards"
    with torch.no_grad():
        if args.fig1:
            fig1(args, args.out)
        elif args.fig2:
            fig2(args, args.out)
        else:
            table1(args, args.out)


if __name__ == "__main__":
    main()

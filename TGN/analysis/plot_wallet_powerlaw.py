"""Figures for wallet_powerlaw.py: log-binned histogram and CCDF with fitted tails.

Reads results/wallet_stats.npz and results/powerlaw_fits.json, writes
figures/wallet_{trades,volume}.{png,pdf}.

    TGN/.venv/bin/python TGN/analysis/plot_wallet_powerlaw.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import powerlaw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"

DATA_C, PL_C, LN_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LABELS = {
    "trades": ("Number of trades per wallet", "trades"),
    "volume": ("Total traded volume per wallet (USD)", "USD"),
}


def style(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def log_hist(x: np.ndarray, discrete: bool):
    lo, hi = x.min(), x.max()
    edges = np.geomspace(lo, hi * 1.0001, 40)
    if discrete:
        edges = np.unique(np.floor(edges))
        edges = np.r_[edges, edges[-1] + 1]
    counts, edges = np.histogram(x, bins=edges)
    width = np.diff(edges)          # for integers: number of integer values in [a, b)
    dens = counts / (len(x) * width)
    centers = np.sqrt(edges[:-1] * edges[1:]) if not discrete else \
        np.sqrt(edges[:-1] * np.maximum(edges[1:] - 1, edges[:-1]))
    keep = counts > 0
    return centers[keep], dens[keep]


def figure(name: str, x: np.ndarray, r: dict) -> None:
    x = x[x > 0]
    discrete = r["discrete"]
    f = powerlaw.Fit(x, xmin=r["xmin"], discrete=discrete, verbose=0)
    share = r["tail_share"]
    title, unit = LABELS[name]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    fig.patch.set_facecolor("white")

    # (a) histogram
    c, dens = log_hist(x, discrete)
    a.plot(c, dens, "o", ms=4, color=DATA_C, label="Empirical (log-binned)", zorder=3)
    xt = np.unique(np.geomspace(r["xmin"], x.max(), 200))
    if discrete:
        xt = np.unique(np.round(xt))
    a.plot(xt, share * f.power_law.pdf(xt), color=PL_C, lw=2,
           label=f"Power law, α = {r['alpha']:.2f}")
    a.plot(xt, share * f.lognormal.pdf(xt), color=LN_C, lw=2, ls="--", label="Lognormal")
    a.axvline(r["xmin"], color=MUTED, lw=1, ls=":")
    a.text(r["xmin"], a.get_ylim()[0] if False else dens.min(), f"  $x_{{min}}$ = {r['xmin']:,.0f}",
           color=MUTED, fontsize=8, va="bottom")
    style(a)
    a.set_xlabel(f"{title}", color=INK, fontsize=9)
    a.set_ylabel("p(x)", color=INK, fontsize=9)
    a.set_title("(a) Histogram", color=INK, fontsize=10, loc="left")

    # (b) CCDF
    # P(X >= v) at each distinct value, thinned to ~log-spaced points so the
    # sparse upper tail is kept in full
    vals, counts = np.unique(x, return_counts=True)
    ccdf = counts[::-1].cumsum()[::-1] / len(x)
    keep = np.unique(np.searchsorted(vals, np.geomspace(vals[0], vals[-1], 600)))
    keep = np.unique(np.r_[keep[keep < len(vals)], np.arange(max(0, len(vals) - 200), len(vals))])
    b.plot(vals[keep], ccdf[keep], "o", ms=2.5, color=DATA_C, label="Empirical", zorder=3)
    b.plot(xt, share * f.power_law.ccdf(xt), color=PL_C, lw=2,
           label=f"Power law, α = {r['alpha']:.2f}")
    b.plot(xt, share * f.lognormal.ccdf(xt), color=LN_C, lw=2, ls="--", label="Lognormal")
    b.axvline(r["xmin"], color=MUTED, lw=1, ls=":")
    style(b)
    b.set_xlabel(f"{title}", color=INK, fontsize=9)
    b.set_ylabel("P(X ≥ x)", color=INK, fontsize=9)
    b.set_title("(b) Complementary CDF", color=INK, fontsize=10, loc="left")

    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.02))
    note = (f"n = {r['n']:,} wallets · tail n = {r['n_tail']:,} ({r['tail_share']:.1%}) · "
            f"KS D = {r['D']:.3f} · bootstrap p = {r['gof_p']:.3f} ({r['gof_sims']} sims)")
    fig.text(0.5, -0.02, note, ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"wallet_{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    stats = np.load(RES / "wallet_stats.npz")
    fits = json.loads((RES / "powerlaw_fits.json").read_text())
    for name in ("trades", "volume"):
        figure(name, stats[name].astype(float), fits[name])
        print("wrote", FIG / f"wallet_{name}.png")


if __name__ == "__main__":
    main()

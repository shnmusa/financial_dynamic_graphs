"""Classic bar histograms (frequency per bin) of per-wallet activity.

For each quantity two panels: (a) equal-width bins on a linear axis, the
textbook histogram, which shows why it is uninformative for heavy-tailed data;
(b) equal-width bins in log10(x), i.e. bins that grow geometrically, drawn
on a log x-axis with a linear frequency axis.

    TGN/.venv/bin/python TGN/analysis/plot_wallet_bar_hist.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
BAR, EDGE = "#2a78d6", "#fcfcfb"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

QUANT = {
    "trades": ("Number of trades per wallet", True),
    "volume": ("Total traded volume per wallet (USD)", False),
}


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    s = np.load(RES / "wallet_stats.npz")
    for name, (label, discrete) in QUANT.items():
        x = s[name].astype(float)
        x = x[x > 0]
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)

        # (a) linear bins
        counts, edges, _ = a.hist(x, bins=30, color=BAR, edgecolor=EDGE, linewidth=0.8)
        style(a)
        a.set_xlabel(label, color=INK, fontsize=9)
        a.set_ylabel("Frequency (wallets)", color=INK, fontsize=9)
        a.set_title("(a) Equal-width bins, linear axis", color=INK, fontsize=10, loc="left")
        a.text(0.97, 0.9, f"first bin: {counts[0] / len(x):.2%} of wallets\n"
               f"bin width: {edges[1] - edges[0]:,.0f}",
               transform=a.transAxes, ha="right", va="top", color=MUTED, fontsize=8)

        # (b) log-spaced bins
        lo = 1 if discrete else max(x.min(), 1e-2)
        edges = np.geomspace(lo, x.max() * 1.0001, 31)
        if discrete:
            edges = np.unique(np.r_[np.floor(edges), x.max() + 1]) - 0.5 + 0.5  # integer-aligned
        xb = np.clip(x, lo, None)
        b.hist(xb, bins=edges, color=BAR, edgecolor=EDGE, linewidth=0.8)
        b.set_xscale("log")
        style(b)
        b.set_xlabel(label + (" (values < $0.01 shown in first bin)" if not discrete else ""),
                     color=INK, fontsize=9)
        b.set_ylabel("Frequency (wallets)", color=INK, fontsize=9)
        b.set_title("(b) Logarithmic bins, log x-axis", color=INK, fontsize=10, loc="left")
        b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

        fig.text(0.5, -0.03, f"n = {len(x):,} wallets · median = {np.median(x):,.0f} · "
                 f"max = {x.max():,.0f}", ha="center", color=MUTED, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"wallet_{name}_bars.{ext}", bbox_inches="tight", dpi=200)
        plt.close(fig)
        print("wrote", FIG / f"wallet_{name}_bars.png")


def paper_figure() -> None:
    """Logarithmic-bin panels only, side by side (figures/wallet_hist_log.*).

    Trades use doubling integer bins [1,2), [2,4), [4,8), ... so every bin is
    twice as wide as the previous one; volume uses 30 geometric bins."""
    s = np.load(RES / "wallet_stats.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)

    t = s["trades"].astype(float)
    t = t[t > 0]
    k = int(np.ceil(np.log2(t.max() + 1)))
    edges = 2.0 ** np.arange(k + 1)
    a.hist(t, bins=edges, color=BAR, edgecolor=EDGE, linewidth=0.8)
    a.set_xscale("log", base=2)
    a.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2, numticks=20))
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if np.log2(v) % 3 == 0 else ""))
    a.set_xlabel("Number of trades per wallet (bins [1,2), [2,4), [4,8), ...)", color=INK, fontsize=9)
    a.set_title("(a) Trades per wallet", color=INK, fontsize=10, loc="left")

    v = s["volume"].astype(float)
    v = v[v > 0]
    lo = 1e-2
    edges = np.geomspace(lo, v.max() * 1.0001, 31)
    b.hist(np.clip(v, lo, None), bins=edges, color=BAR, edgecolor=EDGE, linewidth=0.8)
    b.set_xscale("log")
    b.set_xlabel("Total traded volume per wallet (USD; values < $0.01 in first bin)",
                 color=INK, fontsize=9)
    b.set_title("(b) Volume per wallet", color=INK, fontsize=10, loc="left")
    for ax in (a, b):
        style(ax)
        ax.set_ylabel("Frequency (wallets)", color=INK, fontsize=9)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.text(0.5, -0.03, f"n = {len(t):,} wallets (volume: {len(v):,} with a positive value)",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"wallet_hist_log.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", FIG / "wallet_hist_log.png")


if __name__ == "__main__":
    main()
    paper_figure()

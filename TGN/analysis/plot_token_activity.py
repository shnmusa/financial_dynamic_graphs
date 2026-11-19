"""Per-token activity: number of trade events and USD volume per token node.

Counted on the TGN event set (tgn_dex.pt). Every event is one wallet's net
sale or purchase of one token, so a swap A -> B counts once for A (a sale)
and once for B (a purchase).

  trades  number of events on the token (sales + purchases)
  volume  sum of the events' USD values

Figures (figures/token_*.{png,pdf}):
  token_hist     histograms over the 36 universe tokens, logarithmic bins
  token_ranked   every token node ranked by total, coloured by node type,
                 with the purchase share printed at the end of each bar
Table: results/token_activity.csv

    TGN/.venv/bin/python TGN/analysis/plot_token_activity.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed" / "tgn_dex.pt"
RES, FIG = HERE / "results", HERE / "figures"
BAR = "#2a78d6"
KIND_COLOR = {"universe token": "#2a78d6", "stablecoin": "#1baf7a", "unknown": "#8a8984"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def human(v, _=None):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.3g}{suf}"
    return f"{v:.3g}"


def token_table() -> pd.DataFrame:
    b = torch.load(DATA, weights_only=False)
    d = b["data"]
    side = d.msg[:, 0].numpy()
    usd = np.expm1(d.msg[:, 1].numpy().astype(np.float64))
    tok = np.where(side < 0, d.src.numpy(), d.dst.numpy())
    n = b["unknown_node"] + 1
    buy = side > 0
    names = {t["node"]: t["symbol"] for t in b["token_nodes"]}
    names[b["unknown_node"]] = "UNKNOWN"
    kind = {t["node"]: ("stablecoin" if t["is_stable"] else "universe token") for t in b["token_nodes"]}
    kind[b["unknown_node"]] = "unknown"
    df = pd.DataFrame({
        "node": range(n),
        "symbol": [names[i] for i in range(n)],
        "kind": [kind[i] for i in range(n)],
        "buys": np.bincount(tok[buy], minlength=n),
        "sells": np.bincount(tok[~buy], minlength=n),
        "buy_usd": np.bincount(tok[buy], weights=usd[buy], minlength=n),
        "sell_usd": np.bincount(tok[~buy], weights=usd[~buy], minlength=n),
    })
    df["trades"] = df.buys + df.sells
    df["volume_usd"] = df.buy_usd + df.sell_usd
    return df


def histograms(df: pd.DataFrame) -> None:
    u = df[df.kind == "universe token"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ax, col, label in ((axes[0], "trades", "Number of trade events per token"),
                           (axes[1], "volume_usd", "Total traded volume per token (USD)")):
        x = u[col].to_numpy(float)
        edges = np.geomspace(10 ** np.floor(np.log10(x.min())), 10 ** np.ceil(np.log10(x.max())),
                             int(np.ceil(np.log10(x.max())) - np.floor(np.log10(x.min()))) * 2 + 1)
        ax.hist(x, bins=edges, color=BAR, edgecolor=SURF, linewidth=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(human))
        style(ax)
        ax.set_xlabel(label, color=INK, fontsize=9)
        ax.set_ylabel("Frequency (tokens)", color=INK, fontsize=9)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.text(0.98, 0.95, f"median {human(np.median(x))}\nmax {human(x.max())} "
                f"({u.loc[u[col].idxmax(), 'symbol']})",
                transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8)
    axes[0].set_title("(a) Trade events", color=INK, fontsize=10, loc="left")
    axes[1].set_title("(b) USD volume", color=INK, fontsize=10, loc="left")
    fig.text(0.5, -0.03, f"n = {len(u)} universe tokens (stablecoins and the unknown node excluded) "
             "· bins: two per decade", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"token_hist.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def ranked(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 9), dpi=150)
    for ax, (b_col, tot, label) in zip(axes, (
            ("buys", "trades", "Trade events"),
            ("buy_usd", "volume_usd", "USD volume"))):
        r = df.sort_values(tot)
        y = np.arange(len(r))
        ax.barh(y, r[tot], color=[KIND_COLOR[k] for k in r.kind], edgecolor=SURF, linewidth=0.6)
        for yi, v, share in zip(y, r[tot], r[b_col] / r[tot]):
            ax.text(v * 1.08, yi, f"{human(v)} · {share:.0%} buy", va="center",
                    color=MUTED, fontsize=6.5)
        ax.set_yticks(y, r.symbol, fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(r[tot].min() * 0.5, r[tot].max() * 12)
        ax.xaxis.set_major_formatter(FuncFormatter(human))
        style(ax, "x")
        ax.set_xlabel(label + " (log scale)", color=INK, fontsize=9)
        ax.set_ylim(-0.7, len(r) - 0.3)
    axes[0].set_title("(a) Trade events per token", color=INK, fontsize=10, loc="left")
    axes[1].set_title("(b) USD volume per token", color=INK, fontsize=10, loc="left")
    handles = [matplotlib.patches.Patch(color=c, label=k.capitalize()) for k, c in KIND_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.01), labelcolor=INK)
    fig.text(0.5, -0.01, "Each event is one wallet's net sale or purchase of the token; a swap "
             "A→B counts once for A and once for B. \"buy\" = share of purchases.",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"token_ranked.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    df = token_table()
    df.sort_values("volume_usd", ascending=False).to_csv(RES / "token_activity.csv", index=False)
    histograms(df)
    ranked(df)
    pd.set_option("display.width", 200)
    print(df.sort_values("volume_usd", ascending=False)[
        ["symbol", "kind", "trades", "buys", "sells", "volume_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()

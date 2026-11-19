"""Descriptive statistics of per-wallet activity (reads results/wallet_stats.npz
written by wallet_powerlaw.py; same definitions of a trade and of volume).

  trades          number of trades per wallet
  volume_usd      total traded USD per wallet
  avg_trade_usd   volume_usd / trades (mean trade size of the wallet)

Writes results/descriptives.{json,md}; with --min-trades k only wallets with
at least k trades are kept and the files get a _min{k} suffix; with
--positive-volume wallets with 0 USD total volume are dropped too (_posvol),
with --min-volume v those under v USD (_minvol{v}).

    TGN/.venv/bin/python TGN/analysis/wallet_descriptives.py [--min-trades 3] [--positive-volume]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parent / "results"


def gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def top_share(x: np.ndarray, q: float) -> float:
    x = np.sort(x)[::-1]
    return float(x[: max(1, int(round(len(x) * q)))].sum() / x.sum())


def describe(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    pos = x[x > 0]
    q = np.quantile(x, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
    return {
        "n": int(len(x)), "sum": float(x.sum()),
        "mean": float(x.mean()), "std": float(x.std(ddof=1)), "cv": float(x.std(ddof=1) / x.mean()),
        "sem": float(stats.sem(x)),
        "min": float(x.min()),
        **{f"p{int(p * 1000) / 10:g}": float(v) for p, v in
           zip([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999], q)},
        "max": float(x.max()),
        "iqr": float(q[5] - q[3]),
        "mad": float(stats.median_abs_deviation(x)),
        "geo_mean": float(stats.gmean(pos)),
        "log10_mean": float(np.log10(pos).mean()), "log10_std": float(np.log10(pos).std(ddof=1)),
        "skewness": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x)),
        "log10_skewness": float(stats.skew(np.log10(pos))),
        "log10_excess_kurtosis": float(stats.kurtosis(np.log10(pos))),
        "gini": gini(x),
        "top_0.1pct_share": top_share(x, 0.001), "top_1pct_share": top_share(x, 0.01),
        "top_10pct_share": top_share(x, 0.10), "bottom_50pct_share": 1 - top_share(x, 0.50),
        "zeros": int((x == 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=1)
    ap.add_argument("--positive-volume", action="store_true",
                    help="drop wallets whose total volume is 0 USD (dust only)")
    ap.add_argument("--min-volume", type=float, default=None,
                    help="keep only wallets with total volume >= this many USD")
    args = ap.parse_args()
    k = args.min_trades
    suffix = ("" if k <= 1 else f"_min{k}") + ("_posvol" if args.positive_volume else "") \
        + (f"_minvol{args.min_volume:g}" if args.min_volume is not None else "")
    s = np.load(RES / "wallet_stats.npz")
    trades, volume = s["trades"].astype(np.float64), s["volume"]
    active = trades >= max(k, 1)
    if args.positive_volume:
        active &= volume > 0
    if args.min_volume is not None:
        active &= volume >= args.min_volume
    trades, volume = trades[active], volume[active]
    out = {
        "trades": describe(trades),
        "volume_usd": describe(volume),
        "avg_trade_usd": describe(volume / trades),
        "share_single_trade_wallets": float((trades == 1).mean()),
        "spearman_trades_volume": float(stats.spearmanr(trades, volume).statistic),
        "pearson_log_trades_log_volume": float(np.corrcoef(
            np.log10(trades[volume > 0]), np.log10(volume[volume > 0]))[0, 1]),
    }
    (RES / f"descriptives{suffix}.json").write_text(json.dumps(out, indent=2))

    keys = [k for k in out["trades"] if k not in ("n",)]
    lines = ["| statistic | trades per wallet | total volume per wallet (USD) | mean trade size (USD) |",
             "|---|---|---|---|", f"| n | {out['trades']['n']:,} | {out['volume_usd']['n']:,} | "
             f"{out['avg_trade_usd']['n']:,} |"]
    for k in keys:
        cells = []
        for col in ("trades", "volume_usd", "avg_trade_usd"):
            v = out[col][k]
            cells.append(f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", f"single-trade wallets: {out['share_single_trade_wallets']:.1%}",
              f"Spearman(trades, volume) = {out['spearman_trades_volume']:.3f}",
              f"Pearson(log trades, log volume) = {out['pearson_log_trades_log_volume']:.3f}"]
    (RES / f"descriptives{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

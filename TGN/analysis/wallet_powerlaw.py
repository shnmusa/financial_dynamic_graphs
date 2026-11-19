"""Per-wallet activity distributions and power-law fits (Clauset, Shalizi & Newman 2009).

Two quantities per wallet, from the TGN event set (TGN/data/processed/tgn_dex.pt,
i.e. after cleanup, wallet-side netting and dropping unpriced flows):

  trades  number of trades: distinct (wallet, block timestamp) pairs. One swap
          transaction gives a wallet a sell and a buy event at the same t, so
          counting events would double it; counting distinct t treats a
          wallet's activity within one block as one trade.
  volume  total traded USD: per trade max(sum of sold USD, sum of bought USD),
          summed over the wallet's trades (the notional that changed hands,
          counted once, robust to one leg having been dropped).

For each quantity:
  * x_min by minimising the KS distance D over a log-spaced grid of candidates,
    alpha by maximum likelihood above x_min (discrete for trades, continuous
    for volume), sigma(alpha) = (alpha - 1) / sqrt(n_tail)
  * goodness of fit: semi-parametric bootstrap p-value (Clauset et al. Sec. 4.1):
    synthetic data sets draw from the fitted power law above x_min with
    probability n_tail / n and from the empirical data below x_min otherwise;
    each is re-fitted (x_min included) and p = share with D_synthetic >= D.
    p > 0.1 -> the power law is a plausible model.
  * likelihood-ratio tests against lognormal, exponential, stretched exponential
    and truncated power law (Vuong; normalized R, p). R > 0 favours the power
    law, R < 0 the alternative; meaningful only when p < 0.1.

    TGN/.venv/bin/python TGN/analysis/wallet_powerlaw.py [--sims 200]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import powerlaw
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed" / "tgn_dex.pt"
OUT = HERE / "results"

N_GRID = 120          # candidate x_min values (log-spaced quantiles of the data)
MIN_TAIL = 100        # never fit on fewer tail points than this
ALTERNATIVES = ("lognormal", "exponential", "stretched_exponential", "truncated_power_law")


def wallet_stats() -> tuple[np.ndarray, np.ndarray]:
    b = torch.load(DATA, weights_only=False)
    d, first_wallet = b["data"], b["first_wallet_node"]
    side = d.msg[:, 0].numpy()
    usd = np.expm1(d.msg[:, 1].numpy().astype(np.float64))
    wallet = np.where(side < 0, d.dst.numpy(), d.src.numpy()) - first_wallet
    t = d.t.numpy()
    n_w = int(wallet.max()) + 1

    # one key per (wallet, t) = one trade
    order = np.lexsort((t, wallet))
    w, tt = wallet[order], t[order]
    new = np.r_[True, (w[1:] != w[:-1]) | (tt[1:] != tt[:-1])]
    trade_id = np.cumsum(new) - 1
    n_trades = trade_id[-1] + 1
    s = side[order]
    sold = np.bincount(trade_id, weights=np.where(s < 0, usd[order], 0), minlength=n_trades)
    bought = np.bincount(trade_id, weights=np.where(s > 0, usd[order], 0), minlength=n_trades)
    trade_wallet = w[new]
    trades = np.bincount(trade_wallet, minlength=n_w)
    volume = np.bincount(trade_wallet, weights=np.maximum(sold, bought), minlength=n_w)
    return trades.astype(np.int64), volume


def candidates(x: np.ndarray, discrete: bool) -> np.ndarray:
    xs = np.sort(x)
    upper = xs[-MIN_TAIL]                       # keep at least MIN_TAIL points in the tail
    lo, hi = max(xs[0], 1e-12), upper
    grid = np.unique(np.geomspace(lo, hi, N_GRID))
    if discrete:
        grid = np.unique(np.round(grid))
    # snap to observed values so each candidate is a real x_min
    idx = np.searchsorted(xs, grid)
    return np.unique(xs[np.clip(idx, 0, len(xs) - 1)])


def fit(x: np.ndarray, discrete: bool, grid: np.ndarray | None = None):
    """Return (Fit, D) at the KS-minimising x_min on the grid."""
    grid = candidates(x, discrete) if grid is None else grid
    best = None
    for xm in grid:
        if (x >= xm).sum() < MIN_TAIL:
            continue
        f = powerlaw.Fit(x, xmin=xm, discrete=discrete, verbose=0)
        D = f.power_law.D
        if np.isfinite(D) and (best is None or D < best[1]):
            best = (f, D)
    return best


def _one_sim(args):
    seed, x, discrete, xmin, alpha, n_tail = args
    rng = np.random.default_rng(seed)
    n = len(x)
    body = x[x < xmin]
    k_tail = rng.binomial(n, n_tail / n)
    np.random.seed(seed)   # powerlaw's generator uses the global numpy RNG
    pl = powerlaw.Power_Law(xmin=xmin, parameters=[alpha], discrete=discrete)
    synth = np.concatenate([pl.generate_random(k_tail),
                            rng.choice(body, n - k_tail, replace=True) if len(body) else []])
    synth = synth[np.isfinite(synth)]
    if discrete:
        synth = np.round(synth)
    res = fit(synth, discrete)
    return None if res is None else float(res[1])


def analyse(name: str, x: np.ndarray, discrete: bool, sims: int, workers: int) -> dict:
    x = x[x > 0]
    t0 = time.monotonic()
    f, D = fit(x, discrete)
    xmin, alpha = float(f.power_law.xmin), float(f.power_law.alpha)
    n_tail = int((x >= xmin).sum())
    print(f"[{name}] n={len(x):,} xmin={xmin:.4g} alpha={alpha:.4f} D={D:.4f} "
          f"n_tail={n_tail:,} ({time.monotonic() - t0:.0f}s)", flush=True)

    lr = {}
    for alt in ALTERNATIVES:
        R, p = f.distribution_compare("power_law", alt, normalized_ratio=True)
        lr[alt] = {"R": float(R), "p": float(p)}
        print(f"  vs {alt:22s} R={R:+.3f} p={p:.3g}", flush=True)

    t0 = time.monotonic()
    with ProcessPoolExecutor(workers) as ex:
        Ds = [d for d in ex.map(_one_sim, [(s, x, discrete, xmin, alpha, n_tail)
                                           for s in range(sims)]) if d is not None]
    p_gof = float(np.mean(np.array(Ds) >= D))
    print(f"  bootstrap: p={p_gof:.3f} over {len(Ds)} sims ({time.monotonic() - t0:.0f}s)",
          flush=True)

    return {
        "n": int(len(x)), "discrete": discrete, "xmin": xmin, "alpha": alpha,
        "alpha_se": (alpha - 1) / np.sqrt(n_tail), "n_tail": n_tail,
        "tail_share": n_tail / len(x), "D": float(D),
        "gof_p": p_gof, "gof_sims": len(Ds), "lr": lr,
        "tl_alpha": float(f.truncated_power_law.alpha),
        "tl_lambda": float(f.truncated_power_law.Lambda),
        "ln_mu": float(f.lognormal.mu), "ln_sigma": float(f.lognormal.sigma),
        "summary": {"mean": float(x.mean()), "median": float(np.median(x)),
                    "p99": float(np.quantile(x, 0.99)), "max": float(x.max())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", choices=("trades", "volume"), default=None,
                    help="refit one quantity and merge it into the existing json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    trades, volume = wallet_stats()
    np.savez(OUT / "wallet_stats.npz", trades=trades, volume=volume)
    print(f"wallets {len(trades):,} | trades {trades.sum():,} | volume ${volume.sum():,.0f}",
          flush=True)
    path = OUT / "powerlaw_fits.json"
    res = json.loads(path.read_text()) if args.only and path.exists() else {}
    if args.only in (None, "trades"):
        res["trades"] = analyse("trades", trades.astype(float), True, args.sims, args.workers)
    if args.only in (None, "volume"):
        res["volume"] = analyse("volume", volume, False, args.sims, args.workers)
    (OUT / "powerlaw_fits.json").write_text(json.dumps(res, indent=2))
    print(f"saved {OUT / 'powerlaw_fits.json'}")


if __name__ == "__main__":
    main()

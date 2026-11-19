"""Non-graph baselines for next-hour log-return forecasting.

Same targets, split, volatility scaling and metrics as train.py (it imports
them), so the numbers are directly comparable with the TGN runs.

At the end of hour h each baseline predicts z[h] = r[h] / sigma[h], the
volatility-scaled return over hour h+1:

  zero        z = 0                                   (no-change forecast)
  train_mean  per-token mean of z over training hours
  last        z = lr[h] / sigma[h]                    (last hour's return repeats)
  ar1         z = a + b * lr[h]/sigma[h], one (a, b) pooled over tokens, OLS on train
  flow_ridge  ridge regression on the token's order flow during hour h, built
              from the same events the TGN sees:
                log1p(buy USD), log1p(sell USD), (buy - sell) / (buy + sell),
                log1p(#buy events), log1p(#sell events), lr[h]/sigma[h]
              pooled over tokens, alpha chosen on validation

    TGN/.venv/bin/python TGN/baselines.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from train import HERE, load, make_split, metrics


def flow_features(d: dict, path: Path, sigma: np.ndarray) -> np.ndarray:
    """[H, n_target, 6] order-flow features per token and hour."""
    b = torch.load(path, weights_only=False)
    data, hours = b["data"], b["hours"].numpy()
    H, n = d["H"], d["n_target"]
    side = data.msg[:, 0].numpy()
    usd = np.expm1(data.msg[:, 1].numpy().astype(np.float64))
    token = np.where(side < 0, data.src.numpy(), data.dst.numpy())   # sell: token->wallet
    hour = (data.t.numpy() - hours[0]) // 3600
    keep = (token < n) & (hour < H)
    idx = hour[keep] * n + token[keep]
    buy = side[keep] > 0

    def agg(weights, sel):
        return np.bincount(idx[sel], weights=weights[sel] if weights is not None else None,
                           minlength=H * n).reshape(H, n)

    buy_usd, sell_usd = agg(usd[keep], buy), agg(usd[keep], ~buy)
    n_buy, n_sell = agg(None, buy), agg(None, ~buy)
    imb = (buy_usd - sell_usd) / np.maximum(buy_usd + sell_usd, 1e-9)
    z_last = np.nan_to_num(d["lr"] / sigma)
    return np.stack([np.log1p(buy_usd), np.log1p(sell_usd), imb,
                     np.log1p(n_buy), np.log1p(n_sell), z_last], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data" / "processed" / "tgn_dex.pt")
    ap.add_argument("--vol-window", type=int, default=48)
    ap.add_argument("--exclude", nargs="*", default=["SI"])
    ap.add_argument("--split", type=float, nargs=2, default=(0.70, 0.85))
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    args = ap.parse_args()

    d = load(args.data, torch.device("cpu"), None, args.vol_window, tuple(args.exclude))
    tr, va, te, sigma = make_split(d, tuple(args.split))
    r, mask = d["r"], d["r_mask"]
    z = r / sigma

    def part(rg):
        m = np.zeros_like(mask)
        m[rg.start:rg.stop] = mask[rg.start:rg.stop]
        return m

    m_tr, m_va, m_te = part(tr), part(va), part(te)
    X = flow_features(d, args.data, sigma)
    z_last = X[..., -1]

    preds = {
        "zero": np.zeros_like(r),
        "train_mean": np.broadcast_to(np.nanmean(np.where(m_tr, z, np.nan), axis=0), r.shape),
        "last": z_last,
    }
    ar = LinearRegression().fit(z_last[m_tr][:, None], z[m_tr])
    preds["ar1"] = ar.predict(z_last.reshape(-1, 1)).reshape(r.shape)

    scaler = StandardScaler().fit(X[m_tr])
    Xs = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    best = None
    for alpha in (0.1, 1, 10, 100, 1000, 10000):
        model = Ridge(alpha=alpha).fit(Xs[m_tr], z[m_tr])
        p = model.predict(Xs.reshape(-1, X.shape[-1])).reshape(r.shape)
        v = metrics(p, r, m_va, sigma)["mse_z"]
        if best is None or v < best[0]:
            best = (v, alpha, p, model)
    preds["flow_ridge"] = best[2]

    results = {name: {"val": metrics(np.nan_to_num(p), r, m_va, sigma),
                      "test": metrics(np.nan_to_num(p), r, m_te, sigma)} for name, p in preds.items()}
    results["flow_ridge"]["alpha"] = best[1]
    results["flow_ridge"]["coef"] = dict(zip(
        ["log_buy_usd", "log_sell_usd", "imbalance", "log_n_buy", "log_n_sell", "z_last"],
        best[3].coef_.round(5).tolist()))
    results["ar1"]["coef"] = {"a": float(ar.intercept_), "b": float(ar.coef_[0])}

    out = args.out / f"baselines-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "hours": {"train": [tr.start, tr.stop], "val": [va.start, va.stop], "test": [te.start, te.stop]},
         "results": results}, indent=2))
    print(f"hours train {len(tr)} val {len(va)} test {len(te)} | out {out}")
    for name, res in results.items():
        for split in ("val", "test"):
            m = res[split]
            print(f"{name:11s} {split:4s} | skill {m['skill']:+.4f} | rmse_r {m['rmse_r_pct']:.4f}% "
                  f"(naive {m['rmse_r_pct_naive']:.4f}%) | dir {m['dir_acc']:.3f} | ic {m['ic']:+.4f} | n {m['n']}")
    print("flow_ridge alpha", best[1], results["flow_ridge"]["coef"])


if __name__ == "__main__":
    main()

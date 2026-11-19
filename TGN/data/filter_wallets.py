"""Make a wallet-filtered copy of the TGN bundle.

Keeps only the events of wallets with at least --min-trades trades and total
traded volume above --min-volume USD (exclusive). Trades and volume use the
definitions of TGN/analysis/wallet_powerlaw.py:
  trade   one distinct (wallet, block timestamp)
  volume  per trade max(sold USD, bought USD), summed over the wallet's trades
Both are computed over the whole period (train + val + test), as asked; note
this lets the wallet set depend on activity after the training window.

Token nodes keep their ids; kept wallets are renumbered densely from
first_wallet_node in their original order (= order of first appearance).
Prices, hours and every other bundle field are unchanged.

    TGN/.venv/bin/python TGN/data/filter_wallets.py --min-trades 3 --min-volume 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData

HERE = Path(__file__).resolve().parent / "processed"


def wallet_stats(data, first_wallet: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = data.msg[:, 0].numpy()
    usd = np.expm1(data.msg[:, 1].numpy().astype(np.float64))
    wallet = np.where(side < 0, data.dst.numpy(), data.src.numpy()) - first_wallet
    t = data.t.numpy()
    n_w = int(wallet.max()) + 1
    order = np.lexsort((t, wallet))
    w, tt, s, u = wallet[order], t[order], side[order], usd[order]
    new = np.r_[True, (w[1:] != w[:-1]) | (tt[1:] != tt[:-1])]
    tid = np.cumsum(new) - 1
    sold = np.bincount(tid, weights=np.where(s < 0, u, 0))
    bought = np.bincount(tid, weights=np.where(s > 0, u, 0))
    tw = w[new]
    trades = np.bincount(tw, minlength=n_w)
    volume = np.bincount(tw, weights=np.maximum(sold, bought), minlength=n_w)
    return wallet, trades, volume


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=HERE / "tgn_dex.pt")
    ap.add_argument("--min-trades", type=int, default=3)
    ap.add_argument("--min-volume", type=float, default=0.0, help="keep volume > this (USD)")
    args = ap.parse_args()
    tag = f"min{args.min_trades}_vol{args.min_volume:g}"
    out = HERE / f"tgn_dex_{tag}.pt"

    b = torch.load(args.src, weights_only=False)
    d, fw = b["data"], b["first_wallet_node"]
    wallet, trades, volume = wallet_stats(d, fw)
    keep_w = (trades >= args.min_trades) & (volume > args.min_volume)
    new_id = np.full(len(keep_w), -1, dtype=np.int64)
    new_id[keep_w] = fw + np.arange(keep_w.sum())

    keep_e = keep_w[wallet]
    src, dst = d.src.numpy()[keep_e].copy(), d.dst.numpy()[keep_e].copy()
    sell = d.msg[:, 0].numpy()[keep_e] < 0
    dst[sell] = new_id[dst[sell] - fw]          # sell: token -> wallet
    src[~sell] = new_id[src[~sell] - fw]        # buy:  wallet -> token
    assert (src >= 0).all() and (dst >= 0).all()
    data = TemporalData(src=torch.from_numpy(src), dst=torch.from_numpy(dst),
                        t=d.t[torch.from_numpy(keep_e)], msg=d.msg[torch.from_numpy(keep_e)])

    nb = dict(b)
    nb.update(data=data, num_nodes=fw + int(keep_w.sum()),
              wallet_filter={"min_trades": args.min_trades, "min_volume_exclusive": args.min_volume,
                             "source": str(args.src.name)})
    torch.save(nb, out)

    wallets = pd.read_parquet(HERE / "wallets.parquet").sort_values("node")
    wk = wallets[keep_w].copy()
    wk["orig_node"], wk["node"] = wk["node"], new_id[keep_w]
    wk.to_parquet(HERE / f"wallets_{tag}.parquet", index=False)

    sold_usd = np.expm1(d.msg[:, 1].numpy().astype(np.float64))
    meta = {
        "filter": nb["wallet_filter"],
        "wallets": [int(len(keep_w)), int(keep_w.sum())],
        "events": [int(d.num_events), int(keep_e.sum())],
        "event_share_kept": float(keep_e.mean()),
        "usd_share_kept": float(sold_usd[keep_e].sum() / sold_usd.sum()),
        "num_nodes": nb["num_nodes"],
    }
    (HERE / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()

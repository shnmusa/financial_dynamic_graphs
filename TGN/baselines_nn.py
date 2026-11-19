"""Neural ablations of the TGN: a temporal model without the graph (GRU) and a
graph model without temporal memory (static GNN).

Both reuse train.py's data loading, target (z = next-hour log-return / ex-ante
volatility), split, metrics and training regime, and copy the TGN's building
blocks so that the comparison isolates one ingredient at a time:

  TGN  (train.py)  continuous-time memory (GRU cell, dim 100) + temporal graph
                   attention (TransformerConv, 2 heads x 50, edge attributes =
                   time encoding + event message) + ReturnDecoder.

  gru              The TGN memory without the graph. One GRU cell of the same
                   width (100) per token, weights shared across tokens, updated
                   once per hour with the token's hourly order-flow features
                   (the six flow_ridge features: log1p buy/sell USD, imbalance,
                   log1p #buys/#sells, last z). Same ReturnDecoder. As in the
                   TGN, the state is carried across hours and detached after
                   every update, so gradients flow through one step.

  gnn              The TGN graph attention without memory. For each hour h a
                   static snapshot graph of that hour's events: the 40 token
                   nodes and the wallets active in hour h, one edge per event
                   in both directions. Edge attributes are, as in the TGN, the
                   time encoding of the event's offset from the end of the
                   hour plus the event message (side, log1p USD). Token node
                   inputs are the same six flow features (linear map to 100),
                   wallets share one learned input vector. Two TransformerConv
                   layers of the TGN's shape (2 heads x 50): one layer only
                   lets a token see its own trades; the second lets
                   information pass token -> wallet -> token, which the TGN
                   obtains through its memory. Same ReturnDecoder. No state is
                   carried between hours.

Training as in train.py: Adam (lr 1e-4), one Huber step per hour with targets,
at most 10 epochs, early stopping on validation MSE (patience 3), best epoch
re-run without training for val/test. Outputs go to runs/<model>-<timestamp>/
with the same config.json / metrics.json layout as the TGN runs.

    TGN/.venv/bin/python TGN/baselines_nn.py --model gru --seed 0
    TGN/.venv/bin/python TGN/baselines_nn.py --model gnn --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder

from train import HERE, ReturnDecoder, load, make_split, metrics

N_FEAT = 6


def token_features(d: dict, path: Path, sigma: np.ndarray, n_tok: int) -> np.ndarray:
    """[H, n_tok, 6] hourly order-flow features for every token node (targets,
    stablecoins and the unknown node); z_last only exists for target tokens."""
    b = torch.load(path, weights_only=False)
    data, hours = b["data"], b["hours"].numpy()
    H, n = d["H"], n_tok
    side = data.msg[:, 0].numpy()
    usd = np.expm1(data.msg[:, 1].numpy().astype(np.float64))
    token = np.where(side < 0, data.src.numpy(), data.dst.numpy())
    hour = (data.t.numpy() - hours[0]) // 3600
    keep = hour < H
    idx = hour[keep] * n + token[keep]
    buy = side[keep] > 0

    def agg(w, sel):
        return np.bincount(idx[sel], weights=None if w is None else w[sel],
                           minlength=H * n).reshape(H, n)

    buy_usd, sell_usd = agg(usd[keep], buy), agg(usd[keep], ~buy)
    n_buy, n_sell = agg(None, buy), agg(None, ~buy)
    imb = (buy_usd - sell_usd) / np.maximum(buy_usd + sell_usd, 1e-9)
    z_last = np.zeros((H, n))
    z_last[:, :d["n_target"]] = np.nan_to_num(d["lr"] / sigma)
    return np.stack([np.log1p(buy_usd), np.log1p(sell_usd), imb,
                     np.log1p(n_buy), np.log1p(n_sell), z_last], axis=-1)


class GRUModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.cell = nn.GRUCell(N_FEAT, dim)
        self.decoder = ReturnDecoder(dim)
        self.dim = dim


class GNNModel(nn.Module):
    def __init__(self, dim: int, msg_dim: int, layers: int = 2):
        super().__init__()
        self.tok_in = nn.Linear(N_FEAT, dim)
        self.wallet_in = nn.Parameter(torch.zeros(dim))
        self.time_enc = TimeEncoder(dim)
        self.convs = nn.ModuleList(
            TransformerConv(dim, dim // 2, heads=2, dropout=0.1, edge_dim=msg_dim + dim)
            for _ in range(layers))
        self.decoder = ReturnDecoder(dim)

    def forward(self, x_tok, n_wallet, edge_index, rel_t, msg):
        x = torch.cat([self.tok_in(x_tok), self.wallet_in.expand(n_wallet, -1)])
        e = torch.cat([self.time_enc(rel_t), msg], dim=-1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, e)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


def hour_graphs(d: dict, hours0: int, n_tok: int, device) -> list:
    """Per hour: (n_wallet, edge_index, rel_t, msg) on the device."""
    src, dst = d["src"].cpu().numpy(), d["dst"].cpu().numpy()
    t, msg = d["t"].cpu().numpy(), d["msg"].cpu()
    out = []
    for h in range(d["H"]):
        s, e = int(d["bounds"][h]), int(d["bounds"][h + 1])
        a, b = src[s:e], dst[s:e]
        nodes = np.concatenate([a, b])
        wallets = np.unique(nodes[nodes >= n_tok])
        remap = lambda v: np.where(v < n_tok, v, n_tok + np.searchsorted(wallets, v))  # noqa: E731
        a, b = remap(a), remap(b)
        ei = torch.tensor(np.stack([np.r_[a, b], np.r_[b, a]]), dtype=torch.long)
        hour_end = hours0 + 3600 * (h + 1)
        rel = torch.tensor(np.r_[t[s:e], t[s:e]] - hour_end, dtype=torch.float32)
        m = torch.cat([msg[s:e], msg[s:e]])
        out.append((len(wallets), ei.to(device), rel.to(device), m.to(device)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("gru", "gnn"), required=True)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "processed" / "tgn_dex.pt")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--split", type=float, nargs=2, default=(0.70, 0.85))
    ap.add_argument("--vol-window", type=int, default=48)
    ap.add_argument("--exclude", nargs="*", default=["SI"])
    ap.add_argument("--max-hours", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    d = load(args.data, torch.device("cpu"), args.max_hours, args.vol_window, tuple(args.exclude))
    tr, va, te, sigma_np = make_split(d, tuple(args.split))
    b = torch.load(args.data, weights_only=False)
    n_tok, hours0 = b["first_wallet_node"], int(b["hours"][0])
    nt = d["n_target"]

    X = token_features(d, args.data, sigma_np, n_tok)
    mu = X[tr.start:tr.stop].reshape(-1, N_FEAT).mean(0)
    sd = X[tr.start:tr.stop].reshape(-1, N_FEAT).std(0) + 1e-8
    X = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    z = torch.tensor(np.nan_to_num(d["r"] / sigma_np), dtype=torch.float32, device=device)
    rmask = torch.tensor(d["r_mask"], device=device)
    mask = d["r_mask"]

    if args.model == "gru":
        model = GRUModel(args.dim).to(device)
        graphs = None
    else:
        model = GNNModel(args.dim, d["msg_dim"], args.gnn_layers).to(device)
        graphs = hour_graphs(d, hours0, n_tok, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def predict(h, state):
        """Prediction for the target tokens at the end of hour h; returns (pred, new state)."""
        if args.model == "gru":
            state = model.cell(X[h, :nt], state)
            return model.decoder(state), state
        n_w, ei, rel, m = graphs[h]
        x = model(X[h], n_w, ei, rel, m)
        return model.decoder(x[:nt]), None

    def run(hours_train: range, hours_eval: range) -> dict:
        preds = np.full((d["H"], nt), np.nan)
        state = torch.zeros(nt, args.dim, device=device) if args.model == "gru" else None
        losses = []
        last = max(hours_eval.stop if len(hours_eval) else 0, hours_train.stop)
        for h in range(last):
            training = h in hours_train
            model.train(training)
            has_target = bool(mask[h].any())
            if training and has_target:
                opt.zero_grad()
                pred, state = predict(h, state)
                k = rmask[h]
                loss = F.huber_loss(pred[k], z[h][k], delta=args.huber_delta)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                if state is not None:
                    state = state.detach()
            elif args.model == "gru" or (h in hours_eval and has_target):
                with torch.no_grad():
                    pred, state = predict(h, state)
                if h in hours_eval and has_target:
                    preds[h] = pred.cpu().numpy()
        return {"loss": float(np.mean(losses)) if losses else float("nan"), "preds": preds}

    run_dir = args.out / f"{args.model}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update({"train_hours": [tr.start, tr.stop], "val_hours": [va.start, va.stop],
                "test_hours": [te.start, te.stop],
                "n_params": sum(p.numel() for p in model.parameters())})
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{args.model} | device {device} | params {cfg['n_params']:,} | hours train {len(tr)} "
          f"val {len(va)} test {len(te)} | {run_dir}", flush=True)

    def part(rg):
        m = np.zeros_like(mask)
        m[rg.start:rg.stop] = mask[rg.start:rg.stop]
        return m

    best, best_val, bad, history = None, float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        out = run(tr, va)
        val = metrics(np.nan_to_num(out["preds"]), d["r"], part(va), sigma_np)
        history.append({"epoch": epoch, "reg_loss": out["loss"], "val": val,
                        "minutes": (time.monotonic() - t0) / 60})
        print(f"epoch {epoch:2d} | {history[-1]['minutes']:.1f} min | train huber {out['loss']:.4f} | "
              f"val skill {val['skill']:+.4f} dir {val['dir_acc']:.3f} ic {val['ic']:+.4f}", flush=True)
        if val["mse_z"] < best_val:
            best_val, bad = val["mse_z"], 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs", flush=True)
                break

    model.load_state_dict(best)
    torch.save(best, run_dir / "best.pt")
    out = run(range(0), range(0, te.stop))
    res = {name: metrics(np.nan_to_num(out["preds"]), d["r"], part(rg), sigma_np)
           for name, rg in (("val", va), ("test", te))}
    np.save(run_dir / "preds_z.npy", out["preds"])
    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "final": res}, indent=2))
    for name in ("val", "test"):
        m = res[name]
        print(f"{name:4s} | skill {m['skill']:+.4f} | rmse_r {m['rmse_r_pct']:.4f}% | "
              f"dir_acc {m['dir_acc']:.3f} | ic {m['ic']:+.4f} | n {m['n']}", flush=True)


if __name__ == "__main__":
    main()

"""Train a TGN to forecast next-hour log-returns of the target tokens.

Input:  TGN/data/processed/tgn_dex.pt (see data/build_temporal_data.py)

Setup
-----
Events are streamed in time order, hour by hour. Within an hour they are fed
in batches to the TGN memory and the last-neighbour loader. At the END of hour
h (i.e. at the start of hour h+1) the token nodes are embedded and a small MLP
decoder predicts

    r[h] = log p[h+2] - log p[h+1]

the log-return over the NEXT hour. p[h+1] is the price at the start of hour
h+1, which is exactly the moment the prediction is made, so no event that
moved the predicted return has been seen. (Predicting log p[h+1] - log p[h]
would let the model read the very trades that moved that price.)

Targets are scaled by ex-ante realised volatility: z[h] = r[h] / sigma[h],
where sigma[h] is the token's standard deviation of hourly log-returns over the
--vol-window hours ending at p[h+1] (all known when the prediction is made),
floored at half the token's median sigma over the training hours. A fixed
training-period sigma broke when a token's volatility jumped after training
(AURORA: 0.8% -> 6.8%) and let two tokens dominate the loss. Hours without a
full half-window of history (warm-up, late listings) have no target, and
tokens in --exclude (default SI, listed only in the test period) are never
targets. Loss: Huber on z, masked.

Auxiliary loss: the regression loss fires once per hour, which is too sparse
to train the memory module on its own. Every batch therefore also gets the
standard TGN self-supervised link-prediction loss (does this wallet trade this
token now?), with negatives drawn from the same node type as the true
destination. --aux-weight 0 turns it off.

Split: by prediction hour, 70/15/15 (train/val/test). Memory is carried
across the split boundary, as usual for TGN; val/test hours update it without
gradients. The best epoch by validation loss is re-run end to end for test.

Metrics (val/test), on the original return scale where noted:
    skill      1 - MSE(model) / MSE(predict 0)   on z  (out-of-sample R^2 vs
               the no-change forecast; > 0 means better than no change)
    rmse_r     RMSE of the log-return, in %      (model and naive)
    dir_acc    share of non-zero returns whose sign is predicted correctly
    ic         mean over hours of the cross-sectional Pearson correlation
               between predicted and realised returns (information coefficient)

    TGN/.venv/bin/python TGN/train.py                 # full run on MPS
    TGN/.venv/bin/python TGN/train.py --max-hours 24 --epochs 1   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader

HERE = Path(__file__).resolve().parent


# --- model ------------------------------------------------------------------

class GraphAttentionEmbedding(nn.Module):
    """Temporal graph attention over each node's most recent neighbours."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: nn.Module):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        return self.conv(x, edge_index, torch.cat([rel_t_enc, msg], dim=-1))


class LinkPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin_src, self.lin_dst = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.lin_out = nn.Linear(dim, 1)

    def forward(self, z_src, z_dst):
        return self.lin_out(torch.relu(self.lin_src(z_src) + self.lin_dst(z_dst))).squeeze(-1)


class ReturnDecoder(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


# --- data -------------------------------------------------------------------

def load(path: Path, device: torch.device, max_hours: int | None, vol_window: int = 48,
         exclude: tuple[str, ...] = ("SI",)):
    b = torch.load(path, weights_only=False)
    data = b["data"]
    hours = b["hours"].numpy()
    n_target = sum(1 for x in b["token_nodes"] if not x["is_stable"])
    H = len(hours) if max_hours is None else min(max_hours, len(hours))

    t = data.t.numpy()
    # bounds[h] .. bounds[h+1] are the events of hour h
    bounds = np.searchsorted(t, hours[0] + 3600 * np.arange(H + 1), side="left")

    y = b["y_price"].numpy()[:, :n_target]
    m = b["y_mask"].numpy()[:, :n_target]
    logp = np.log(np.where(m, y, np.nan))
    # lr[k] = log p[k+1] - log p[k]: the return over hour k, known at the end of hour k
    lr = np.full((len(hours), n_target), np.nan)
    lr[:-1] = logp[1:] - logp[:-1]
    # target for the state at the end of hour h: the NEXT hour's return, lr[h+1]
    r = np.full((H, n_target), np.nan)
    k = min(H, len(hours) - 1)
    r[:k] = lr[1:k + 1]
    # ex-ante volatility: std of lr[h-W+1 .. h]
    sigma_raw = pd.DataFrame(lr).rolling(vol_window, min_periods=vol_window // 2).std().to_numpy()[:H]
    symbols = [x["symbol"] for x in b["token_nodes"]][:n_target]
    r_mask = ~np.isnan(r) & ~np.isnan(sigma_raw)
    for sym in exclude:
        r_mask[:, symbols.index(sym)] = False

    return {
        "src": data.src.to(device), "dst": data.dst.to(device),
        "t": data.t.to(device), "msg": data.msg.to(device),
        "num_nodes": b["num_nodes"], "first_wallet": b["first_wallet_node"],
        "n_target": n_target, "H": H, "bounds": bounds, "r": r, "r_mask": r_mask,
        "sigma_raw": sigma_raw, "lr": lr[:H], "symbols": symbols, "exclude": list(exclude),
        "msg_dim": data.msg.size(1),
    }


def make_split(d: dict, split: tuple[float, float]):
    """Hour ranges for train/val/test over the hours that have any target, plus
    the floored sigma matrix (floor from training hours only)."""
    valid_h = np.where(d["r_mask"].any(axis=1))[0]
    c1, c2 = int(len(valid_h) * split[0]), int(len(valid_h) * split[1])
    tr = range(0, int(valid_h[c1]))
    va = range(int(valid_h[c1]), int(valid_h[c2]))
    te = range(int(valid_h[c2]), int(valid_h[-1]) + 1)
    s = d["sigma_raw"]
    floor = 0.5 * np.nanmedian(np.where(d["r_mask"][tr.start:tr.stop], s[tr.start:tr.stop], np.nan), axis=0)
    floor = np.where(np.isfinite(floor), floor, np.nanmedian(floor))
    sigma = np.fmax(s, floor[None, :])
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)   # only where there is no target anyway
    return tr, va, te, sigma


# --- metrics ----------------------------------------------------------------

def metrics(pred_z: np.ndarray, r: np.ndarray, mask: np.ndarray, sigma: np.ndarray) -> dict:
    """sigma: [H, n] (or broadcastable) scale used for z = r / sigma."""
    z = r / sigma
    pz, z_, rr = pred_z[mask], z[mask], r[mask]
    pr = (pred_z * sigma)[mask]
    mse, mse_naive = float(np.mean((pz - z_) ** 2)), float(np.mean(z_ ** 2))
    nz = rr != 0
    ics = []
    for h in range(pred_z.shape[0]):
        k = mask[h]
        if k.sum() >= 5 and np.std(pred_z[h, k]) > 0 and np.std(r[h, k]) > 0:
            ics.append(np.corrcoef(pred_z[h, k], r[h, k])[0, 1])
    return {
        "n": int(mask.sum()),
        "mse_z": mse, "mse_z_naive": mse_naive, "skill": 1 - mse / mse_naive,
        "rmse_r_pct": float(np.sqrt(np.mean((pr - rr) ** 2)) * 100),
        "rmse_r_pct_naive": float(np.sqrt(np.mean(rr ** 2)) * 100),
        "dir_acc": float(np.mean(np.sign(pr[nz]) == np.sign(rr[nz]))),
        "ic": float(np.mean(ics)) if ics else float("nan"),
        "pooled_corr": float(np.corrcoef(pr, rr)[0, 1]) if pr.std() > 0 else float("nan"),
    }


# --- one streaming pass -----------------------------------------------------

class Runner:
    def __init__(self, d: dict, args, device: torch.device):
        self.d, self.args, self.device = d, args, device
        dim = args.dim
        self.memory = TGNMemory(d["num_nodes"], d["msg_dim"], dim, dim,
                                IdentityMessage(d["msg_dim"], dim, dim), LastAggregator()).to(device)
        self.gnn = GraphAttentionEmbedding(dim, dim, d["msg_dim"], self.memory.time_enc).to(device)
        self.link = LinkPredictor(dim).to(device)
        self.decoder = ReturnDecoder(dim).to(device)
        self.loader = LastNeighborLoader(d["num_nodes"], size=args.neighbors, device=device)
        # The time encoder is shared by memory and gnn: dedupe, keep a fixed order.
        self.params = list(dict.fromkeys(
            [*self.memory.parameters(), *self.gnn.parameters(),
             *self.link.parameters(), *self.decoder.parameters()]))
        self.opt = torch.optim.Adam(self.params, lr=args.lr)
        self.assoc = torch.empty(d["num_nodes"], dtype=torch.long, device=device)
        self.token_ids = torch.arange(d["n_target"], device=device)
        self.gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def modules(self):
        return [self.memory, self.gnn, self.link, self.decoder]

    def state_dict(self):
        return {k: m.state_dict() for k, m in zip(["memory", "gnn", "link", "decoder"], self.modules())}

    def load_state_dict(self, sd):
        for k, m in zip(["memory", "gnn", "link", "decoder"], self.modules()):
            m.load_state_dict(sd[k])

    def embed(self, n_id):
        d = self.d
        n_id, edge_index, e_id = self.loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        z, last_update = self.memory(n_id)
        return self.gnn(z, last_update, edge_index, d["t"][e_id], d["msg"][e_id])

    def negatives(self, dst):
        """Same node type as the true destination: token for buys, wallet for sells."""
        d, n = self.d, dst.size(0)
        to_token = dst < d["first_wallet"]
        tok = torch.randint(0, d["first_wallet"], (n,), generator=self.gen)
        wal = torch.randint(d["first_wallet"], d["num_nodes"], (n,), generator=self.gen)
        return torch.where(to_token.cpu(), tok, wal).to(self.device)

    def run(self, hours_train: range, hours_eval: range, sigma: torch.Tensor) -> dict:
        """One pass from hour 0: train on hours_train, then predict hours_eval
        without gradients. Returns losses and eval predictions."""
        d, a = self.d, self.args
        self.memory.reset_state()
        self.loader.reset_state()
        preds = np.full((d["H"], d["n_target"]), np.nan)
        r = torch.tensor(np.nan_to_num(d["r"]), dtype=torch.float32, device=self.device)
        rmask = torch.tensor(d["r_mask"], device=self.device)
        reg_losses, aux_losses = [], []
        last = max(hours_eval.stop if len(hours_eval) else 0, hours_train.stop)

        for h in range(last):
            training = h in hours_train
            predicting = training or h in hours_eval
            for m in self.modules():
                m.train(training)
            s, e = int(d["bounds"][h]), int(d["bounds"][h + 1])

            for b0 in range(s, e, a.batch_size):
                b1 = min(b0 + a.batch_size, e)
                src, dst, t, msg = d["src"][b0:b1], d["dst"][b0:b1], d["t"][b0:b1], d["msg"][b0:b1]
                if training and a.aux_weight > 0:
                    self.opt.zero_grad()
                    neg = self.negatives(dst)
                    n_id = torch.cat([src, dst, neg]).unique()
                    z = self.embed(n_id)
                    zs = z[self.assoc[src]]
                    pos_out = self.link(zs, z[self.assoc[dst]])
                    neg_out = self.link(zs, z[self.assoc[neg]])
                    loss = a.aux_weight * (
                        F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
                        + F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out)))
                    self.memory.update_state(src, dst, t, msg)
                    self.loader.insert(src, dst)
                    loss.backward()
                    self.opt.step()
                    self.memory.detach()
                    aux_losses.append(loss.item())
                else:
                    with torch.no_grad():
                        self.memory.update_state(src, dst, t, msg)
                        self.loader.insert(src, dst)

            # end of hour h: forecast the return over hour h+1 -> h+2
            if not predicting or not d["r_mask"][h].any():
                continue
            if training:
                self.opt.zero_grad()
                z = self.embed(self.token_ids)
                pred = self.decoder(z[self.assoc[self.token_ids]])
                k = rmask[h]
                loss = F.huber_loss(pred[k], (r[h] / sigma[h])[k], delta=a.huber_delta)
                loss.backward()
                self.opt.step()
                self.memory.detach()
                reg_losses.append(loss.item())
            else:
                with torch.no_grad():
                    z = self.embed(self.token_ids)
                    preds[h] = self.decoder(z[self.assoc[self.token_ids]]).cpu().numpy()

        return {"reg_loss": float(np.mean(reg_losses)) if reg_losses else float("nan"),
                "aux_loss": float(np.mean(aux_losses)) if aux_losses else float("nan"),
                "preds": preds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data" / "processed" / "tgn_dex.pt")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--neighbors", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--split", type=float, nargs=2, default=(0.70, 0.85))
    ap.add_argument("--vol-window", type=int, default=48, help="hours of history for ex-ante sigma")
    ap.add_argument("--exclude", nargs="*", default=["SI"], help="tokens never used as targets")
    ap.add_argument("--max-hours", type=int, default=None, help="debug: use only the first N hours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    d = load(args.data, device, args.max_hours, args.vol_window, tuple(args.exclude))
    tr, va, te, sigma_np = make_split(d, tuple(args.split))
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)

    run_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update({"train_hours": [tr.start, tr.stop], "val_hours": [va.start, va.stop],
                "test_hours": [te.start, te.stop], "n_events": int(d["src"].numel()),
                "symbols": d["symbols"]})
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    np.save(run_dir / "sigma.npy", sigma_np)
    print(f"device {device} | events {d['src'].numel():,} | nodes {d['num_nodes']:,} | "
          f"hours train {len(tr)} val {len(va)} test {len(te)} | run dir {run_dir}", flush=True)

    runner = Runner(d, args, device)
    best, best_val, bad, history = None, float("inf"), 0, []
    mask = d["r_mask"]

    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        out = runner.run(tr, va, sigma)
        vm = np.zeros_like(mask)
        vm[va.start:va.stop] = mask[va.start:va.stop]
        val = metrics(np.nan_to_num(out["preds"]), d["r"], vm, sigma_np)
        rec = {"epoch": epoch, "reg_loss": out["reg_loss"], "aux_loss": out["aux_loss"],
               "val": val, "minutes": (time.monotonic() - t0) / 60}
        history.append(rec)
        print(f"epoch {epoch:2d} | {rec['minutes']:.1f} min | train huber {out['reg_loss']:.4f} "
              f"aux {out['aux_loss']:.4f} | val mse_z {val['mse_z']:.4f} (naive {val['mse_z_naive']:.4f}) "
              f"skill {val['skill']:+.4f} dir {val['dir_acc']:.3f} ic {val['ic']:+.4f}", flush=True)
        if val["mse_z"] < best_val:
            best_val, bad = val["mse_z"], 0
            best = {k: {kk: vv.detach().cpu().clone() for kk, vv in v.items()}
                    for k, v in runner.state_dict().items()}
            torch.save(best, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs", flush=True)
                break

    # final: best weights, one pass with no training, predict val and test
    runner.load_state_dict(best)
    out = runner.run(range(0), range(0, te.stop), sigma)
    res = {}
    for name, rg in (("val", va), ("test", te)):
        mm = np.zeros_like(mask)
        mm[rg.start:rg.stop] = mask[rg.start:rg.stop]
        res[name] = metrics(np.nan_to_num(out["preds"]), d["r"], mm, sigma_np)
    np.save(run_dir / "preds_z.npy", out["preds"])
    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "final": res}, indent=2))
    for name in ("val", "test"):
        m = res[name]
        print(f"{name:4s} | skill {m['skill']:+.4f} | rmse_r {m['rmse_r_pct']:.4f}% "
              f"(naive {m['rmse_r_pct_naive']:.4f}%) | dir_acc {m['dir_acc']:.3f} | "
              f"ic {m['ic']:+.4f} | pooled corr {m['pooled_corr']:+.4f} | n {m['n']}", flush=True)


if __name__ == "__main__":
    main()

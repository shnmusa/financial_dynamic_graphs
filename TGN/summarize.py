"""Collect TGN, GRU and static-GNN runs and the latest baselines into one results table.

TGN runs are grouped by (dataset file, aux_weight, vol_window, exclude), one table
per dataset, and reported as
mean ± std over seeds. Only runs whose config has vol_window (the rolling-
volatility setup) are included, so the first fixed-sigma run is left out.

    TGN/.venv/bin/python TGN/summarize.py            # prints and writes runs/results.md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
COLS = [("skill", "{:+.4f}"), ("rmse_r_pct", "{:.4f}"), ("dir_acc", "{:.3f}"), ("ic", "{:+.4f}")]


def fmt(vals, f):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f.format(vals[0])
    return f"{f.format(np.mean(vals))} ± {np.std(vals):.4f}"


def dataset(cfg: dict) -> str:
    return Path(cfg.get("data", "tgn_dex.pt")).name


def table(rows) -> list[str]:
    header = ("| model | seeds | " + " | ".join(f"{s} {c}" for s in ("val", "test") for c, _ in COLS)
              + " |")
    lines = [header, "|" + "---|" * (2 + 2 * len(COLS))]
    for name, n, res in rows:
        cells = [fmt([m[c] for m in res[s]], f) for s in ("val", "test") for c, f in COLS]
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    groups = defaultdict(list)
    for run in sorted(RUNS.glob("2*")):
        cfg_p, met_p = run / "config.json", run / "metrics.json"
        if not (cfg_p.exists() and met_p.exists()):
            continue
        cfg = json.loads(cfg_p.read_text())
        if "vol_window" not in cfg or cfg.get("max_hours"):
            continue
        key = (dataset(cfg), cfg["aux_weight"], cfg["vol_window"], tuple(cfg.get("exclude", [])))
        groups[key].append((cfg["seed"], json.loads(met_p.read_text())["final"], run.name))

    baselines = {}                      # dataset -> latest baseline results
    for b in sorted(RUNS.glob("baselines-*")):
        m = json.loads((b / "metrics.json").read_text())
        baselines[dataset(m["config"])] = m["results"]

    nn_runs = defaultdict(list)        # (dataset, gru|gnn) -> final metrics per seed
    for run in sorted(list(RUNS.glob("gru-*")) + list(RUNS.glob("gnn-*"))):
        cfg_p, met_p = run / "config.json", run / "metrics.json"
        if not met_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        if cfg.get("max_hours"):
            continue
        nn_runs[(dataset(cfg), cfg["model"])].append(json.loads(met_p.read_text())["final"])

    out = []
    for ds in sorted(set(baselines) | {k[0] for k in groups}):
        rows = []
        if ds in baselines:
            res = baselines[ds]
            for name in ("zero", "train_mean", "last", "ar1", "flow_ridge"):
                rows.append((name, 1, {s: [res[name][s]] for s in ("val", "test")}))
        for (_, aux, vw, exc), runs in sorted(((k, v) for k, v in groups.items() if k[0] == ds),
                                              key=lambda x: -x[0][1]):
            label = f"TGN (aux link loss {'on' if aux > 0 else 'off'})"
            rows.append((label, len(runs), {s: [r[1][s] for r in runs] for s in ("val", "test")}))
        for kind, label in (("gru", "GRU (temporal, no graph)"), ("gnn", "Static GNN (graph, no memory)")):
            runs = nn_runs.get((ds, kind), [])
            if runs:
                rows.append((label, len(runs), {s: [r[s] for r in runs] for s in ("val", "test")}))
        out += [f"### dataset: {ds}", ""] + table(rows) + [""]
    note = ("skill = 1 - MSE(z) / MSE(z, predict 0); rmse_r in % log-return; dir_acc on non-zero "
            "returns (0 for the zero forecast by construction); ic = mean hourly cross-sectional "
            "Pearson correlation. Targets: next-hour log-return / ex-ante rolling volatility.\n")
    text = "\n".join(out) + "\n" + note
    (RUNS / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

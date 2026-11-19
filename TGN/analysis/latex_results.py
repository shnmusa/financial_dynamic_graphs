"""Write the forecasting-results table for the paper (TGN/paper/tables/results.tex).

Reads the runs in TGN/runs for one dataset (default: the wallet-filtered
tgn_dex_min3_vol0.pt): the latest baselines run (zero, flow_ridge), and all
TGN, GRU and static-GNN seeds. Neural models are reported as mean (standard
deviation over seeds). Skill is printed in percent.

    TGN/.venv/bin/python TGN/analysis/latex_results.py [--dataset tgn_dex.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS, OUT = ROOT / "runs", ROOT / "paper" / "tables"

# (metric, split, column header, scale, decimals)
COLS = [("dir_acc", "val", "Val.", 1, 3), ("dir_acc", "test", "Test", 1, 3),
        ("skill", "val", "Val.", 100, 2), ("skill", "test", "Test", 100, 2),
        ("ic", "val", "Val.", 1, 3), ("ic", "test", "Test", 1, 3),
        ("rmse_r_pct", "test", "Test", 1, 4)]
# Each model takes two table rows: means on the first, seed standard deviations
# (small, in parentheses) on the second. Model names are split over the same two
# rows, which keeps both the name column and the number columns narrow.
DISPLAY = {"Zero forecast": ("Zero", "forecast"), "Flow ridge regression": ("Flow ridge", "regression"),
           "Static GNN": ("Static", "GNN"), "TGN + link loss": ("TGN +", "link loss")}
HIGHER_IS_BETTER = {"dir_acc": True, "skill": True, "ic": True, "rmse_r_pct": False}


def ds(cfg: dict) -> str:
    return Path(cfg.get("data", "tgn_dex.pt")).name


def collect(dataset: str) -> list[tuple[str, list[dict]]]:
    base = None
    for b in sorted(RUNS.glob("baselines-*")):
        m = json.loads((b / "metrics.json").read_text())
        if ds(m["config"]) == dataset:
            base = m["results"]
    tgn = {True: [], False: []}
    nn = {"gru": [], "gnn": []}
    for run in sorted(RUNS.iterdir()):
        cp, mp = run / "config.json", run / "metrics.json"
        if not (cp.exists() and mp.exists()):
            continue
        cfg = json.loads(cp.read_text())
        if ds(cfg) != dataset or cfg.get("max_hours"):
            continue
        final = json.loads(mp.read_text())["final"]
        if "model" in cfg:
            nn[cfg["model"]].append(final)
        elif "vol_window" in cfg:
            tgn[cfg["aux_weight"] > 0].append(final)
    return [("Zero forecast", [base["zero"]]),
            ("Flow ridge regression", [base["flow_ridge"]]),
            ("GRU", nn["gru"]),
            ("Static GNN", nn["gnn"]),
            ("TGN", tgn[False]),
            ("TGN + link loss", tgn[True])]


def fmt(v: float, dec: int) -> str:
    s = f"{abs(v):.{dec}f}"
    return f"$-${s}" if v < 0 and round(abs(v), dec) > 0 else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tgn_dex_min3_vol0.pt")
    args = ap.parse_args()
    rows = collect(args.dataset)

    stats = {}
    for name, runs in rows:
        for c in COLS:
            key, split, _, scale, _ = c
            vals = [r[split][key] * scale for r in runs]
            if name == "Zero forecast" and key in ("dir_acc", "ic"):
                stats[name, c] = None
            else:
                stats[name, c] = (np.mean(vals), np.std(vals) if len(vals) > 1 else None)
    best = {}
    for c in COLS:
        means = [stats[n, c][0] for n, _ in rows if stats[n, c] is not None and n != "Zero forecast"]
        best[c] = max(means) if HIGHER_IS_BETTER[c[0]] else min(means)

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Forecasting performance on the wallet-filtered data set. Targets are next-hour "
        r"log-returns of the 35 target tokens. Direction accuracy is computed on non-zero returns; "
        r"$R^2_{\mathrm{OOS}}$ (in \%) is measured against the zero forecast; IC is the mean hourly "
        r"cross-sectional correlation; RMSE is the root-mean-squared error of the log-return in "
        r"percentage points. Neural models: mean over three seeds, with the standard deviation in "
        r"parentheses below. Bold: best model in each column (zero forecast excluded).}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "r" * len(COLS) + "}", r"\toprule",
        r"& \multicolumn{2}{c}{Direction accuracy} & \multicolumn{2}{c}{$R^2_{\mathrm{OOS}}$ (\%)} "
        r"& \multicolumn{2}{c}{IC} & RMSE (\%) \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-8}",
        "Model & " + " & ".join(c[2] for c in COLS) + r" \\", r"\midrule"]
    for k, (name, _) in enumerate(rows):
        means, sds = [], []
        for c in COLS:
            st = stats[name, c]
            if st is None:
                means.append("--")
                sds.append("")
                continue
            m, s = st
            txt = fmt(m, c[4])
            if name != "Zero forecast" and round(m, c[4]) == round(best[c], c[4]):
                txt = rf"\textbf{{{txt}}}"
            means.append(txt)
            sds.append(rf"{{\scriptsize ({s:.{c[4]}f})}}" if s is not None else "")
        top, bottom = DISPLAY.get(name, (name, ""))
        lines.append(top + " & " + " & ".join(means) + r" \\")
        if bottom or any(sds):
            lines.append(bottom + " & " + " & ".join(sds) + r" \\")
        if k < len(rows) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.tex").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

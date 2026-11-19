"""Write the descriptive-statistics tables for the paper as LaTeX.

Reads results/descriptives*.json (wallet_descriptives.py) and writes
TGN/paper/tables/desc_{trades,volume}.tex, one column per wallet subset.

    TGN/.venv/bin/python TGN/analysis/latex_descriptives.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parent / "paper" / "tables"

SUBSETS = [  # (file suffix, column header)
    ("", r"All"),
    ("_min2", r"$\geq 2$"),
    ("_min3", r"$\geq 3$"),
    ("_min3_posvol", r"$\geq 3$, $V>0$"),
    ("_min3_minvol1", r"$\geq 3$, $V\geq 1$"),
]

ROWS = [  # (key, label, kind)
    ("n", r"Wallets", "int"),
    ("mean", r"Mean", "num"),
    ("std", r"Std.\ dev.", "num"),
    ("min", r"Min", "num"),
    ("p25", r"25th pct.", "num"),
    ("p50", r"Median", "num"),
    ("p75", r"75th pct.", "num"),
    ("p90", r"90th pct.", "num"),
    ("p99", r"99th pct.", "num"),
    ("max", r"Max", "num"),
    ("geo_mean", r"Geometric mean", "num"),
    ("skewness", r"Skewness", "num"),
    ("excess_kurtosis", r"Excess kurtosis", "num"),
    ("log10_skewness", r"Skewness of $\log_{10}$", "dec"),
    ("log10_excess_kurtosis", r"Excess kurtosis of $\log_{10}$", "dec"),
    ("gini", r"Gini coefficient", "dec3"),
    ("top_1pct_share", r"Share of top 1\%", "pct"),
    ("top_10pct_share", r"Share of top 10\%", "pct"),
    ("bottom_50pct_share", r"Share of bottom 50\%", "pct"),
]


def num(v: float) -> str:
    """Large values abbreviated, integers as integers, otherwise 3 significant-ish digits."""
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.2f}\\,B"
    if a >= 1e6:
        return f"{v / 1e6:.2f}\\,M"
    if a >= 1000:
        return f"{v:,.0f}".replace(",", "{,}")
    if float(v).is_integer():
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.01:
        return f"{v:.2f}"
    return f"${v:.1e}".replace("e-0", r"\times 10^{-") + "}$"


def signed(v: float, digits: int) -> str:
    return (f"$-${abs(v):.{digits}f}" if v < 0 else f"{v:.{digits}f}")


def cell(v: float, kind: str) -> str:
    if kind == "int":
        return f"{int(v):,}".replace(",", "{,}")
    if kind == "dec":
        return signed(v, 2)
    if kind == "dec3":
        return f"{v:.3f}"
    if kind == "pct":
        return f"{100 * v:.1f}\\%" if v >= 0.001 else f"{100 * v:.2f}\\%"
    return num(v)


def table(var: str, caption: str, label: str, data: list[dict], extra: dict) -> str:
    cols = "l" + "r" * len(SUBSETS)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             r"& \multicolumn{" + str(len(SUBSETS)) + r"}{c}{Wallets with trades $\geq k$ and volume $V$} \\",
             r"\cmidrule(lr){2-" + str(len(SUBSETS) + 1) + "}",
             "Statistic & " + " & ".join(h for _, h in SUBSETS) + r" \\", r"\midrule"]
    for key, lab, kind in ROWS:
        lines.append(lab + " & " + " & ".join(cell(d[var][key], kind) for d in data) + r" \\")
        if key in ("n", "geo_mean", "log10_excess_kurtosis"):
            lines.append(r"\addlinespace")
    for lab, vals in extra.items():
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def powerlaw_table() -> str | None:
    path = RES / "powerlaw_fits.json"
    if not path.exists():
        return None
    f = json.loads(path.read_text())
    alts = [("lognormal", "Lognormal"), ("exponential", "Exponential"),
            ("stretched_exponential", "Stretched exp."), ("truncated_power_law", "Truncated power law")]

    def p(v):
        return r"$<10^{-3}$" if v < 1e-3 else f"{v:.3f}"

    rows = []
    for key, lab in (("trades", "Trades per wallet"), ("volume", "Volume per wallet (USD)")):
        r = f[key]
        xmin = f"{r['xmin']:,.0f}".replace(",", "{,}")
        rows.append(f"{lab} & {xmin} & {r['n_tail']:,}".replace(",", "{,}").replace("{{,}}", "{,}")
                    + f" ({100 * r['tail_share']:.1f}\\%) & {r['alpha']:.3f} ({r['alpha_se']:.3f}) & "
                    f"{r['D']:.4f} & {p(r['gof_p'])} \\\\")
    lr_rows = []
    for key, lab in (("trades", "Trades per wallet"), ("volume", "Volume per wallet (USD)")):
        cells = []
        for a, _ in alts:
            R, pv = f[key]["lr"][a]["R"], f[key]["lr"][a]["p"]
            cells.append(f"{signed(R, 2)} ({p(pv)})")
        lr_rows.append(lab + r" & \multicolumn{2}{r}{" + cells[0] + "} & "
                       + " & ".join(cells[1:]) + r" \\")
    sims = f["trades"]["gof_sims"]
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Power-law fits to the upper tails of the per-wallet distributions "
        r"\citep{clauset2009power}. $x_{\min}$ minimises the Kolmogorov--Smirnov distance $D$; "
        r"$\hat\alpha$ is the maximum-likelihood exponent above $x_{\min}$ (standard error in "
        rf"parentheses); $p_{{\mathrm{{GoF}}}}$ is the bootstrap goodness-of-fit $p$-value from {sims} "
        r"semi-parametric replications. Panel B reports the normalised log-likelihood ratio $R$ "
        r"(power law vs.\ alternative; $p$-value in parentheses): $R>0$ favours the power law, "
        r"$R<0$ the alternative.}",
        r"\label{tab:powerlaw}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"\multicolumn{6}{l}{\textit{Panel A: power-law fit}} \\",
        r"& $x_{\min}$ & $n_{\mathrm{tail}}$ & $\hat\alpha$ & $D$ & $p_{\mathrm{GoF}}$ \\", r"\midrule",
        *rows, r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel B: likelihood-ratio tests, $R$ ($p$)}} \\",
        r"& \multicolumn{2}{r}{Lognormal} & Exponential & Stretched exp. & Truncated PL \\", r"\midrule",
        *lr_rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = [json.loads((RES / f"descriptives{s}.json").read_text()) for s, _ in SUBSETS]
    single = {r"Single-trade wallets": [f"{100 * d['share_single_trade_wallets']:.1f}\\%" for d in data]}
    rho = {r"Spearman $\rho$ (trades, volume)": [f"{d['spearman_trades_volume']:.2f}" for d in data]}
    (OUT / "desc_trades.tex").write_text(table(
        "trades",
        r"Descriptive statistics of the number of trades per wallet. A trade is one "
        r"wallet's activity within one block. Columns restrict the sample to wallets "
        r"with at least $k$ trades and total volume $V$ (USD).",
        "tab:desc_trades", data, single))
    (OUT / "desc_volume.tex").write_text(table(
        "volume_usd",
        r"Descriptive statistics of the total traded volume per wallet (USD). The "
        r"volume of a trade is the larger of the USD values the wallet sold and bought "
        r"in that block. Columns as in Table~\ref{tab:desc_trades}.",
        "tab:desc_volume", data, rho))
    pl = powerlaw_table()
    if pl:
        (OUT / "powerlaw.tex").write_text(pl)
        print(pl)
    print((OUT / "desc_trades.tex").read_text())
    print((OUT / "desc_volume.tex").read_text())


if __name__ == "__main__":
    main()

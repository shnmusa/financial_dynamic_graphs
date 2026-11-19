"""Build the TGN input (PyG TemporalData + hourly price targets) from dex_events.

Graph
-----
Nodes (one id space; counts come from tokens.target at build time --
currently 36 + 3):
    0 .. T-1        the T non-stable target tokens -> price-prediction targets,
                    column j of `y_price` is node j
    T .. T+S-1      the S target stablecoins (USDC, USDT, DAI)
    T+S             UNKNOWN: every token outside tokens.target
    T+S+1 ..        wallets: the non-pool side of a swap leg (EOAs, bots,
                    routers, aggregators -- anything that is not a pool)

Events (edges), from the wallet's point of view of one transaction:
    For every (tx, wallet, token) the wallet-side legs are netted:
        paid     = sum of the token the wallet sent into pools
        received = sum of the token the wallet got out of pools
    Pool-to-pool rows are hops of a route and are skipped. A token whose net
    flow is under 1% of its gross flow is a routing intermediate (a router
    receiving and forwarding it), not a trade, and produces no event.
        net sold   (paid > received):  token  -> wallet   side = -1
        net bought (received > paid):  wallet -> token    side = +1
    usd = |paid_usd - received_usd|, from dex_events.usd_amount (CoinGecko).
    A (tx, wallet, token) with any unpriced leg is dropped, so every event has
    a dollar value. dex_events_removed is never read.

    msg = [side, log1p(usd)]          (float32, D = 2)
    t   = block timestamp, unix seconds; events are ordered by
          (t, block, first log_index, tx, sell-before-buy)

Targets
-------
    y_price[h, j]  CoinGecko USD price of token node j at hour hours[h]
                   (token_prices_hourly, hop 1); NaN where CoinGecko has no
                   price (e.g. SI before its 2026-09-22 listing)
    y_mask[h, j]   True where y_price is present
They are hour-indexed, not event-indexed, so they are stored next to the
TemporalData rather than in it (TemporalData slicing assumes one row per event).

Output: TGN/data/processed/tgn_dex.pt  (torch.load(..., weights_only=False))
        TGN/data/processed/wallets.parquet  (node id -> wallet address)
        TGN/data/processed/meta.json

    TGN/.venv/bin/python TGN/data/build_temporal_data.py
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import torch
from dotenv import dotenv_values
from torch_geometric.data import TemporalData

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "processed"
DSN = dotenv_values(ROOT / "dex_data" / ".env")["DEX_PG_DSN"]

INTERMEDIATE_TOLERANCE = 0.01

T0 = time.monotonic()


def step(msg: str) -> None:
    print(f"[{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


def copy_df(conn: psycopg.Connection, query: str) -> pd.DataFrame:
    buf = io.BytesIO()
    with conn.cursor().copy(f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER true)") as cp:
        for chunk in cp:
            buf.write(chunk)
    buf.seek(0)
    return pd.read_csv(buf)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = psycopg.connect(DSN)
    conn.execute("SET max_parallel_workers_per_gather = 0")   # small /dev/shm in the container
    conn.execute("SET work_mem = '512MB'")

    # --- token nodes ------------------------------------------------------
    tokens = conn.execute("""
        SELECT address, symbol, is_stable FROM tokens WHERE target
        ORDER BY is_stable, market_cap_usd DESC NULLS LAST, address""").fetchall()
    targets = [t for t in tokens if not t[2]]
    stables = [t for t in tokens if t[2]]
    N_TARGET, N_STABLE = len(targets), len(stables)
    UNKNOWN_ID = N_TARGET + N_STABLE
    FIRST_WALLET_ID = UNKNOWN_ID + 1
    token_nodes = targets + stables
    conn.execute("CREATE TEMP TABLE token_node (address text PRIMARY KEY, node int)")
    with conn.cursor().copy("COPY token_node FROM STDIN") as cp:
        for i, (addr, _, _) in enumerate(token_nodes):
            cp.write_row((addr, i))
    step(f"token nodes: {N_TARGET} targets + {N_STABLE} stables + UNKNOWN")

    # --- net (tx, wallet, token) flows ------------------------------------
    conn.execute("""
        CREATE TEMP TABLE flows AS
        SELECT e.tx_hash,
               CASE WHEN pd.address IS NOT NULL THEN e.source ELSE e.destination END AS wallet,
               e.token_address,
               min(extract(epoch FROM e.timestamp))::bigint AS t,
               min(e.block_number) AS block_number,
               min(e.log_index)    AS log_index,
               coalesce(sum(e.amount)     FILTER (WHERE pd.address IS NOT NULL), 0) AS paid,
               coalesce(sum(e.amount)     FILTER (WHERE pd.address IS NULL), 0)     AS received,
               coalesce(sum(e.usd_amount) FILTER (WHERE pd.address IS NOT NULL), 0) AS paid_usd,
               coalesce(sum(e.usd_amount) FILTER (WHERE pd.address IS NULL), 0)     AS received_usd,
               bool_and(e.usd_amount IS NOT NULL) AS all_priced
        FROM dex_events e
        LEFT JOIN pools ps ON ps.address = e.source
        LEFT JOIN pools pd ON pd.address = e.destination
        WHERE NOT (ps.address IS NOT NULL AND pd.address IS NOT NULL)   -- route hops
        GROUP BY 1, 2, 3""")
    n_flows = conn.execute("SELECT count(*) FROM flows").fetchone()[0]
    step(f"net (tx, wallet, token) flows: {n_flows:,}")

    stats = conn.execute("""
        SELECT count(*) FILTER (WHERE paid + received = 0
                                   OR abs(paid - received) / nullif(paid + received, 0) < %(tol)s),
               count(*) FILTER (WHERE NOT (paid + received = 0
                                   OR abs(paid - received) / nullif(paid + received, 0) < %(tol)s)
                                AND NOT all_priced)
        FROM flows""", {"tol": INTERMEDIATE_TOLERANCE}).fetchone()
    step(f"dropped: {stats[0]:,} intermediate (nets to ~0), {stats[1]:,} with an unpriced leg")

    conn.execute("""
        CREATE TEMP TABLE ev AS
        SELECT f.*,
               CASE WHEN f.paid > f.received THEN -1 ELSE 1 END AS side,
               abs(f.paid_usd - f.received_usd) AS usd,
               coalesce(tn.node, %(unknown)s) AS token_node
        FROM flows f LEFT JOIN token_node tn ON tn.address = f.token_address
        WHERE f.paid + f.received > 0
          AND abs(f.paid - f.received) / nullif(f.paid + f.received, 0) >= %(tol)s
          AND f.all_priced""", {"tol": INTERMEDIATE_TOLERANCE, "unknown": UNKNOWN_ID})

    # Wallet ids in order of first appearance.
    conn.execute("""
        CREATE TEMP TABLE wallet_node AS
        SELECT wallet, %(first)s - 1 + row_number() OVER (ORDER BY min(t), min(block_number),
                                                         min(log_index), wallet) AS node
        FROM ev GROUP BY wallet""", {"first": FIRST_WALLET_ID})
    conn.execute("CREATE INDEX ON wallet_node (wallet)")
    step("events and wallet ids built in Postgres")

    ev = copy_df(conn, """
        SELECT ev.t, ev.block_number, ev.log_index, ev.side, ev.usd, ev.token_node,
               w.node AS wallet_node, ev.tx_hash
        FROM ev JOIN wallet_node w USING (wallet)
        ORDER BY ev.t, ev.block_number, ev.log_index, ev.tx_hash, ev.side""")
    wallets = copy_df(conn, "SELECT node, wallet FROM wallet_node ORDER BY node")
    step(f"fetched {len(ev):,} events, {len(wallets):,} wallets")

    # --- TemporalData -------------------------------------------------------
    sell = ev["side"].to_numpy() == -1
    tok = ev["token_node"].to_numpy(np.int64)
    wal = ev["wallet_node"].to_numpy(np.int64)
    src = np.where(sell, tok, wal)          # sold token -> wallet
    dst = np.where(sell, wal, tok)          # wallet -> bought token
    t = ev["t"].to_numpy(np.int64)
    assert np.all(np.diff(t) >= 0), "events must be in time order"
    usd = ev["usd"].to_numpy(np.float64)
    assert np.isfinite(usd).all() and (usd >= 0).all()
    msg = np.stack([ev["side"].to_numpy(np.float32), np.log1p(usd).astype(np.float32)], axis=1)

    data = TemporalData(
        src=torch.from_numpy(src), dst=torch.from_numpy(dst),
        t=torch.from_numpy(t), msg=torch.from_numpy(msg))
    num_nodes = FIRST_WALLET_ID + len(wallets)

    # --- hourly price targets ----------------------------------------------
    hours = pd.date_range(
        pd.to_datetime(t.min(), unit="s", utc=True).floor("h"),
        pd.to_datetime(t.max(), unit="s", utc=True).floor("h"), freq="h")
    px = pd.DataFrame(conn.execute("""
        SELECT tn.node, p.hour, p.usd_price::float8
        FROM token_prices_hourly p JOIN token_node tn ON tn.address = p.token
        WHERE tn.node < %(n)s""", {"n": N_TARGET}).fetchall(), columns=["node", "hour", "price"])
    px["hour"] = pd.to_datetime(px["hour"], utc=True)
    grid = px.pivot(index="hour", columns="node", values="price").reindex(
        index=hours, columns=range(N_TARGET))
    y_price = torch.tensor(grid.to_numpy(), dtype=torch.float64)
    y_mask = ~torch.isnan(y_price)
    step(f"price targets: {tuple(y_price.shape)} (hours x tokens), {int((~y_mask).sum())} missing")

    # --- save ---------------------------------------------------------------
    bundle = {
        "data": data,
        "num_nodes": num_nodes,
        "y_price": y_price,
        "y_mask": y_mask,
        "hours": torch.tensor(hours.as_unit("s").asi8, dtype=torch.long),   # unix seconds, hour start
        "token_nodes": [{"node": i, "address": a, "symbol": s, "is_stable": st}
                        for i, (a, s, st) in enumerate(token_nodes)],
        "unknown_node": UNKNOWN_ID,
        "first_wallet_node": FIRST_WALLET_ID,
        "msg_features": ["side (+1 buy: wallet->token, -1 sell: token->wallet)", "log1p(usd)"],
    }
    torch.save(bundle, OUT / "tgn_dex.pt")
    wallets.to_parquet(OUT / "wallets.parquet", index=False)

    n_sell, n_buy = int(sell.sum()), int((~sell).sum())
    meta = {
        "events": len(ev), "sell_events": n_sell, "buy_events": n_buy,
        "num_nodes": num_nodes, "wallets": len(wallets),
        "target_token_nodes": N_TARGET, "stable_nodes": N_STABLE, "unknown_node": UNKNOWN_ID,
        "t_min": int(t.min()), "t_max": int(t.max()),
        "hours": len(hours), "missing_prices": int((~y_mask).sum()),
        "dropped_intermediate": stats[0], "dropped_unpriced": stats[1],
        "events_touching_unknown": int((tok == UNKNOWN_ID).sum()),
        "usd_quantiles": {q: float(np.quantile(usd, q)) for q in (0.01, 0.5, 0.99)},
        "msg_features": bundle["msg_features"],
        "token_nodes": [(i, s) for i, (_, s, _) in enumerate(token_nodes)],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    step(f"saved {OUT / 'tgn_dex.pt'}")
    print(json.dumps({k: v for k, v in meta.items() if k != "token_nodes"}, indent=2))


if __name__ == "__main__":
    main()

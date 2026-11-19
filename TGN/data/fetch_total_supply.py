"""Hourly on-chain totalSupply() for every tokens.target token -> token_supply_hourly.

For each hour in the blocks table window, reads totalSupply() at the LAST block
of that hour (an end-of-hour snapshot) through Multicall3 using dex_data's own
RpcClient (rate limiting, endpoint rotation). One multicall per hour.

    cd dex_data && PYTHONPATH=src .venv/bin/python ../TGN/data/fetch_total_supply.py

Re-runnable: rows are upserted on (token, hour).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from dex_data import abi, db
from dex_data.rpc import get_client

TOTAL_SUPPLY = "0x18160ddd"

DDL = """
CREATE TABLE IF NOT EXISTS token_supply_hourly (
    token            TEXT NOT NULL REFERENCES tokens(address),
    hour             TIMESTAMPTZ NOT NULL,
    block_number     BIGINT NOT NULL,
    total_supply_raw NUMERIC(78,0),          -- NULL if the call reverted
    total_supply     NUMERIC(78,18),         -- raw / 10^decimals
    PRIMARY KEY (token, hour)
)"""


def main() -> int:
    db.execute(DDL)
    tokens = db.fetch_all(
        "SELECT address, symbol, decimals FROM tokens WHERE target ORDER BY address")
    hours = db.fetch_all("""
        SELECT date_trunc('hour', timestamp) AS hour, max(number) AS block
        FROM blocks GROUP BY 1 ORDER BY 1""")
    print(f"{len(tokens)} tokens x {len(hours)} hours", flush=True)

    rpc = get_client()
    calls = [(addr, TOTAL_SUPPLY) for addr, _, _ in tokens]

    def fetch(item):
        hour, block = item
        return hour, block, rpc.multicall(calls, block)

    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, (hour, block, results) in enumerate(pool.map(fetch, hours), 1):
            for (addr, _, dec), res in zip(tokens, results):
                raw = abi.decode_uint(res) if res else None
                failed += raw is None
                human = None if raw is None or dec is None else Decimal(raw).scaleb(-dec)
                rows.append((addr, hour, block, raw, human))
            if i % 100 == 0:
                print(f"  {i}/{len(hours)} hours", flush=True)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE s (LIKE token_supply_hourly) ON COMMIT DROP")
            with cur.copy("COPY s FROM STDIN") as cp:
                for r in rows:
                    cp.write_row(r)
            cur.execute("""
                INSERT INTO token_supply_hourly SELECT * FROM s
                ON CONFLICT (token, hour) DO UPDATE SET
                    block_number = EXCLUDED.block_number,
                    total_supply_raw = EXCLUDED.total_supply_raw,
                    total_supply = EXCLUDED.total_supply""")
        conn.commit()
    print(f"wrote {len(rows)} rows ({failed} reverted/empty calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""USD pricing from CoinGecko hourly prices.

Two steps, both re-runnable:

  build()              rewrites token_prices_hourly for the blocks window:
                         hop 0  every `is_stable` token at exactly $1
                         hop 1  every non-stable `tokens.target` token, from
                                CoinGecko's /coins/{id}/market_chart/range,
                                which is hourly for windows of 1-90 days
  apply_usd_amounts()  clears every usd_amount in the table, then values each
                       row as amount x the price of its token in that hour, and
                       finally mirrors a priced leg onto the unpriced other leg
                       of the same swap (always a non-target token, since only
                       targets and stablecoins have a price), then runs
                       fill_gaps() to carry values along multi-hop routes.

This replaced an on-chain pricing pass that derived prices from the swaps
themselves. That pass left whole days unpriced when it did not finish and let
broken-decimals long-tail tokens through at 1e25 dollars; an external hourly
price has neither problem for the tokens that matter (the target universe).

Why the last price at or before the row's hour, up to PRICE_STALENESS old, and
not an exact-hour match: CoinGecko's points land a few seconds either side of
the hour, so a rare hour has no point of its own. A token CoinGecko only lists
part-way through the window (cirBTC, SI) is NULL before its listing, and its
legs then take the other leg's value through the mirror step.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import requests
from psycopg import sql

from . import db
from .config import CONFIG, Config
from .marketcap import CG_API, _get

log = logging.getLogger(__name__)

# hop column of token_prices_hourly: where the price came from.
HOP_STABLE = 0
HOP_COINGECKO = 1

# A price may be carried forward at most this long when valuing a row.
PRICE_STALENESS = "24 hours"

# The keyless public tier allows ~5 calls/min; a demo key ~30/min.
CG_PAUSE_KEYLESS = 13.0
CG_PAUSE_KEYED = 2.5

# Tables apply_usd_amounts may write. dex_events_removed holds rows taken out
# of dex_events by cleanup passes, with the same columns.
PRICEABLE_TABLES = ("dex_events", "dex_events_removed")

# (coingecko_id, start, end) -> [[unix_ms, price], ...]
PriceFetcher = Callable[[str, datetime, datetime], list]


def coingecko_fetcher(api_key: str | None = None) -> PriceFetcher:
    """Hourly USD prices for one CoinGecko id over [start, end]."""
    session = requests.Session()
    session.headers["User-Agent"] = "dex-data/0.1"
    if api_key:
        session.headers["x-cg-demo-api-key"] = api_key
    pause = CG_PAUSE_KEYED if api_key else CG_PAUSE_KEYLESS

    def fetch(cg_id: str, start: datetime, end: datetime) -> list:
        response = _get(session, f"coins/{cg_id}/market_chart/range", {
            "vs_currency": "usd",
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
        })
        response.raise_for_status()
        time.sleep(pause)
        return response.json().get("prices") or []

    return fetch


def _hourly(points: list, start: datetime, end: datetime) -> dict[datetime, tuple[Decimal, int]]:
    """Floor each point to its hour; keep the hour's last price and the count."""
    out: dict[datetime, tuple[Decimal, int]] = {}
    for ms, price in points:
        if price is None or price <= 0:
            continue
        hour = datetime.fromtimestamp(ms / 1000, timezone.utc).replace(
            minute=0, second=0, microsecond=0)
        if start <= hour <= end:
            out[hour] = (Decimal(str(price)), out.get(hour, (0, 0))[1] + 1)
    return out


def build(fetch: PriceFetcher | None = None, api_key: str | None = None) -> dict[str, int]:
    """Rewrite token_prices_hourly: stablecoins at $1, targets from CoinGecko.

    All network calls happen first; the table is truncated and refilled in a
    single transaction afterwards, so a failed fetch leaves it untouched.
    """
    lo, hi = db.fetch_one("SELECT min(timestamp), max(timestamp) FROM blocks")
    if lo is None:
        log.warning("blocks is empty; no window to price")
        return {}
    start = lo.replace(minute=0, second=0, microsecond=0)
    end = hi.replace(minute=0, second=0, microsecond=0)

    tokens = db.fetch_all(
        "SELECT address, symbol, coingecko_id FROM tokens "
        "WHERE target AND NOT is_stable ORDER BY address")
    fetch = fetch or coingecko_fetcher(api_key)

    rows: list[tuple] = []
    missing: list[str] = []
    for i, (address, symbol, cg_id) in enumerate(tokens, 1):
        if not cg_id:
            missing.append(symbol)
            continue
        # Padded so the first and last hour of the window get a point too.
        hourly = _hourly(fetch(cg_id, start - timedelta(hours=2), end + timedelta(hours=2)),
                         start, end)
        if not hourly:
            missing.append(symbol)
        rows += [(address, h, p, n, HOP_COINGECKO) for h, (p, n) in hourly.items()]
        log.info("coingecko %d/%d %s: %d hours", i, len(tokens), symbol, len(hourly))

    with db.connection() as conn:
        conn.execute("TRUNCATE token_prices_hourly")
        stable = conn.execute("""
            INSERT INTO token_prices_hourly (token, hour, usd_price, n_obs, hop)
            SELECT t.address, h.hour, 1, 0, %(hop)s
            FROM tokens t
            CROSS JOIN generate_series(%(start)s::timestamptz, %(end)s::timestamptz,
                                       interval '1 hour') AS h(hour)
            WHERE t.is_stable""",
            {"hop": HOP_STABLE, "start": start, "end": end}).rowcount
        with conn.cursor().copy(
            "COPY token_prices_hourly (token, hour, usd_price, n_obs, hop) FROM STDIN"
        ) as copy:
            for row in rows:
                copy.write_row(row)
        conn.commit()

    if missing:
        log.warning("no CoinGecko price for %d target tokens: %s", len(missing), missing)
    log.info("token_prices_hourly: %d stablecoin rows, %d CoinGecko rows", stable, len(rows))
    return {"stable": stable, "coingecko": len(rows), "missing": len(missing)}


# Every statement below takes the table as {table} and a block range, so the
# same SQL serves dex_events and dex_events_removed.
_CLEAR_USD = """
UPDATE {table} SET usd_amount = NULL
WHERE block_number BETWEEN %(lo)s AND %(hi)s AND usd_amount IS NOT NULL
"""

# Most recent price at or before the row's hour, never older than
# PRICE_STALENESS. The lateral join hangs off `d`, a plain FROM entry: Postgres
# does not let a FROM/LATERAL item reference the UPDATE target itself.
_APPLY_USD = """
UPDATE {table} e
SET usd_amount = px.usd_amount
FROM (
    SELECT d.transaction_id,
           round(d.amount * p.usd_price, 6) AS usd_amount
    FROM {table} d
    CROSS JOIN LATERAL (
        SELECT pr.usd_price
        FROM token_prices_hourly pr
        WHERE pr.token = d.token_address
          AND pr.hour <= date_trunc('hour', d.timestamp)
          AND pr.hour >  date_trunc('hour', d.timestamp) - %(staleness)s::interval
        ORDER BY pr.hour DESC
        LIMIT 1
    ) p
    WHERE d.block_number BETWEEN %(lo)s AND %(hi)s
      AND d.amount IS NOT NULL
) px
WHERE e.transaction_id = px.transaction_id
"""

# The other leg of the same swap, for whatever _APPLY_USD left NULL. Same tx,
# same pool, opposite direction: both legs of one trade move (near enough) the
# same dollar amount. Joined to the pool through source/destination, because a
# multi-hop route's pool-to-pool row is a leg of both pools.
_MIRROR_SIBLING_USD = """
WITH legs AS (
    SELECT e.transaction_id, e.tx_hash, p.address AS pool, e.usd_amount,
           (e.destination = p.address) AS into_pool
    FROM {table} e
    JOIN pools p ON p.address IN (e.source, e.destination)
    WHERE e.block_number BETWEEN %(lo)s AND %(hi)s
)
UPDATE {table} e
SET usd_amount = sib.usd_amount
FROM legs mine
JOIN legs sib
  ON  sib.tx_hash    = mine.tx_hash
  AND sib.pool       = mine.pool
  AND sib.into_pool != mine.into_pool
WHERE e.transaction_id = mine.transaction_id
  AND e.usd_amount IS NULL
  AND sib.usd_amount IS NOT NULL
"""


# --- gap filling -------------------------------------------------------------
#
# _MIRROR_SIBLING_USD is one UPDATE, and an UPDATE does not see the rows it is
# itself changing, so a value travels exactly one leg per run. A route like
#   X -> poolA -> Y -> router -> poolB -> WETH
# leaves X empty: WETH's value reaches the router's Y leg into poolB, but the
# Y leg *out of* poolA is a different row, and X sits behind it. Even without
# a router, A -> B -> C chains need one run per hop. fill_gaps() closes both
# by alternating two
# steps, only over transactions that still have an empty row, until neither
# changes anything:
#   mirror      the same _MIRROR_SIBLING_USD rule, repeated
#   same token  a token priced on one leg of a transaction prices its other
#               legs in that transaction at the same unit price (same token,
#               same block -- the median if the legs disagree)
# Neither step estimates anything; both only carry a dollar value that is
# already in the transaction to a leg that moved the same money.

_GAP_TXS = """
CREATE TEMP TABLE gap_tx ON COMMIT DROP AS
SELECT DISTINCT tx_hash FROM {table} WHERE usd_amount IS NULL
"""

_MIRROR_GAPS = """
WITH legs AS (
    SELECT e.transaction_id, e.tx_hash, p.address AS pool, e.usd_amount,
           (e.destination = p.address) AS into_pool
    FROM {table} e
    JOIN gap_tx g ON g.tx_hash = e.tx_hash
    JOIN pools p ON p.address IN (e.source, e.destination)
)
UPDATE {table} e
SET usd_amount = sib.usd_amount
FROM legs mine
JOIN legs sib
  ON  sib.tx_hash    = mine.tx_hash
  AND sib.pool       = mine.pool
  AND sib.into_pool != mine.into_pool
WHERE e.transaction_id = mine.transaction_id
  AND e.usd_amount IS NULL
  AND sib.usd_amount IS NOT NULL
"""

_SAME_TOKEN_GAPS = """
UPDATE {table} e
SET usd_amount = round(e.amount * px.unit_price, 6)
FROM (
    SELECT d.tx_hash, d.token_address,
           -- _disc, not _cont: it works on NUMERIC and returns an observed price.
           percentile_disc(0.5) WITHIN GROUP (ORDER BY d.usd_amount / d.amount) AS unit_price
    FROM {table} d
    JOIN gap_tx g ON g.tx_hash = d.tx_hash
    WHERE d.usd_amount IS NOT NULL AND d.amount > 0
    GROUP BY 1, 2
) px
WHERE e.tx_hash = px.tx_hash
  AND e.token_address = px.token_address
  AND e.usd_amount IS NULL
  AND e.amount > 0
"""

MAX_FILL_PASSES = 20


def fill_gaps(table: str = "dex_events") -> dict[str, int]:
    """Carry dollar values across the legs of a transaction until nothing moves.

    Runs after apply_usd_amounts. Only transactions with an empty row are
    touched, so it is cheap to re-run on its own (`price --fill-only`).
    """
    if table not in PRICEABLE_TABLES:
        raise ValueError(f"table must be one of {PRICEABLE_TABLES}, not {table!r}")
    ident = sql.Identifier(table)
    counts = {"mirrored": 0, "same_token": 0, "passes": 0}
    started = time.monotonic()
    with db.connection() as conn:
        conn.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        conn.execute(sql.SQL(_GAP_TXS).format(table=ident))
        conn.execute("CREATE INDEX ON gap_tx (tx_hash)")
        conn.execute("ANALYZE gap_tx")
        for _ in range(MAX_FILL_PASSES):
            m = conn.execute(sql.SQL(_MIRROR_GAPS).format(table=ident)).rowcount
            s = conn.execute(sql.SQL(_SAME_TOKEN_GAPS).format(table=ident)).rowcount
            counts["mirrored"] += m
            counts["same_token"] += s
            counts["passes"] += 1
            log.info("%s gap fill pass %d: %d mirrored, %d by same-token price",
                     table, counts["passes"], m, s)
            if m == 0 and s == 0:
                break
        conn.commit()
    counts["unpriced"] = db.fetch_one(
        sql.SQL("SELECT count(*) FROM {t} WHERE usd_amount IS NULL").format(t=ident))[0]
    log.info("%s gap fill: %d mirrored + %d same-token in %d passes, %.1f min "
             "(%d still without a price)", table, counts["mirrored"], counts["same_token"],
             counts["passes"], (time.monotonic() - started) / 60, counts["unpriced"])
    return counts


def _apply_range(table: sql.Identifier, lo: int, hi: int,
                 counter: dict, lock: threading.Lock) -> None:
    params = {"lo": lo, "hi": hi, "staleness": PRICE_STALENESS}
    with db.connection() as conn:
        # The Postgres container's /dev/shm is small; parallel hash joins over a
        # wide range fail with "could not resize shared memory segment".
        conn.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cleared = conn.execute(sql.SQL(_CLEAR_USD).format(table=table), params).rowcount
        priced = conn.execute(sql.SQL(_APPLY_USD).format(table=table), params).rowcount
        mirrored = conn.execute(sql.SQL(_MIRROR_SIBLING_USD).format(table=table), params).rowcount
        conn.commit()
    with lock:
        counter["cleared"] += cleared
        counter["priced"] += priced
        counter["mirrored"] += mirrored


def apply_usd_amounts(config: Config = CONFIG, table: str = "dex_events",
                      range_size: int = 5_000) -> dict[str, int]:
    """Clear and refill usd_amount across `table`, in parallel over block ranges.

    Each range is cleared and refilled in one transaction, so a range is never
    left half-valued, and a row that no longer has a price ends up NULL rather
    than keeping a value from an earlier pricing run.
    """
    if table not in PRICEABLE_TABLES:
        raise ValueError(f"table must be one of {PRICEABLE_TABLES}, not {table!r}")
    ident = sql.Identifier(table)

    bounds = db.fetch_one(
        sql.SQL("SELECT min(block_number), max(block_number) FROM {t}").format(t=ident))
    if not bounds or bounds[0] is None:
        log.warning("%s is empty; nothing to price", table)
        return {"cleared": 0, "priced": 0, "mirrored": 0, "unpriced": 0}

    lo, hi = int(bounds[0]), int(bounds[1])
    ranges = [(s, min(s + range_size - 1, hi)) for s in range(lo, hi + 1, range_size)]
    counter = {"cleared": 0, "priced": 0, "mirrored": 0}
    lock = threading.Lock()
    started = time.monotonic()
    failed = 0

    with ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="price") as pool:
        futures = {pool.submit(_apply_range, ident, a, b, counter, lock): (a, b)
                   for a, b in ranges}
        for done, future in enumerate(as_completed(futures), 1):
            a, b = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.warning("pricing %s range %d-%d failed: %s", table, a, b, exc)
            # Each range is several full UPDATEs; without this a long run is
            # silent for most of an hour.
            log.info("%s: %d/%d block ranges done (%.1f min)", table, done, len(ranges),
                     (time.monotonic() - started) / 60)

    filled = fill_gaps(table)
    counter["gap_mirrored"] = filled["mirrored"]
    counter["gap_same_token"] = filled["same_token"]

    counter["unpriced"] = filled["unpriced"]
    counter["failed_ranges"] = failed
    log.info(
        "%s: cleared %d, valued %d directly + %d mirrored from the swap's other leg "
        "in %.1f min (%d still without a price, %d ranges failed)",
        table, counter["cleared"], counter["priced"], counter["mirrored"],
        (time.monotonic() - started) / 60, counter["unpriced"], failed,
    )
    return counter

"""Integrity checks over dex_events.

The headline one is swap pairing. A swap is an exchange, so for every pool that
took part in a transaction there must be BOTH a row moving a token into that
pool and a row moving a different token out of it -- those two rows are the
swap.

A leg is matched to its pool through `source`/`destination` rather than through
the `pool` column. A multi-hop route (USDC -> WETH -> LINK) settles pool to
pool, so one row is simultaneously the first pool's outbound leg and the
second's inbound leg, and only the join to `pools` sees both roles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from . import db

log = logging.getLogger(__name__)

# Every (transaction, pool) that moved tokens, with the two directions it must
# have. `counterpart_expected` is FALSE where the chain could not have emitted
# the opposite leg -- the swap's amount on that side was zero, or the token
# rebases and so moves without emitting Transfer. Both causes were found by
# running this check against live data, not assumed; see ingest._counterpart_exempt.
PAIRING_DETAIL_SQL = """
SELECT
    e.tx_hash,
    p.address                                   AS pool,
    min(e.block_number)                          AS block_number,
    min(e.timestamp)                             AS timestamp,
    bool_or(e.destination = p.address)           AS has_in_leg,
    bool_or(e.source = p.address)                AS has_out_leg,
    count(DISTINCT e.token_address)              AS distinct_tokens,
    bool_and(e.counterpart_expected)             AS counterpart_expected,
    string_agg(DISTINCT e.token, ', ')           AS tokens
FROM dex_events e
JOIN pools p ON p.address IN (e.source, e.destination)
GROUP BY e.tx_hash, p.address
"""

PAIRING_SUMMARY_SQL = f"""
WITH checked AS ({PAIRING_DETAIL_SQL})
SELECT
    count(*)                                                       AS total_swaps,
    count(*) FILTER (WHERE NOT counterpart_expected)               AS one_sided,
    count(*) FILTER (WHERE counterpart_expected)                   AS checked_swaps,
    count(*) FILTER (WHERE counterpart_expected
                       AND has_in_leg AND has_out_leg)             AS paired,
    count(*) FILTER (WHERE counterpart_expected AND NOT has_in_leg)  AS missing_in_leg,
    count(*) FILTER (WHERE counterpart_expected AND NOT has_out_leg) AS missing_out_leg,
    count(*) FILTER (WHERE counterpart_expected AND has_in_leg
                       AND has_out_leg AND distinct_tokens < 2)    AS single_token_pair
FROM checked
"""

UNPAIRED_SAMPLE_SQL = f"""
WITH checked AS ({PAIRING_DETAIL_SQL})
SELECT tx_hash, pool, block_number, timestamp, tokens,
       has_in_leg, has_out_leg, distinct_tokens
FROM checked
WHERE counterpart_expected AND NOT (has_in_leg AND has_out_leg)
ORDER BY block_number DESC
LIMIT %s
"""

#: Column order of UNPAIRED_SAMPLE_SQL, so callers name fields instead of
#: indexing positions that shift whenever the query gains a column.
UNPAIRED_SAMPLE_COLUMNS = (
    "tx_hash", "pool", "block_number", "timestamp", "tokens",
    "has_in_leg", "has_out_leg", "distinct_tokens",
)

# Rows whose two endpoints are both pools: legitimate multi-hop routing, where
# one transfer serves two pools. Reported, not asserted.
POOL_TO_POOL_SQL = """
SELECT count(*)
FROM dex_events e
WHERE EXISTS (SELECT 1 FROM pools p WHERE p.address = e.source)
  AND EXISTS (SELECT 1 FROM pools p WHERE p.address = e.destination)
"""

# A leg's token must be one of its pool's two tokens. Anything else is not a
# swap: most often a v2 pair's own LP token, which the pair emits Transfers for
# when liquidity moves. Such a row names the pool address as its token, and
# that address is not in `tokens` -- which is how it took the pricing pass down.
FOREIGN_TOKEN_SQL = """
SELECT count(*) AS rows,
       count(DISTINCT e.token_address) AS tokens,
       count(*) FILTER (WHERE e.token_address = e.pool) AS lp_tokens
FROM dex_events e
JOIN pools p ON p.address = e.pool
WHERE e.token_address <> p.token0 AND e.token_address <> p.token1
"""

# Rows whose token was never resolved into `tokens`. These cannot be priced and
# would break token_prices_hourly's foreign key.
UNKNOWN_TOKEN_SQL = """
SELECT count(*) AS rows, count(DISTINCT e.token_address) AS tokens
FROM dex_events e
LEFT JOIN tokens t ON t.address = e.token_address
WHERE t.address IS NULL
"""

COVERAGE_SQL = """
SELECT
    count(*)                                             AS rows_total,
    count(DISTINCT tx_hash)                              AS transactions,
    count(*) FILTER (WHERE usd_amount IS NULL)           AS unpriced,
    count(DISTINCT token_address)                        AS tokens,
    count(DISTINCT pool) FILTER (WHERE pool IS NOT NULL) AS pools,
    min(timestamp), max(timestamp),
    min(block_number), max(block_number)
FROM dex_events
"""


@dataclass
class PairingResult:
    #: (transaction, pool) groups, i.e. individual swaps.
    total_swaps: int
    #: Swaps the chain could not have emitted both legs for: a zero amount on
    #: one side, or a rebasing token that moves without emitting Transfer.
    one_sided: int
    #: Swaps the invariant actually applies to.
    checked_swaps: int
    paired: int
    missing_in_leg: int
    missing_out_leg: int
    single_token_pair: int

    @property
    def unpaired(self) -> int:
        return self.checked_swaps - self.paired

    @property
    def ratio(self) -> float:
        return self.paired / self.checked_swaps if self.checked_swaps else 1.0


def check_pairing() -> PairingResult:
    row = db.fetch_one(PAIRING_SUMMARY_SQL)
    return PairingResult(*(int(v or 0) for v in row))


def unpaired_sample(limit: int = 20) -> list[dict]:
    return [
        dict(zip(UNPAIRED_SAMPLE_COLUMNS, row))
        for row in db.fetch_all(UNPAIRED_SAMPLE_SQL, (limit,))
    ]


def pool_to_pool_rows() -> int:
    row = db.fetch_one(POOL_TO_POOL_SQL)
    return int(row[0]) if row else 0


def foreign_token_rows() -> dict:
    row = db.fetch_one(FOREIGN_TOKEN_SQL)
    return dict(zip(("rows", "tokens", "lp_tokens"), (int(v or 0) for v in row)))


def unknown_token_rows() -> dict:
    row = db.fetch_one(UNKNOWN_TOKEN_SQL)
    return dict(zip(("rows", "tokens"), (int(v or 0) for v in row)))


def coverage() -> dict:
    row = db.fetch_one(COVERAGE_SQL)
    keys = (
        "rows_total", "transactions", "unpriced", "tokens", "pools",
        "first_timestamp", "last_timestamp", "first_block", "last_block",
    )
    return dict(zip(keys, row))


def report() -> PairingResult:
    cov = coverage()
    log.info("dex_events: %s rows across %s transactions",
             cov["rows_total"], cov["transactions"])
    log.info("  window: %s .. %s  (blocks %s .. %s)",
             cov["first_timestamp"], cov["last_timestamp"],
             cov["first_block"], cov["last_block"])
    log.info("  %s distinct tokens, %s pools, %s rows without a USD price",
             cov["tokens"], cov["pools"], cov["unpriced"])

    pairing = check_pairing()
    log.info("swap pairing: %d/%d paired (%.4f%%)",
             pairing.paired, pairing.checked_swaps, 100 * pairing.ratio)
    if pairing.one_sided:
        log.info("  %d one-sided swaps excluded (zero amount on a side, or a "
                 "rebasing token that moves without emitting Transfer)",
                 pairing.one_sided)
    if pairing.unpaired:
        log.warning("  missing in-leg: %d | missing out-leg: %d | single-token: %d",
                    pairing.missing_in_leg, pairing.missing_out_leg,
                    pairing.single_token_pair)
        for row in unpaired_sample(5):
            log.warning(
                "  unpaired: tx=%s pool=%s block=%s tokens=[%s] in_leg=%s out_leg=%s",
                row["tx_hash"], row["pool"], row["block_number"],
                row["tokens"], row["has_in_leg"], row["has_out_leg"],
            )

    log.info("pool-to-pool rows (multi-hop routing, one transfer serving two "
             "pools): %d", pool_to_pool_rows())

    foreign = foreign_token_rows()
    if foreign["rows"]:
        log.warning(
            "%d rows carry a token that is not one of their pool's two tokens "
            "(%d of them the pool's own LP token). These are not swaps; ingest "
            "should exclude them.", foreign["rows"], foreign["lp_tokens"],
        )
    unknown = unknown_token_rows()
    if unknown["rows"]:
        log.warning(
            "%d rows reference %d tokens missing from `tokens`; these cannot be "
            "priced and break token_prices_hourly's foreign key.",
            unknown["rows"], unknown["tokens"],
        )
    return pairing

"""Remove swaps that carry no information about the target universe.

A swap here is one (tx, pool) pair of dex_events legs. Only the simplest shape
is ever removed, so no multi-hop route is cut in half:

  * exactly two legs: wallet -> pool and pool -> wallet, in two different tokens
  * neither leg is pool-to-pool
  * the transaction touches no other pool

Such a swap is moved to dex_events_removed (same columns plus removed_reason,
removed_at) when its two tokens are, with `target` and `is_stable` read from
tokens (a token missing from tokens counts as non-target, non-stable):

  stable_stable_standalone_swap   both stablecoins           (USDC -> USDT)
  stable_vs_nontarget             a stablecoin and a non-target token
  both_nontarget                  two non-target tokens

The same shapes inside a multi-pool transaction are kept: they are hops of a
route whose other end may well be a target token, and removing them would
leave the route half-recorded.

Re-runnable: rows already removed are no longer in dex_events. Re-run it after
an ingest (which re-inserts removed rows) or after the target universe changes
(un-targeting a token makes its swaps against stablecoins removable).
"""
from __future__ import annotations

import logging

from . import db

log = logging.getLogger(__name__)

_CANDIDATES = """
CREATE TEMP TABLE cleanup_candidates ON COMMIT DROP AS
WITH legs AS (
    SELECT e.transaction_id, e.tx_hash, p.address AS pool, e.token_address,
           (e.destination = p.address) AS into_pool,
           (ps.address IS NOT NULL AND pd.address IS NOT NULL) AS pool_to_pool,
           coalesce(t.is_stable, FALSE) AS stable,
           NOT coalesce(t.target, FALSE) AS nontarget
    FROM dex_events e
    JOIN pools p ON p.address IN (e.source, e.destination)
    LEFT JOIN pools ps ON ps.address = e.source
    LEFT JOIN pools pd ON pd.address = e.destination
    LEFT JOIN tokens t ON t.address = e.token_address
)
SELECT tx_hash, pool, array_agg(transaction_id) AS ids,
       CASE WHEN bool_and(stable) THEN 'stable_stable_standalone_swap'
            WHEN bool_or(stable)  THEN 'stable_vs_nontarget'
            ELSE                       'both_nontarget'
       END AS reason
FROM legs
GROUP BY tx_hash, pool
HAVING count(*) = 2
   AND count(*) FILTER (WHERE into_pool) = 1
   AND NOT bool_or(pool_to_pool)
   AND count(DISTINCT token_address) = 2
   -- every leg is a stablecoin or outside the universe: a non-stable target
   -- token on either side keeps the swap
   AND bool_and(stable OR nontarget)
"""

# The standalone test runs on the (much smaller) candidate set through the
# tx_hash index; folding it into the CTE above made the planner pick a plan
# that did not finish.
_STANDALONE = """
CREATE TEMP TABLE cleanup_ids ON COMMIT DROP AS
SELECT unnest(c.ids) AS transaction_id, c.reason
FROM cleanup_candidates c
WHERE NOT EXISTS (
    SELECT 1 FROM dex_events e
    JOIN pools p ON p.address IN (e.source, e.destination)
    WHERE e.tx_hash = c.tx_hash AND p.address <> c.pool)
"""


def run() -> dict[str, int]:
    """Move removable standalone swaps into dex_events_removed, atomically."""
    with db.connection() as conn:
        # The Postgres container's /dev/shm is too small for parallel hash
        # joins over dex_events.
        conn.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        conn.execute("SET LOCAL work_mem = '512MB'")
        before = conn.execute("SELECT count(*) FROM dex_events").fetchone()[0]

        conn.execute(_CANDIDATES)
        conn.execute("ANALYZE cleanup_candidates")
        conn.execute(_STANDALONE)
        conn.execute("ANALYZE cleanup_ids")

        counts = dict(conn.execute(
            "SELECT reason, count(*) / 2 FROM cleanup_ids GROUP BY 1").fetchall())
        n = conn.execute("SELECT count(*) FROM cleanup_ids").fetchone()[0]
        moved = conn.execute("""
            INSERT INTO dex_events_removed
            SELECT e.*, c.reason FROM dex_events e JOIN cleanup_ids c USING (transaction_id)
        """).rowcount
        deleted = conn.execute("""
            DELETE FROM dex_events e USING cleanup_ids c
            WHERE e.transaction_id = c.transaction_id""").rowcount
        if not (moved == deleted == n):
            conn.rollback()
            raise RuntimeError(f"cleanup mismatch: {n} selected, {moved} backed up, "
                               f"{deleted} deleted; rolled back")
        conn.commit()

    counts = {k: int(v) for k, v in counts.items()}
    counts["rows_removed"] = deleted
    counts["dex_events_before"] = before
    counts["dex_events_after"] = before - deleted
    log.info("cleanup: removed %d rows (%s); dex_events %d -> %d", deleted,
             ", ".join(f"{k} {v} swaps" for k, v in sorted(counts.items())
                       if k not in ("rows_removed", "dex_events_before", "dex_events_after")),
             before, before - deleted)
    return counts

-- ===========================================================================
-- Swap pairing check
--
--   "Tabloda bir swap varsa karşılığının da olması gerekir. Yani adresten
--    pool'a para gitmiş, aynı şekilde pool'dan da adrese para gidecek."
--
-- Every row in dex_events is one token movement into or out of a Uniswap pool.
-- A swap is therefore TWO rows -- wallet -> pool and pool -> wallet -- and
-- those two rows are the swap. So for every (transaction, pool) pair there
-- must be both a row whose destination is the pool and a row whose source is
-- the pool, in two different tokens.
--
-- Legs are matched to a pool through source/destination, not through the
-- `pool` column: a multi-hop route (USDC -> WETH -> LINK) settles pool to pool,
-- so one row is at once the first pool's outbound leg and the second pool's
-- inbound leg, and only the join to `pools` sees both roles.
--
-- counterpart_expected is FALSE where the chain could not have emitted the
-- opposite leg. Two causes, both found by running this query against live data
-- rather than assumed:
--   1. The swap's amount on one side rounds to zero, so no tokens move there
--      and no Transfer is emitted. Dust probes, typically 1 wei in.
--   2. A rebasing / reflection token, which changes balances without emitting
--      Transfer. A v2 pair accumulates it silently and anyone can call swap()
--      to skim the surplus, so no inbound Transfer ever exists. Confirmed on
--      the KISHU/WETH pair.
-- Anything else missing a leg is a failure: section 5 raises.
--
-- Run:  psql "$DEX_PG_DSN" -f sql/swap_pairing_check.sql
-- ===========================================================================

\set ON_ERROR_STOP on

CREATE OR REPLACE TEMP VIEW swap_pairing AS
SELECT
    e.tx_hash,
    p.address                          AS pool,
    min(e.block_number)                 AS block_number,
    min(e.timestamp)                    AS timestamp,
    bool_or(e.destination = p.address)  AS has_in_leg,
    bool_or(e.source = p.address)       AS has_out_leg,
    count(DISTINCT e.token_address)     AS distinct_tokens,
    bool_and(e.counterpart_expected)    AS counterpart_expected,
    string_agg(DISTINCT e.token, ', ')  AS tokens
FROM dex_events e
JOIN pools p ON p.address IN (e.source, e.destination)
GROUP BY e.tx_hash, p.address;


\echo ''
\echo '=== 1. Summary ==='

SELECT
    count(*)                                                        AS total_swaps,
    count(*) FILTER (WHERE NOT counterpart_expected)                AS one_sided_excluded,
    count(*) FILTER (WHERE counterpart_expected)                    AS checked_swaps,
    count(*) FILTER (WHERE counterpart_expected
                       AND has_in_leg AND has_out_leg)              AS paired,
    count(*) FILTER (WHERE counterpart_expected AND NOT has_in_leg)   AS missing_in_leg,
    count(*) FILTER (WHERE counterpart_expected AND NOT has_out_leg)  AS missing_out_leg,
    count(*) FILTER (WHERE counterpart_expected AND has_in_leg
                       AND has_out_leg AND distinct_tokens < 2)     AS single_token_pair,
    round(100.0 * count(*) FILTER (WHERE counterpart_expected
                                     AND has_in_leg AND has_out_leg)
          / NULLIF(count(*) FILTER (WHERE counterpart_expected), 0), 4) AS paired_pct
FROM swap_pairing;


\echo ''
\echo '=== 2. Unpaired swaps (should be empty) ==='

SELECT tx_hash, pool, block_number, tokens, has_in_leg, has_out_leg
FROM swap_pairing
WHERE counterpart_expected AND NOT (has_in_leg AND has_out_leg)
ORDER BY block_number DESC
LIMIT 25;


\echo ''
\echo '=== 3. Every unpaired swap must fall into a known category ==='

SELECT
    CASE WHEN has_in_leg AND has_out_leg THEN 'paired' ELSE 'unpaired' END AS status,
    count(*)                                            AS swaps,
    count(*) FILTER (WHERE NOT counterpart_expected)    AS one_sided_by_design,
    count(*) FILTER (WHERE counterpart_expected)        AS unexplained
FROM swap_pairing
GROUP BY 1
ORDER BY 1;


\echo ''
\echo '=== 4. One swap, both of its rows ==='

WITH one AS (
    SELECT tx_hash, pool FROM swap_pairing
    WHERE counterpart_expected AND has_in_leg AND has_out_leg
    ORDER BY block_number DESC LIMIT 1
)
SELECT
    CASE WHEN e.destination = one.pool THEN 'wallet -> pool'
         ELSE 'pool -> wallet' END AS direction,
    e.source,
    e.destination,
    e.token,
    e.amount,
    e.usd_amount,
    e.block_number
FROM dex_events e
JOIN one ON one.tx_hash = e.tx_hash
WHERE e.source = one.pool OR e.destination = one.pool
ORDER BY e.log_index;


\echo ''
\echo '=== 5. Assertion ==='

DO $$
DECLARE
    unpaired  bigint;
    one_sided bigint;
BEGIN
    SELECT count(*) FILTER (WHERE counterpart_expected
                              AND NOT (has_in_leg AND has_out_leg)),
           count(*) FILTER (WHERE NOT counterpart_expected)
      INTO unpaired, one_sided
      FROM swap_pairing;

    IF unpaired > 0 THEN
        RAISE EXCEPTION
            'SWAP PAIRING FAILED: % swaps have money moving one way but not the other',
            unpaired;
    END IF;

    RAISE NOTICE
        'SWAP PAIRING OK: every swap has both directions (% one-sided swaps classified separately)',
        one_sided;
END $$;

-- ===========================================================================
-- dex_data schema
--
--   reference    tokens, pools, blocks
--   derived      token_prices_hourly
--   output       dex_events
--   bookkeeping  ingest_progress
--
-- Addresses are stored lowercase hex with the 0x prefix, everywhere.
-- ===========================================================================

-- Earlier versions staged decoded logs in raw_transfers / raw_swaps before
-- projecting them. Ingest now writes dex_events directly, so these are dropped.
DROP TABLE IF EXISTS raw_transfers;
DROP TABLE IF EXISTS raw_swaps;

CREATE TABLE IF NOT EXISTS tokens (
    address          TEXT PRIMARY KEY,
    symbol           TEXT,
    name             TEXT,
    decimals         SMALLINT,
    -- Priced at exactly 1 USD and used as the root of the pricing graph.
    is_stable        BOOLEAN NOT NULL DEFAULT FALSE,
    -- Reflection/rebasing tokens redistribute balances WITHOUT emitting a
    -- Transfer event, so a pool holding one accumulates it silently and a swap
    -- against it can legitimately have only one leg. See verify.py.
    is_rebasing      BOOLEAN NOT NULL DEFAULT FALSE,
    first_seen_block BIGINT
);
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS is_rebasing BOOLEAN NOT NULL DEFAULT FALSE;

-- Market cap, filled by marketcap.py from CoinGecko. NULL means CoinGecko does
-- not list the token (about two thirds of what a 24h window sees) or lists it
-- without a circulating figure. That NULL is a verdict, not a gap: analysis
-- eliminates those rather than substituting an estimate. An earlier version
-- valued them from totalSupply() and it could not distinguish a real small cap
-- from a broken one, so it let junk through the very filter it fed.
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS market_cap_usd  NUMERIC(38,2);
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS fdv_usd         NUMERIC(38,2);
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS coingecko_id    TEXT;
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS cg_rank         INTEGER;
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS market_data_at  TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS tokens_market_cap_idx ON tokens(market_cap_usd);

-- The modelling universe, decided ONCE and then left alone.
--
-- Market caps move every day. If the dataset selected its tokens by reading
-- market_cap_usd live, a token drifting from $101M to $99M would silently leave
-- the universe and every swap it appears in would turn into an anonymous
-- UNKNOWN -- the training set changing shape underneath a model for reasons
-- that have nothing to do with the data. So `cli target` writes this flag once
-- and `cli dataset` reads only the flag. Re-running `marketcap` refreshes the
-- caps and never touches it; re-pinning takes an explicit --force.
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS target BOOLEAN NOT NULL DEFAULT FALSE;
-- The market cap this token had when it was pinned, kept so the choice stays
-- auditable after the live figure has moved on.
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS target_market_cap_usd NUMERIC(38,2);
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS target_pinned_at      TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS tokens_target_idx ON tokens(target) WHERE target;

-- Dropped with the on-chain estimate they belonged to.
ALTER TABLE tokens DROP COLUMN IF EXISTS total_supply;
ALTER TABLE tokens DROP COLUMN IF EXISTS burned_supply;
ALTER TABLE tokens DROP COLUMN IF EXISTS onchain_value_usd;
ALTER TABLE tokens DROP COLUMN IF EXISTS cg_market_cap_usd;
ALTER TABLE tokens DROP COLUMN IF EXISTS cg_fdv_usd;
ALTER TABLE tokens DROP COLUMN IF EXISTS market_cap_source;

CREATE TABLE IF NOT EXISTS pools (
    address          TEXT PRIMARY KEY,
    dex              TEXT NOT NULL,             -- 'uniswap'
    version          TEXT NOT NULL,             -- 'v2' | 'v3'
    factory          TEXT,
    token0           TEXT REFERENCES tokens(address),
    token1           TEXT REFERENCES tokens(address),
    fee              INTEGER,                   -- v3 only, hundredths of a bip
    first_seen_block BIGINT
);
CREATE INDEX IF NOT EXISTS pools_token0_idx ON pools(token0);
CREATE INDEX IF NOT EXISTS pools_token1_idx ON pools(token1);

-- Denormalized from tokens.target (true when token0 OR token1 is pinned).
-- Kept in sync by pools.refresh_target_flags -- called from `cli target` and
-- after every newly discovered pool is stored. Lets ingest scope eth_getLogs
-- to a flat, already-verified address list with no join and no factory()
-- call ever touching a pool outside the pinned universe.
ALTER TABLE pools ADD COLUMN IF NOT EXISTS is_target BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS pools_target_idx ON pools(is_target) WHERE is_target;

CREATE TABLE IF NOT EXISTS blocks (
    number     BIGINT PRIMARY KEY,
    timestamp  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS blocks_timestamp_idx ON blocks(timestamp);

CREATE TABLE IF NOT EXISTS token_prices_hourly (
    token     TEXT NOT NULL REFERENCES tokens(address),
    hour      TIMESTAMPTZ NOT NULL,
    usd_price NUMERIC(38,18) NOT NULL,
    n_obs     INTEGER NOT NULL DEFAULT 0,
    hop       SMALLINT NOT NULL,
    PRIMARY KEY (token, hour)
);

-- ---------------------------------------------------------------------------
-- Output table.
--
-- One row per token movement into or out of a Uniswap pool. A swap therefore
-- produces TWO rows -- wallet -> pool (what the trader sold) and pool -> wallet
-- (what they bought) -- and those two rows ARE the swap. There is no separate
-- marker row, and `source` never equals `destination`.
--
-- A multi-hop route (USDC -> WETH -> LINK) settles pool to pool: one transfer
-- is simultaneously the first pool's outbound leg and the second pool's
-- inbound leg. Such rows are kept; dropping them would leave both pools of
-- every multi-hop swap missing a side.
--
-- transaction_id is '<tx_hash>-<log_index>'. A transaction hash alone cannot be
-- the primary key: one Ethereum transaction routinely moves tokens through
-- several pools, so the hash is not unique per row. Suffixing the log index
-- keeps the key unique, traceable to the source event, and re-runs idempotent.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dex_events (
    transaction_id TEXT PRIMARY KEY,
    -- Only 'swap' today. The column stays so wallet-to-wallet transfers can be
    -- added later as a second type without a schema change.
    type           TEXT NOT NULL DEFAULT 'swap' CHECK (type IN ('swap')),
    source         TEXT NOT NULL,
    destination    TEXT NOT NULL,
    -- Human-readable ticker, e.g. 'WETH'. token_address carries the identity;
    -- symbols are neither unique nor guaranteed to exist on chain.
    token          TEXT NOT NULL,
    usd_amount     NUMERIC(38,6),
    timestamp      TIMESTAMPTZ NOT NULL,
    block_number   BIGINT NOT NULL,

    -- Tracking columns.
    tx_hash        TEXT NOT NULL,
    log_index      INTEGER NOT NULL,
    token_address  TEXT NOT NULL,
    amount         NUMERIC(78,18),        -- decimal-adjusted
    amount_raw     NUMERIC(78,0),         -- as emitted on chain
    -- The pool this leg was attributed to. For a pool-to-pool leg this is the
    -- sending pool; the receiving pool still matches through `destination`,
    -- which is why the pairing check joins on pools rather than on this column.
    pool           TEXT,
    -- FALSE when the chain cannot have emitted the opposite leg: the swap's
    -- amount on that side was zero, or the token rebases and so moves without
    -- emitting Transfer. Set from the Swap event at ingest time.
    counterpart_expected BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT dex_events_distinct_endpoints CHECK (source <> destination)
);
CREATE INDEX IF NOT EXISTS dex_events_tx_idx        ON dex_events(tx_hash);
CREATE INDEX IF NOT EXISTS dex_events_source_idx    ON dex_events(source);
CREATE INDEX IF NOT EXISTS dex_events_dest_idx      ON dex_events(destination);
CREATE INDEX IF NOT EXISTS dex_events_token_idx     ON dex_events(token_address);
CREATE INDEX IF NOT EXISTS dex_events_timestamp_idx ON dex_events(timestamp);
CREATE INDEX IF NOT EXISTS dex_events_block_idx     ON dex_events(block_number);
CREATE INDEX IF NOT EXISTS dex_events_pool_idx      ON dex_events(pool);
-- Serves the pairing check, which groups a transaction's legs by pool.
CREATE INDEX IF NOT EXISTS dex_events_tx_pool_idx   ON dex_events(tx_hash, pool);

-- Rows taken out of dex_events by cleanup passes (e.g. stablecoin-for-
-- stablecoin swaps), kept so a removal can be audited or undone. Same columns
-- as dex_events plus why and when. NOTE: re-running ingest over the same
-- blocks re-inserts these rows into dex_events; the cleanup must be re-run.
CREATE TABLE IF NOT EXISTS dex_events_removed (
    LIKE dex_events INCLUDING DEFAULTS,
    removed_reason TEXT NOT NULL,
    removed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS dex_events_removed_id_idx    ON dex_events_removed(transaction_id);
CREATE INDEX IF NOT EXISTS dex_events_removed_block_idx ON dex_events_removed(block_number);
CREATE INDEX IF NOT EXISTS dex_events_removed_tx_idx    ON dex_events_removed(tx_hash);

CREATE TABLE IF NOT EXISTS ingest_progress (
    job         TEXT   NOT NULL,
    from_block  BIGINT NOT NULL,
    to_block    BIGINT NOT NULL,
    status      TEXT   NOT NULL DEFAULT 'done',
    n_logs      INTEGER NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job, from_block)
);

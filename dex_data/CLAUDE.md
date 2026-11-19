# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An Ethereum indexer that extracts Uniswap swaps into Postgres. USD values come
from CoinGecko hourly prices for the target universe (stablecoins at $1). See [README.md](../README.md)
for the full design rationale (table shape, why the RPC layer is built the
way it is, pricing model, concurrency race that was fixed, swap-pairing
invariant) — it's unusually detailed and worth reading before touching
`ingest.py`, `pricing.py`, `rpc.py`, or `verify.py`.

## Setup

```bash
docker compose up -d postgres                    # Postgres on localhost:5433
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                              # set DEX_RPC_URLS
```

## Commands

```bash
PYTHONPATH=src .venv/bin/python -m dex_data.cli initdb                 # create schema
PYTHONPATH=src .venv/bin/python -m dex_data.cli ingest --days 30        # log scan (resumable)
PYTHONPATH=src .venv/bin/python -m dex_data.cli ingest --from-block N --to-block M
PYTHONPATH=src .venv/bin/python -m dex_data.cli price                   # CoinGecko hourly prices, clear + refill usd_amount
PYTHONPATH=src .venv/bin/python -m dex_data.cli price --no-fetch --table dex_events_removed
PYTHONPATH=src .venv/bin/python -m dex_data.cli price --fill-only       # only fill still-empty rows along each tx
PYTHONPATH=src .venv/bin/python -m dex_data.cli marketcap               # fill tokens.market_cap_usd from CoinGecko
PYTHONPATH=src .venv/bin/python -m dex_data.cli target --top 100        # pin the modelling universe (once)
PYTHONPATH=src .venv/bin/python -m dex_data.cli cleanup                 # move standalone stable/non-target swaps to dex_events_removed
PYTHONPATH=src .venv/bin/python -m dex_data.cli verify                  # swap-pairing integrity report, exit 1 on failure
PYTHONPATH=src .venv/bin/python -m dex_data.cli all --days 30           # ingest -> price -> verify
PYTHONPATH=src .venv/bin/python -m dex_data.cli reset --yes             # truncate dex_events/prices/progress/blocks (--all: pools/tokens too)
```

Settings come from `.env`: `DEX_RPC_URLS`, `DEX_PG_DSN`, `DEX_DAYS`,
`DEX_MAX_WORKERS`, `DEX_LOG_CHUNK_BLOCKS`, `DEX_RPC_RATE_LIMIT`, `DEX_ARCHIVE`.
`Config` (`src/dex_data/config.py`) is a frozen dataclass built **once**, at
import time, from `os.environ` — mutating env vars after `dex_data.config`
has been imported (e.g. inside a test) has no effect on `CONFIG` itself.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ingest.py -v
PYTHONPATH=src .venv/bin/python -m pytest tests/test_swap_pairing.py::test_a_swap_is_exactly_two_rows_wallet_to_pool_and_back -v
```

- `test_decoders.py`, `test_rpc.py`, `test_multicall.py`, `test_marketcap.py` — no network/DB required.
- `test_ingest.py` — includes `TestPoolClassificationRace`, three concurrency tests locking in a real production bug (see README "Parallelism and the pool-classification race").
- `test_dataset.py`, `test_swap_pairing.py` — require Postgres (`docker compose up -d postgres`).

### DB test isolation — read this before adding a DB-backed test

DB tests run against **real Postgres**, not a mock (the SQL under test is the
product, not a reimplementation of it worth mocking). Isolation is schema-based:
`tests/conftest.py`'s `pg_schema` fixture creates a throwaway `dex_test_*`
schema and calls `db.set_schema_override(schema)` *before* the connection pool
is (re)opened — the pool's own `configure` callback then forces every
connection it ever opens onto that schema via `SET search_path`. Always depend
on `clean_db` (function-scoped, truncates everything, asserts
`current_schema()` really is the throwaway schema before truncating) or
`live_db` (read-only checks against whatever is actually ingested; skips if
`dex_events` is empty) — never write to `db.connection()` in a test without
going through one of these. A DB test that bypasses both fixtures runs against
the real `public` schema, i.e. real ingested data, silently.

`@pytest.mark.live` tests are skipped both when Postgres is down and when
`dex_events` is empty — a green run alone doesn't prove they executed; check
with `-rs`.

## Architecture

Everything is orchestrated by `cli.py` as subcommands; each subcommand is a
thin wrapper around one module's `run`/`build`/`report` entry point. Pipeline
order (`all` = `ingest` → `price` → `verify`):

1. **`rpc.py`** — the shared substrate. A single `RpcClient` round-robins
   across configured endpoints, learns per-endpoint quirks at runtime (max
   `eth_getLogs` span, unsupported methods, safe batch size) rather than
   hardcoding them, and rate-limits with AIMD (halve on 429, ease back on
   success). Batched `eth_call`s go through Multicall3 (`abi.py` has the raw
   encoding) — this is the difference between ~6k rate-limited calls and ~4.
2. **`blocks.py`** — wall-clock → block number via binary search; resolves
   the `--days` lookback window into a block range.
3. **`pools.py`** / **`tokens.py`** — discovered on demand during ingest, not
   backfilled from factory events. Any address seen emitting a Swap log is a
   pool *candidate*; confirmed by calling `factory()` and comparing against
   known Uniswap factories (byte-identical forks like SushiSwap emit the same
   Swap topic, so topic match alone isn't enough). `tokens.py` holds the
   stablecoin set that roots the pricing graph.
4. **`ingest.py`** — one `eth_getLogs` per block chunk with an OR'd topic0
   filter (Transfer + V2 Swap + V3 Swap in one call), distributed over a
   `ThreadPoolExecutor` (network I/O, GIL-free). Writes `dex_events` directly;
   Swap events themselves aren't persisted, only used to classify pools and
   set `counterpart_expected`. Progress is checkpointed per chunk in
   `ingest_progress`, so a run is resumable and safely re-runnable (idempotent
   upsert on `(tx_hash, log_index)`).
5. **`pricing.py`** — `build()` rewrites `token_prices_hourly` (hop 0:
   every `is_stable` token at $1; hop 1: non-stable `tokens.target` from
   CoinGecko `market_chart/range`, keyless, ~13 s between calls). Network
   first, then TRUNCATE + refill in one transaction. `apply_usd_amounts(table=)`
   clears every `usd_amount` and refills it (`amount × price`, last price ≤ 24h
   old), then mirrors a priced leg onto the unpriced leg of the same swap,
   then `fill_gaps()` repeats that mirror plus a same-token-in-tx unit price
   until nothing changes (an UPDATE can't see its own writes, so values move
   one hop per run; routed multi-hops need several). `price --fill-only` runs
   just `fill_gaps()`. Works on `dex_events` or `dex_events_removed`. Tests inject a fake fetcher
   via `build(fetch=...)` — never hit the network in tests.
6. **`marketcap.py`** — fills `tokens.market_cap_usd` from CoinGecko;
   unlisted is treated as "eliminated", never estimated from `totalSupply()`
   (that was tried and produced a $57T token from a bad decimals guess).
7. **`dataset.py`** — `pin()` freezes a modelling universe once into
   `tokens.target` (deliberately not a live market-cap threshold, so the token
   set doesn't reshape itself between runs).
8. **`verify.py`** — the swap-pairing invariant: every `(tx, pool)` must have
   both an inbound and outbound leg. Matched via `pools` join on
   `source`/`destination`, not the `pool` column, because a multi-hop route
   settles pool-to-pool and one row is simultaneously two legs.
9. **`db.py`** — connection pooling (`psycopg_pool`), schema bootstrap
   (`schema.sql`), and the `COPY`-into-temp-table-then-`INSERT ... ON CONFLICT`
   upsert pattern used everywhere bulk writes happen.

`dex_events_removed` holds rows taken out of `dex_events` by `cleanup.py`
(`removed_reason`: `stable_stable_standalone_swap`, `stable_vs_nontarget`,
`both_nontarget`). Only a two-leg wallet<->pool swap in a tx that touches no
other pool is ever removed, so routes are never cut. Re-ingesting the same
blocks puts removed rows back, and un-targeting a token makes more swaps
removable: re-run `cleanup` (then `price`) after either. Typical order after
changing `tokens.target`: `cleanup` -> `price` -> TGN/data/build_temporal_data.py.

`dex_events` is the one output table: a swap is two rows (wallet→pool,
pool→wallet), not a row plus a marker. `source` is never equal to
`destination` (schema-enforced via `CHECK`).

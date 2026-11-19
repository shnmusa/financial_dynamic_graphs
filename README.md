# Dynamic financial graph: Uniswap swaps and temporal graph networks

This repository has two parts:

| folder | what it does |
|---|---|
| [`dex_data/`](dex_data) | Indexes Uniswap swaps on Ethereum into Postgres and values every leg in USD with hourly CoinGecko prices. |
| [`TGN/`](TGN) | Turns those swaps into a wallet–token temporal graph and trains a Temporal Graph Network (TGN) to forecast next-hour token returns, together with baselines, descriptive analysis and the LaTeX sources of the paper. |

The data flow is one-directional: `dex_data` writes Postgres tables, and `TGN`
reads them once to build a self-contained PyTorch bundle. Everything after that
step runs offline.

```
Ethereum RPC ──► dex_data (Postgres: dex_events, token_prices_hourly, tokens …)
                     │
                     ▼  TGN/data/build_temporal_data.py
              TGN/data/processed/tgn_dex.pt  ──►  training, baselines, analysis, paper
```

---

## 1. `dex_data`: Uniswap swap indexer

### Quick start

```bash
cd dex_data
docker compose up -d postgres                 # Postgres on localhost:5433
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                          # put your RPC URL(s) here
PYTHONPATH=src .venv/bin/python -m dex_data.cli all --days 30
```

### Output table: `dex_events`

Every row is **one token movement into or out of a Uniswap pool**. A swap is
therefore two rows: wallet → pool (the token sold) and pool → wallet (the token
bought). There is no separate "swap marker" row, and `source` never equals
`destination` (enforced by a CHECK constraint).

| column | meaning |
|---|---|
| `transaction_id` | primary key, `<tx_hash>-<log_index>` |
| `type` | `swap` |
| `source` / `destination` | sending and receiving address (wallet or pool) |
| `token` | readable symbol, e.g. `WETH` (falls back to a shortened address) |
| `usd_amount` | USD value of the movement |
| `timestamp` | block time (UTC) |
| `block_number` | block the row was read from |

Extra columns for traceability: `tx_hash`, `log_index`, `token_address`,
`amount`, `amount_raw`, `pool`, `counterpart_expected`. `token_address` is kept
next to the symbol because symbols are not unique.

The transaction hash alone cannot be the key: a single transaction routinely
passes tokens through many pools (one real example has 22 swaps), so the log
index is appended.

### Where the data comes from

* **Raw `eth_getLogs`, one call per block range.** Transfer, Uniswap V2 Swap and
  Uniswap V3 Swap logs are requested together (`topics[0]` as an OR list). This
  works on free endpoints and does not depend on third-party indexers.
* **Swap events are read but not stored.** They identify which addresses are
  pools and whether a counter-leg is expected. The stored rows are token
  transfers.
* **Pools are discovered, not backfilled.** Every address that emits a Swap log
  in the window is a pool candidate; its `factory()` is then checked against the
  known Uniswap factories. This matters because SushiSwap, PancakeSwap and many
  forks emit byte-identical V2 Swap events. About 20% of candidates are
  rejected at this step.
* **Excluded:** liquidity mints and burns (a row requires the pool to have
  emitted a Swap in the same transaction) and wallet-to-wallet transfers.
* **Kept:** pool-to-pool rows. In a multi-hop route (USDC → WETH → LINK) one
  transfer is both the output leg of the first pool and the input leg of the
  second; dropping it would break both swaps.

### USD valuation

`price` rebuilds `token_prices_hourly` and then fills `usd_amount`:

* stablecoins (`is_stable`) are fixed at exactly $1 per hour;
* every non-stable token in the modelling universe (`tokens.target`) gets
  hourly prices from CoinGecko `/coins/{id}/market_chart/range`;
* each row is valued as `amount × price of its hour`; a leg without a price
  (always a non-universe token) copies the value of the other leg of the same
  swap.

Network calls happen first, and the table is cleared and refilled in a single
database transaction, so an interrupted CoinGecko download never leaves it half
priced. Without a key CoinGecko allows about 5 calls per minute; set
`COINGECKO_API_KEY` to go faster.

### Modelling universe and cleanup

* `marketcap` fetches market caps from CoinGecko. Tokens CoinGecko does not
  list get no market cap and cannot enter the universe.
* `target` pins the universe once into `tokens.target` (by default every token
  above the configured market-cap threshold). It is intentionally frozen, so
  the token set does not change when caps drift.
* `cleanup` moves swaps that carry no information about the universe to
  `dex_events_removed`: stand-alone stablecoin↔stablecoin swaps, stablecoin↔
  non-universe swaps and swaps between two non-universe tokens. Only simple
  two-leg swaps in single-pool transactions are removed, so no multi-hop route
  is cut in half.

### Swap-pairing check

Every `(transaction, pool)` must have one leg into and one leg out of the pool,
in different tokens. `verify` checks this and exits with status 1 if it fails.
Two legitimate exceptions are classified at ingest time
(`counterpart_expected = FALSE`) rather than tolerated with a threshold:

1. **Swaps with a zero side.** If an amount rounds to zero, no Transfer is
   emitted for it (typical of arbitrage bots probing with 1 wei).
2. **Rebasing / reflection tokens** (e.g. KISHU), whose balances change without
   Transfer events.

On real data every unpaired swap falls into one of these categories, with no
unexplained cases.

```bash
docker exec -i dex_data_pg psql -U dex -d dex_data -q < sql/swap_pairing_check.sql
PYTHONPATH=src .venv/bin/python -m dex_data.cli verify
```

### RPC client

The client is provider-agnostic. It learns at run time, per endpoint, the
largest accepted batch, unsupported methods and the widest `getLogs` range;
halves its request rate on HTTP 429 and recovers additively (AIMD); and fails
over between endpoints. Argument-less view calls (`factory()`, `symbol()`,
`decimals()`, …) are bundled through
[Multicall3](https://www.multicall3.com/): in one measurement 900 billed
`eth_call`s became 4, 122 times faster. Alchemy's free tier is recommended; it
includes archive state, which historical `eth_call`s need.

### Parallelism and the pool-classification race

Block ranges are independent and are processed on a `ThreadPoolExecutor` (the
work is network I/O). Pool candidates are claimed per address, so a cold start
with tens of thousands of unseen pools does not serialise. A real bug was found
and fixed here: when one thread's pool verification failed, it released its
candidates, and another thread waiting on the same addresses treated them as
"not a pool", skipped their swaps and marked its chunk as done. One thread's RPC
error silently corrupted another thread's output. Now the waiting thread takes
over the work, and a candidate that cannot be resolved fails the chunk so it is
retried. Three concurrency tests lock this in
(`tests/test_ingest.py::TestPoolClassificationRace`).

Measured speed with 3 workers on Alchemy's free tier: about 100 blocks per 4
minutes from a cold start, faster as the pool table fills. Progress is stored
in `ingest_progress`, so an ingest can be stopped and resumed at any time.

### Commands

```bash
python -m dex_data.cli initdb                # create the schema
python -m dex_data.cli ingest --days 30      # log scan (resumable)
python -m dex_data.cli ingest --from-block N --to-block M
python -m dex_data.cli marketcap             # CoinGecko market caps
python -m dex_data.cli target                # pin the modelling universe (once)
python -m dex_data.cli price                 # hourly prices -> usd_amount
python -m dex_data.cli cleanup               # move uninformative swaps to dex_events_removed
python -m dex_data.cli verify                # integrity report, exit 1 on failure
python -m dex_data.cli all --days 30         # ingest -> price -> verify
python -m dex_data.cli reset --yes           # clear output (--all: pools and tokens too)
```

Run with `PYTHONPATH=src .venv/bin/python -m dex_data.cli …` from `dex_data/`,
or install the package (`pip install -e .`) and use the `dex-data` script.

Settings come from `.env` (see `.env.example`): `DEX_RPC_URLS`, `DEX_ARCHIVE`,
`DEX_PG_DSN`, `DEX_DAYS`, `DEX_MAX_WORKERS`, `DEX_LOG_CHUNK_BLOCKS`,
`DEX_RPC_TIMEOUT`, `DEX_RPC_MAX_RETRIES`, `DEX_RPC_RATE_LIMIT`.
`DEX_LOG_CHUNK_BLOCKS` is a starting hint: a range the provider rejects as too
large is split in half automatically.

### Schema

```
reference    tokens, pools, blocks
derived      token_prices_hourly
output       dex_events, dex_events_removed
bookkeeping  ingest_progress
```

### Tests and exploratory analysis

```bash
.venv/bin/python -m pytest tests/ -v -rs
```

Database tests run against a real Postgres in an isolated schema, not mocks.
Tests marked `@pytest.mark.live` need indexed data and are **skipped** when it
is missing or Postgres is down, so check the skip report (`-rs`).

```bash
.venv/bin/pip install -e ".[notebook]"
.venv/bin/jupyter lab notebooks/eda.ipynb
```

When summing volumes, remember that each swap is two rows: `sum(usd_amount)`
double-counts traded volume unless you keep only one direction.

---

## 2. `TGN`: return forecasting on the wallet–token graph

### Setup

```bash
cd TGN
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

All commands below are run from the repository root. Training uses Apple's MPS
backend when it is available and falls back to the CPU.

### Data

Window: 28 August 2026 10:20 UTC to 26 September 2026 12:16 UTC (699 hours).

```bash
TGN/.venv/bin/python TGN/data/build_temporal_data.py      # reads Postgres (needs dex_data's .env)
TGN/.venv/bin/python TGN/data/filter_wallets.py --min-trades 3 --min-volume 0
```

`build_temporal_data.py` writes `TGN/data/processed/tgn_dex.pt`, a PyG
`TemporalData` plus hourly price targets:

* **Nodes:** 36 universe tokens (ids 0–35), 3 stablecoins (36–38), one
  `UNKNOWN` node for every token outside the universe (39), and wallets (40+).
  In total 214,625 nodes, of which 214,585 are wallets.
* **Events:** for each (transaction, wallet, token) the wallet's legs are
  netted. A net sale is an edge token → wallet, a net purchase wallet → token.
  Routing intermediates (net flow under 1% of gross flow) and events with an
  unpriced leg are dropped. 7,983,119 events.
* **Message:** `[side, log1p(USD value)]`; time is the block timestamp.
* **Targets:** hourly CoinGecko prices of the universe tokens. SI is excluded as
  a target (CoinGecko lists it only from 22 September) but stays in the graph,
  leaving 35 target tokens.

`filter_wallets.py` makes the wallet-filtered variant
(`tgn_dex_min3_vol0.pt`): wallets with at least 3 trades and positive total
volume. It keeps 23.4% of the wallets but 97.3% of the events.

### Task

At the end of every hour, forecast each target token's log-return over the next
hour, scaled by its ex-ante volatility (standard deviation of the last 48
hourly returns). The prediction hours are split chronologically 70/15/15 into
train, validation and test (test: 22–26 September, 3,570 token-hours).

### Models

| script | model |
|---|---|
| `train.py` | TGN: memory (GRU cell, dim 100) + temporal graph attention (TransformerConv, 2 heads × 50, last 10 neighbours) + MLP decoder; optional auxiliary link-prediction loss (`--aux-weight`) |
| `baselines.py` | zero forecast, training mean, last return, pooled AR(1), flow ridge regression on six hourly order-flow features |
| `baselines_nn.py --model gru` | GRU on the hourly flow features: the TGN's memory without the graph |
| `baselines_nn.py --model gnn` | static GNN on hourly snapshot graphs: the TGN's attention without memory |

The neural models share the TGN's building blocks and training protocol (Adam,
learning rate 1e-4, one Huber step per hour, early stopping on validation MSE).

```bash
TGN/.venv/bin/python TGN/train.py --data TGN/data/processed/tgn_dex_min3_vol0.pt --aux-weight 0 --seed 0
TGN/.venv/bin/python TGN/baselines.py --data TGN/data/processed/tgn_dex_min3_vol0.pt
TGN/.venv/bin/python TGN/baselines_nn.py --model gru --data TGN/data/processed/tgn_dex_min3_vol0.pt --seed 0
TGN/.venv/bin/python TGN/baselines_nn.py --model gnn --data TGN/data/processed/tgn_dex_min3_vol0.pt --seed 0
TGN/.venv/bin/python TGN/summarize.py         # -> TGN/runs/results.md
```

Each run writes `config.json`, `metrics.json` and predictions to
`TGN/runs/<timestamp>/` (`gru-…`, `gnn-…`, `baselines-…` for the others).
`summarize.py` groups the runs by dataset and reports mean ± standard deviation
over seeds.

### Metrics

* **R²_OOS (skill):** 1 − MSE(model) / MSE(zero forecast) on scaled returns.
* **RMSE:** on log-returns, in percent.
* **Direction accuracy:** share of correctly predicted signs among non-zero
  returns.
* **IC:** mean hourly cross-sectional Pearson correlation between forecast and
  realised return.

### Results (wallet-filtered data, test period)

| model | direction acc. | R²_OOS (%) | IC |
|---|---|---|---|
| flow ridge regression | 0.511 | **+0.08** | **0.057** |
| GRU | **0.514** ± 0.008 | −0.25 ± 0.06 | 0.049 ± 0.005 |
| static GNN | 0.506 ± 0.001 | −0.19 ± 0.02 | 0.041 ± 0.005 |
| TGN | 0.507 ± 0.001 | 0.00 ± 0.13 | 0.002 ± 0.014 |
| TGN + link-prediction loss | 0.506 ± 0.001 | −0.40 ± 0.37 | 0.005 ± 0.011 |

No model improves materially on the no-change forecast. The small ranking
signal (IC ≈ 0.05) comes from short-term reversal of the last hour's return.
Neither temporal memory nor the graph adds to simple order-flow features. The
unfiltered data gives the same picture. Full tables are in
[`TGN/runs/results.md`](TGN/runs/results.md).

### Descriptive analysis

| script | output |
|---|---|
| `analysis/wallet_powerlaw.py [--sims 1000]` | per-wallet trades and volume; power-law fits (Clauset–Shalizi–Newman: KS-optimal x_min, MLE α, bootstrap goodness of fit, likelihood-ratio tests) |
| `analysis/plot_wallet_powerlaw.py` | density and CCDF plots with the fitted power law and lognormal |
| `analysis/plot_wallet_bar_hist.py` | wallet histograms with logarithmic bins |
| `analysis/plot_token_activity.py` | events and USD volume per token node |
| `analysis/wallet_descriptives.py [--min-trades k] [--positive-volume] [--min-volume v]` | summary statistics per wallet subset |
| `analysis/latex_descriptives.py`, `analysis/latex_results.py` | LaTeX tables for the paper |

Results go to `TGN/analysis/results/`, figures to `TGN/analysis/figures/`.

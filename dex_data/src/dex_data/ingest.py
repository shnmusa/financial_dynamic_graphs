"""Parallel log ingestion, writing dex_events directly.

Strategy
--------
One `eth_getLogs` per block chunk, with a topic0 OR-filter that pulls Transfer,
UniswapV2 Swap and UniswapV3 Swap logs together. Fetching all three in a single
call halves the request count versus separate passes, and it is *correct* to
filter transfers against pools discovered in the same chunk: the transfer legs
of a swap are emitted by the same transaction, so they always land in the same
block, and therefore the same chunk.

What gets written
-----------------
Every token movement into or out of a verified Uniswap pool becomes one
dex_events row. A swap is therefore two rows -- wallet -> pool and pool ->
wallet -- and those two rows *are* the swap; there is no separate marker row.

The Swap events themselves are not stored. They are read for two things and
then discarded: to discover which addresses are pools, and to decide whether
the chain could have emitted the opposite leg (`counterpart_expected`).

Chunks are independent, so they fan out across a thread pool. The work is pure
network I/O -- the GIL is released for the whole request -- so threads, not
processes, are the right tool, and they avoid having to ship decoded logs
across a process boundary.

Every chunk is checkpointed in ingest_progress, so an interrupted run resumes
instead of rescanning.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from . import abi, db, pools, tokens
from .config import CONFIG, Config
from .rpc import RpcClient, get_client

log = logging.getLogger(__name__)

JOB_NAME = "logs"

# topics[0] as an OR-array: "any of these three event signatures".
LOG_TOPICS: list = [[abi.TRANSFER_TOPIC, abi.V2_SWAP_TOPIC, abi.V3_SWAP_TOPIC]]

# Upper bound on resolving one chunk's pool candidates. Exceeding it fails
# the chunk, which is then retried, rather than writing it incomplete.
CLASSIFY_TIMEOUT = 300


class PoolRegistry:
    """Thread-safe cache of which addresses are (and are not) Uniswap pools.

    Two things matter here:

    * The negative cache. Without it every chunk would re-issue factory() calls
      for the same few thousand non-Uniswap forks it just rejected.
    * Not holding one global lock across the verification round trip. Chunks
      run concurrently and mostly see *different* new pools, so verification is
      tracked per-address: a thread claims the addresses nobody else is
      resolving, verifies just those off-lock, and waits on a condition for any
      claimed by someone else. Serialising it instead would make the cold start
      (tens of thousands of unseen pools) single-threaded.
    """

    def __init__(self, client: RpcClient):
        self._client = client
        self._cond = threading.Condition()
        self._pools: dict[str, dict] = pools.load_all()
        self._not_pools: set[str] = set()
        self._in_flight: set[str] = set()
        # symbol/decimals/is_rebasing, needed to write a row: the output table
        # carries a readable ticker and a decimal-adjusted amount.
        self._tokens: dict[str, dict] = tokens.load_metadata()

    def snapshot(self) -> set[str]:
        with self._cond:
            return set(self._pools)

    def pool(self, address: str) -> dict | None:
        with self._cond:
            return self._pools.get(address)

    def token(self, address: str) -> dict:
        """Token metadata, falling back to a shortened address as the label.

        A token with no readable symbol() still has to produce a usable row, so
        the fallback keeps the address recognisable rather than leaving a NULL.
        """
        with self._cond:
            meta = self._tokens.get(address)
        if meta is None:
            meta = {"symbol": None, "decimals": 18, "is_rebasing": False}
        symbol = meta.get("symbol") or f"{address[:8]}…{address[-4:]}"
        return {
            "symbol": symbol,
            "decimals": meta.get("decimals") if meta.get("decimals") is not None else 18,
            "is_rebasing": bool(meta.get("is_rebasing")),
        }

    def classify(self, candidates: set[str], first_seen: dict[str, int]) -> set[str]:
        """Return the subset of candidates that are Uniswap pools.

        Every candidate is resolved to a definite answer -- pool or not -- before
        returning. That matters more than it looks: if one thread's verification
        fails, it releases its claim, and a thread waiting on those same
        addresses used to see them as merely "unknown" and carry on. It then
        wrote a chunk missing those pools' swaps and checkpointed it as done, so
        one thread's RPC error silently corrupted another thread's output. Now
        the waiter takes the work over instead, and a candidate that cannot be
        resolved at all raises, failing the chunk so it is retried.
        """
        pending = set(candidates)
        deadline = time.monotonic() + CLASSIFY_TIMEOUT

        while True:
            with self._cond:
                pending = {
                    a for a in pending
                    if a not in self._pools and a not in self._not_pools
                }
                if not pending:
                    return {a for a in candidates if a in self._pools}

                mine = [a for a in pending if a not in self._in_flight]
                if mine:
                    self._in_flight.update(mine)
                else:
                    # Everything left is being verified by another thread. Wait
                    # for it to publish an answer or release its claim.
                    self._cond.wait_for(
                        lambda: any(
                            a in self._pools or a in self._not_pools
                            or a not in self._in_flight
                            for a in pending
                        ),
                        timeout=5,
                    )
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"timed out classifying {len(pending)} pool candidates"
                        )
                    continue

            # Verify off-lock so concurrent chunks resolve disjoint sets in
            # parallel; serialising this would make the cold start (tens of
            # thousands of unseen pools) single-threaded.
            try:
                verified = pools.verify_candidates(mine, self._client)
                if verified:
                    pools.store(verified, first_seen)
            except Exception:
                # Release the claim so another thread can retry these, rather
                # than leaving them permanently unresolvable.
                with self._cond:
                    self._in_flight.difference_update(mine)
                    self._cond.notify_all()
                raise

            refreshed = tokens.load_metadata() if verified else None
            with self._cond:
                self._pools.update(verified)
                if refreshed:
                    self._tokens.update(refreshed)
                self._not_pools.update(a for a in mine if a not in verified)
                self._in_flight.difference_update(mine)
                self._cond.notify_all()

            if time.monotonic() > deadline:
                raise RuntimeError("timed out classifying pool candidates")


@dataclass
class IngestStats:
    chunks_done: int = 0
    chunks_failed: int = 0
    #: dex_events rows written (two per swap).
    rows: int = 0
    #: Swap events seen; read for pool discovery, never stored.
    swaps: int = 0
    logs_seen: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, *, rows: int = 0, swaps: int = 0, logs: int = 0,
            done: int = 0, failed: int = 0) -> None:
        with self._lock:
            self.rows += rows
            self.swaps += swaps
            self.logs_seen += logs
            self.chunks_done += done
            self.chunks_failed += failed


def chunk_ranges(from_block: int, to_block: int, size: int) -> list[tuple[int, int]]:
    return [(lo, min(lo + size - 1, to_block)) for lo in range(from_block, to_block + 1, size)]


def _timestamps_for_chunk(
    logs: list[dict], client: RpcClient
) -> dict[int, datetime]:
    """Resolve block timestamps, preferring any the provider volunteered.

    Erigon-backed endpoints return `blockTimestamp` on log objects; Geth-backed
    ones do not. Using it when present removes a whole batch round trip per
    chunk, and falling back keeps us provider-agnostic.
    """
    resolved: dict[int, datetime] = {}
    missing: set[int] = set()

    for entry in logs:
        number = abi.to_uint(entry["blockNumber"])
        if number in resolved:
            continue
        raw_ts = entry.get("blockTimestamp")
        if raw_ts:
            ts = int(raw_ts, 16) if isinstance(raw_ts, str) else int(raw_ts)
            resolved[number] = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            missing.add(number)

    if missing:
        for number, ts in client.get_block_timestamps(sorted(missing)).items():
            resolved[number] = datetime.fromtimestamp(ts, tz=timezone.utc)

    return resolved


def _counterpart_exempt(
    swap_rows: list[dict], registry: PoolRegistry
) -> set[tuple[str, str]]:
    """(tx_hash, pool) pairs where the chain cannot have emitted both legs.

    Two causes, both established from live data rather than assumed:

    * The swap's amount on one side is zero. No tokens move there, so no
      Transfer is emitted. These are dust probes -- typically 1 wei in, nothing
      out. In a 3149-swap sample every unpaired swap had a zero side and no
      paired swap did.
    * One side is a rebasing/reflection token, which changes balances without
      emitting Transfer. A v2 pair accumulates it silently and anyone can call
      swap() to skim the surplus, so the Swap event reports an input for which
      no Transfer exists anywhere in the transaction. Confirmed on KISHU/WETH.

    Anything else missing a leg is a real gap and is reported as a failure.
    """
    exempt: set[tuple[str, str]] = set()
    for row in swap_rows:
        pool = registry.pool(row["pool"])
        if pool is None:
            continue
        if row["amount0_raw"] == 0 or row["amount1_raw"] == 0:
            exempt.add((row["tx_hash"], row["pool"]))
            continue
        if any(
            registry.token(t)["is_rebasing"] for t in (pool["token0"], pool["token1"])
        ):
            exempt.add((row["tx_hash"], row["pool"]))
    return exempt


def _build_rows(
    transfer_rows: list[dict],
    swapped: set[tuple[str, str]],
    exempt: set[tuple[str, str]],
    block_times: dict[int, datetime],
    registry: PoolRegistry,
) -> list[dict]:
    """Turn swap-related pool transfers into dex_events rows.

    Two rules decide whether a transfer is a swap leg.

    First, the pool it touches must have emitted a Swap event in that same
    transaction. Pools also receive and pay out tokens for liquidity mints,
    burns, skims and donations, and those are not exchanges: a burn sends BOTH
    tokens out and a mint takes BOTH in, so admitting them would put rows in a
    table of swaps that can never have an opposite leg.

    Second, the token moved must be one of that pool's own two tokens. The
    first rule alone is not enough, because a v2 pair is itself an ERC20: it
    emits Transfer events for its own LP token when liquidity is added or
    removed, and if that happens in a transaction where the pool also swapped,
    the LP movement looks like a leg. Those rows are not swaps, they name the
    pool address as their "token", and that address is never in `tokens` --
    which broke the whole pricing pass on a foreign key violation. Anything a
    router happens to dump on a pool is excluded by the same rule.
    """
    out: list[dict] = []
    for row in transfer_rows:
        # A same-address Transfer moves nothing anywhere -- some tokens emit
        # one anyway (reward/rebase bookkeeping). It can never be a swap leg,
        # and writing it would violate dex_events' own source != destination
        # invariant, so it is dropped rather than let it fail the whole chunk.
        if row["from_addr"] == row["to_addr"]:
            continue

        timestamp = block_times.get(row["block_number"])
        if timestamp is None:
            continue

        tx = row["tx_hash"]
        from_swapped = (tx, row["from_addr"]) in swapped
        to_swapped = (tx, row["to_addr"]) in swapped
        if not (from_swapped or to_swapped):
            continue

        # A pool-to-pool leg (multi-hop routing) belongs to both pools; it is
        # attributed to the sender, and the receiving pool still matches
        # through `destination` in the pairing check.
        pool = row["from_addr"] if from_swapped else row["to_addr"]

        # The moved token must be one of this pool's two tokens; see above.
        pool_meta = registry.pool(pool)
        if pool_meta is None or row["token"] not in (
            pool_meta["token0"], pool_meta["token1"]
        ):
            continue

        meta = registry.token(row["token"])
        decimals = meta["decimals"]

        # Exempt if either pool this leg touches is exempt for this transaction.
        expected = (tx, pool) not in exempt
        if expected and from_swapped and to_swapped:
            expected = (tx, row["to_addr"]) not in exempt

        out.append(
            {
                "transaction_id": f"{row['tx_hash']}-{row['log_index']}",
                "type": "swap",
                "source": row["from_addr"],
                "destination": row["to_addr"],
                "token": meta["symbol"],
                "usd_amount": None,  # filled in by the pricing pass
                "timestamp": timestamp,
                "block_number": row["block_number"],
                "tx_hash": row["tx_hash"],
                "log_index": row["log_index"],
                "token_address": row["token"],
                "amount": Decimal(row["amount_raw"]) / (Decimal(10) ** decimals),
                "amount_raw": row["amount_raw"],
                "pool": pool,
                "counterpart_expected": expected,
            }
        )
    return out


def process_chunk(
    lo: int,
    hi: int,
    client: RpcClient,
    registry: PoolRegistry,
    stats: IngestStats,
) -> int:
    """Fetch, decode, filter and persist one block range. Returns rows written."""
    logs = client.get_logs_adaptive(lo, hi, topics=LOG_TOPICS)

    swap_rows: list[dict] = []
    transfer_rows: list[dict] = []
    for entry in logs:
        decoded = abi.decode_log(entry)
        if decoded is None:
            continue
        kind, row = decoded
        (swap_rows if kind == "swap" else transfer_rows).append(row)

    # Swap events are read to discover pools and to judge leg expectations,
    # then discarded -- they are not stored.
    candidates = {r["pool"] for r in swap_rows}
    first_seen: dict[str, int] = {}
    for row in swap_rows:
        first_seen.setdefault(row["pool"], row["block_number"])
    if candidates:
        registry.classify(candidates, first_seen)

    known_pools = registry.snapshot()
    swap_rows = [r for r in swap_rows if r["pool"] in known_pools]

    # Scope: only transfers that touch a pool. Both legs of every swap match
    # here -- one has the pool as destination, the other as source.
    transfer_rows = [
        r for r in transfer_rows
        if r["from_addr"] in known_pools or r["to_addr"] in known_pools
    ]

    # The (transaction, pool) pairs that genuinely swapped. Everything written
    # below is scoped to these, which is what keeps mints and burns out.
    swapped = {(r["tx_hash"], r["pool"]) for r in swap_rows}

    block_times = _timestamps_for_chunk(logs, client) if transfer_rows else {}
    exempt = _counterpart_exempt(swap_rows, registry)
    event_rows = _build_rows(transfer_rows, swapped, exempt, block_times, registry)

    with db.connection() as conn:
        if block_times:
            db._copy_upsert(
                conn, "blocks", ("number", "timestamp"),
                sorted(block_times.items()), ("number",),
            )
        written = db.insert_events(conn, event_rows) if event_rows else 0
        db.mark_chunk_done(conn, JOB_NAME, lo, hi, len(logs))
        conn.commit()

    stats.add(rows=written, swaps=len(swap_rows), logs=len(logs), done=1)
    return written


def run(
    from_block: int,
    to_block: int,
    config: Config = CONFIG,
    client: RpcClient | None = None,
    resume: bool = True,
) -> IngestStats:
    client = client or get_client()
    registry = PoolRegistry(client)
    stats = IngestStats()

    ranges = chunk_ranges(from_block, to_block, config.log_chunk_blocks)
    if resume:
        done = db.completed_chunks(JOB_NAME)
        pending = [(lo, hi) for lo, hi in ranges if lo not in done]
        skipped = len(ranges) - len(pending)
        if skipped:
            log.info("resuming: %d/%d chunks already done", skipped, len(ranges))
    else:
        pending = ranges

    if not pending:
        log.info("nothing to do")
        return stats

    total = len(pending)
    log.info(
        "ingesting %d chunks (%d blocks, %d per chunk) with %d workers",
        total, to_block - from_block + 1, config.log_chunk_blocks, config.max_workers,
    )

    started = time.monotonic()
    last_report = started

    with ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="ingest") as pool:
        futures = {
            pool.submit(process_chunk, lo, hi, client, registry, stats): (lo, hi)
            for lo, hi in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            lo, hi = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not stop the run
                stats.add(failed=1)
                log.warning("chunk %d-%d failed: %s", lo, hi, exc)

            now = time.monotonic()
            if now - last_report >= 15 or completed == total:
                elapsed = now - started
                rate = completed / elapsed if elapsed else 0
                eta = (total - completed) / rate if rate else 0
                log.info(
                    "%d/%d chunks (%.1f%%) | %d rows from %d swap events | "
                    "%.1f chunks/s | ETA %.0f min",
                    completed, total, 100 * completed / total,
                    stats.rows, stats.swaps, rate, eta / 60,
                )
                last_report = now

    log.info(
        "ingest done in %.1f min: %d rows from %d swap events, %d chunks failed",
        (time.monotonic() - started) / 60, stats.rows, stats.swaps, stats.chunks_failed,
    )
    return stats

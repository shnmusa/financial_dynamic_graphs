"""Mapping wall-clock time to block numbers, and caching block timestamps."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from . import db
from .rpc import RpcClient, get_client

log = logging.getLogger(__name__)

# Post-merge Ethereum produces a block every 12s exactly (modulo missed slots),
# which makes the initial guess for the binary search very close to correct.
SECONDS_PER_BLOCK = 12


def block_at_timestamp(target: datetime, client: RpcClient | None = None) -> int:
    """First block whose timestamp is >= target, via binary search."""
    client = client or get_client()
    target_ts = int(target.timestamp())

    latest = client.get_block("latest")
    hi = int(latest["number"], 16)
    hi_ts = int(latest["timestamp"], 16)

    if target_ts >= hi_ts:
        return hi

    # Seed the lower bound from the constant block time, with slack, so the
    # search starts inside a few hundred blocks instead of at genesis.
    estimate = hi - (hi_ts - target_ts) // SECONDS_PER_BLOCK
    lo = max(1, estimate - 20_000)

    lo_block = client.get_block(lo)
    while lo_block and int(lo_block["timestamp"], 16) > target_ts and lo > 1:
        lo = max(1, lo - 100_000)
        lo_block = client.get_block(lo)

    while lo < hi:
        mid = (lo + hi) // 2
        block = client.get_block(mid)
        if block is None:
            lo = mid + 1
            continue
        if int(block["timestamp"], 16) < target_ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def resolve_window(days: int, client: RpcClient | None = None) -> tuple[int, int, datetime, datetime]:
    """Block range covering the last `days` days, plus the wall-clock bounds."""
    client = client or get_client()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    end_block = client.block_number()
    start_block = block_at_timestamp(start, client)
    log.info(
        "window: %s .. %s  ->  blocks %d .. %d (%d blocks)",
        start.isoformat(timespec="seconds"),
        now.isoformat(timespec="seconds"),
        start_block,
        end_block,
        end_block - start_block + 1,
    )
    return start_block, end_block, start, now


def backfill_block_timestamps(
    numbers: list[int], client: RpcClient | None = None, batch_size: int = 100
) -> int:
    """Fetch and store timestamps for blocks not already cached."""
    client = client or get_client()
    numbers = sorted(set(numbers))
    if not numbers:
        return 0

    known = {
        r[0]
        for r in db.fetch_all(
            "SELECT number FROM blocks WHERE number = ANY(%s)", (numbers,)
        )
    }
    missing = [n for n in numbers if n not in known]
    if not missing:
        return 0

    log.info("fetching timestamps for %d blocks", len(missing))
    fetched = client.get_block_timestamps(missing, batch_size=batch_size)
    rows = [
        (number, datetime.fromtimestamp(ts, tz=timezone.utc))
        for number, ts in sorted(fetched.items())
    ]
    db.upsert("blocks", ("number", "timestamp"), rows, ("number",))
    return len(rows)

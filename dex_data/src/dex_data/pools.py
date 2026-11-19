"""Uniswap pool discovery and verification.

Pools are not backfilled from factory PairCreated/PoolCreated events -- there
are ~400k of those going back to 2020 and we only care about pools that traded
inside the window. Instead every address that emits a Swap log in the window is
a pool candidate, and each candidate is then asked for its factory() to confirm
it really is Uniswap rather than one of the many byte-identical v2 forks.
"""
from __future__ import annotations

import logging
from typing import Iterable

from . import abi, db, tokens
from .rpc import RpcClient, get_client

log = logging.getLogger(__name__)


def verify_candidates(
    candidates: Iterable[str], client: RpcClient | None = None, batch_size: int = 500
) -> dict[str, dict]:
    """Resolve factory/token0/token1/fee for candidate pool addresses.

    Addresses whose factory() is not a known Uniswap factory are dropped, which
    is what keeps SushiSwap et al. out of a table that claims to be Uniswap.
    """
    client = client or get_client()
    candidates = [a.lower() for a in dict.fromkeys(candidates)]
    verified: dict[str, dict] = {}
    rejected = 0

    for start in range(0, len(candidates), batch_size):
        window = candidates[start : start + batch_size]

        factories = client.multicall([(a, abi.SEL_FACTORY) for a in window])

        keep: list[tuple[str, str, str]] = []  # (pool, dex, version)
        for address, raw in zip(window, factories):
            factory = abi.decode_address(raw) if raw else None
            entry = abi.KNOWN_FACTORIES.get(factory or "")
            if entry is None:
                rejected += 1
                continue
            keep.append((address, *entry))

        if not keep:
            continue

        calls: list[tuple[str, str]] = []
        for address, _dex, version in keep:
            calls.append((address, abi.SEL_TOKEN0))
            calls.append((address, abi.SEL_TOKEN1))
            calls.append((address, abi.SEL_FEE))  # reverts on v2; batch tolerates it

        results = client.multicall(calls)

        for index, (address, dex, version) in enumerate(keep):
            token0_raw, token1_raw, fee_raw = results[index * 3 : index * 3 + 3]
            token0 = abi.decode_address(token0_raw) if token0_raw else None
            token1 = abi.decode_address(token1_raw) if token1_raw else None
            if not token0 or not token1:
                rejected += 1
                continue

            verified[address] = {
                "address": address,
                "dex": dex,
                "version": version,
                "factory": abi.UNISWAP_V2_FACTORY if version == "v2" else abi.UNISWAP_V3_FACTORY,
                "token0": token0,
                "token1": token1,
                "fee": abi.decode_uint(fee_raw) if (version == "v3" and fee_raw) else None,
            }

    log.info("verified %d Uniswap pools, rejected %d non-Uniswap candidates",
             len(verified), rejected)
    return verified


def store(pools: dict[str, dict], first_seen: dict[str, int] | None = None) -> int:
    """Persist pools. Tokens are inserted first to satisfy the FK."""
    if not pools:
        return 0

    first_seen = first_seen or {}
    token_addresses = {p["token0"] for p in pools.values()} | {p["token1"] for p in pools.values()}
    tokens.ensure_known(sorted(token_addresses))

    rows = [
        (
            p["address"], p["dex"], p["version"], p["factory"],
            p["token0"], p["token1"], p["fee"], first_seen.get(p["address"]),
        )
        for p in pools.values()
    ]
    n = db.upsert(
        "pools",
        ("address", "dex", "version", "factory", "token0", "token1", "fee", "first_seen_block"),
        rows,
        ("address",),
        update_cols=("dex", "version", "factory", "token0", "token1", "fee"),
    )
    # A newly stored pool may already touch a pinned target token (the
    # universe is pinned once, before most pools are ever discovered), so
    # is_target has to be (re)computed here too, not just from `cli target`.
    # Scoped to just these addresses -- store() runs once per chunk during
    # discovery, and a full-table rewrite every time would be its own waste.
    refresh_target_flags(pools.keys())
    return n


def discover_and_store(
    candidates: Iterable[str],
    client: RpcClient | None = None,
    first_seen: dict[str, int] | None = None,
) -> int:
    """Verify and persist only candidates we have not already classified."""
    candidates = [a.lower() for a in dict.fromkeys(candidates)]
    if not candidates:
        return 0

    known = {
        r[0]
        for r in db.fetch_all("SELECT address FROM pools WHERE address = ANY(%s)", (candidates,))
    }
    unknown = [a for a in candidates if a not in known]
    if not unknown:
        return 0

    log.info("classifying %d new pool candidates", len(unknown))
    return store(verify_candidates(unknown, client), first_seen)


def load_all() -> dict[str, dict]:
    rows = db.fetch_all(
        "SELECT address, version, token0, token1, fee FROM pools"
    )
    return {
        r[0]: {"address": r[0], "version": r[1], "token0": r[2], "token1": r[3], "fee": r[4]}
        for r in rows
    }


def load_addresses() -> set[str]:
    return {r[0] for r in db.fetch_all("SELECT address FROM pools")}


def refresh_target_flags(addresses: Iterable[str] | None = None) -> int:
    """Recompute pools.is_target from the current tokens.target set.

    Requires a target token on at least one side: a target traded against
    something outside the universe is still a real flow. But a stablecoin
    against a non-target is excluded even then -- that side can only ever be
    priced as $1 for $unknown, so scoping ingest to such a pool would just
    spend requests on rows that carry no price information.

    With `addresses`, scoped to just those pools (what `store` calls after
    discovering new ones -- a full-table rewrite on every chunk would be its
    own waste). Without it, the whole table -- what `dataset.pin` calls, since
    pinning (or re-pinning) the universe can flip the flag for pools that were
    already sitting in the table long before this token was ever a target.
    """
    where = "AND p.address = ANY(%(addresses)s)" if addresses is not None else ""
    return db.execute(
        f"""
        UPDATE pools p SET is_target = (
            (t0.target OR t1.target)
            AND NOT (t0.is_stable AND NOT t1.target)
            AND NOT (t1.is_stable AND NOT t0.target)
        )
        FROM tokens t0, tokens t1
        WHERE t0.address = p.token0 AND t1.address = p.token1
        {where}
        """,
        {"addresses": list(addresses)} if addresses is not None else None,
    )


def target_addresses() -> set[str]:
    """Pools flagged is_target -- see refresh_target_flags."""
    return {r[0] for r in db.fetch_all("SELECT address FROM pools WHERE is_target")}

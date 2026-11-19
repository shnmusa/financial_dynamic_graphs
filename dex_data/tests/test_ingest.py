"""Row-building decisions made during ingest.

These run against the real functions with a stub registry -- no network, no
database. They cover the judgements that decide whether a chain event becomes a
row at all, which the DB tests cannot reach because those seed dex_events
directly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from dex_data import ingest

from .factories import (
    KISHU, POOL_2, POOL_KISHU, POOL_V3, TOKEN_META, TRADER, USDC, USDC_1000,
    WETH, WETH_HALF,
)

BLOCK = 100
WHEN = datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc)
TIMES = {BLOCK: WHEN}


class StubRegistry:
    """Stands in for PoolRegistry without a database or an RPC endpoint."""

    #: A token with no readable symbol(), paired against WETH. Real pools hold
    #: plenty of these, and a row for one still has to be usable.
    NAMELESS = "0xdeadbeef00000000000000000000000000001234"
    POOL_NAMELESS = "0xaaaa000000000000000000000000000000000002"

    POOLS = {
        POOL_V3: {"token0": USDC, "token1": WETH},
        POOL_2: {"token0": USDC, "token1": WETH},
        POOL_KISHU: {"token0": KISHU, "token1": WETH},
        POOL_NAMELESS: {"token0": NAMELESS, "token1": WETH},
    }

    def pool(self, address):
        return self.POOLS.get(address)

    def token(self, address):
        meta = TOKEN_META.get(address)
        if meta is None:
            return {"symbol": f"{address[:8]}…{address[-4:]}", "decimals": 18,
                    "is_rebasing": False}
        symbol, decimals, _stable, rebasing = meta
        return {"symbol": symbol, "decimals": decimals, "is_rebasing": rebasing}


@pytest.fixture
def registry():
    return StubRegistry()


def transfer(tx, log_index, token, sender, recipient, amount):
    return {
        "tx_hash": tx, "log_index": log_index, "block_number": BLOCK,
        "token": token, "from_addr": sender, "to_addr": recipient,
        "amount_raw": amount,
    }


def swap_event(tx, pool, amount0, amount1, log_index=9):
    return {
        "tx_hash": tx, "log_index": log_index, "block_number": BLOCK,
        "pool": pool, "version": "v3", "amount0_raw": amount0, "amount1_raw": amount1,
    }


def build(transfers, swaps, registry):
    swapped = {(s["tx_hash"], s["pool"]) for s in swaps}
    exempt = ingest._counterpart_exempt(swaps, registry)
    return ingest._build_rows(transfers, swapped, exempt, TIMES, registry)


class TestWhatBecomesARow:
    def test_a_swaps_two_legs_both_become_rows(self, registry):
        rows = build(
            [
                transfer("0x1", 0, USDC, TRADER, POOL_V3, USDC_1000),
                transfer("0x1", 1, WETH, POOL_V3, TRADER, WETH_HALF),
            ],
            [swap_event("0x1", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert len(rows) == 2
        assert [(r["source"], r["destination"], r["token"]) for r in rows] == [
            (TRADER, POOL_V3, "USDC"),
            (POOL_V3, TRADER, "WETH"),
        ]
        assert all(r["type"] == "swap" for r in rows)
        assert all(r["pool"] == POOL_V3 for r in rows)

    def test_a_liquidity_mint_is_not_a_swap(self, registry):
        """Adding liquidity sends BOTH tokens into the pool and emits no Swap.
        Admitting it would create rows that can never have an opposite leg."""
        rows = build(
            [
                transfer("0x2", 0, USDC, TRADER, POOL_V3, USDC_1000),
                transfer("0x2", 1, WETH, TRADER, POOL_V3, WETH_HALF),
            ],
            [],  # no Swap event in this transaction
            registry,
        )
        assert rows == []

    def test_a_liquidity_burn_is_not_a_swap(self, registry):
        """Removing liquidity sends BOTH tokens out. This is what first showed
        up on live data as a swap with two out-legs and no in-leg."""
        rows = build(
            [
                transfer("0x3", 0, USDC, POOL_V3, TRADER, USDC_1000),
                transfer("0x3", 1, WETH, POOL_V3, TRADER, WETH_HALF),
            ],
            [],
            registry,
        )
        assert rows == []

    def test_a_mint_alongside_a_swap_at_another_pool_is_still_excluded(self, registry):
        """Scoping is per (transaction, pool), not per transaction."""
        rows = build(
            [
                transfer("0x4", 0, USDC, TRADER, POOL_V3, USDC_1000),   # swap leg
                transfer("0x4", 1, WETH, POOL_V3, TRADER, WETH_HALF),   # swap leg
                transfer("0x4", 2, USDC, TRADER, POOL_2, USDC_1000),    # mint
                transfer("0x4", 3, WETH, TRADER, POOL_2, WETH_HALF),    # mint
            ],
            [swap_event("0x4", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert {r["pool"] for r in rows} == {POOL_V3}
        assert len(rows) == 2

    def test_transfers_between_two_wallets_are_out_of_scope(self, registry):
        rows = build(
            [transfer("0x5", 0, USDC, TRADER, "0x2222222222222222222222222222222222222222", 5)],
            [swap_event("0x5", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert rows == []


class TestMultiHopRouting:
    def test_a_pool_to_pool_leg_is_kept_and_attributed_to_the_sender(self, registry):
        """One transfer is at once the first pool's out-leg and the second's
        in-leg. Dropping it would leave both pools missing a side."""
        rows = build(
            [
                transfer("0x6", 0, USDC, TRADER, POOL_V3, USDC_1000),
                transfer("0x6", 1, WETH, POOL_V3, POOL_2, WETH_HALF),
                transfer("0x6", 2, USDC, POOL_2, TRADER, USDC_1000 - 3_000_000),
            ],
            [
                swap_event("0x6", POOL_V3, USDC_1000, -WETH_HALF),
                swap_event("0x6", POOL_2, -(USDC_1000 - 3_000_000), WETH_HALF,
                           log_index=10),
            ],
            registry,
        )
        assert len(rows) == 3
        shared = next(r for r in rows if r["source"] == POOL_V3
                      and r["destination"] == POOL_2)
        assert shared["pool"] == POOL_V3


class TestCounterpartExpectations:
    def test_a_zero_amount_side_means_no_opposite_leg_is_expected(self, registry):
        """1 wei in, nothing out: the chain emits no Transfer for a side that
        rounds to zero, so the missing leg is not a gap in our data."""
        rows = build(
            [transfer("0x7", 0, USDC, TRADER, POOL_V3, 1)],
            [swap_event("0x7", POOL_V3, 1, 0)],
            registry,
        )
        assert len(rows) == 1
        assert rows[0]["counterpart_expected"] is False

    def test_a_rebasing_token_means_no_opposite_leg_is_expected(self, registry):
        """KISHU redistributes balances without emitting Transfer, so the pool
        accumulates it silently and the inbound leg never exists on chain."""
        rows = build(
            [transfer("0x8", 0, WETH, POOL_KISHU, TRADER, 118_124_753_328_066)],
            [swap_event("0x8", POOL_KISHU, 2_438_075_269_184_966_621,
                        -118_124_753_328_066)],
            registry,
        )
        assert len(rows) == 1
        assert rows[0]["counterpart_expected"] is False

    def test_an_ordinary_swap_does_expect_its_counterpart(self, registry):
        rows = build(
            [
                transfer("0x9", 0, USDC, TRADER, POOL_V3, USDC_1000),
                transfer("0x9", 1, WETH, POOL_V3, TRADER, WETH_HALF),
            ],
            [swap_event("0x9", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert all(r["counterpart_expected"] is True for r in rows)


class TestRowContents:
    def test_amounts_are_decimal_adjusted_per_token(self, registry):
        rows = build(
            [
                transfer("0xa", 0, USDC, TRADER, POOL_V3, USDC_1000),   # 6 decimals
                transfer("0xa", 1, WETH, POOL_V3, TRADER, WETH_HALF),   # 18 decimals
            ],
            [swap_event("0xa", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        by_token = {r["token"]: r for r in rows}
        assert by_token["USDC"]["amount"] == Decimal(1000)
        assert by_token["WETH"]["amount"] == Decimal("0.5")
        # The unscaled value is kept too, so nothing is lost to rounding.
        assert by_token["USDC"]["amount_raw"] == USDC_1000

    def test_transaction_id_combines_hash_and_log_index(self, registry):
        rows = build(
            [
                transfer("0xb", 4, USDC, TRADER, POOL_V3, USDC_1000),
                transfer("0xb", 5, WETH, POOL_V3, TRADER, WETH_HALF),
            ],
            [swap_event("0xb", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert [r["transaction_id"] for r in rows] == ["0xb-4", "0xb-5"]

    def test_a_token_without_a_symbol_falls_back_to_its_address(self, registry):
        """A row must stay usable even when symbol() is unreadable, rather than
        carrying a NULL where the ticker should be."""
        unknown = StubRegistry.NAMELESS
        pool = StubRegistry.POOL_NAMELESS
        rows = build(
            [transfer("0xc", 0, unknown, TRADER, pool, 10**18)],
            [swap_event("0xc", pool, 10**18, -1)],
            registry,
        )
        assert rows[0]["token"].startswith("0xdeadbe")
        assert rows[0]["token_address"] == unknown

    def test_usd_amount_is_left_for_the_pricing_pass(self, registry):
        rows = build(
            [transfer("0xd", 0, USDC, TRADER, POOL_V3, USDC_1000)],
            [swap_event("0xd", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert rows[0]["usd_amount"] is None


class TestPoolClassificationRace:
    """Concurrency around pool verification.

    The bug these lock down: when one thread's verification failed it released
    its claim on those addresses, and a thread waiting on the same addresses
    read them as merely "unknown", dropped their swaps, and checkpointed the
    chunk as done. One thread's RPC error silently corrupted another thread's
    output -- invisible in any single run, and it cost real rows on live data
    (a three-hop arbitrage lost the leg entering the cycle).
    """

    def _registry(self, monkeypatch, verify_impl):
        from dex_data import ingest as ingest_mod, pools as pools_mod, tokens as tokens_mod

        monkeypatch.setattr(pools_mod, "load_all", lambda: {})
        monkeypatch.setattr(pools_mod, "store", lambda *a, **k: 0)
        monkeypatch.setattr(pools_mod, "verify_candidates", verify_impl)
        monkeypatch.setattr(tokens_mod, "load_metadata", lambda: {})
        return ingest_mod.PoolRegistry(client=None)

    def test_a_failed_verification_is_taken_over_not_skipped(self, monkeypatch):
        import threading

        pool = POOL_V3
        attempts: list[str] = []
        first = threading.Event()

        def verify(candidates, client=None, **kw):
            attempts.append("call")
            if len(attempts) == 1:
                first.set()
                raise RuntimeError("transient RPC failure")
            return {a: {"address": a, "dex": "uniswap", "version": "v3",
                        "factory": "0x0", "token0": USDC, "token1": WETH, "fee": 500}
                    for a in candidates}

        registry = self._registry(monkeypatch, verify)

        failures: list[Exception] = []

        def failing_thread():
            try:
                registry.classify({pool}, {})
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)

        t = threading.Thread(target=failing_thread)
        t.start()
        first.wait(timeout=5)
        t.join(timeout=5)

        assert failures, "the thread whose verification failed must raise"

        # The next caller must resolve the pool itself, not treat it as unknown.
        assert registry.classify({pool}, {}) == {pool}
        assert len(attempts) == 2

    def test_a_definite_negative_is_cached_and_not_re_verified(self, monkeypatch):
        """Non-Uniswap forks emit a byte-identical v2 Swap event. Without a
        negative cache every chunk would re-ask factory() for the same
        thousands of rejects."""
        calls: list[int] = []

        def verify(candidates, client=None, **kw):
            calls.append(len(candidates))
            return {}  # none of them are Uniswap

        registry = self._registry(monkeypatch, verify)
        fork = "0x1234567890123456789012345678901234567890"

        assert registry.classify({fork}, {}) == set()
        assert registry.classify({fork}, {}) == set()
        assert len(calls) == 1, "a settled negative must not be re-verified"

    def test_concurrent_callers_verify_a_shared_pool_only_once(self, monkeypatch):
        import threading

        pool = POOL_V3
        calls: list[int] = []
        barrier = threading.Barrier(4, timeout=5)

        def verify(candidates, client=None, **kw):
            calls.append(len(candidates))
            return {a: {"address": a, "dex": "uniswap", "version": "v3",
                        "factory": "0x0", "token0": USDC, "token1": WETH, "fee": 500}
                    for a in candidates}

        registry = self._registry(monkeypatch, verify)
        results: list[set] = []

        def worker():
            barrier.wait()
            results.append(registry.classify({pool}, {}))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert results == [{pool}] * 4, "every caller must see the pool"
        assert len(calls) == 1, "the pool must be verified once, not four times"


class TestPoolOwnTokensOnly:
    """A swap leg's token must be one of its pool's two tokens.

    The bug this locks down: a Uniswap v2 pair is itself an ERC20 and emits
    Transfer events for its own LP token when liquidity is added or removed.
    When that happened in a transaction where the pool also swapped, the LP
    movement passed the "did this pool swap here" test and became a row whose
    `token` was the pool address itself. That address is never in `tokens`, so
    the pricing pass died on a foreign key violation -- 68 such rows left
    402,518 rows unpriced.
    """

    def test_an_lp_token_transfer_is_not_a_swap_leg(self, registry):
        """The v2 burn flow: LP tokens are sent to the pair, which then emits
        the two real token transfers. Only the latter are legs."""
        rows = build(
            [
                # LP token: emitted BY the pair, so token address == pool.
                transfer("0x1", 0, POOL_V3, TRADER, POOL_V3, 10**18),
                # The genuine swap legs in the same transaction.
                transfer("0x1", 1, USDC, TRADER, POOL_V3, USDC_1000),
                transfer("0x1", 2, WETH, POOL_V3, TRADER, WETH_HALF),
            ],
            [swap_event("0x1", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert [r["token"] for r in rows] == ["USDC", "WETH"]
        assert all(r["token_address"] != r["pool"] for r in rows)

    def test_an_lp_burn_to_the_zero_address_is_excluded(self, registry):
        rows = build(
            [transfer("0x2", 0, POOL_V3, POOL_V3,
                      "0x0000000000000000000000000000000000000000", 10**18)],
            [swap_event("0x2", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert rows == []

    def test_an_unrelated_token_dumped_on_a_pool_is_excluded(self, registry):
        """Anyone can send any ERC20 to a pool address. That is not a swap."""
        stray = "0xdeadbeef00000000000000000000000000009999"
        rows = build(
            [transfer("0x3", 0, stray, TRADER, POOL_V3, 10**18)],
            [swap_event("0x3", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert rows == []

    def test_every_emitted_row_names_a_token_of_its_pool(self, registry):
        """The invariant stated directly, over a mixed batch."""
        stray = "0xdeadbeef00000000000000000000000000009999"
        rows = build(
            [
                transfer("0x4", 0, USDC, TRADER, POOL_V3, USDC_1000),
                transfer("0x4", 1, WETH, POOL_V3, TRADER, WETH_HALF),
                transfer("0x4", 2, POOL_V3, TRADER, POOL_V3, 1),      # LP token
                transfer("0x4", 3, stray, TRADER, POOL_V3, 1),        # unrelated
            ],
            [swap_event("0x4", POOL_V3, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert len(rows) == 2
        for r in rows:
            pool = registry.pool(r["pool"])
            assert r["token_address"] in (pool["token0"], pool["token1"])

    def test_a_multi_hop_leg_is_kept_when_the_token_belongs_to_its_pool(self, registry):
        """The rule must not cost the pool-to-pool legs of a real route."""
        rows = build(
            [
                transfer("0x5", 0, USDC, TRADER, POOL_V3, USDC_1000),
                transfer("0x5", 1, WETH, POOL_V3, POOL_2, WETH_HALF),
                transfer("0x5", 2, USDC, POOL_2, TRADER, USDC_1000 - 3_000_000),
            ],
            [
                swap_event("0x5", POOL_V3, USDC_1000, -WETH_HALF),
                swap_event("0x5", POOL_2, -(USDC_1000 - 3_000_000), WETH_HALF,
                           log_index=10),
            ],
            registry,
        )
        assert len(rows) == 3

    def test_a_leg_at_an_unverified_pool_is_dropped(self, registry):
        """Without pool metadata there is no way to check the token, and a row
        that cannot be checked must not be written."""
        unknown_pool = "0xaaaa000000000000000000000000000000000001"
        rows = build(
            [transfer("0x6", 0, USDC, TRADER, unknown_pool, USDC_1000)],
            [swap_event("0x6", unknown_pool, USDC_1000, -WETH_HALF)],
            registry,
        )
        assert rows == []

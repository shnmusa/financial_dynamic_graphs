"""Which swaps cleanup.run() moves out of dex_events, and which it must keep."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dex_data import cleanup

from .factories import (
    KISHU, POOL_KISHU, POOL_V3, TRADER, USDC, USDC_1000, WETH, WETH_HALF, leg,
    seed_reference, seed_swap,
)

pytestmark = pytest.mark.db

WHEN = datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc)
POOL_USDC_KISHU = "0x5555555555555555555555555555555555555555"


@pytest.fixture
def seeded(clean_db):
    seed_reference(clean_db, block=100, when=WHEN)
    clean_db.execute(
        "INSERT INTO pools (address, dex, version, token0, token1) "
        "VALUES (%s, 'uniswap', 'v2', %s, %s)", (POOL_USDC_KISHU, USDC, KISHU))
    return clean_db


def remaining(db, tx):
    return db.fetch_one("SELECT count(*) FROM dex_events WHERE tx_hash = %s", (tx,))[0]


def removed(db, tx):
    return db.fetch_all(
        "SELECT DISTINCT removed_reason FROM dex_events_removed WHERE tx_hash = %s", (tx,))


def kishu_for_usdc(db, tx):
    leg(db, tx=tx, log_index=0, source=TRADER, destination=POOL_USDC_KISHU,
        token=KISHU, amount_raw=10**15, pool=POOL_USDC_KISHU)
    leg(db, tx=tx, log_index=1, source=POOL_USDC_KISHU, destination=TRADER,
        token=USDC, amount_raw=USDC_1000, pool=POOL_USDC_KISHU)


def test_stablecoin_against_non_target_is_moved_whole(seeded):
    kishu_for_usdc(seeded, "0xc1")
    counts = cleanup.run()
    assert remaining(seeded, "0xc1") == 0
    assert removed(seeded, "0xc1") == [("stable_vs_nontarget",)]
    assert seeded.fetch_one(
        "SELECT count(*) FROM dex_events_removed WHERE tx_hash = '0xc1'")[0] == 2
    assert counts["stable_vs_nontarget"] == 1


def test_stablecoin_against_target_is_kept(seeded):
    seed_swap(seeded, tx="0xc2")  # USDC -> WETH, WETH is a target
    cleanup.run()
    assert remaining(seeded, "0xc2") == 2
    assert removed(seeded, "0xc2") == []


def test_the_same_swap_inside_a_route_is_kept(seeded):
    """KISHU -> USDC is a hop of KISHU -> USDC -> WETH; removing it would
    leave the route without its first half."""
    kishu_for_usdc(seeded, "0xc3")
    leg(seeded, tx="0xc3", log_index=2, source=TRADER, destination=POOL_V3,
        token=USDC, amount_raw=USDC_1000, pool=POOL_V3)
    leg(seeded, tx="0xc3", log_index=3, source=POOL_V3, destination=TRADER,
        token=WETH, amount_raw=WETH_HALF, pool=POOL_V3)
    cleanup.run()
    assert remaining(seeded, "0xc3") == 4


def test_untargeting_a_token_makes_its_swaps_removable(seeded):
    seed_swap(seeded, tx="0xc4")
    cleanup.run()
    assert remaining(seeded, "0xc4") == 2
    seeded.execute("UPDATE tokens SET target = FALSE WHERE address = %s", (WETH,))
    cleanup.run()
    assert remaining(seeded, "0xc4") == 0
    assert removed(seeded, "0xc4") == [("stable_vs_nontarget",)]


def test_rerun_is_a_no_op(seeded):
    kishu_for_usdc(seeded, "0xc5")
    cleanup.run()
    assert cleanup.run()["rows_removed"] == 0

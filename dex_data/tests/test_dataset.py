"""The modelling universe: `pin` chooses it once and refuses to move it."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from dex_data import dataset

pytestmark = pytest.mark.db

WHEN = datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc)
POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

# Four tokens spanning the classification: a major, a major stablecoin, a
# second major stablecoin, a listed-but-small token, and one nobody lists.
BIG = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"      # WETH-like
BIG2 = "0x4444444444444444444444444444444444444444"     # a second major, non-stable
STABLE = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"   # USDC-like
STABLE2 = "0xdac17f958d2ee523a2206206994597c13d831ec7"  # USDT-like
SMALL = "0x2222222222222222222222222222222222222222"    # listed, under threshold
UNLISTED = "0x3333333333333333333333333333333333333333"  # market cap NULL

TOKENS = {
    BIG:      ("BIG", 5_000_000_000, False),
    BIG2:     ("BIG2", 4_000_000_000, False),
    STABLE:   ("STABLE", 70_000_000_000, True),
    STABLE2:  ("STABLE2", 60_000_000_000, True),
    SMALL:    ("SMALL", 1_000_000, False),
    UNLISTED: ("UNLISTED", None, False),
}


@pytest.fixture
def seeded(clean_db):
    """Reference rows only; each test adds the legs it cares about."""
    db = clean_db
    with db.connection() as conn:
        for address, (symbol, cap, stable) in TOKENS.items():
            conn.execute(
                "INSERT INTO tokens (address, symbol, decimals, is_stable, market_cap_usd) "
                "VALUES (%s, %s, 18, %s, %s)",
                (address, symbol, stable, cap),
            )
        conn.execute(
            "INSERT INTO pools (address, dex, version, token0, token1) "
            "VALUES (%s, 'uniswap', 'v3', %s, %s)", (POOL, BIG, STABLE),
        )
        conn.execute("INSERT INTO blocks (number, timestamp) VALUES (100, %s)", (WHEN,))
        conn.commit()
    return db


class TestPin:
    def test_threshold_boundary_is_strictly_greater_than(self, seeded):
        """A token exactly at the threshold is below the universe, not in it."""
        assert len(dataset.pin(5e9, force=True)) == 2    # BIG at its own cap: out
        assert len(dataset.pin(4.9e9, force=True)) == 3  # BIG in, BIG2 still out

    def test_universe_is_everything_above_threshold(self, seeded):
        chosen = dataset.pin(1e8)
        assert sorted(r[0] for r in chosen) == ["BIG", "BIG2", "STABLE", "STABLE2"]

    def test_unlisted_token_is_never_in_the_universe(self, seeded):
        """NULL market cap is a verdict, not a gap: it must not slip through."""
        dataset.pin(0)
        assert seeded.fetch_one(
            "SELECT target FROM tokens WHERE address = %s", (UNLISTED,))[0] is False

    def test_pinning_twice_is_refused(self, seeded):
        dataset.pin(1e8)
        with pytest.raises(dataset.UniverseError, match="already pinned"):
            dataset.pin(1e9)

    def test_force_repins(self, seeded):
        dataset.pin(1e8)
        dataset.pin(1e10, force=True)
        assert seeded.fetch_one("SELECT count(*) FROM tokens WHERE target")[0] == 2

    def test_pin_records_the_cap_it_chose_on(self, seeded):
        dataset.pin(1e8)
        row = seeded.fetch_one(
            "SELECT target_market_cap_usd, target_pinned_at FROM tokens "
            "WHERE symbol = 'BIG'")
        assert row[0] == Decimal(5_000_000_000)
        assert row[1] is not None

    def test_top_n_pins_by_rank(self, seeded):
        dataset.pin(top=2, force=True)
        pinned = seeded.fetch_all(
            "SELECT symbol FROM tokens WHERE target ORDER BY market_cap_usd DESC")
        assert [r[0] for r in pinned] == ["STABLE", "STABLE2"]

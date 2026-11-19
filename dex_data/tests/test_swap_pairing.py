"""The swap-pairing invariant.

    "Tabloda bir swap varsa karşılığının da olması gerekir. Yani adresten
     pool'a para gitmiş, aynı şekilde pool'dan da adrese para gidecek."

Every dex_events row is one token movement into or out of a pool, so a swap is
two rows and those two rows *are* the swap. The invariant: for every
(transaction, pool), money must move both ways, in two different tokens.

These tests exercise the real check (dex_data.verify) and the real pricing SQL
against a live Postgres, because both are SQL -- a mocked version would test a
reimplementation, not the shipped code.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from dex_data import pricing, verify

from .factories import (
    KISHU, POOL_2, POOL_KISHU, POOL_V3, TRADER, USDC, USDC_1000, WETH, WETH_HALF,
    leg, seed_reference, seed_swap,
)

pytestmark = pytest.mark.db

BLOCK = 100
WHEN = datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc)


WETH_USD = Decimal(2000)


def fake_coingecko(prices):
    """A PriceFetcher answering from a {coingecko_id: usd} map, one point per
    hour at a few seconds past the hour, like the real endpoint."""
    def fetch(cg_id, start, end):
        if cg_id not in prices:
            return []
        hour = start.replace(minute=0, second=0, microsecond=0)
        points = []
        while hour <= end:
            points.append([int(hour.timestamp() * 1000) + 7_000, float(prices[cg_id])])
            hour += timedelta(hours=1)
        return points
    return fetch


def run_pricing(db, prices=None):
    db.execute("UPDATE tokens SET coingecko_id = 'weth' WHERE address = %s", (WETH,))
    pricing.build(fetch=fake_coingecko({"weth": WETH_USD} if prices is None else prices))
    pricing.apply_usd_amounts()


class TestTableShape:
    def test_a_swap_is_exactly_two_rows_wallet_to_pool_and_back(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xaa1")

        rows = clean_db.fetch_all(
            "SELECT type, source, destination, token, amount FROM dex_events "
            "ORDER BY log_index"
        )
        assert len(rows) == 2, "a swap is its two legs, with no extra marker row"

        in_leg, out_leg = rows
        assert in_leg[:4] == ("swap", TRADER, POOL_V3, "USDC")
        assert out_leg[:4] == ("swap", POOL_V3, TRADER, "WETH")

    def test_there_is_no_transaction_type(self, clean_db):
        """Wallet-to-wallet transfers are out of scope, so 'swap' is the only
        type, and the CHECK constraint enforces it."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xaa2")

        types = clean_db.fetch_all("SELECT DISTINCT type FROM dex_events")
        assert types == [("swap",)]

        with pytest.raises(Exception) as exc:
            with clean_db.connection() as conn:
                conn.execute(
                    "INSERT INTO dex_events (transaction_id, type, source, destination, "
                    "token, timestamp, block_number, tx_hash, log_index, token_address) "
                    "VALUES ('x-0', 'transaction', %s, %s, 'USDC', %s, 1, 'x', 0, %s)",
                    (TRADER, POOL_V3, WHEN, USDC),
                )
                conn.commit()
        assert "dex_events_type_check" in str(exc.value)

    def test_no_row_has_the_same_source_and_destination(self, clean_db):
        """The old design emitted a pool -> itself marker row. It is gone, and
        the schema now refuses to store one."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xaa3")

        assert clean_db.fetch_one(
            "SELECT count(*) FROM dex_events WHERE source = destination"
        )[0] == 0

        with pytest.raises(Exception) as exc:
            leg(clean_db, tx="0xbad", log_index=0, source=POOL_V3,
                destination=POOL_V3, token=USDC, amount_raw=1)
        assert "dex_events_distinct_endpoints" in str(exc.value)

    def test_token_column_holds_a_readable_symbol(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xaa4")

        rows = clean_db.fetch_all(
            "SELECT token, token_address FROM dex_events ORDER BY log_index"
        )
        assert [r[0] for r in rows] == ["USDC", "WETH"]
        # The address is kept alongside: symbols are neither unique nor
        # guaranteed to exist, so joins still need the identity.
        assert [r[1] for r in rows] == [USDC, WETH]

    def test_transaction_id_is_unique_per_log_not_per_transaction(self, clean_db):
        """A tx hash cannot be the primary key: one transaction routinely moves
        tokens through several pools."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xaa5")

        ids = [r[0] for r in clean_db.fetch_all(
            "SELECT transaction_id FROM dex_events WHERE tx_hash = '0xaa5' "
            "ORDER BY log_index"
        )]
        assert ids == ["0xaa5-0", "0xaa5-1"]

    def test_the_raw_staging_tables_are_gone(self, clean_db):
        present = {
            r[0] for r in clean_db.fetch_all(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema()"
            )
        }
        assert "raw_swaps" not in present
        assert "raw_transfers" not in present
        assert {"dex_events", "tokens", "pools"} <= present


class TestPairingHolds:
    def test_verify_reports_full_pairing(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        for i in range(5):
            seed_swap(clean_db, tx=f"0xbb{i}")

        result = verify.check_pairing()
        assert result.total_swaps == 5
        assert result.paired == 5
        assert result.unpaired == 0
        assert result.ratio == pytest.approx(1.0)

    def test_money_moves_both_ways_for_every_swap(self, clean_db):
        """The invariant written as the query a reviewer would reach for."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        for i in range(3):
            seed_swap(clean_db, tx=f"0xcc{i}")

        unpaired = clean_db.fetch_all(
            """
            SELECT e.tx_hash, p.address
            FROM dex_events e
            JOIN pools p ON p.address IN (e.source, e.destination)
            GROUP BY e.tx_hash, p.address
            HAVING NOT (
                bool_or(e.destination = p.address)
                AND bool_or(e.source = p.address)
                AND count(DISTINCT e.token_address) >= 2
            )
            """
        )
        assert unpaired == []

    def test_several_pools_in_one_transaction_each_pair_independently(self, clean_db):
        """Multi-hop routes touch several pools under one tx hash. Pairing is
        per (transaction, pool), so each hop resolves on its own."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xdd0", pool=POOL_V3, base_log_index=0)
        seed_swap(clean_db, tx="0xdd0", pool=POOL_2, base_log_index=2)

        result = verify.check_pairing()
        assert result.total_swaps == 2
        assert result.paired == 2


class TestPoolToPoolRouting:
    """A multi-hop route settles pool to pool: one transfer is at once the
    first pool's outbound leg and the second pool's inbound leg. Dropping such
    rows would leave both pools of every multi-hop swap missing a side, so they
    are kept -- and the pairing check joins on `pools` rather than on the `pool`
    column precisely so it sees both roles.
    """

    def test_one_shared_transfer_satisfies_both_pools(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)

        # USDC -> (pool A) -> WETH -> (pool B) -> USDC, WETH never touching the
        # trader's wallet.
        leg(clean_db, tx="0xee0", log_index=0, source=TRADER, destination=POOL_V3,
            token=USDC, amount_raw=USDC_1000, pool=POOL_V3)
        leg(clean_db, tx="0xee0", log_index=1, source=POOL_V3, destination=POOL_2,
            token=WETH, amount_raw=WETH_HALF, pool=POOL_V3)
        leg(clean_db, tx="0xee0", log_index=2, source=POOL_2, destination=TRADER,
            token=USDC, amount_raw=USDC_1000 - 3_000_000, pool=POOL_2)

        result = verify.check_pairing()
        assert result.total_swaps == 2, "one per pool"
        assert result.paired == 2, "the shared transfer serves both pools"

        assert verify.pool_to_pool_rows() == 1

    def test_the_shared_row_is_attributed_to_the_sending_pool(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        leg(clean_db, tx="0xee1", log_index=0, source=POOL_V3, destination=POOL_2,
            token=WETH, amount_raw=WETH_HALF, pool=POOL_V3)

        row = clean_db.fetch_one(
            "SELECT source, destination, pool FROM dex_events WHERE tx_hash = '0xee1'"
        )
        assert row == (POOL_V3, POOL_2, POOL_V3)


class TestPairingCatchesBreakage:
    """A check that cannot fail is not a check. These prove it can."""

    def test_missing_out_leg_is_detected(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xf00")                        # complete
        seed_swap(clean_db, tx="0xf01", include_out_leg=False)  # broken

        result = verify.check_pairing()
        assert result.total_swaps == 2
        assert result.paired == 1
        assert result.missing_out_leg == 1
        assert result.missing_in_leg == 0
        assert result.ratio == pytest.approx(0.5)

    def test_missing_in_leg_is_detected(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xf10", include_in_leg=False)

        result = verify.check_pairing()
        assert result.missing_in_leg == 1
        assert result.paired == 0

    def test_unpaired_sample_points_at_the_offending_swap(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xf20", include_out_leg=False)

        sample = verify.unpaired_sample(limit=10)
        assert len(sample) == 1
        offender = sample[0]
        assert offender["tx_hash"] == "0xf20"
        assert offender["pool"] == POOL_V3
        assert offender["has_in_leg"] is True
        assert offender["has_out_leg"] is False
        assert "USDC" in offender["tokens"]

    def test_legs_in_another_transaction_do_not_satisfy_each_other(self, clean_db):
        """Pairing is scoped to the transaction: two half-swaps in two different
        transactions must not count as one whole."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xf30", include_out_leg=False)
        seed_swap(clean_db, tx="0xf31", include_in_leg=False)

        result = verify.check_pairing()
        assert result.paired == 0
        assert result.missing_out_leg == 1
        assert result.missing_in_leg == 1

    def test_legs_at_another_pool_do_not_satisfy_each_other(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0xf40", pool=POOL_V3, include_out_leg=False,
                  base_log_index=0)
        seed_swap(clean_db, tx="0xf40", pool=POOL_2, include_in_leg=False,
                  base_log_index=2)

        result = verify.check_pairing()
        assert result.total_swaps == 2
        assert result.paired == 0

    def test_both_legs_in_the_same_token_is_not_a_swap(self, clean_db):
        """Money in and out of a pool in one token is a mint, burn or skim --
        not an exchange."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        leg(clean_db, tx="0xf50", log_index=0, source=TRADER, destination=POOL_V3,
            token=USDC, amount_raw=USDC_1000, pool=POOL_V3)
        leg(clean_db, tx="0xf50", log_index=1, source=POOL_V3, destination=TRADER,
            token=USDC, amount_raw=USDC_1000, pool=POOL_V3)

        result = verify.check_pairing()
        assert result.single_token_pair == 1


class TestOneSidedSwaps:
    """Two causes make a one-legged swap genuine, both found by running the
    check against live data rather than assumed:

      * the swap's amount on one side rounds to zero, so nothing moves there
        and no Transfer is emitted (dust probes, typically 1 wei in);
      * a rebasing / reflection token changes balances without emitting
        Transfer, so a v2 pair accumulates it silently and anyone can call
        swap() to skim the surplus -- confirmed on KISHU/WETH, where the Swap
        event reported 2.4 billion KISHU in and the receipt held no KISHU
        transfer at all.

    These are classified, not tolerated: the threshold stays at 1.0.
    """

    def test_a_zero_amount_swap_is_classified_not_failed(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0x200")  # normal, both legs

        # 1 wei of USDC in, nothing out: exactly the dust probe seen on chain.
        leg(clean_db, tx="0x201", log_index=0, source=TRADER, destination=POOL_V3,
            token=USDC, amount_raw=1, pool=POOL_V3, counterpart_expected=False)

        result = verify.check_pairing()
        assert result.total_swaps == 2
        assert result.one_sided == 1
        assert result.checked_swaps == 1
        assert result.paired == 1
        assert result.unpaired == 0
        assert result.ratio == pytest.approx(1.0)

    def test_a_rebasing_token_swap_is_classified(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        # WETH leaves the pool; the KISHU that paid for it never emitted a
        # Transfer, so there is no inbound leg to find.
        leg(clean_db, tx="0x210", log_index=0, source=POOL_KISHU, destination=TRADER,
            token=WETH, amount_raw=118_124_753_328_066, pool=POOL_KISHU,
            counterpart_expected=False)

        result = verify.check_pairing()
        assert result.one_sided == 1
        assert result.checked_swaps == 0
        assert result.unpaired == 0

    def test_a_real_gap_still_fails_alongside_a_one_sided_swap(self, clean_db):
        """The classification must not become a blanket excuse."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0x220", include_out_leg=False)  # genuine gap
        leg(clean_db, tx="0x221", log_index=0, source=TRADER, destination=POOL_V3,
            token=USDC, amount_raw=1, pool=POOL_V3, counterpart_expected=False)

        result = verify.check_pairing()
        assert result.one_sided == 1
        assert result.checked_swaps == 1
        assert result.unpaired == 1, "a real missing leg must survive the carve-out"


class TestDerivedValues:
    def test_target_is_priced_from_coingecko_and_stables_at_one(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0x300")
        run_pricing(clean_db)

        price = clean_db.fetch_one(
            "SELECT usd_price, hop FROM token_prices_hourly WHERE token = %s", (WETH,))
        assert price == (WETH_USD, pricing.HOP_COINGECKO)
        stable = clean_db.fetch_one(
            "SELECT usd_price, hop FROM token_prices_hourly WHERE token = %s", (USDC,))
        assert stable == (Decimal(1), pricing.HOP_STABLE)

    def test_non_target_gets_no_price_of_its_own(self, clean_db):
        """KISHU is not a target: no CoinGecko call, no price row."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        asked = []
        def fetch(cg_id, start, end):
            asked.append(cg_id)
            return []
        clean_db.execute("UPDATE tokens SET coingecko_id = 'kishu-inu' WHERE address = %s",
                         (KISHU,))
        pricing.build(fetch=fetch)
        assert "kishu-inu" not in asked
        assert clean_db.fetch_one(
            "SELECT count(*) FROM token_prices_hourly WHERE token = %s", (KISHU,))[0] == 0

    def test_both_legs_are_valued_at_amount_times_price(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0x310")  # 1000 USDC -> 0.5 WETH
        run_pricing(clean_db)

        rows = dict(clean_db.fetch_all(
            "SELECT token, usd_amount FROM dex_events WHERE tx_hash = '0x310'"))
        assert rows["USDC"] == Decimal(1000)
        assert rows["WETH"] == Decimal(1000)  # 0.5 x $2000

    def test_unpriced_leg_is_mirrored_from_the_other_leg(self, clean_db):
        """KISHU has no price; selling it for WETH values it at the WETH leg."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        leg(clean_db, tx="0x330", log_index=0, source=TRADER, destination=POOL_KISHU,
            token=KISHU, amount_raw=10**15, pool=POOL_KISHU)
        leg(clean_db, tx="0x330", log_index=1, source=POOL_KISHU, destination=TRADER,
            token=WETH, amount_raw=WETH_HALF // 5, pool=POOL_KISHU)   # 0.1 WETH
        run_pricing(clean_db)

        rows = dict(clean_db.fetch_all(
            "SELECT token, usd_amount FROM dex_events WHERE tx_hash = '0x330'"))
        assert rows["WETH"] == Decimal(200)
        assert rows["KISHU"] == Decimal(200)

    def test_old_usd_amounts_are_cleared_not_kept(self, clean_db):
        """A value from an earlier pricing run must not survive a re-price that
        has no price for the row -- the 1e25-dollar long-tail values came from
        exactly that kind of leftover."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        leg(clean_db, tx="0x340", log_index=0, source=TRADER, destination=POOL_KISHU,
            token=KISHU, amount_raw=10**15, pool=POOL_KISHU, counterpart_expected=False)
        clean_db.execute("UPDATE dex_events SET usd_amount = 1e25 WHERE tx_hash = '0x340'")
        run_pricing(clean_db)

        assert clean_db.fetch_one(
            "SELECT usd_amount FROM dex_events WHERE tx_hash = '0x340'")[0] is None

    def test_no_coingecko_price_leaves_the_target_leg_to_the_mirror(self, clean_db):
        """A target CoinGecko has no data for (not listed yet) is NULL, and its
        leg then takes the stablecoin leg's value."""
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0x350")
        run_pricing(clean_db, prices={})

        rows = dict(clean_db.fetch_all(
            "SELECT token, usd_amount FROM dex_events WHERE tx_hash = '0x350'"))
        assert rows["USDC"] == Decimal(1000)
        assert rows["WETH"] == Decimal(1000)

    def test_value_travels_along_a_routed_multi_hop(self, clean_db):
        """KISHU -> POOL_KISHU -> WETH -> router -> POOL_V3 -> USDC, with no
        CoinGecko price for WETH. Only USDC is priced up front. The WETH leg
        into POOL_V3 mirrors USDC; the WETH leg out of POOL_KISHU is a
        different row and needs the same-token step; KISHU, the token the
        wallet actually sold, needs one more mirror after that."""
        router = "0x2222222222222222222222222222222222222222"
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        leg(clean_db, tx="0x360", log_index=0, source=TRADER, destination=POOL_KISHU,
            token=KISHU, amount_raw=10**15, pool=POOL_KISHU)
        leg(clean_db, tx="0x360", log_index=1, source=POOL_KISHU, destination=router,
            token=WETH, amount_raw=WETH_HALF, pool=POOL_KISHU)
        leg(clean_db, tx="0x360", log_index=2, source=router, destination=POOL_V3,
            token=WETH, amount_raw=WETH_HALF, pool=POOL_V3)
        leg(clean_db, tx="0x360", log_index=3, source=POOL_V3, destination=TRADER,
            token=USDC, amount_raw=USDC_1000, pool=POOL_V3)
        run_pricing(clean_db, prices={})

        rows = clean_db.fetch_all(
            "SELECT log_index, usd_amount FROM dex_events WHERE tx_hash = '0x360' "
            "ORDER BY log_index")
        assert [r[1] for r in rows] == [Decimal(1000)] * 4

    def test_same_token_step_uses_the_unit_price_not_the_total(self, clean_db):
        """Two legs of WETH in one tx with different sizes: the unpriced one is
        valued at its own amount times the priced leg's unit price."""
        router = "0x2222222222222222222222222222222222222222"
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        # router receives 0.5 WETH from a KISHU pool but forwards only 0.25
        leg(clean_db, tx="0x370", log_index=0, source=TRADER, destination=POOL_KISHU,
            token=KISHU, amount_raw=10**15, pool=POOL_KISHU)
        leg(clean_db, tx="0x370", log_index=1, source=POOL_KISHU, destination=router,
            token=WETH, amount_raw=WETH_HALF, pool=POOL_KISHU)
        leg(clean_db, tx="0x370", log_index=2, source=router, destination=POOL_V3,
            token=WETH, amount_raw=WETH_HALF // 2, pool=POOL_V3)
        leg(clean_db, tx="0x370", log_index=3, source=POOL_V3, destination=TRADER,
            token=USDC, amount_raw=USDC_1000 // 2, pool=POOL_V3)
        run_pricing(clean_db, prices={})

        rows = dict(clean_db.fetch_all(
            "SELECT log_index, usd_amount FROM dex_events WHERE tx_hash = '0x370'"))
        assert rows[2] == Decimal(500)    # mirrored from 500 USDC
        assert rows[1] == Decimal(1000)   # 0.5 WETH at the implied $2000
        assert rows[0] == Decimal(1000)   # KISHU mirrors its pool's WETH leg

    def test_fill_gaps_leaves_priced_rows_alone(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0x380")
        run_pricing(clean_db)
        before = clean_db.fetch_all("SELECT transaction_id, usd_amount FROM dex_events ORDER BY 1")
        counts = pricing.fill_gaps()
        assert counts["mirrored"] == counts["same_token"] == 0
        assert clean_db.fetch_all(
            "SELECT transaction_id, usd_amount FROM dex_events ORDER BY 1") == before

    def test_pricing_is_idempotent(self, clean_db):
        seed_reference(clean_db, block=BLOCK, when=WHEN)
        seed_swap(clean_db, tx="0x320")
        run_pricing(clean_db)
        first = clean_db.fetch_all(
            "SELECT transaction_id, usd_amount FROM dex_events ORDER BY 1")
        run_pricing(clean_db)
        assert clean_db.fetch_all(
            "SELECT transaction_id, usd_amount FROM dex_events ORDER BY 1") == first

    def test_unknown_table_is_refused(self, clean_db):
        with pytest.raises(ValueError, match="table must be one of"):
            pricing.apply_usd_amounts(table="tokens; DROP TABLE dex_events")


@pytest.mark.live
class TestIngestedData:
    """Runs against whatever has actually been ingested. Skipped until it has."""

    def test_swap_pairing_is_exact_on_real_data(self, live_db):
        """No tolerance. The only legitimate exceptions are excluded by the
        check itself, so anything left over is a real gap."""
        result = verify.check_pairing()
        assert result.total_swaps > 0
        assert result.checked_swaps > 0
        assert result.unpaired == 0, (
            f"{result.unpaired}/{result.checked_swaps} swaps move money one way "
            f"only ({result.missing_in_leg} missing in-leg, "
            f"{result.missing_out_leg} missing out-leg); "
            f"{result.one_sided} one-sided swaps were classified"
        )

    def test_no_row_has_identical_endpoints(self, live_db):
        assert live_db.fetch_one(
            "SELECT count(*) FROM dex_events WHERE source = destination"
        )[0] == 0

    def test_every_row_touches_a_pool(self, live_db):
        """Scope is pool-touching movements only."""
        assert live_db.fetch_one(
            "SELECT count(*) FROM dex_events e WHERE NOT EXISTS ("
            "  SELECT 1 FROM pools p WHERE p.address IN (e.source, e.destination))"
        )[0] == 0

    def test_token_symbols_are_populated(self, live_db):
        total, blank = live_db.fetch_one(
            "SELECT count(*), count(*) FILTER (WHERE token IS NULL OR token = '') "
            "FROM dex_events"
        )
        assert total > 0 and blank == 0

    def test_most_rows_carry_a_usd_amount(self, live_db):
        total, unpriced = live_db.fetch_one(
            "SELECT count(*), count(*) FILTER (WHERE usd_amount IS NULL) FROM dex_events"
        )
        assert total > 0
        # Long-tail tokens that never trade against a stable stay unpriced; a
        # large share of them means the pricing graph is broken, not sparse.
        assert unpriced / total < 0.35, f"{unpriced}/{total} rows have no USD value"

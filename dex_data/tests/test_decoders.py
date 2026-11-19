"""Log decoding tests. Pure functions, no network and no database."""
from __future__ import annotations

import pytest

from dex_data import abi

from .factories import POOL_V2, POOL_V3, TRADER, USDC, WETH, addr_topic, transfer_log
from .factories import v2_swap_log, v3_swap_log, word


class TestPrimitives:
    def test_to_int_decodes_two_complement(self):
        assert abi.to_int(word(0)) == 0
        assert abi.to_int(word(12345)) == 12345
        assert abi.to_int(word(-1)) == -1
        assert abi.to_int(word(-(10**18))) == -(10**18)

    def test_to_uint_ignores_sign(self):
        assert abi.to_uint(word(-1)) == (1 << 256) - 1

    def test_to_int24_handles_negative_ticks(self):
        assert abi.to_int24(word(-200_000)) == -200_000
        assert abi.to_int24(word(887_272)) == 887_272

    def test_topic_to_address_lowercases_and_trims_padding(self):
        assert abi.topic_to_address(addr_topic("0xAbCdEf" + "11" * 17)) == (
            "0xabcdef" + "11" * 17
        )

    def test_decode_string_handles_dynamic_and_bytes32(self):
        dynamic = "0x" + word(32) + word(4) + b"USDC".hex().ljust(64, "0")
        assert abi.decode_string(dynamic) == "USDC"

        # MKR-style: symbol() declared as bytes32, not string.
        bytes32 = "0x" + b"MKR".hex().ljust(64, "0")
        assert abi.decode_string(bytes32) == "MKR"

    def test_decode_string_survives_garbage(self):
        assert abi.decode_string("0x") is None
        assert abi.decode_string("") is None

    def test_decode_address_rejects_zero(self):
        assert abi.decode_address("0x" + "0" * 64) is None
        assert abi.decode_address("0x" + word(int(WETH, 16))) == WETH


class TestTransferDecoding:
    def test_decodes_a_standard_transfer(self):
        decoded = abi.decode_transfer(
            transfer_log(token=USDC, sender=TRADER, recipient=POOL_V3,
                         amount=1_000_000_000, block=21_000_000, log_index=7, tx="0xdeadbeef")
        )
        assert decoded == {
            "tx_hash": "0xdeadbeef",
            "log_index": 7,
            "block_number": 21_000_000,
            "token": USDC,
            "from_addr": TRADER,
            "to_addr": POOL_V3,
            "amount_raw": 1_000_000_000,
        }

    def test_rejects_erc721_transfers(self):
        """ERC721 shares Transfer's topic0 but indexes tokenId as a 4th topic.

        Letting those through would put NFT ids in a usd_amount column.
        """
        log = transfer_log(token=USDC, sender=TRADER, recipient=POOL_V3, amount=1)
        log["topics"].append(addr_topic("0x" + "0" * 39 + "1"))
        log["data"] = "0x"
        assert abi.decode_transfer(log) is None

    def test_rejects_truncated_logs(self):
        log = transfer_log(token=USDC, sender=TRADER, recipient=POOL_V3, amount=1)
        log["topics"] = log["topics"][:2]
        assert abi.decode_transfer(log) is None

        log2 = transfer_log(token=USDC, sender=TRADER, recipient=POOL_V3, amount=1)
        log2["data"] = "0x"
        assert abi.decode_transfer(log2) is None


class TestSwapDecoding:
    def test_v2_swap_normalises_to_signed_amounts(self):
        """v2 emits four unsigned legs; we collapse them to `in - out` per token
        so v2 and v3 rows share one sign convention (positive = into the pool)."""
        decoded = abi.decode_v2_swap(
            v2_swap_log(
                pool=POOL_V2,
                amount0_in=1_000_000_000,   # 1000 USDC sold to the pool
                amount1_in=0,
                amount0_out=0,
                amount1_out=5 * 10**17,     # 0.5 WETH paid out
            )
        )
        assert decoded["amount0_raw"] == 1_000_000_000
        assert decoded["amount1_raw"] == -5 * 10**17
        assert decoded["version"] == "v2"
        assert decoded["pool"] == POOL_V2

    def test_v2_swap_in_the_other_direction(self):
        decoded = abi.decode_v2_swap(
            v2_swap_log(pool=POOL_V2, amount0_in=0, amount1_in=10**18,
                        amount0_out=2_000_000_000, amount1_out=0)
        )
        assert decoded["amount0_raw"] == -2_000_000_000
        assert decoded["amount1_raw"] == 10**18

    def test_v3_swap_decodes_signed_amounts_and_pool_state(self):
        decoded = abi.decode_v3_swap(
            v3_swap_log(pool=POOL_V3, amount0=1_000_000_000, amount1=-5 * 10**17,
                        liquidity=12345, tick=-201_000)
        )
        assert decoded["amount0_raw"] == 1_000_000_000
        assert decoded["amount1_raw"] == -5 * 10**17
        assert decoded["liquidity"] == 12345
        assert decoded["tick"] == -201_000
        assert decoded["version"] == "v3"

    def test_exactly_one_side_flows_into_the_pool(self):
        """The invariant the whole pairing check rests on: a swap moves one
        token in and a different token out, never both the same way."""
        for log in (
            v3_swap_log(pool=POOL_V3, amount0=1_000_000_000, amount1=-5 * 10**17),
            v3_swap_log(pool=POOL_V3, amount0=-2_000_000_000, amount1=10**18),
            v2_swap_log(pool=POOL_V2, amount0_in=10**9, amount1_in=0,
                        amount0_out=0, amount1_out=10**17),
        ):
            decoded = abi.decode_log(log)[1]
            assert (decoded["amount0_raw"] > 0) != (decoded["amount1_raw"] > 0)


class TestDispatch:
    @pytest.mark.parametrize(
        "log,expected_kind",
        [
            (transfer_log(token=USDC, sender=TRADER, recipient=POOL_V3, amount=1), "transfer"),
            (v2_swap_log(pool=POOL_V2, amount0_in=1, amount1_in=0,
                         amount0_out=0, amount1_out=1), "swap"),
            (v3_swap_log(pool=POOL_V3, amount0=1, amount1=-1), "swap"),
        ],
    )
    def test_dispatches_by_topic0(self, log, expected_kind):
        kind, decoded = abi.decode_log(log)
        assert kind == expected_kind
        assert decoded is not None

    def test_ignores_unknown_events(self):
        log = transfer_log(token=USDC, sender=TRADER, recipient=POOL_V3, amount=1)
        log["topics"][0] = "0x" + "ab" * 32
        assert abi.decode_log(log) is None

    def test_ignores_empty_topics(self):
        assert abi.decode_log({"topics": [], "data": "0x"}) is None

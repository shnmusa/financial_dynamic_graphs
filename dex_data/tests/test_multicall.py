"""Multicall3 encoding and the fallbacks around it.

Encoding ABI by hand is unforgiving, and a mis-decode here does not raise --
it silently reports "this call reverted", which would discard real pools and
tokens. So the shapes are pinned down directly.
"""
from __future__ import annotations

import pytest

from dex_data import abi
from dex_data.config import Config
from dex_data.rpc import MULTICALL_CHUNK, RpcClient, RpcError

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DEX_RPC_URLS", "https://a.example")
    monkeypatch.setenv("DEX_RPC_RATE_LIMIT", "0")
    return RpcClient(Config())


def encode_result(entries: list[tuple[bool, str]]) -> str:
    """Build an aggregate3 return value: (bool success, bytes returnData)[]."""
    elements = []
    for success, payload in entries:
        body = abi.strip0x(payload)
        length = len(body) // 2
        elements.append(
            f"{1 if success else 0:064x}"
            + f"{64:064x}"
            + f"{length:064x}"
            + body.ljust(((length + 31) // 32) * 64, "0")
        )
    offsets, cursor = [], 32 * len(elements)
    for element in elements:
        offsets.append(f"{cursor:064x}")
        cursor += len(element) // 2
    return "0x" + f"{32:064x}" + f"{len(elements):064x}" + "".join(offsets) + "".join(elements)


class TestEncoding:
    def test_calldata_starts_with_the_aggregate3_selector(self):
        data = abi.encode_aggregate3([(WETH, abi.SEL_DECIMALS)])
        assert data.startswith(abi.SEL_AGGREGATE3)

    def test_every_target_and_selector_appears_in_the_payload(self):
        data = abi.encode_aggregate3(
            [(WETH, abi.SEL_DECIMALS), (USDC, abi.SEL_SYMBOL)]
        ).lower()
        assert abi.strip0x(WETH) in data
        assert abi.strip0x(USDC) in data
        assert abi.strip0x(abi.SEL_DECIMALS) in data
        assert abi.strip0x(abi.SEL_SYMBOL) in data

    def test_payload_is_word_aligned(self):
        data = abi.strip0x(abi.encode_aggregate3([(WETH, abi.SEL_DECIMALS)] * 3))[8:]
        assert len(data) % 64 == 0

    def test_allow_failure_is_set_on_every_call(self):
        """Without it one reverting sub-call (fee() on a v2 pair) would revert
        the whole aggregate and lose the window."""
        data = abi.strip0x(abi.encode_aggregate3([(WETH, abi.SEL_DECIMALS)] * 2))[8:]
        words = [data[i:i + 64] for i in range(0, len(data), 64)]
        assert words.count(f"{1:064x}") >= 2

    def test_encoding_is_empty_safe(self):
        assert abi.decode_aggregate3(None) == []
        assert abi.decode_aggregate3("0x") == []


class TestDecoding:
    def test_round_trips_values_positionally(self):
        raw = encode_result([
            (True, "0x" + f"{18:064x}"),
            (True, "0x" + f"{6:064x}"),
            (True, "0x" + f"{500:064x}"),
        ])
        assert [abi.decode_uint(r) for r in abi.decode_aggregate3(raw)] == [18, 6, 500]

    def test_a_failed_subcall_becomes_none(self):
        """A v2 pair asked for fee() reverts. That must read as None, exactly
        like eth_call_batch reports a revert."""
        raw = encode_result([
            (True, "0x" + f"{18:064x}"),
            (False, "0x"),
            (True, "0x" + f"{8:064x}"),
        ])
        decoded = abi.decode_aggregate3(raw)
        assert decoded[1] is None
        assert [abi.decode_uint(decoded[0]), abi.decode_uint(decoded[2])] == [18, 8]

    def test_variable_length_returns_stay_aligned(self):
        """Dynamic strings shift every later offset; a mistake here pairs a
        token with another token's symbol."""
        symbol = "0x" + f"{32:064x}" + f"{4:064x}" + b"WETH".hex().ljust(64, "0")
        raw = encode_result([
            (True, "0x" + f"{18:064x}"),
            (True, symbol),
            (True, "0x" + f"{6:064x}"),
        ])
        decoded = abi.decode_aggregate3(raw)
        assert abi.decode_uint(decoded[0]) == 18
        assert abi.decode_string(decoded[1]) == "WETH"
        assert abi.decode_uint(decoded[2]) == 6


class TestClientBehaviour:
    def test_calls_are_chunked_into_separate_eth_calls(self, client):
        sizes: list[int] = []

        def fake_eth_call(to, data, block="latest"):
            n = int(abi.strip0x(data)[8 + 64 : 8 + 128], 16)
            sizes.append(n)
            return encode_result([(True, "0x" + f"{18:064x}")] * n)

        client.eth_call = fake_eth_call
        out = client.multicall([(WETH, abi.SEL_DECIMALS)] * (MULTICALL_CHUNK + 40))

        assert len(out) == MULTICALL_CHUNK + 40
        assert sizes == [MULTICALL_CHUNK, 40]

    def test_an_oversized_aggregate_is_split_not_lost(self, client):
        seen: list[int] = []

        def fake_eth_call(to, data, block="latest"):
            n = int(abi.strip0x(data)[8 + 64 : 8 + 128], 16)
            if n > 2:
                raise RpcError("request entity too large")
            seen.append(n)
            return encode_result([(True, "0x" + f"{18:064x}")] * n)

        client.eth_call = fake_eth_call
        out = client.multicall([(WETH, abi.SEL_DECIMALS)] * 8)

        assert len(out) == 8 and all(abi.decode_uint(r) == 18 for r in out)
        assert max(seen) <= 2

    def test_a_truncated_response_falls_back_instead_of_reporting_reverts(self, client):
        """The dangerous failure: fewer results than calls, read positionally,
        looks like 'these reverted' and would silently drop real pools."""
        client.eth_call = lambda to, data, block="latest": encode_result(
            [(True, "0x" + f"{18:064x}")]  # one result for three calls
        )
        fallback: list[int] = []

        def fake_batch(calls, block="latest"):
            fallback.append(len(calls))
            return ["0x" + f"{18:064x}"] * len(calls)

        client.eth_call_batch = fake_batch
        out = client.multicall([(WETH, abi.SEL_DECIMALS)] * 3)

        assert fallback == [3], "must redo the window the unambiguous way"
        assert len(out) == 3 and all(r is not None for r in out)

    def test_historical_blocks_before_deployment_use_plain_calls(self, client):
        """Multicall3 did not exist before block 14353601, so asking it there
        returns empty data that would decode as three reverts."""
        used: list[str] = []
        client.eth_call_batch = lambda calls, block="latest": (
            used.append("batch") or [None] * len(calls)
        )
        client.eth_call = lambda *a, **k: pytest.fail("must not reach Multicall3")

        client.multicall([(WETH, abi.SEL_DECIMALS)] * 3,
                         block=abi.MULTICALL3_DEPLOY_BLOCK - 1)
        assert used == ["batch"]

    def test_empty_input_makes_no_request(self, client):
        client.eth_call = lambda *a, **k: pytest.fail("no request expected")
        assert client.multicall([]) == []

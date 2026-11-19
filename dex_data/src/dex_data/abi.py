"""Event signatures, call encodings and hand-rolled ABI decoding.

Only fixed, no-argument calls and three event shapes are needed, so decoding is
done directly rather than pulling in web3/eth-abi. That keeps the dependency
list to requests + psycopg, which matters when this has to run anywhere.
"""
from __future__ import annotations

from typing import Sequence

# --- event topic0 hashes ----------------------------------------------------

# Transfer(address indexed from, address indexed to, uint256 value)
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# UniswapV2Pair.Swap(address indexed sender, uint amount0In, uint amount1In,
#                    uint amount0Out, uint amount1Out, address indexed to)
V2_SWAP_TOPIC = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"

# UniswapV3Pool.Swap(address indexed sender, address indexed recipient,
#                    int256 amount0, int256 amount1, uint160 sqrtPriceX96,
#                    uint128 liquidity, int24 tick)
V3_SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# UniswapV2Pair.Sync(uint112 reserve0, uint112 reserve1) -- not ingested today,
# but this is the free path to historical v2 liquidity when that task lands.
V2_SYNC_TOPIC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"

SWAP_TOPICS = (V2_SWAP_TOPIC, V3_SWAP_TOPIC)

# --- known factories --------------------------------------------------------
# SushiSwap, PancakeSwap and dozens of other forks emit a byte-identical v2 Swap
# event, so a topic match alone does not mean "Uniswap". Every discovered pool
# is asked for its factory() and checked against this map.

UNISWAP_V2_FACTORY = "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f"
UNISWAP_V3_FACTORY = "0x1f98431c8ad98523631ae4a59f267346ea31f984"

KNOWN_FACTORIES: dict[str, tuple[str, str]] = {
    UNISWAP_V2_FACTORY: ("uniswap", "v2"),
    UNISWAP_V3_FACTORY: ("uniswap", "v3"),
}

# --- function selectors (first 4 bytes of keccak of the signature) ----------

SEL_DECIMALS = "0x313ce567"  # decimals()
SEL_SYMBOL = "0x95d89b41"    # symbol()
SEL_NAME = "0x06fdde03"      # name()
SEL_FACTORY = "0xc45a0155"   # factory()
SEL_TOKEN0 = "0x0dfe1681"    # token0()
SEL_TOKEN1 = "0xd21220a7"    # token1()
SEL_FEE = "0xddca3f43"       # fee()
SEL_GET_RESERVES = "0x0902f1ac"  # getReserves()
SEL_BALANCE_OF = "0x70a08231"    # balanceOf(address)

# Multicall3, deployed at the same address on every major chain since block
# 14353601 on mainnet. Packing many argument-less view calls into ONE eth_call
# is the difference between thousands of metered requests and a handful:
# resolving 1008 pools and 762 tokens costs 6318 separate eth_calls, which
# saturates a free tier's per-second compute budget on its own.
MULTICALL3_ADDRESS = "0xca11bde05977b3631167028862be2a173976ca11"
MULTICALL3_DEPLOY_BLOCK = 14353601

# aggregate3((address target, bool allowFailure, bytes callData)[])
SEL_AGGREGATE3 = "0x82ad56cb"

ZERO_ADDRESS = "0x" + "0" * 40

UINT256_MAX = 1 << 256
INT256_SIGN_BIT = 1 << 255


# --- primitives -------------------------------------------------------------


def strip0x(value: str) -> str:
    return value[2:] if value.startswith(("0x", "0X")) else value


def words(data: str) -> list[str]:
    """Split ABI-encoded hex payload into 32-byte words."""
    body = strip0x(data)
    return [body[i : i + 64] for i in range(0, len(body) - len(body) % 64, 64)]


def to_uint(word: str) -> int:
    word = strip0x(word)
    return int(word, 16) if word else 0


def to_int(word: str) -> int:
    """Decode a two's-complement signed 256-bit word (v3 amounts are int256)."""
    value = to_uint(word)
    return value - UINT256_MAX if value & INT256_SIGN_BIT else value


def to_int24(word: str) -> int:
    value = to_uint(word) & 0xFFFFFF
    return value - (1 << 24) if value & (1 << 23) else value


def topic_to_address(topic: str) -> str:
    """A 32-byte topic holding a left-padded address -> lowercase 0x address."""
    return "0x" + strip0x(topic)[-40:].lower()


def encode_address_arg(address: str) -> str:
    return strip0x(address).lower().rjust(64, "0")


# --- return-value decoders --------------------------------------------------


def decode_address(result: str) -> str | None:
    body = strip0x(result or "")
    if len(body) < 40:
        return None
    address = "0x" + body[-40:].lower()
    return None if address == ZERO_ADDRESS else address


def decode_uint(result: str) -> int | None:
    body = strip0x(result or "")
    if not body:
        return None
    try:
        return int(body[:64], 16)
    except ValueError:
        return None


def decode_string(result: str) -> str | None:
    """Decode a `string` return, tolerating the bytes32 variant.

    MKR, SAI and other pre-ERC20-standard tokens declare `symbol()` as returning
    bytes32, so the well-formed dynamic-string decode has to fall back to
    "treat the single word as padded ASCII".
    """
    body = strip0x(result or "")
    if not body:
        return None

    # Canonical dynamic string: offset word, length word, then the bytes.
    if len(body) >= 128:
        try:
            offset = int(body[:64], 16) * 2
            if 0 <= offset < len(body) - 64:
                length = int(body[offset : offset + 64], 16) * 2
                if 0 < length <= len(body) - offset - 64:
                    raw = bytes.fromhex(body[offset + 64 : offset + 64 + length])
                    text = raw.decode("utf-8", errors="replace").strip("\x00").strip()
                    if text:
                        return text
        except ValueError:
            pass

    # bytes32 fallback.
    try:
        raw = bytes.fromhex(body[:64])
    except ValueError:
        return None
    text = raw.decode("utf-8", errors="replace").strip("\x00").strip()
    return text or None


# --- log decoders -----------------------------------------------------------


def decode_transfer(log: dict) -> dict | None:
    """Decode an ERC20 Transfer log. Returns None for shapes we cannot trust."""
    topics = log.get("topics") or []
    if len(topics) < 3:
        return None  # ERC721 or a malformed emitter

    # ERC721 Transfer has the same topic0 but a 4th indexed topic (tokenId) and
    # an empty data field. Those are NFTs, not money movement -- skip them.
    if len(topics) >= 4:
        return None

    data_words = words(log.get("data") or "")
    if not data_words:
        return None

    return {
        "tx_hash": log["transactionHash"].lower(),
        "log_index": to_uint(log["logIndex"]),
        "block_number": to_uint(log["blockNumber"]),
        "token": log["address"].lower(),
        "from_addr": topic_to_address(topics[1]),
        "to_addr": topic_to_address(topics[2]),
        "amount_raw": to_uint(data_words[0]),
    }


def decode_v2_swap(log: dict) -> dict | None:
    """Decode a UniswapV2 Swap log into the v3 sign convention.

    v2 reports four unsigned legs (amount0In, amount1In, amount0Out, amount1Out).
    Collapsing them to `in - out` per token gives the same signed amounts v3
    emits directly: positive means the token flowed into the pool.
    """
    topics = log.get("topics") or []
    data_words = words(log.get("data") or "")
    if len(topics) < 3 or len(data_words) < 4:
        return None

    amount0_in, amount1_in, amount0_out, amount1_out = (to_uint(w) for w in data_words[:4])

    return {
        "tx_hash": log["transactionHash"].lower(),
        "log_index": to_uint(log["logIndex"]),
        "block_number": to_uint(log["blockNumber"]),
        "pool": log["address"].lower(),
        "version": "v2",
        "sender": topic_to_address(topics[1]),
        "recipient": topic_to_address(topics[2]),
        "amount0_raw": amount0_in - amount0_out,
        "amount1_raw": amount1_in - amount1_out,
        "liquidity": None,
        "sqrt_price_x96": None,
        "tick": None,
    }


def decode_v3_swap(log: dict) -> dict | None:
    topics = log.get("topics") or []
    data_words = words(log.get("data") or "")
    if len(topics) < 3 or len(data_words) < 5:
        return None

    return {
        "tx_hash": log["transactionHash"].lower(),
        "log_index": to_uint(log["logIndex"]),
        "block_number": to_uint(log["blockNumber"]),
        "pool": log["address"].lower(),
        "version": "v3",
        "sender": topic_to_address(topics[1]),
        "recipient": topic_to_address(topics[2]),
        "amount0_raw": to_int(data_words[0]),
        "amount1_raw": to_int(data_words[1]),
        "sqrt_price_x96": to_uint(data_words[2]),
        "liquidity": to_uint(data_words[3]),
        "tick": to_int24(data_words[4]),
    }


def decode_log(log: dict) -> tuple[str, dict] | None:
    """Dispatch one log to its decoder. Returns (kind, decoded) or None."""
    topics = log.get("topics") or []
    if not topics:
        return None
    topic0 = topics[0].lower()

    if topic0 == TRANSFER_TOPIC:
        decoded = decode_transfer(log)
        return ("transfer", decoded) if decoded else None
    if topic0 == V2_SWAP_TOPIC:
        decoded = decode_v2_swap(log)
        return ("swap", decoded) if decoded else None
    if topic0 == V3_SWAP_TOPIC:
        decoded = decode_v3_swap(log)
        return ("swap", decoded) if decoded else None
    return None


# --- Multicall3 ------------------------------------------------------------


def encode_aggregate3(calls: Sequence[tuple[str, str]]) -> str:
    """ABI-encode `aggregate3` for (target, calldata) pairs.

    allowFailure is always true: a reverting sub-call (asking a v2 pair for
    fee(), say) is an expected outcome that must not take the batch down with
    it. Those come back as success=false and decode to None.
    """
    elements: list[str] = []
    for target, data in calls:
        body = strip0x(data)
        length = len(body) // 2
        padded = body.ljust(((length + 31) // 32) * 64, "0")
        elements.append(
            strip0x(target).lower().rjust(64, "0")  # address target
            + f"{1:064x}"                           # bool allowFailure
            + f"{96:064x}"                          # offset to callData in struct
            + f"{length:064x}"                      # callData length
            + padded                                # callData
        )

    # Call3 is a dynamic struct, so the array holds an offset per element,
    # measured from the first word after the array length.
    offsets: list[str] = []
    cursor = 32 * len(elements)
    for element in elements:
        offsets.append(f"{cursor:064x}")
        cursor += len(element) // 2

    return (
        SEL_AGGREGATE3
        + f"{32:064x}"                 # offset to the array
        + f"{len(elements):064x}"      # array length
        + "".join(offsets)
        + "".join(elements)
    )


def decode_aggregate3(result: str | None) -> list[str | None]:
    """Decode `(bool success, bytes returnData)[]`.

    A failed sub-call becomes None, matching what eth_call_batch returns for a
    revert, so callers cannot tell the two paths apart.
    """
    body = strip0x(result or "")
    if len(body) < 128:
        return []

    try:
        array_at = int(body[:64], 16) * 2
        count = int(body[array_at : array_at + 64], 16)
    except ValueError:
        return []

    base = array_at + 64
    out: list[str | None] = []
    for index in range(count):
        try:
            element_at = base + int(body[base + index * 64 : base + (index + 1) * 64], 16) * 2
            success = int(body[element_at : element_at + 64], 16) == 1
            data_at = element_at + int(body[element_at + 64 : element_at + 128], 16) * 2
            length = int(body[data_at : data_at + 64], 16) * 2
            payload = body[data_at + 64 : data_at + 64 + length]
        except ValueError:
            out.append(None)
            continue
        out.append(("0x" + payload) if success else None)
    return out

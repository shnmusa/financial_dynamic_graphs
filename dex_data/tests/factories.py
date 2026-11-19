"""Builders for synthetic on-chain logs and rows, shared by the test modules."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from dex_data import abi

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"  # 6 decimals, stablecoin
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"  # 18 decimals
KISHU = "0xa2b4c0af19cc16a6cfacce81f192b024d625817d"  # 9 decimals, rebasing
TRADER = "0x1111111111111111111111111111111111111111"
POOL_V3 = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"  # USDC/WETH 0.05%
POOL_V2 = "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc"  # USDC/WETH v2
POOL_2 = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"   # a second USDC/WETH pool
POOL_KISHU = "0xf82d8ec196fb0d56c6b82a8b1870f09502a49f88"  # KISHU/WETH v2

USDC_1000 = 1_000_000_000                # 1000 USDC (6 decimals)
WETH_HALF = 500_000_000_000_000_000      # 0.5 WETH (18 decimals)


def word(value: int) -> str:
    """Encode an int as one 32-byte ABI word, two's complement when negative."""
    if value < 0:
        value += 1 << 256
    return f"{value:064x}"


def addr_topic(address: str) -> str:
    return "0x" + abi.strip0x(address).lower().rjust(64, "0")


def transfer_log(
    *, token: str, sender: str, recipient: str, amount: int,
    block: int = 100, log_index: int = 0, tx: str = "0xaa",
) -> dict:
    return {
        "address": token,
        "topics": [abi.TRANSFER_TOPIC, addr_topic(sender), addr_topic(recipient)],
        "data": "0x" + word(amount),
        "blockNumber": hex(block),
        "transactionHash": tx,
        "logIndex": hex(log_index),
    }


def v2_swap_log(
    *, pool: str, amount0_in: int, amount1_in: int, amount0_out: int, amount1_out: int,
    sender: str = TRADER, to: str = TRADER,
    block: int = 100, log_index: int = 2, tx: str = "0xaa",
) -> dict:
    return {
        "address": pool,
        "topics": [abi.V2_SWAP_TOPIC, addr_topic(sender), addr_topic(to)],
        "data": "0x" + word(amount0_in) + word(amount1_in) + word(amount0_out) + word(amount1_out),
        "blockNumber": hex(block),
        "transactionHash": tx,
        "logIndex": hex(log_index),
    }


def v3_swap_log(
    *, pool: str, amount0: int, amount1: int,
    sqrt_price: int = 1 << 96, liquidity: int = 10**20, tick: int = -1000,
    sender: str = TRADER, recipient: str = TRADER,
    block: int = 100, log_index: int = 2, tx: str = "0xaa",
) -> dict:
    return {
        "address": pool,
        "topics": [abi.V3_SWAP_TOPIC, addr_topic(sender), addr_topic(recipient)],
        "data": "0x" + word(amount0) + word(amount1) + word(sqrt_price)
                + word(liquidity) + word(tick),
        "blockNumber": hex(block),
        "transactionHash": tx,
        "logIndex": hex(log_index),
    }


# --- direct row seeding -----------------------------------------------------

TOKEN_META = {
    USDC: ("USDC", 6, True, False),
    WETH: ("WETH", 18, False, False),
    KISHU: ("KISHU", 9, False, True),
}

# pricing.py only ever prices tokens.target (stablecoins are the one
# exception, seeded at hop 0 regardless). WETH stands in for "a real target
# token" here, same as it is in production.
TARGET_TOKENS = {WETH}


def seed_reference(db, *, block: int = 100, when: datetime | None = None) -> None:
    """Insert the tokens, pools and block a pairing test needs."""
    when = when or datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc)
    with db.connection() as conn:
        for address, (symbol, decimals, stable, rebasing) in TOKEN_META.items():
            conn.execute(
                "INSERT INTO tokens (address, symbol, decimals, is_stable, is_rebasing, target) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (address) DO NOTHING",
                (address, symbol, decimals, stable, rebasing, address in TARGET_TOKENS),
            )
        for pool, version, t0, t1 in (
            (POOL_V3, "v3", USDC, WETH),
            (POOL_V2, "v2", USDC, WETH),
            (POOL_2, "v3", USDC, WETH),
            (POOL_KISHU, "v2", KISHU, WETH),
        ):
            factory = abi.UNISWAP_V3_FACTORY if version == "v3" else abi.UNISWAP_V2_FACTORY
            conn.execute(
                "INSERT INTO pools (address, dex, version, factory, token0, token1) "
                "VALUES (%s, 'uniswap', %s, %s, %s, %s) ON CONFLICT (address) DO NOTHING",
                (pool, version, factory, t0, t1),
            )
        conn.execute(
            "INSERT INTO blocks (number, timestamp) VALUES (%s, %s) "
            "ON CONFLICT (number) DO NOTHING",
            (block, when),
        )
        conn.commit()


def leg(
    db,
    *,
    tx: str,
    log_index: int,
    source: str,
    destination: str,
    token: str,
    amount_raw: int,
    pool: str | None = None,
    block: int = 100,
    when: datetime | None = None,
    counterpart_expected: bool = True,
) -> None:
    """Insert one dex_events row: a single token movement touching a pool."""
    when = when or datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc)
    symbol, decimals = TOKEN_META[token][0], TOKEN_META[token][1]
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO dex_events (transaction_id, type, source, destination, token, "
            "usd_amount, timestamp, block_number, tx_hash, log_index, token_address, "
            "amount, amount_raw, pool, counterpart_expected) "
            "VALUES (%s, 'swap', %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (transaction_id) DO NOTHING",
            (
                f"{tx}-{log_index}", source, destination, symbol, when, block, tx,
                log_index, token, Decimal(amount_raw) / (Decimal(10) ** decimals),
                amount_raw, pool or (source if source != destination else None),
                counterpart_expected,
            ),
        )
        conn.commit()


def seed_swap(
    db,
    *,
    tx: str,
    pool: str = POOL_V3,
    block: int = 100,
    when: datetime | None = None,
    usdc_amount: int = USDC_1000,
    weth_amount: int = WETH_HALF,
    include_in_leg: bool = True,
    include_out_leg: bool = True,
    counterpart_expected: bool = True,
    base_log_index: int = 0,
) -> None:
    """A complete USDC -> WETH swap as its two dex_events rows.

    include_in_leg / include_out_leg exist so a test can deliberately break the
    pairing invariant and confirm the check notices.
    """
    if include_in_leg:
        leg(db, tx=tx, log_index=base_log_index, source=TRADER, destination=pool,
            token=USDC, amount_raw=usdc_amount, pool=pool, block=block, when=when,
            counterpart_expected=counterpart_expected)
    if include_out_leg:
        leg(db, tx=tx, log_index=base_log_index + 1, source=pool, destination=TRADER,
            token=WETH, amount_raw=weth_amount, pool=pool, block=block, when=when,
            counterpart_expected=counterpart_expected)

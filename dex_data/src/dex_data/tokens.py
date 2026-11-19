"""ERC20 metadata resolution and the stablecoin set that anchors pricing."""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

from . import abi, db
from .rpc import RpcClient, get_client

log = logging.getLogger(__name__)

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

# Priced at exactly $1. These are the root of the pricing graph -- every other
# token's USD value is derived by walking pools back to one of these.
STABLECOINS: dict[str, int] = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,   # USDT
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
    "0x853d955acef822db058eb8505911ed77f175b99e": 18,  # FRAX
    "0x0000000000085d4780b73119b644ae5ecd22b376": 18,  # TUSD
    "0x4c9edd5852cd905f086c759e8383e09bff1e68b3": 18,  # USDe
    "0x8292bb45bf1ee4d140127049757c2e0ff06317ed": 6,   # RLUSD
    "0x6c3ea9036406852006290770bedfcaba0e23a0e8": 6,   # PYUSD
}

# Tokens that change balances without emitting Transfer (reflection / rebasing).
#
# Established from data, not assumed: an ingest run flagged swaps on the
# KISHU/WETH v2 pair whose Swap event reported 2.4 billion KISHU flowing in
# while the transaction contained no KISHU Transfer at all. That is the pair's
# balance growing by reflection and someone calling swap() to skim the surplus.
#
# Extend this list only on the same evidence -- `dex_data.cli verify` names the
# input token of every unpaired swap precisely so an unknown offender is
# reported loudly rather than absorbed into a tolerance.
REBASING_TOKENS: set[str] = {
    "0xa2b4c0af19cc16a6cfacce81f192b024d625817d",  # KISHU
    # The following were all found the same way, in one `ingest --days 1` run:
    # each pool's Swap event reported a nonzero amount in on one side while the
    # transaction's logs carried zero Transfer events crediting that pool for
    # that token. Confirmed individually by decoding each Swap event's raw
    # amount0/amount1 and checking for a Transfer log whose `to` is the pool.
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",  # stETH -- rebases; a pool's
        # balanceOf can grow between transactions with no Transfer in the swap
        # that observes it. A known Uniswap V2 / stETH integration hazard.
    "0x4d5f47fa6a74757f35c14fd3a6ef8e3c9bc514e8",  # aEthWETH (Aave interest-
        # bearing WETH) -- same rebasing-balance behaviour as stETH.
    "0x106552c11272420aad5d7e94f8acab9095a6c952",  # KEANU
    "0x6fa0be17e4bea2fcfa22ef89bf8ac9aab0ab0fc9",  # A7A5
    "0xcd7492db29e2ab436e819b249452ee1bbdf52214",  # SMI
    "0xe2a083397521968eb05585932750634bed4b7d56",  # HODL
    "0xd555498a524612c67f286df0e0a9a64a73a7cdc7",  # DEFROGS
    "0xa8c16fbecf1ae1bc1dcc87e6892f18f777e92c69",  # ANDROT
    "0xc6e64729931f60d2c8bc70a27d66d9e0c28d1bf9",  # FLOW
    "0x52263e871674fb1c71f4a4b3575c0cc43d027dd7",  # 404Cats
    "0xecd48f326e70388d993694de59b4542ce8af7649",  # TADPOLE
    "0x348b7f3106b5da47405332534d06069ff9ce4d1b",  # ELongD
}

# Fallbacks for the handful of tokens whose on-chain metadata calls are
# non-standard enough that a plain eth_call cannot read them.
KNOWN_METADATA: dict[str, tuple[str, int]] = {
    WETH: ("WETH", 18),
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": ("WBTC", 8),
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": ("MKR", 18),
    "0x89d24a6b4ccb1b6faa2625fe562bdd9a23260359": ("SAI", 18),
}


def resolve_metadata(
    addresses: Iterable[str], client: RpcClient | None = None, batch_size: int = 300
) -> dict[str, dict]:
    """Read symbol/name/decimals for tokens, three view calls each.

    Packed through Multicall3, so 300 tokens cost one eth_call rather than 900.
    """
    client = client or get_client()
    addresses = [a.lower() for a in dict.fromkeys(addresses)]
    out: dict[str, dict] = {}

    for start in range(0, len(addresses), batch_size):
        window = addresses[start : start + batch_size]
        calls: list[tuple[str, str]] = []
        for address in window:
            calls.append((address, abi.SEL_SYMBOL))
            calls.append((address, abi.SEL_NAME))
            calls.append((address, abi.SEL_DECIMALS))

        results = client.multicall(calls)

        for index, address in enumerate(window):
            symbol_raw, name_raw, decimals_raw = results[index * 3 : index * 3 + 3]
            known = KNOWN_METADATA.get(address)

            symbol = abi.decode_string(symbol_raw) if symbol_raw else None
            name = abi.decode_string(name_raw) if name_raw else None
            decimals = abi.decode_uint(decimals_raw) if decimals_raw else None

            if known:
                symbol = symbol or known[0]
                decimals = known[1] if decimals is None else decimals
            if address in STABLECOINS and decimals is None:
                decimals = STABLECOINS[address]

            # A token reporting absurd decimals is broken or hostile; 18 is the
            # ERC20 default and the only safe assumption.
            if decimals is None or not (0 <= decimals <= 36):
                decimals = 18

            out[address] = {
                "address": address,
                "symbol": (symbol or "")[:64] or None,
                "name": (name or "")[:128] or None,
                "decimals": decimals,
                "is_stable": address in STABLECOINS,
                "is_rebasing": address in REBASING_TOKENS,
            }

    return out


def store(metadata: dict[str, dict], first_seen: dict[str, int] | None = None) -> int:
    first_seen = first_seen or {}
    rows = [
        (
            meta["address"], meta["symbol"], meta["name"], meta["decimals"],
            meta["is_stable"], meta["is_rebasing"], first_seen.get(meta["address"]),
        )
        for meta in metadata.values()
    ]
    # is_stable is set on insert from STABLECOINS above and then left alone.
    # marketcap.refresh() widens it from CoinGecko's stablecoin category -- it
    # finds around thirty more than this file hardcodes -- and re-running ingest
    # must not quietly reset those to the seed list.
    return db.upsert(
        "tokens",
        ("address", "symbol", "name", "decimals", "is_stable", "is_rebasing",
         "first_seen_block"),
        rows,
        ("address",),
        update_cols=("symbol", "name", "decimals", "is_rebasing"),
    )


def ensure_known(
    addresses: Sequence[str], client: RpcClient | None = None, first_seen: dict[str, int] | None = None
) -> int:
    """Resolve and persist any token addresses not already in the tokens table."""
    addresses = [a.lower() for a in dict.fromkeys(addresses)]
    if not addresses:
        return 0

    known = {
        r[0]
        for r in db.fetch_all("SELECT address FROM tokens WHERE address = ANY(%s)", (addresses,))
    }
    missing = [a for a in addresses if a not in known]
    if not missing:
        return 0

    log.info("resolving metadata for %d tokens", len(missing))
    return store(resolve_metadata(missing, client), first_seen)


def load_decimals() -> dict[str, int]:
    return {r[0]: r[1] for r in db.fetch_all("SELECT address, decimals FROM tokens")}


def load_metadata() -> dict[str, dict]:
    """symbol/decimals/is_rebasing for every known token, keyed by address."""
    return {
        r[0]: {"symbol": r[1], "decimals": r[2], "is_rebasing": r[3]}
        for r in db.fetch_all(
            "SELECT address, symbol, decimals, is_rebasing FROM tokens"
        )
    }

"""Market cap per token, from CoinGecko.

Listing is itself the filter. Roughly two thirds of the addresses that touch a
Uniswap pool in a 24h window are launched-that-day tokens nobody lists, and they
carry under 8% of the volume between them. An earlier version priced those from
`totalSupply()` so every token had *some* number; that was dropped on purpose.
It could not tell a real small cap from a broken one -- a token with 1e30 units
and a bad price came out at $57 trillion and sailed through a "keep the big
ones" filter, which is the exact opposite of what the filter is for. Absence
from CoinGecko is now taken as the verdict it is.

CoinGecko is queried backwards. The obvious endpoint, `simple/token_price`,
takes contract addresses -- but the public tier caps it at ONE address per
request and 429s after about five requests, so 5.5k tokens would take hours.
`coins/list?include_platform=true` returns every listed coin with its Ethereum
contract in a single 3 MB response, and `coins/markets?ids=` then accepts 250
ids at a time. The whole set costs about ten requests that way.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable

import requests

from . import db
from .config import PROJECT_ROOT
from .tokens import STABLECOINS

log = logging.getLogger(__name__)

CG_API = "https://api.coingecko.com/api/v3"

# Public tier: ~5 requests before a 429, with no Retry-After header to read.
CG_PAGE_SIZE = 250
CG_PAUSE_SECONDS = 6.0
CG_MAX_RETRIES = 4
CG_CACHE = PROJECT_ROOT / ".cache" / "coingecko_contracts.json"

# How far from $1.00 a coin may drift and still be treated as a dollar. See
# usd_stablecoins for why 2% and not something tighter or looser.
STABLE_TOLERANCE = 0.02
# The stablecoins category is a few hundred coins; this bounds the paging.
CG_MAX_CATEGORY_PAGES = 6


def _get(session: requests.Session, path: str, params: dict, timeout: int = 60):
    """GET with backoff, because the public tier 429s without a Retry-After.

    Its window is long enough that a single doubled pause is not always
    sufficient, and a dropped window silently deletes 250 tokens from the
    result, so it is worth waiting out.
    """
    response = None
    for attempt in range(CG_MAX_RETRIES):
        response = session.get(f"{CG_API}/{path}", params=params, timeout=timeout)
        if response.status_code != 429:
            return response
        backoff = CG_PAUSE_SECONDS * 2 ** (attempt + 1)
        log.info("coingecko rate limited on %s; waiting %.0fs", path, backoff)
        time.sleep(backoff)
    return response


def contract_map(
    session: requests.Session | None = None, max_age_hours: float = 24.0
) -> dict[str, str]:
    """Ethereum contract address -> CoinGecko coin id, for every listed coin.

    Cached on disk: it is a 3 MB response that changes on the timescale of new
    listings, and re-fetching it on every run is what pushes the free tier into
    a 429 before the market-cap pages have even started.
    """
    session = session or requests.Session()
    if CG_CACHE.exists():
        age = time.time() - CG_CACHE.stat().st_mtime
        if age < max_age_hours * 3600:
            return json.loads(CG_CACHE.read_text())

    response = _get(session, "coins/list", {"include_platform": "true"}, timeout=120)
    if not response.ok:
        # A stale cache beats no mapping at all: without it every token looks
        # unlisted, and unlisted now means eliminated.
        if CG_CACHE.exists():
            log.warning("coins/list %s; using cache from %s",
                        response.status_code,
                        datetime.fromtimestamp(CG_CACHE.stat().st_mtime, timezone.utc))
            return json.loads(CG_CACHE.read_text())
        response.raise_for_status()

    mapping = {
        coin["platforms"]["ethereum"].lower(): coin["id"]
        for coin in response.json()
        if coin.get("platforms", {}).get("ethereum")
    }
    CG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CG_CACHE.write_text(json.dumps(mapping))
    return mapping


def market_caps(
    ids: Iterable[str], session: requests.Session | None = None
) -> dict[str, dict]:
    """Circulating market cap, FDV and rank for each CoinGecko id."""
    session = session or requests.Session()
    ids = list(dict.fromkeys(ids))
    out: dict[str, dict] = {}

    for start in range(0, len(ids), CG_PAGE_SIZE):
        window = ids[start : start + CG_PAGE_SIZE]
        if start:
            time.sleep(CG_PAUSE_SECONDS)
        response = _get(session, "coins/markets",
                        {"vs_currency": "usd", "ids": ",".join(window),
                         "per_page": CG_PAGE_SIZE, "page": 1})
        if not response.ok:
            log.warning("coingecko %s for %d ids; skipping window",
                        response.status_code, len(window))
            continue

        for coin in response.json():
            out[coin["id"]] = {
                "market_cap": coin.get("market_cap"),
                "fdv": coin.get("fully_diluted_valuation"),
                "rank": coin.get("market_cap_rank"),
            }
        log.info("coingecko %d/%d", min(start + CG_PAGE_SIZE, len(ids)), len(ids))

    return out


def usd_stablecoins(
    session: requests.Session | None = None, tolerance: float = STABLE_TOLERANCE
) -> set[str]:
    """Ethereum contracts of USD stablecoins currently trading at their peg.

    Category membership alone is not the test. CoinGecko files EUR, CHF, JPY,
    SGD and BRL tokens under `stablecoins` too -- EURC sits at $1.16, JPYC at
    $0.0095 -- and it keeps coins there long after they have broken (USR at
    $0.12, msUSD at $0.38). `is_stable` means "worth exactly one dollar, use as
    a root of the pricing graph", so a coin has to still be worth a dollar.

    The tolerance is not arbitrary: on this data the observed prices leave a
    clean gap either side of 2%. The nearest coins outside are ARGN at 1.06 and
    the FX pegs; the nearest inside are USDM1 at 1.016, LUSD at 1.012 and FRAX
    at 0.992. Nothing sits in between, so the cut is not balanced on an edge.
    """
    session = session or requests.Session()
    known = contract_map(session)
    stable: set[str] = set()
    by_id = {cid: address for address, cid in known.items()}

    for page in range(1, CG_MAX_CATEGORY_PAGES + 1):
        if page > 1:
            time.sleep(CG_PAUSE_SECONDS)
        response = _get(session, "coins/markets",
                        {"vs_currency": "usd", "category": "stablecoins",
                         "per_page": CG_PAGE_SIZE, "page": page})
        if not response.ok:
            log.warning("coingecko %s on stablecoin page %d", response.status_code, page)
            break
        coins = response.json()
        for coin in coins:
            price = coin.get("current_price")
            if price and abs(price - 1.0) <= tolerance and coin["id"] in by_id:
                stable.add(by_id[coin["id"]])
        if len(coins) < CG_PAGE_SIZE:
            break

    return stable


def refresh() -> dict[str, int]:
    """Fill market_cap_usd and is_stable on `tokens` from CoinGecko."""
    addresses = [row[0] for row in db.fetch_all("SELECT address FROM tokens")]

    session = requests.Session()
    session.headers["User-Agent"] = "dex_data/0.1"
    known = contract_map(session)
    listed = {a: known[a] for a in addresses if a in known}
    caps = market_caps(listed.values(), session)
    stable = usd_stablecoins(session)

    def money(value) -> Decimal | None:
        """Zero is CoinGecko's way of saying it has no circulating figure.

        Storing it as 0.00 would make `market_cap_usd IS NOT NULL` read as
        "we know how big this is", and let the token through any `>= 0` filter.
        """
        if value in (None, "") or not Decimal(str(value)):
            return None
        return Decimal(str(value)).quantize(Decimal("0.01"))

    now = datetime.now(timezone.utc)
    rows = []
    counts = {"listed": 0, "unlisted": 0, "stable": 0}

    for address in addresses:
        cg_id = listed.get(address)
        # A listed coin with a null market_cap has no circulating figure on
        # CoinGecko either. It is not evidence of size, so it stays NULL and
        # gets eliminated alongside the unlisted.
        cap = caps.get(cg_id) or {} if cg_id else {}
        market_cap = money(cap.get("market_cap"))
        counts["listed" if market_cap else "unlisted"] += 1
        # The hardcoded seed in tokens.STABLECOINS stays authoritative: pricing
        # has to work on a machine that has never reached the API, and those
        # eight are the graph's floor.
        is_stable = address in stable or address in STABLECOINS
        counts["stable"] += is_stable
        rows.append((address, market_cap, money(cap.get("fdv")),
                     cg_id, cap.get("rank"), is_stable, now))

    columns = ("address", "market_cap_usd", "fdv_usd", "coingecko_id",
               "cg_rank", "is_stable", "market_data_at")
    db.upsert("tokens", columns, rows, ("address",), update_cols=columns[1:])
    return counts

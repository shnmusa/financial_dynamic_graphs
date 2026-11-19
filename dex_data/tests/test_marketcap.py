"""What CoinGecko's answers are allowed to mean.

Two judgements in marketcap.py are not obvious from the API and would fail
silently if they regressed, so they are pinned here.

The first is `is_stable`. CoinGecko's `stablecoins` category is a category, not
a peg check: EUR, CHF and JPY tokens sit in it, and so do coins that broke their
peg months ago. `is_stable` means "worth exactly one dollar, use as a root of
the pricing graph", so trusting the category alone would seed the graph with
EURC at $1.16 and mis-price everything derived from it.

The second is a zero market cap. CoinGecko returns `market_cap: 0` for a few
hundred listed coins that have no circulating figure. Stored as 0.00 it would
read as "we know how big this is" and pass any `>= 0` threshold.
"""
from __future__ import annotations

import json

import pytest

from dex_data import marketcap


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for requests.Session, answering by endpoint path."""

    def __init__(self, markets):
        self.markets = markets
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        return FakeResponse(self.markets)


def coin(cid, price, market_cap=1_000_000):
    return {"id": cid, "current_price": price, "market_cap": market_cap,
            "fully_diluted_valuation": market_cap, "market_cap_rank": None}


@pytest.fixture
def contracts(tmp_path, monkeypatch):
    """A contract map on disk, fresh enough that no HTTP call is attempted."""
    mapping = {
        "0xaaa": "usd-coin", "0xbbb": "euro-coin", "0xccc": "resolv-usr",
        "0xddd": "frax", "0xeee": "jpy-coin", "0xfff": "argentine-token",
    }
    cache = tmp_path / "coingecko_contracts.json"
    cache.write_text(json.dumps(mapping))
    monkeypatch.setattr(marketcap, "CG_CACHE", cache)
    return mapping


class TestStablecoinSelection:
    def test_keeps_usd_coins_at_peg(self, contracts):
        session = FakeSession([coin("usd-coin", 0.9998)])
        assert marketcap.usd_stablecoins(session) == {"0xaaa"}

    def test_rejects_foreign_currency_pegs(self, contracts):
        """EURC is in the stablecoins category and is not a dollar."""
        session = FakeSession([coin("euro-coin", 1.16), coin("jpy-coin", 0.0095)])
        assert marketcap.usd_stablecoins(session) == set()

    def test_rejects_broken_pegs(self, contracts):
        """The category keeps coins long after they stop being worth a dollar."""
        session = FakeSession([coin("resolv-usr", 0.123)])
        assert marketcap.usd_stablecoins(session) == set()

    def test_tolerance_admits_normal_drift(self, contracts):
        """FRAX at 0.992 is a dollar; a token at 1.06 is not."""
        session = FakeSession([coin("frax", 0.9916), coin("argentine-token", 1.06)])
        assert marketcap.usd_stablecoins(session) == {"0xddd"}

    def test_ignores_coins_without_an_ethereum_contract(self, contracts):
        session = FakeSession([coin("solana-only-usd", 1.0)])
        assert marketcap.usd_stablecoins(session) == set()

    def test_missing_price_is_not_a_dollar(self, contracts):
        session = FakeSession([coin("usd-coin", None)])
        assert marketcap.usd_stablecoins(session) == set()


class TestPaging:
    def test_stops_on_a_short_page(self, contracts):
        session = FakeSession([coin("usd-coin", 1.0)])
        marketcap.usd_stablecoins(session)
        assert len(session.calls) == 1

    def test_asks_for_the_stablecoin_category(self, contracts):
        session = FakeSession([])
        marketcap.usd_stablecoins(session)
        assert session.calls[0]["category"] == "stablecoins"

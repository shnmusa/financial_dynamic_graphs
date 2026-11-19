"""The modelling universe: which tokens a model built on dex_events is about.

The universe is `tokens.target`, a flag set once by `pin` and never moved by
anything else. It is deliberately not a live `market_cap_usd > x` test: caps
move daily, so a token drifting from $101M to $99M would drop out on its own
and the token set a model trains on would change shape for reasons that are
not in the data.
"""
from __future__ import annotations

import logging

from . import db, pools
from .config import CONFIG, Config

log = logging.getLogger(__name__)

class UniverseError(RuntimeError):
    """The universe is missing, or already pinned and would be overwritten."""


def pin(min_market_cap: float | None = None, top: int | None = None,
        force: bool = False, config: Config = CONFIG) -> list[tuple]:
    """Choose the modelling universe once and freeze it into `tokens.target`.

    Refuses to move an existing universe without `force`, because a dataset
    whose token set changed between two runs is not comparable with itself, and
    that is exactly the failure this flag exists to prevent.

    Selection is on market cap alone. `cg_rank` looks like a tempting quality
    filter and is not one: CoinGecko withholds a rank from wrapped and staked
    derivatives so their cap is not counted twice against the asset they wrap.
    Thirty of the 128 tokens here are unranked for that reason, WETH among
    them -- and WETH is one side of 92% of the swaps that survive.
    """
    existing = db.fetch_one("SELECT count(*) FROM tokens WHERE target")[0]
    if existing and not force:
        raise UniverseError(
            f"{existing} tokens are already pinned as target; pass force=True "
            "to re-pin. Every dataset built before and after would differ."
        )

    if top is not None:
        criterion = f"top {top} by market cap"
        selector = """
            WITH ranked AS (
                SELECT address FROM tokens WHERE market_cap_usd IS NOT NULL
                ORDER BY market_cap_usd DESC LIMIT %(top)s
            )
            UPDATE tokens SET target = TRUE,
                              target_market_cap_usd = market_cap_usd,
                              target_pinned_at = now()
            WHERE address IN (SELECT address FROM ranked)"""
        params = {"top": top}
    else:
        threshold = config.min_market_cap if min_market_cap is None else min_market_cap
        criterion = f"market cap > ${threshold:,.0f}"
        selector = """
            UPDATE tokens SET target = TRUE,
                              target_market_cap_usd = market_cap_usd,
                              target_pinned_at = now()
            WHERE market_cap_usd > %(threshold)s"""
        params = {"threshold": threshold}

    db.execute("UPDATE tokens SET target = FALSE, target_market_cap_usd = NULL, "
               "target_pinned_at = NULL WHERE target")
    db.execute(selector, params)
    # Whole-table: pinning can flip is_target for pools that were discovered
    # long before this token was ever a target.
    pools.refresh_target_flags()

    chosen = db.fetch_all(
        "SELECT symbol, target_market_cap_usd, is_stable FROM tokens "
        "WHERE target ORDER BY target_market_cap_usd DESC"
    )
    log.info("pinned %d tokens as target (%s)", len(chosen), criterion)
    return chosen

"""Command line entry point.

    python -m dex_data.cli all            # full pipeline, last DEX_DAYS days
    python -m dex_data.cli initdb
    python -m dex_data.cli ingest --days 30
    python -m dex_data.cli price                        # CoinGecko prices + usd_amount
    python -m dex_data.cli price --no-fetch --table dex_events_removed
    python -m dex_data.cli verify
    python -m dex_data.cli reset
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from . import blocks, cleanup, dataset, db, ingest, marketcap, pricing, verify
from .config import CONFIG
from .rpc import get_client


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


log = logging.getLogger("dex_data")


def cmd_initdb(_args) -> int:
    db.init_schema()
    return 0


def cmd_reset(args) -> int:
    if not args.yes:
        answer = input("This truncates all ingested data. Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            print("aborted")
            return 1
    tables = None if not args.all else [
        "dex_events", "dex_events_removed", "token_prices_hourly",
        "ingest_progress", "blocks", "pools", "tokens",
    ]
    db.reset_data(tables)
    return 0


def cmd_ingest(args) -> int:
    db.init_schema()
    client = get_client()

    if args.from_block is not None and args.to_block is not None:
        from_block, to_block = args.from_block, args.to_block
        log.info("explicit block range %d .. %d", from_block, to_block)
    else:
        from_block, to_block, _, _ = blocks.resolve_window(args.days or CONFIG.days, client)

    ingest.run(from_block, to_block, CONFIG, client, resume=not args.no_resume)
    return 0


def cmd_price(args) -> int:
    """Fetch CoinGecko hourly prices, then clear and refill usd_amount."""
    table = getattr(args, "table", "dex_events")
    if getattr(args, "fill_only", False):
        pricing.fill_gaps(table)
        return 0
    if not getattr(args, "no_fetch", False):
        pricing.build(api_key=os.environ.get("COINGECKO_API_KEY"))
    pricing.apply_usd_amounts(CONFIG, table=table)
    return 0


def cmd_marketcap(args) -> int:
    """Fill market_cap_usd on tokens from CoinGecko."""
    counts = marketcap.refresh()
    log.info("market cap: %d listed on coingecko, %d unlisted (eliminated); "
             "%d stablecoins", counts["listed"], counts["unlisted"], counts["stable"])
    return 0


def cmd_target(args) -> int:
    """Choose the modelling universe once and freeze it into tokens.target."""
    chosen = dataset.pin(args.min_market_cap, args.top, args.force)
    log.info("%d tokens pinned as target", len(chosen))
    for symbol, cap, stable in chosen[:10]:
        log.info("  %-12s $%s%s", symbol, f"{float(cap or 0):,.0f}",
                 "  (stablecoin)" if stable else "")
    if len(chosen) > 10:
        log.info("  ... and %d more", len(chosen) - 10)
    return 0


def cmd_cleanup(_args) -> int:
    """Move standalone stable/non-target swaps into dex_events_removed."""
    cleanup.run()
    return 0


def cmd_verify(args) -> int:
    result = verify.report()
    if result.checked_swaps and result.ratio < args.min_ratio:
        log.error("swap pairing ratio %.4f is below the required %.4f",
                  result.ratio, args.min_ratio)
        return 1
    return 0


def cmd_all(args) -> int:
    started = time.monotonic()
    rc = cmd_ingest(args)
    if rc:
        return rc
    cmd_price(args)
    rc = cmd_verify(args)
    log.info("pipeline finished in %.1f min", (time.monotonic() - started) / 60)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dex_data", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_ingest_args(p):
        p.add_argument("--days", type=int, default=None,
                       help=f"lookback window in days (default {CONFIG.days})")
        p.add_argument("--from-block", type=int, default=None)
        p.add_argument("--to-block", type=int, default=None)
        p.add_argument("--no-resume", action="store_true",
                       help="reprocess chunks already marked done")

    def add_verify_args(p):
        p.add_argument("--min-ratio", type=float, default=1.0,
                       help="fail if fewer than this share of swaps are paired "
                            "(zero-amount swaps are excluded, so 1.0 is achievable)")

    sub.add_parser("initdb").set_defaults(func=cmd_initdb)

    p_reset = sub.add_parser("reset")
    p_reset.add_argument("--yes", action="store_true")
    p_reset.add_argument("--all", action="store_true",
                         help="also truncate pools and tokens")
    p_reset.set_defaults(func=cmd_reset)

    p_ingest = sub.add_parser("ingest")
    add_ingest_args(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    p_price = sub.add_parser(
        "price", help="CoinGecko hourly prices -> token_prices_hourly -> usd_amount")
    p_price.add_argument("--table", choices=pricing.PRICEABLE_TABLES, default="dex_events",
                         help="table whose usd_amount is cleared and refilled")
    p_price.add_argument("--no-fetch", action="store_true",
                         help="reuse token_prices_hourly as it is; skip CoinGecko")
    p_price.add_argument("--fill-only", action="store_true",
                         help="only carry values into still-empty rows along each "
                              "transaction's legs; keeps existing usd_amounts")
    p_price.set_defaults(func=cmd_price)

    sub.add_parser("marketcap").set_defaults(func=cmd_marketcap)

    sub.add_parser(
        "cleanup", help="move standalone stable/non-target swaps to dex_events_removed"
    ).set_defaults(func=cmd_cleanup)

    p_target = sub.add_parser(
        "target", help="pin the modelling universe (run once)")
    p_target.add_argument("--min-market-cap", type=float, default=None,
                          help=f"pin every token above this cap "
                               f"(default ${CONFIG.min_market_cap:,.0f})")
    p_target.add_argument("--top", type=int, default=None,
                          help="pin the top N tokens by market cap instead")
    p_target.add_argument("--force", action="store_true",
                          help="re-pin an existing universe; datasets built "
                               "before and after will not be comparable")
    p_target.set_defaults(func=cmd_target)


    p_verify = sub.add_parser("verify")
    add_verify_args(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    p_all = sub.add_parser("all")
    add_ingest_args(p_all)
    add_verify_args(p_all)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    try:
        return args.func(args)
    finally:
        db.close_pool()


if __name__ == "__main__":
    raise SystemExit(main())

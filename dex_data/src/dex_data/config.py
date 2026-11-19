"""Environment-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_RPC_URLS = (
    "https://eth.llamarpc.com,"
    "https://ethereum-rpc.publicnode.com,"
    "https://eth.drpc.org"
)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    rpc_urls: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            u.strip() for u in os.getenv("DEX_RPC_URLS", DEFAULT_RPC_URLS).split(",") if u.strip()
        )
    )
    archive: bool = field(default_factory=lambda: _bool("DEX_ARCHIVE", False))
    pg_dsn: str = field(
        default_factory=lambda: os.getenv(
            "DEX_PG_DSN", "postgresql://dex:dex@localhost:5433/dex_data"
        )
    )
    days: int = field(default_factory=lambda: _int("DEX_DAYS", 30))
    max_workers: int = field(default_factory=lambda: _int("DEX_MAX_WORKERS", 4))
    log_chunk_blocks: int = field(default_factory=lambda: _int("DEX_LOG_CHUNK_BLOCKS", 20))
    rpc_timeout: int = field(default_factory=lambda: _int("DEX_RPC_TIMEOUT", 60))
    rpc_max_retries: int = field(default_factory=lambda: _int("DEX_RPC_MAX_RETRIES", 8))
    rpc_rate_limit: float = field(default_factory=lambda: _float("DEX_RPC_RATE_LIMIT", 8.0))
    # Market cap above which a token is modelled by name rather than folded into
    # the anonymous "unknown" bucket. See dataset.py.
    min_market_cap: float = field(
        default_factory=lambda: _float("DEX_MIN_MARKET_CAP", 1e8)
    )
    # Below this chunk size a "response too large" error is unrecoverable rather
    # than something to keep splitting.
    min_chunk_blocks: int = 1

    def __post_init__(self) -> None:
        if not self.rpc_urls:
            raise ValueError("DEX_RPC_URLS is empty; at least one endpoint is required")


CONFIG = Config()

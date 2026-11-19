"""Postgres access: pooling, schema bootstrap and bulk upserts."""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool

from .config import CONFIG, Config

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()

# Test-only hook (see tests/conftest.py). Pointing every connection at a
# throwaway schema used to be done by exporting PGOPTIONS before opening the
# pool, but that relied on the pool's background worker threads not having
# opened a connection yet -- a race that silently lost the race at least once
# and pointed live test writes at `public`. Baking the schema into the pool's
# own `configure` hook removes the timing dependency: every connection this
# pool ever opens is forced onto the override schema by the pool itself,
# synchronously, no matter when that happens.
_schema_override: str | None = None


def set_schema_override(schema: str | None) -> None:
    """Force every future connection onto `schema` (test isolation only).

    Must be called before the pool is (re)opened -- pair with `close_pool()`.
    """
    global _schema_override
    _schema_override = schema


def get_schema_override() -> str | None:
    return _schema_override


def get_pool(config: Config = CONFIG) -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                schema = _schema_override

                def _configure(conn: psycopg.Connection) -> None:
                    if schema is not None:
                        conn.execute(
                            sql.SQL("SET search_path TO {}, public").format(
                                sql.Identifier(schema)
                            )
                        )
                        conn.commit()

                _pool = ConnectionPool(
                    config.pg_dsn,
                    min_size=2,
                    # Each worker thread holds one connection while flushing.
                    max_size=max(4, config.max_workers + 4),
                    open=True,
                    timeout=60,
                    configure=_configure,
                )
    return _pool


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


def init_schema() -> None:
    ddl = SCHEMA_PATH.read_text()
    with connection() as conn:
        conn.execute(ddl)
        conn.commit()
    log.info("schema ready")


def reset_data(tables: Sequence[str] | None = None) -> None:
    """Truncate ingest output. Reference tables are kept unless asked for."""
    tables = tables or [
        "dex_events",
        "dex_events_removed",
        "token_prices_hourly",
        "ingest_progress",
        "blocks",
    ]
    with connection() as conn:
        conn.execute(
            sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(
                sql.SQL(", ").join(sql.Identifier(t) for t in tables)
            )
        )
        conn.commit()
    log.info("truncated: %s", ", ".join(tables))


# ---------------------------------------------------------------------------
# Bulk upserts
#
# COPY is the fast path but cannot express ON CONFLICT, and re-runs must stay
# idempotent. So: COPY into an UNLOGGED temp table, then a single INSERT ... 
# SELECT ... ON CONFLICT DO NOTHING. That keeps COPY's throughput and the
# upsert's safety.
# ---------------------------------------------------------------------------


def _copy_upsert(
    conn: psycopg.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str] = (),
) -> int:
    rows = list(rows)
    if not rows:
        return 0

    temp = f"_stage_{table}"
    col_list = sql.SQL(", ").join(sql.Identifier(c) for c in columns)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE TEMP TABLE {} (LIKE {} INCLUDING DEFAULTS) ON COMMIT DROP"
            ).format(sql.Identifier(temp), sql.Identifier(table))
        )
        copy_stmt = sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(temp), col_list)
        with cur.copy(copy_stmt) as copy:
            for row in rows:
                copy.write_row(row)

        if update_cols:
            action = sql.SQL("DO UPDATE SET {}").format(
                sql.SQL(", ").join(
                    sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in update_cols
                )
            )
        else:
            action = sql.SQL("DO NOTHING")

        cur.execute(
            sql.SQL(
                "INSERT INTO {table} ({cols}) SELECT {cols} FROM {temp} "
                "ON CONFLICT ({conflict}) {action}"
            ).format(
                table=sql.Identifier(table),
                cols=col_list,
                temp=sql.Identifier(temp),
                conflict=sql.SQL(", ").join(sql.Identifier(c) for c in conflict_cols),
                action=action,
            )
        )
        return cur.rowcount


def upsert(
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str] = (),
) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with connection() as conn:
        n = _copy_upsert(conn, table, columns, rows, conflict_cols, update_cols)
        conn.commit()
        return n


DEX_EVENT_COLUMNS = (
    "transaction_id", "type", "source", "destination", "token", "usd_amount",
    "timestamp", "block_number", "tx_hash", "log_index", "token_address",
    "amount", "amount_raw", "pool", "counterpart_expected",
)


def insert_events(conn: psycopg.Connection, rows: Iterable[dict]) -> int:
    """Bulk-insert dex_events rows, skipping ones already present.

    (tx_hash, log_index) is the chain's own identity for a log, so a re-run of
    the same block range is a no-op rather than a duplicate.
    """
    return _copy_upsert(
        conn, "dex_events", DEX_EVENT_COLUMNS,
        ([r[c] for c in DEX_EVENT_COLUMNS] for r in rows),
        ("transaction_id",),
    )


def mark_chunk_done(conn: psycopg.Connection, job: str, lo: int, hi: int, n_logs: int) -> None:
    conn.execute(
        "INSERT INTO ingest_progress (job, from_block, to_block, status, n_logs, updated_at) "
        "VALUES (%s, %s, %s, 'done', %s, now()) "
        "ON CONFLICT (job, from_block) DO UPDATE SET "
        "to_block = EXCLUDED.to_block, status = 'done', "
        "n_logs = EXCLUDED.n_logs, updated_at = now()",
        (job, lo, hi, n_logs),
    )


def completed_chunks(job: str) -> set[int]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT from_block FROM ingest_progress WHERE job = %s AND status = 'done'",
            (job,),
        ).fetchall()
    return {r[0] for r in rows}


def fetch_all(query: str, params: Sequence | None = None) -> list[tuple]:
    with connection() as conn:
        return conn.execute(query, params).fetchall()


def fetch_one(query: str, params: Sequence | None = None):
    with connection() as conn:
        return conn.execute(query, params).fetchone()


def execute(query: str, params: Sequence | None = None) -> int:
    with connection() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.rowcount

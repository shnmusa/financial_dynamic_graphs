"""Test fixtures.

DB-backed tests run against a throwaway schema inside the real Postgres, not a
mock. That matters here: the projection from raw logs into dex_events is pure
SQL, so testing it against anything other than Postgres would be testing a
reimplementation rather than the code that actually ships.

Isolation is done via `db.set_schema_override`, which bakes the throwaway
schema into the connection pool's own `configure` hook (see db.py) rather than
exporting PGOPTIONS and hoping the pool opens its connections afterwards --
that env-var approach lost that race at least once and let a test write
straight into `public`, wiping real ingested data. `clean_db` also asserts
`current_schema()` right before truncating, as a second, independent guard:
even if isolation breaks again in some new way, that assertion refuses to
touch anything that isn't the throwaway schema.
"""
from __future__ import annotations

import os
import uuid

import pytest

TEST_SCHEMA = f"dex_test_{uuid.uuid4().hex[:10]}"


@pytest.fixture(scope="session")
def pg_schema():
    """Create an isolated schema, point every connection at it, drop it after."""
    dsn = os.environ.get("DEX_PG_DSN", "postgresql://dex:dex@localhost:5433/dex_data")

    try:
        import psycopg
    except ImportError:  # pragma: no cover
        pytest.skip("psycopg is not installed")

    try:
        admin = psycopg.connect(dsn, connect_timeout=5)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable at {dsn}: {exc}")

    with admin:
        admin.execute(f'CREATE SCHEMA "{TEST_SCHEMA}"')
        admin.commit()

    os.environ["DEX_PG_DSN"] = dsn

    from dex_data import db

    # Must happen before the pool (re)opens -- it decides how every
    # connection this pool creates gets configured.
    db.set_schema_override(TEST_SCHEMA)
    db.close_pool()
    db.init_schema()

    yield TEST_SCHEMA

    db.close_pool()
    db.set_schema_override(None)

    with psycopg.connect(dsn, connect_timeout=5) as admin:
        admin.execute(f'DROP SCHEMA IF EXISTS "{TEST_SCHEMA}" CASCADE')
        admin.commit()


@pytest.fixture
def clean_db(pg_schema):
    """Empty every table before each DB test."""
    from dex_data import db

    with db.connection() as conn:
        # Independent of set_schema_override: even if that mechanism ever
        # regresses, this refuses to truncate anything but the throwaway
        # schema -- it must never be able to reach the real `public` data.
        current = conn.execute("SELECT current_schema()").fetchone()[0]
        assert current == pg_schema, (
            f"test isolation broken: current_schema() is {current!r}, expected "
            f"the throwaway schema {pg_schema!r} -- refusing to truncate"
        )
        conn.execute(
            "TRUNCATE dex_events, dex_events_removed, token_prices_hourly, ingest_progress, "
            "blocks, pools, tokens RESTART IDENTITY CASCADE"
        )
        conn.commit()
    return db


@pytest.fixture
def live_db():
    """For checks against real ingested data; skips when there is none.

    The isolated test schema must be off the connection *before* probing,
    otherwise this sees the synthetic rows the DB tests just created rather than
    what was actually ingested.
    """
    import psycopg

    from dex_data import db

    saved = db.get_schema_override()
    db.set_schema_override(None)
    db.close_pool()

    dsn = os.environ.get("DEX_PG_DSN", "postgresql://dex:dex@localhost:5433/dex_data")
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            row = conn.execute("SELECT count(*) FROM dex_events").fetchone()
        has_data = bool(row and row[0])
        reason = "dex_events is empty; run `python -m dex_data.cli all` first"
    except psycopg.errors.UndefinedTable:
        has_data, reason = False, "dex_events does not exist yet; run `dex_data.cli initdb`"
    except Exception as exc:  # noqa: BLE001
        has_data, reason = False, f"no live database: {exc}"

    if not has_data:
        db.close_pool()
        db.set_schema_override(saved)
        pytest.skip(reason)

    yield db

    db.close_pool()
    db.set_schema_override(saved)

"""Create the schema and (re)load the fake data into ClickHouse Cloud.

Idempotent: drop the three tables, recreate them from schema.sql, insert the generated rows. Column
lists come from `generate_data` so the insert order always matches `schema.sql`
(campaigns carry country/objective; ad_spend and purchases carry device).
Run with `uv run --extra seed python warehouse/seed.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import clickhouse_connect
from dotenv import load_dotenv

from warehouse.generate_data import AD_SPEND_COLUMNS, CAMPAIGN_COLUMNS, PURCHASE_COLUMNS, build_rows

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
TABLES = {"campaigns": CAMPAIGN_COLUMNS, "ad_spend": AD_SPEND_COLUMNS, "purchases": PURCHASE_COLUMNS}


def connect():
    """Build a clickhouse-connect client from CLICKHOUSE_* environment variables."""
    load_dotenv()
    host, password = os.getenv("CLICKHOUSE_HOST"), os.getenv("CLICKHOUSE_PASSWORD")
    if not host or not password:
        sys.exit("seed: CLICKHOUSE_HOST and CLICKHOUSE_PASSWORD must be set (see .env.example)")
    return clickhouse_connect.get_client(
        host=host,
        port=int(os.getenv("CLICKHOUSE_PORT", "8443")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=password,
        secure=os.getenv("CLICKHOUSE_SECURE", "true").lower() == "true",
    )


def main() -> None:
    client = connect()
    db = "marketing"  # the database named in schema.sql

    # Rebuild from scratch: the mock warehouse is fully owned by this script, and
    # CREATE TABLE IF NOT EXISTS would silently keep an outdated schema.
    for table in TABLES:
        client.command(f"DROP TABLE IF EXISTS {db}.{table}")
    for statement in SCHEMA_PATH.read_text().split(";"):
        if statement.strip():
            client.command(statement)

    rows = build_rows()
    for table, columns in TABLES.items():
        client.insert(f"{db}.{table}", rows[table], column_names=columns)
        print(f"{db}.{table}: {client.command(f'SELECT count() FROM {db}.{table}')} rows")


if __name__ == "__main__":
    main()

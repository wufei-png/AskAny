#!/usr/bin/env python3
"""Check insertion progress for docs vector store.

This script queries the database to show current progress of node insertion.
Useful when insert_nodes() is running without batch_size and you want to monitor progress.
"""

import os
import sys
import time
from pathlib import Path

import psycopg2

# Try to import settings, but fall back to environment variables
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from askany.config import settings

    TABLE_NAME = f"data_{settings.docs_vector_table_name}"
    DB_URL = settings.database_url
except Exception:
    # Fall back to environment variables or defaults
    TABLE_NAME = os.getenv("DOCS_VECTOR_TABLE", "data_askany2_docs_vectors")
    DB_URL = os.getenv("DATABASE_URL", None)


def check_progress():
    """Check current insertion progress."""
    # Get database connection info
    db_url = DB_URL

    # Parse connection string or use individual settings
    # Format: postgresql://user:password@host:port/database
    if db_url.startswith("postgresql://"):
        # Parse URL
        url = db_url.replace("postgresql://", "")
        if "@" in url:
            auth, rest = url.split("@", 1)
            if ":" in auth:
                user, password = auth.split(":", 1)
            else:
                user = auth
                password = None
            if "/" in rest:
                host_port, database = rest.split("/", 1)
                if ":" in host_port:
                    host, port = host_port.split(":", 1)
                    port = int(port)
                else:
                    host = host_port
                    port = 5432
            else:
                host = rest
                port = 5432
                database = None
        else:
            user = None
            password = None
            host = "localhost"
            port = 5432
            database = None
    else:
        # Use environment variables or defaults
        user = os.getenv("PGUSER", "root")
        password = os.getenv("PGPASSWORD", "123456")
        host = os.getenv("PGHOST", "localhost")
        port = int(os.getenv("PGPORT", "5432"))
        database = os.getenv("PGDATABASE", "askany")

    # Connect to database
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        cur = conn.cursor()

        # Get table name
        table_name = TABLE_NAME

        # Query current row count
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        current_rows = cur.fetchone()[0]

        # Query table size
        cur.execute(
            f"""
            SELECT pg_size_pretty(pg_relation_size('{table_name}'::regclass)) as table_size
            """
        )
        table_size = cur.fetchone()[0] if cur.rowcount > 0 else "N/A"

        # Query index size (if exists)
        try:
            index_name = f"{table_name}_embedding_idx"
            cur.execute(
                f"""
                SELECT pg_size_pretty(pg_relation_size('{index_name}'::regclass)) as index_size
                """
            )
            index_size = cur.fetchone()[0] if cur.rowcount > 0 else "N/A"
        except Exception:
            index_size = "N/A"

        # Query recent insertions (last 5 minutes)
        cur.execute(
            f"""
            SELECT COUNT(*) 
            FROM {table_name} 
            WHERE id > (
                SELECT COALESCE(MAX(id) - 1000, 0) 
                FROM {table_name}
            );
            """
        )
        recent_rows = cur.fetchone()[0]

        # Get max ID to see if insertion is progressing
        cur.execute(f"SELECT MAX(id) FROM {table_name};")
        max_id = cur.fetchone()[0] or 0

        print(f"\n{'=' * 60}")
        print(f"Insertion Progress for {table_name}")
        print(f"{'=' * 60}")
        print(f"Current rows: {current_rows:,}")
        print(f"Max ID: {max_id:,}")
        print(f"Recent rows (last ~1000): {recent_rows:,}")
        print(f"Table size: {table_size}")
        print(f"Index size: {index_size}")
        print(f"{'=' * 60}\n")

        cur.close()
        conn.close()

        return current_rows, max_id

    except Exception as e:
        print(f"Error checking progress: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def monitor_progress(interval=10):
    """Monitor progress continuously."""
    print(f"Monitoring insertion progress (checking every {interval} seconds)...")
    print("Press Ctrl+C to stop\n")

    prev_rows = None
    prev_max_id = None
    start_time = time.time()

    try:
        while True:
            current_rows, max_id = check_progress()

            if current_rows is not None:
                if prev_rows is not None:
                    rows_diff = current_rows - prev_rows
                    id_diff = max_id - prev_max_id if max_id and prev_max_id else 0
                    elapsed = time.time() - start_time

                    if rows_diff > 0:
                        rate = rows_diff / interval if interval > 0 else 0
                        print(
                            f"Progress: +{rows_diff:,} rows in last {interval}s ({rate:.1f} rows/sec)"
                        )
                        if id_diff > 0:
                            print(f"         Max ID increased by {id_diff:,}")
                    else:
                        print(
                            "Progress: No new rows detected (insertion may be stuck or completed)"
                        )

                prev_rows = current_rows
                prev_max_id = max_id

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check insertion progress for docs vector store"
    )
    parser.add_argument(
        "--monitor",
        "-m",
        action="store_true",
        help="Monitor progress continuously",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=10,
        help="Monitoring interval in seconds (default: 10)",
    )

    args = parser.parse_args()

    if args.monitor:
        monitor_progress(args.interval)
    else:
        check_progress()

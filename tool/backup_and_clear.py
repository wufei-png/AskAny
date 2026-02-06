#!/usr/bin/env python3
"""Backup original tables to test tables, clear original tables, then import."""

import logging
import sys
from pathlib import Path

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


def backup_table_to_test(original_table: str, test_table: str) -> bool:
    """Copy original table to test table."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )
        cur = conn.cursor()

        # Check if original table exists
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
            """,
            (original_table,),
        )
        exists = cur.fetchone()[0]

        if not exists:
            logger.warning(f"⚠️  Table {original_table} does not exist, skipping backup")
            cur.close()
            conn.close()
            return True

        # Drop test table if exists
        cur.execute(f"DROP TABLE IF EXISTS {test_table} CASCADE;")
        conn.commit()

        # Copy table structure and data
        logger.info(f"📋 Copying {original_table} to {test_table}...")
        cur.execute(
            f"""
            CREATE TABLE {test_table} (LIKE {original_table} INCLUDING ALL);
            """
        )
        conn.commit()

        cur.execute(
            f"""
            INSERT INTO {test_table}
            SELECT * FROM {original_table};
            """
        )
        conn.commit()

        # Copy sequence if exists
        sequence_name = f"{original_table}_id_seq"
        test_sequence_name = f"{test_table}_id_seq"
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM pg_sequences 
                WHERE schemaname = 'public' 
                AND sequencename = %s
            );
            """,
            (sequence_name,),
        )
        seq_exists = cur.fetchone()[0]

        if seq_exists:
            # Get current sequence value
            cur.execute(f"SELECT last_value FROM {sequence_name};")
            last_value = cur.fetchone()[0]

            # Create new sequence for test table
            cur.execute(
                f"""
                CREATE SEQUENCE IF NOT EXISTS {test_sequence_name}
                START WITH {last_value + 1};
                """
            )
            conn.commit()

            # Set the sequence to the same value
            cur.execute(f"SELECT setval('{test_sequence_name}', {last_value}, true);")
            conn.commit()

        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM {test_table};")
        count = cur.fetchone()[0]

        logger.info(f"✅ Copied {count} rows from {original_table} to {test_table}")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Error backing up table {original_table}: {e}")
        import traceback

        traceback.print_exc()
        return False


def clear_table(table_name: str) -> bool:
    """Clear all data from table."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )
        cur = conn.cursor()

        # Check if table exists
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
            """,
            (table_name,),
        )
        exists = cur.fetchone()[0]

        if not exists:
            logger.warning(f"⚠️  Table {table_name} does not exist, skipping clear")
            cur.close()
            conn.close()
            return True

        # Get row count before clearing
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        count_before = cur.fetchone()[0]

        # Clear table
        logger.info(f"🗑️  Clearing {table_name}...")
        cur.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
        conn.commit()

        # Verify
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        count_after = cur.fetchone()[0]

        logger.info(
            f"✅ Cleared {count_before} rows from {table_name} (now has {count_after} rows)"
        )
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Error clearing table {table_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("Backup original tables to test tables")
    logger.info("=" * 80)

    # Get table names
    tables = [
        ("FAQ", f"data_{settings.faq_vector_table_name}"),
        ("Docs", f"data_{settings.docs_vector_table_name}"),
    ]

    # Step 1: Backup tables
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Backing up original tables to test tables...")
    logger.info("=" * 80)

    for table_type, original_table in tables:
        test_table = f"{original_table}_test"
        logger.info(
            f"\nBacking up {table_type} table: {original_table} -> {test_table}"
        )
        if not backup_table_to_test(original_table, test_table):
            logger.error(f"❌ Failed to backup {table_type} table")
            return

    # Step 2: Clear original tables
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Clearing original tables...")
    logger.info("=" * 80)

    for table_type, original_table in tables:
        logger.info(f"\nClearing {table_type} table: {original_table}")
        if not clear_table(original_table):
            logger.error(f"❌ Failed to clear {table_type} table")
            return

    logger.info("\n" + "=" * 80)
    logger.info("✅ Backup and clear completed!")
    logger.info("=" * 80)
    logger.info(
        "\nNext step: Run import_vector_data.py to import data to original tables"
    )
    logger.info("Then run compare_table_data.py to compare the data")


if __name__ == "__main__":
    main()

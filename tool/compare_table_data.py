#!/usr/bin/env python3
"""Compare data between original tables and test tables."""

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


def compare_tables(original_table: str, test_table: str) -> bool:
    """Compare data between two tables."""
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

        # Check if both tables exist
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
        original_exists = cur.fetchone()[0]

        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
            """,
            (test_table,),
        )
        test_exists = cur.fetchone()[0]

        if not original_exists:
            logger.error(f"❌ Original table {original_table} does not exist")
            return False

        if not test_exists:
            logger.error(f"❌ Test table {test_table} does not exist")
            return False

        # Get row counts
        cur.execute(f"SELECT COUNT(*) FROM {original_table};")
        original_count = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM {test_table};")
        test_count = cur.fetchone()[0]

        logger.info(f"📊 {original_table}: {original_count} rows")
        logger.info(f"📊 {test_table}: {test_count} rows")

        if original_count != test_count:
            logger.warning(
                f"⚠️  Row count mismatch: {original_table} has {original_count} rows, {test_table} has {test_count} rows"
            )
            return False

        # Compare data row by row
        logger.info(f"🔍 Comparing data between {original_table} and {test_table}...")

        # Get all rows from both tables ordered by id
        cur.execute(
            f"""
            SELECT id, text, metadata_::text, node_id, embedding::text
            FROM {original_table}
            ORDER BY id;
            """
        )
        original_rows = cur.fetchall()

        cur.execute(
            f"""
            SELECT id, text, metadata_::text, node_id, embedding::text
            FROM {test_table}
            ORDER BY id;
            """
        )
        test_rows = cur.fetchall()

        if len(original_rows) != len(test_rows):
            logger.error(
                f"❌ Row count mismatch after fetching: {len(original_rows)} vs {len(test_rows)}"
            )
            return False

        differences = []
        for i, (orig_row, test_row) in enumerate(zip(original_rows, test_rows)):
            if orig_row != test_row:
                differences.append(
                    {
                        "index": i,
                        "id": orig_row[0],
                        "original": orig_row,
                        "test": test_row,
                    }
                )

        if differences:
            logger.error(f"❌ Found {len(differences)} differences:")
            for diff in differences[:10]:  # Show first 10 differences
                logger.error(f"  Row {diff['index']} (id={diff['id']}):")
                logger.error(f"    Original: {diff['original']}")
                logger.error(f"    Test: {diff['test']}")
            if len(differences) > 10:
                logger.error(f"  ... and {len(differences) - 10} more differences")
            return False
        else:
            logger.info(f"✅ All {len(original_rows)} rows match perfectly!")
            return True

        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Error comparing tables: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("Comparing table data")
    logger.info("=" * 80)

    # Get table names
    tables = [
        (
            "FAQ",
            f"data_{settings.faq_vector_table_name}",
            f"data_{settings.faq_vector_table_name}_test",
        ),
        (
            "Docs",
            f"data_{settings.docs_vector_table_name}",
            f"data_{settings.docs_vector_table_name}_test",
        ),
    ]

    all_match = True
    for table_type, original_table, test_table in tables:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Comparing {table_type} tables:")
        logger.info(f"  Original: {original_table}")
        logger.info(f"  Test: {test_table}")
        logger.info("=" * 80)

        if compare_tables(original_table, test_table):
            logger.info(f"✅ {table_type} tables match!")
        else:
            logger.error(f"❌ {table_type} tables do not match!")
            all_match = False

    logger.info("\n" + "=" * 80)
    if all_match:
        logger.info("✅ All tables match perfectly!")
    else:
        logger.error("❌ Some tables have differences!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

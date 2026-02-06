#!/usr/bin/env python3
"""Delete migrated data from data_askany_docs_vectors (MOVE operation).

This script deletes data where id > 3720 that has already been migrated to data_hybrid_askany_docs_vectors.
WARNING: This will permanently delete data from data_askany_docs_vectors.
Only execute this after verifying the migration was successful.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psycopg2
    from psycopg2 import sql

    PSYCOPG2_AVAILABLE = True
except ImportError:
    print(
        "⚠️  Warning: psycopg2 not available. Install it with: pip install psycopg2-binary"
    )
    PSYCOPG2_AVAILABLE = False

# Try to import settings, but provide defaults if not available
try:
    from askany.config import settings
except (ImportError, ModuleNotFoundError):
    # Provide defaults if askany module is not available
    import os

    class Settings:
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
        postgres_user = os.getenv("POSTGRES_USER", "root")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "123456")
        postgres_db = os.getenv("POSTGRES_DB", "askany")

    settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


def delete_migrated_data(
    source_table: str,
    target_table: str,
    cutoff_id: int = 3720,
    db_host: str = None,
    db_port: int = None,
    db_user: str = None,
    db_password: str = None,
    db_name: str = None,
    dry_run: bool = False,
) -> bool:
    """Delete migrated data from source table.

    Args:
        source_table: Source table name (e.g., 'data_askany_docs_vectors')
        target_table: Target table name (e.g., 'data_hybrid_askany_docs_vectors')
        cutoff_id: ID cutoff (data with id > this will be deleted)
        dry_run: If True, only show what would be deleted without actually deleting

    Returns:
        True if successful, False otherwise
    """
    if not PSYCOPG2_AVAILABLE:
        logger.error("❌ Cannot delete: psycopg2 not available")
        return False

    # Use provided parameters or fall back to settings
    db_host = db_host or settings.postgres_host
    db_port = db_port or settings.postgres_port
    db_user = db_user or settings.postgres_user
    db_password = db_password or settings.postgres_password
    db_name = db_name or settings.postgres_db

    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
        )
        cur = conn.cursor()

        # Check if target table exists and has data
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
            """,
            (target_table,),
        )
        target_exists = cur.fetchone()[0]

        if not target_exists:
            logger.error(f"❌ Target table {target_table} does not exist")
            logger.error("   Migration may not have been completed yet.")
            cur.close()
            conn.close()
            return False

        # Check how many rows exist in target table with id > cutoff_id
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*) FROM {}
                WHERE id > %s
                """
            ).format(sql.Identifier(target_table)),
            (cutoff_id,),
        )
        target_count = cur.fetchone()[0]

        if target_count == 0:
            logger.warning(
                f"⚠️  Target table {target_table} has no rows with id > {cutoff_id}"
            )
            logger.warning(
                "   Nothing to delete. Migration may not have been completed."
            )
            cur.close()
            conn.close()
            return False

        # Check how many rows will be deleted from source table
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*) FROM {}
                WHERE id > %s
                """
            ).format(sql.Identifier(source_table)),
            (cutoff_id,),
        )
        rows_to_delete = cur.fetchone()[0]

        logger.info(
            f"📊 Target table {target_table} has {target_count} rows with id > {cutoff_id}"
        )
        logger.info(
            f"📊 Source table {source_table} has {rows_to_delete} rows with id > {cutoff_id}"
        )

        if rows_to_delete == 0:
            logger.info("✅ No rows to delete. Data may have already been deleted.")
            cur.close()
            conn.close()
            return True

        # Show sample rows that will be deleted
        cur.execute(
            sql.SQL(
                """
                SELECT id, metadata_->>'last_updated' as last_updated, LEFT(text, 50) as text_preview
                FROM {}
                WHERE id > %s
                LIMIT 5
                """
            ).format(sql.Identifier(source_table)),
            (cutoff_id,),
        )
        sample_rows = cur.fetchall()

        logger.info("📋 Sample rows that will be deleted:")
        for row in sample_rows:
            row_id, last_updated, text_preview = row
            logger.info(
                f"  - ID: {row_id}, Last Updated: {last_updated}, "
                f"Text: {text_preview[:50] if text_preview else 'N/A'}..."
            )

        if dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")
            logger.info(f"   Would delete {rows_to_delete} rows from {source_table}")
            cur.close()
            conn.close()
            return True

        # Confirm deletion
        logger.warning(
            "⚠️  WARNING: This will permanently delete data from the source table!"
        )
        response = input(
            f"⚠️  Are you sure you want to delete {rows_to_delete} rows from {source_table}? (yes/no): "
        )
        if response.lower() not in ["yes", "y"]:
            logger.info("❌ Deletion cancelled.")
            cur.close()
            conn.close()
            return False

        # Delete from source table
        logger.info(f"🗑️  Deleting {rows_to_delete} rows from {source_table}...")
        cur.execute(
            sql.SQL(
                """
                DELETE FROM {}
                WHERE id > %s
                """
            ).format(sql.Identifier(source_table)),
            (cutoff_id,),
        )
        rows_deleted = cur.rowcount
        conn.commit()
        logger.info(f"✅ Deleted {rows_deleted} rows from {source_table}")

        # Verify deletion
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(source_table))
        )
        remaining_count = cur.fetchone()[0]
        logger.info(
            f"📊 Source table {source_table} now has {remaining_count} rows remaining"
        )

        cur.close()
        conn.close()

        logger.info("✅ Deletion completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Error during deletion: {e}", exc_info=True)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Delete migrated data from source table (MOVE operation)"
    )
    parser.add_argument(
        "--source-table",
        type=str,
        default="data_askany_docs_vectors",
        help="Source table name (default: data_askany_docs_vectors)",
    )
    parser.add_argument(
        "--target-table",
        type=str,
        default="data_hybrid_askany_docs_vectors",
        help="Target table name (default: data_hybrid_askany_docs_vectors)",
    )
    parser.add_argument(
        "--cutoff-id",
        type=int,
        default=3720,
        help="Cutoff ID (delete rows with ID > this value, default: 3720)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="Database host (default: from config or localhost)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="Database port (default: from config or 5432)",
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default=None,
        help="Database user (default: from config or root)",
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default=None,
        help="Database password (default: from config or 123456)",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="Database name (default: from config or askany)",
    )

    args = parser.parse_args()

    success = delete_migrated_data(
        args.source_table,
        args.target_table,
        cutoff_id=args.cutoff_id,
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        dry_run=args.dry_run,
    )

    if success:
        logger.info("✅ Operation completed successfully!")
    else:
        logger.error("❌ Operation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

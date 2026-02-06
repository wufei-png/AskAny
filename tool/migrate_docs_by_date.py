#!/usr/bin/env python3
"""Migrate docs vectors data from data_askany_docs_vectors to hybrid_askany_docs_vectors based on date.

This script:
1. Creates a new table data_hybrid_askany_docs_vectors (PGVectorStore adds 'data_' prefix)
2. Migrates data from data_askany_docs_vectors where last_updated >= December 1, 2024
3. Creates the same indexes and sequences
"""

import argparse
import logging
import sys
from datetime import datetime
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


def get_december_timestamp(year: int = 2024, month: int = 12, day: int = 1) -> float:
    """Get Unix timestamp for December 1st of the specified year."""
    dt = datetime(year, month, day)
    return dt.timestamp()


def create_hybrid_table(
    source_table: str,
    target_table: str,
    cutoff_timestamp: float = None,
    cutoff_id: int = None,
    use_id_range: bool = False,
    delete_source: bool = False,
    db_host: str = None,
    db_port: int = None,
    db_user: str = None,
    db_password: str = None,
    db_name: str = None,
) -> bool:
    """Create hybrid table and migrate data from source table.

    Args:
        source_table: Source table name (e.g., 'data_askany_docs_vectors')
        target_table: Target table name (e.g., 'data_hybrid_askany_docs_vectors')
        cutoff_timestamp: Timestamp cutoff (data with last_updated >= this will be migrated)
        cutoff_id: ID cutoff (data with id > this will be migrated)
        use_id_range: Whether to use ID range instead of timestamp
        delete_source: If True, delete migrated data from source table (MOVE operation)

    Returns:
        True if successful, False otherwise
    """
    if not PSYCOPG2_AVAILABLE:
        logger.error("❌ Cannot migrate: psycopg2 not available")
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

        # Check if source table exists
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
            """,
            (source_table,),
        )
        exists = cur.fetchone()[0]

        if not exists:
            logger.error(f"❌ Source table {source_table} does not exist")
            cur.close()
            conn.close()
            return False

        # Check how many rows will be migrated
        if use_id_range and cutoff_id is not None:
            cur.execute(
                sql.SQL(
                    """
                    SELECT COUNT(*) FROM {}
                    WHERE id > %s
                    """
                ).format(sql.Identifier(source_table)),
                (cutoff_id,),
            )
            rows_to_migrate = cur.fetchone()[0]
            logger.info(
                f"📊 Found {rows_to_migrate} rows to migrate (id > {cutoff_id})"
            )
            where_clause = sql.SQL("id > %s")
            where_params = (cutoff_id,)
        elif cutoff_timestamp is not None:
            cur.execute(
                sql.SQL(
                    """
                    SELECT COUNT(*) FROM {}
                    WHERE (metadata_->>'last_updated')::float >= %s
                    """
                ).format(sql.Identifier(source_table)),
                (cutoff_timestamp,),
            )
            rows_to_migrate = cur.fetchone()[0]
            logger.info(
                f"📊 Found {rows_to_migrate} rows to migrate (last_updated >= {cutoff_timestamp})"
            )
            where_clause = sql.SQL("(metadata_->>'last_updated')::float >= %s")
            where_params = (cutoff_timestamp,)
        else:
            logger.error("❌ Either cutoff_timestamp or cutoff_id must be provided")
            cur.close()
            conn.close()
            return False

        if rows_to_migrate == 0:
            logger.warning("⚠️  No rows to migrate. Exiting.")
            cur.close()
            conn.close()
            return True

        # Check if target table already exists
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

        if target_exists:
            logger.warning(f"⚠️  Target table {target_table} already exists")
            response = input("Do you want to drop and recreate it? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                logger.info("❌ Operation cancelled.")
                cur.close()
                conn.close()
                return False
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(
                    sql.Identifier(target_table)
                )
            )
            conn.commit()
            logger.info(f"🗑️  Dropped existing table {target_table}")

        # Create target table with same structure as source
        logger.info(f"📋 Creating table {target_table}...")
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE {} (LIKE {} INCLUDING ALL);
                """
            ).format(sql.Identifier(target_table), sql.Identifier(source_table))
        )
        conn.commit()
        logger.info(f"✅ Created table {target_table}")

        # Migrate data
        logger.info(f"📦 Migrating data from {source_table} to {target_table}...")
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {}
                SELECT * FROM {}
                WHERE {}
                """
            ).format(
                sql.Identifier(target_table),
                sql.Identifier(source_table),
                where_clause,
            ),
            where_params,
        )
        rows_migrated = cur.rowcount
        conn.commit()
        logger.info(f"✅ Migrated {rows_migrated} rows to {target_table}")

        # Verify migration
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(target_table))
        )
        count = cur.fetchone()[0]
        logger.info(f"📊 Target table {target_table} now has {count} rows")

        # Copy sequence if exists
        source_sequence = f"{source_table}_id_seq"
        target_sequence = f"{target_table}_id_seq"
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM pg_sequences 
                WHERE schemaname = 'public' 
                AND sequencename = %s
            );
            """,
            (source_sequence,),
        )
        seq_exists = cur.fetchone()[0]

        if seq_exists:
            # Get current sequence value from source
            cur.execute(
                sql.SQL("SELECT last_value FROM {}").format(
                    sql.Identifier(source_sequence)
                )
            )
            last_value = cur.fetchone()[0]

            # Check if target sequence exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM pg_sequences 
                    WHERE schemaname = 'public' 
                    AND sequencename = %s
                );
                """,
                (target_sequence,),
            )
            target_seq_exists = cur.fetchone()[0]

            if not target_seq_exists:
                # Create new sequence for target table
                cur.execute(
                    sql.SQL(
                        """
                        CREATE SEQUENCE {}
                        START WITH {};
                        """
                    ).format(
                        sql.Identifier(target_sequence),
                        sql.SQL(str(last_value + 1)),
                    )
                )
                conn.commit()
                logger.info(
                    f"✅ Created sequence {target_sequence} starting from {last_value + 1}"
                )
            else:
                # Update existing sequence
                cur.execute(
                    sql.SQL("SELECT setval(%s, %s, true)").format(),
                    (target_sequence, last_value + 1),
                )
                conn.commit()
                logger.info(
                    f"✅ Updated sequence {target_sequence} to {last_value + 1}"
                )

        # Delete migrated data from source table if requested (MOVE operation)
        if delete_source:
            logger.warning(
                f"🗑️  Deleting migrated data from source table {source_table}..."
            )
            logger.warning("⚠️  This is a destructive operation!")

            # Verify target table has the data before deleting
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(target_table))
            )
            target_count = cur.fetchone()[0]

            if target_count == 0:
                logger.error(
                    "❌ Target table is empty! Aborting deletion to prevent data loss."
                )
                cur.close()
                conn.close()
                return False

            # Confirm deletion
            response = input(
                f"⚠️  Are you sure you want to delete {rows_migrated} rows from {source_table}? (yes/no): "
            )
            if response.lower() not in ["yes", "y"]:
                logger.info("❌ Deletion cancelled.")
                cur.close()
                conn.close()
                return True

            # Delete from source table
            cur.execute(
                sql.SQL(
                    """
                    DELETE FROM {}
                    WHERE {}
                    """
                ).format(
                    sql.Identifier(source_table),
                    where_clause,
                ),
                where_params,
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

        logger.info("✅ Migration completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Error during migration: {e}", exc_info=True)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Migrate docs vectors data based on date or ID range"
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
        "--year",
        type=int,
        default=2025,
        help="Year for cutoff date (default: 2025)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=12,
        help="Month for cutoff date (default: 12)",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=1,
        help="Day for cutoff date (default: 1)",
    )
    parser.add_argument(
        "--by-id",
        action="store_true",
        help="Use ID range instead of date (migrate rows with ID > cutoff_id)",
    )
    parser.add_argument(
        "--cutoff-id",
        type=int,
        default=3720,
        help="Cutoff ID when using --by-id (default: 3720)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: show what would be migrated without actually migrating",
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
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete migrated data from source table after migration (MOVE operation)",
    )

    args = parser.parse_args()

    if args.by_id:
        cutoff_timestamp = None
        cutoff_id = args.cutoff_id
        logger.info(f"📊 Using ID range: migrating rows with id > {cutoff_id}")
    else:
        cutoff_timestamp = get_december_timestamp(args.year, args.month, args.day)
        cutoff_date = datetime.fromtimestamp(cutoff_timestamp).strftime("%Y-%m-%d")
        logger.info(f"📅 Cutoff date: {cutoff_date} (timestamp: {cutoff_timestamp})")
        cutoff_id = None

    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No changes will be made")
        if not PSYCOPG2_AVAILABLE:
            logger.error("❌ Cannot check: psycopg2 not available")
            return

        # Use provided parameters or fall back to settings
        db_host = args.db_host or settings.postgres_host
        db_port = args.db_port or settings.postgres_port
        db_user = args.db_user or settings.postgres_user
        db_password = args.db_password or settings.postgres_password
        db_name = args.db_name or settings.postgres_db

        try:
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name,
            )
            cur = conn.cursor()

            # Check how many rows would be migrated
            if args.by_id:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT COUNT(*) FROM {}
                        WHERE id > %s
                        """
                    ).format(sql.Identifier(args.source_table)),
                    (args.cutoff_id,),
                )
                rows_to_migrate = cur.fetchone()[0]

                # Get sample of rows that would be migrated
                cur.execute(
                    sql.SQL(
                        """
                        SELECT id, metadata_->>'last_updated' as last_updated, LEFT(text, 50) as text_preview
                        FROM {}
                        WHERE id > %s
                        LIMIT 5
                        """
                    ).format(sql.Identifier(args.source_table)),
                    (args.cutoff_id,),
                )
            else:
                cutoff_ts = get_december_timestamp(args.year, args.month, args.day)
                cur.execute(
                    sql.SQL(
                        """
                        SELECT COUNT(*) FROM {}
                        WHERE (metadata_->>'last_updated')::float >= %s
                        """
                    ).format(sql.Identifier(args.source_table)),
                    (cutoff_ts,),
                )
                rows_to_migrate = cur.fetchone()[0]

                # Get sample of rows that would be migrated
                cur.execute(
                    sql.SQL(
                        """
                        SELECT id, metadata_->>'last_updated' as last_updated, LEFT(text, 50) as text_preview
                        FROM {}
                        WHERE (metadata_->>'last_updated')::float >= %s
                        LIMIT 5
                        """
                    ).format(sql.Identifier(args.source_table)),
                    (cutoff_ts,),
                )
            sample_rows = cur.fetchall()

            logger.info(f"📊 Would migrate {rows_to_migrate} rows")
            logger.info("📋 Sample rows that would be migrated:")
            for row in sample_rows:
                row_id, last_updated, text_preview = row
                if last_updated:
                    dt = datetime.fromtimestamp(float(last_updated))
                    logger.info(
                        f"  - ID: {row_id}, Last Updated: {dt.strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"Text: {text_preview[:50]}..."
                    )

            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Error during dry run: {e}", exc_info=True)
    else:
        success = create_hybrid_table(
            args.source_table,
            args.target_table,
            cutoff_timestamp=cutoff_timestamp,
            cutoff_id=cutoff_id,
            use_id_range=args.by_id,
            delete_source=args.delete_source,
            db_host=args.db_host,
            db_port=args.db_port,
            db_user=args.db_user,
            db_password=args.db_password,
            db_name=args.db_name,
        )
        if success:
            logger.info("✅ Migration completed successfully!")
            logger.info(
                "💡 Note: To use the new table, update config.py: "
                "docs_vector_table_name = 'hybrid_askany_docs_vectors'"
            )
            logger.info(
                f"   (PGVectorStore will automatically add 'data_' prefix, "
                f"so the actual table name will be {args.target_table})"
            )
        else:
            logger.error("❌ Migration failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()

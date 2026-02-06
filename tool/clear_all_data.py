#!/usr/bin/env python3
"""Clear all data from vector stores and keyword index.

This script clears:
1. data_askany_faq_vectors table
2. data_askany_docs_vectors table
3. FAQ keyword index storage directory
"""

import argparse
import shutil
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

from askany.config import settings


def clear_vector_tables(confirm: bool = False, table_names: list = None):
    """Clear all data from vector tables.

    Args:
        confirm: If True, skip confirmation prompt
        table_names: List of table names to clear. If None, uses default tables.
    """
    if not PSYCOPG2_AVAILABLE:
        print("❌ Cannot clear vector tables: psycopg2 not available")
        return False

    # Table names - try both with and without data_ prefix
    if table_names is None:
        # Default: try both naming conventions
        tables = [
            "data_askany_faq_vectors",  # User mentioned this format
            "data_askany_docs_vectors",
            settings.faq_vector_table_name,  # From config
            settings.docs_vector_table_name,
        ]
        # Remove duplicates while preserving order
        seen = set()
        tables = [t for t in tables if not (t in seen or seen.add(t))]
    else:
        tables = table_names

    if not confirm:
        print("\n⚠️  WARNING: This will delete ALL data from the following tables:")
        for table in tables:
            print(f"  - {table}")
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("❌ Operation cancelled.")
            return False

    try:
        # Connect to database
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )
        cur = conn.cursor()

        cleared_tables = []
        for table in tables:
            try:
                # Check if table exists
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                    """,
                    (table,),
                )
                table_exists = cur.fetchone()[0]

                if not table_exists:
                    print(f"⚠️  Table '{table}' does not exist, skipping...")
                    continue

                # Get row count before deletion
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
                )
                row_count = cur.fetchone()[0]

                # Delete all rows
                cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table)))
                conn.commit()

                cleared_tables.append((table, row_count))
                print(f"✅ Cleared {row_count} rows from table '{table}'")

            except Exception as e:
                print(f"❌ Error clearing table '{table}': {e}")
                conn.rollback()
                continue

        cur.close()
        conn.close()

        if cleared_tables:
            total_rows = sum(count for _, count in cleared_tables)
            print(
                f"\n✅ Successfully cleared {total_rows} total rows from {len(cleared_tables)} table(s)"
            )
            return True
        else:
            print("\n⚠️  No tables were cleared")
            return False

    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return False


def clear_keyword_index(confirm: bool = False):
    """Clear FAQ keyword index storage directory.

    Args:
        confirm: If True, skip confirmation prompt
    """
    # Keyword index storage directory
    keyword_storage_dir = (
        Path(settings.storage_dir) / settings.faq_keyword_storage_index
    )

    if not keyword_storage_dir.exists():
        print(
            f"⚠️  Keyword index directory '{keyword_storage_dir}' does not exist, nothing to clear"
        )
        return True

    if not confirm:
        print("\n⚠️  WARNING: This will delete the keyword index directory:")
        print(f"  - {keyword_storage_dir}")
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("❌ Operation cancelled.")
            return False

    try:
        # List files before deletion
        files_before = list(keyword_storage_dir.rglob("*"))
        file_count = len([f for f in files_before if f.is_file()])

        # Remove directory and all contents
        shutil.rmtree(keyword_storage_dir)
        print(
            f"✅ Deleted keyword index directory '{keyword_storage_dir}' ({file_count} files)"
        )
        return True

    except Exception as e:
        print(f"❌ Error deleting keyword index directory: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Clear all data from vector stores and keyword index"
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only clear vector tables, not keyword index",
    )
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="Only clear keyword index, not vector tables",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Custom table names to clear (overrides default tables)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Clear All Data Script")
    print("=" * 80)

    success = True

    # Clear vector tables
    if not args.keyword_only:
        print("\n[1/2] Clearing vector tables...")
        success = (
            clear_vector_tables(confirm=args.yes, table_names=args.tables) and success
        )

    # Clear keyword index
    if not args.tables_only:
        print("\n[2/2] Clearing keyword index...")
        success = clear_keyword_index(confirm=args.yes) and success

    print("\n" + "=" * 80)
    if success:
        print("✅ All data cleared successfully!")
    else:
        print("⚠️  Some operations may have failed. Please check the output above.")
    print("=" * 80)


if __name__ == "__main__":
    main()

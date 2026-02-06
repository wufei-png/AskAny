#!/usr/bin/env python3
"""Import pgvector data from vector_data directory to PostgreSQL."""

import argparse
import json
import logging
import subprocess
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


def check_pgvector_extension() -> bool:
    """Check if pgvector extension exists, create if not."""
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

        # Check if extension exists
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            );
            """
        )
        exists = cur.fetchone()[0]

        if not exists:
            logger.info("Creating pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            logger.info("✅ pgvector extension created")
        else:
            logger.info("✅ pgvector extension already exists")

        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Error checking/creating pgvector extension: {e}")
        return False


def import_table_from_dump(
    table_name: str, dump_file: Path, drop_existing: bool = False
) -> bool:
    """Import table from pg_dump custom format file."""
    try:
        if drop_existing:
            # Drop table if exists
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
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                conn.commit()
                cur.close()
                conn.close()
                logger.info(f"Dropped existing table {table_name}")
            except Exception as e:
                logger.warning(f"Error dropping table {table_name}: {e}")

        # Use pg_restore to import
        cmd = [
            "pg_restore",
            "-h",
            settings.postgres_host,
            "-p",
            str(settings.postgres_port),
            "-U",
            settings.postgres_user,
            "-d",
            settings.postgres_db,
            "--no-owner",
            "--no-privileges",
            "--verbose",
            str(dump_file),
        ]

        env = {"PGPASSWORD": settings.postgres_password}
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info(f"✅ Imported table {table_name} from {dump_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error importing table {table_name}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ pg_restore not found. Please install PostgreSQL client tools.")
        return False


def import_table_from_schema_and_data(
    table_name: str, schema_file: Path, data_file: Path, drop_existing: bool = False
) -> bool:
    """Import table from separate schema and data files."""
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

        if drop_existing:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            conn.commit()
            logger.info(f"Dropped existing table {table_name}")

        # Import schema
        logger.info(f"Importing schema from {schema_file}...")
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()
            # Remove CREATE EXTENSION statements (we handle that separately)
            schema_sql = "\n".join(
                line
                for line in schema_sql.split("\n")
                if not line.strip().upper().startswith("CREATE EXTENSION")
            )
            cur.execute(schema_sql)
            conn.commit()

        logger.info(f"✅ Imported schema for {table_name}")

        # Import data
        if data_file.exists():
            logger.info(f"Importing data from {data_file}...")
            import csv

            with open(data_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                batch_size = 1000
                batch = []
                total_rows = 0

                for row in reader:
                    # Parse vector from text format (e.g., "[0.1,0.2,0.3]")
                    embedding_text = row.get("embedding", "")
                    # Vector type can accept text format directly

                    # Handle NULL values
                    metadata_val = row.get("metadata_", None)
                    if metadata_val and metadata_val.lower() != "null":
                        # Parse JSON if it's a JSON string
                        try:
                            import json

                            metadata_val = json.loads(metadata_val)
                        except:
                            pass
                    else:
                        metadata_val = None

                    node_id_val = row.get("node_id", None)
                    if node_id_val and node_id_val.lower() == "null":
                        node_id_val = None

                    batch.append(
                        (
                            int(row["id"]),
                            row["text"],
                            metadata_val,
                            node_id_val,
                            embedding_text,  # PostgreSQL will cast text to vector
                        )
                    )

                    if len(batch) >= batch_size:
                        cur.executemany(
                            f"""
                            INSERT INTO {table_name} (id, text, metadata_, node_id, embedding)
                            VALUES (%s, %s, %s, %s, %s::vector)
                            """,
                            batch,
                        )
                        conn.commit()
                        total_rows += len(batch)
                        batch = []
                        logger.info(f"  Imported {total_rows} rows...")

                # Insert remaining rows
                if batch:
                    cur.executemany(
                        f"""
                        INSERT INTO {table_name} (id, text, metadata_, node_id, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        """,
                        batch,
                    )
                    conn.commit()
                    total_rows += len(batch)

            logger.info(f"✅ Imported {total_rows} rows for {table_name}")
        else:
            logger.warning(f"⚠️  Data file {data_file} not found, skipping data import")

        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Error importing table {table_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def import_sequences(sequences_file: Path) -> bool:
    """Import sequence information."""
    if not sequences_file.exists():
        logger.info("ℹ️  No sequences file found, skipping")
        return True

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

        with open(sequences_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("--"):
                    try:
                        cur.execute(line)
                        conn.commit()
                    except Exception as e:
                        logger.warning(
                            f"Error executing sequence command: {line} - {e}"
                        )

        cur.close()
        conn.close()

        logger.info(f"✅ Imported sequences from {sequences_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Error importing sequences: {e}")
        return False


def load_metadata(input_dir: Path) -> dict:
    """Load metadata from export."""
    metadata_file = input_dir / "metadata.json"
    if not metadata_file.exists():
        logger.warning("⚠️  Metadata file not found, using defaults")
        return {}

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading metadata: {e}")
        return {}


def find_table_files(input_dir: Path, table_name: str):
    """Find dump file or schema/data files for a table."""
    dump_file = input_dir / f"{table_name}.dump"
    schema_file = input_dir / f"{table_name}_schema.sql"
    data_file = input_dir / f"{table_name}_data.csv"

    if dump_file.exists():
        return ("dump", dump_file, None)
    elif schema_file.exists():
        return ("separate", schema_file, data_file)
    else:
        return (None, None, None)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Import pgvector data from vector_data directory to PostgreSQL"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="vector_data",
        help="Input directory containing exported data (default: vector_data)",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before importing",
    )
    parser.add_argument(
        "--skip-sequences",
        action="store_true",
        help="Skip importing sequences",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"❌ Input directory does not exist: {input_dir.absolute()}")
        return

    logger.info("=" * 80)
    logger.info("Starting vector data import")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir.absolute()}")
    logger.info(
        f"Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    # Check/create pgvector extension
    logger.info("\n" + "=" * 80)
    logger.info("Checking pgvector extension...")
    logger.info("=" * 80)
    if not check_pgvector_extension():
        logger.error("❌ Failed to setup pgvector extension")
        return

    # Load metadata
    metadata = load_metadata(input_dir)
    if metadata:
        logger.info("\n" + "=" * 80)
        logger.info("Export metadata:")
        logger.info("=" * 80)
        logger.info(
            f"  Database: {metadata.get('database', {}).get('database', 'N/A')}"
        )
        logger.info(
            f"  Vector dimension: {metadata.get('vector_config', {}).get('vector_dimension', 'N/A')}"
        )
        tables_info = metadata.get("tables", {})
        for table_type, info in tables_info.items():
            if info.get("exists"):
                logger.info(
                    f"  {table_type} table: {info.get('actual_name', 'N/A')} ({info.get('row_count', 'N/A')} rows)"
                )

    # Import tables
    logger.info("\n" + "=" * 80)
    logger.info("Importing tables...")
    logger.info("=" * 80)

    # Try to import tables from metadata or use defaults
    tables_to_import = []
    if metadata and "tables" in metadata:
        for table_type, info in metadata["tables"].items():
            if info.get("exists"):
                tables_to_import.append((table_type, info.get("actual_name")))
    else:
        # Fallback to default table names
        tables_to_import = [
            ("FAQ", f"data_{settings.faq_vector_table_name}"),
            ("Docs", f"data_{settings.docs_vector_table_name}"),
            ("Legacy", f"data_{settings.vector_table_name}"),
        ]

    tables_imported = 0
    for table_type, table_name in tables_to_import:
        file_type, file1, file2 = find_table_files(input_dir, table_name)

        if file_type is None:
            logger.info(f"ℹ️  No files found for {table_name} ({table_type}), skipping")
            continue

        logger.info(f"\nImporting {table_type} table: {table_name}")

        if file_type == "dump":
            success = import_table_from_dump(
                table_name, file1, drop_existing=args.drop_existing
            )
        else:
            success = import_table_from_schema_and_data(
                table_name, file1, file2, drop_existing=args.drop_existing
            )

        if success:
            tables_imported += 1
        else:
            logger.warning(f"⚠️  Failed to import {table_name}")

    # Import sequences
    if not args.skip_sequences:
        logger.info("\n" + "=" * 80)
        logger.info("Importing sequences...")
        logger.info("=" * 80)
        sequences_file = input_dir / "sequences.sql"
        import_sequences(sequences_file)

    logger.info("\n" + "=" * 80)
    logger.info("Import completed!")
    logger.info(f"Imported {tables_imported} table(s)")
    logger.info("=" * 80)

    # Verify import
    logger.info("\n" + "=" * 80)
    logger.info("Verifying import...")
    logger.info("=" * 80)
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

        for table_type, table_name in tables_to_import:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cur.fetchone()[0]
                logger.info(f"✅ {table_name}: {count} rows")
            except Exception as e:
                logger.warning(f"⚠️  {table_name}: {e}")

        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"⚠️  Error verifying import: {e}")


if __name__ == "__main__":
    main()

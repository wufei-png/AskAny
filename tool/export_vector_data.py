#!/usr/bin/env python3
"""Export pgvector data from PostgreSQL to vector_data directory."""

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


def get_dump_version(dump_file: Path) -> dict:
    """Extract PostgreSQL version information from a pg_dump custom format file."""
    try:
        # Use pg_restore --list to read dump header
        result = subprocess.run(
            ["pg_restore", "--list", str(dump_file)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse version information from header
        import re

        version_info = {}

        # Extract "Dumped from database version"
        db_version_match = re.search(
            r"Dumped from database version:\s*(.+)", result.stdout
        )
        if db_version_match:
            version_info["database_version"] = db_version_match.group(1).strip()
            # Extract version number (e.g., "14.19")
            version_num_match = re.search(
                r"(\d+\.\d+)", version_info["database_version"]
            )
            if version_num_match:
                version_info["version_number"] = version_num_match.group(1)

        # Extract "Dumped by pg_dump version"
        pg_dump_version_match = re.search(
            r"Dumped by pg_dump version:\s*(.+)", result.stdout
        )
        if pg_dump_version_match:
            version_info["pg_dump_version"] = pg_dump_version_match.group(1).strip()

        # Extract archive creation date
        archive_date_match = re.search(r"Archive created at\s*(.+)", result.stdout)
        if archive_date_match:
            version_info["archive_created_at"] = archive_date_match.group(1).strip()

        # Extract dump version
        dump_version_match = re.search(r"Dump Version:\s*(.+)", result.stdout)
        if dump_version_match:
            version_info["dump_version"] = dump_version_match.group(1).strip()

        return version_info
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error reading dump file {dump_file}: {e.stderr}")
        return {"error": str(e)}
    except FileNotFoundError:
        logger.warning("pg_restore not found. Cannot read dump file version.")
        return {"error": "pg_restore not found"}
    except Exception as e:
        logger.warning(f"Error extracting version from dump file: {e}")
        return {"error": str(e)}


def get_table_names():
    """Get all vector table names that might exist."""
    # PGVectorStore uses "data_{table_name}" as the actual table name
    tables = []

    # Check for FAQ table
    faq_table = settings.faq_vector_table_name
    faq_actual_table = f"data_{faq_table}"
    tables.append(("FAQ", faq_table, faq_actual_table))

    # Check for Docs table
    docs_table = settings.docs_vector_table_name
    docs_actual_table = f"data_{docs_table}"
    tables.append(("Docs", docs_table, docs_actual_table))

    # Check for legacy table
    legacy_table = settings.vector_table_name
    legacy_actual_table = f"data_{legacy_table}"
    tables.append(("Legacy", legacy_table, legacy_actual_table))

    return tables


def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
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
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        logger.error(f"Error checking table {table_name}: {e}")
        return False


def export_table_schema(table_name: str, output_dir: Path) -> bool:
    """Export table schema (structure only) to SQL file."""
    schema_file = output_dir / f"{table_name}_schema.sql"

    try:
        # Use pg_dump to export schema only
        cmd = [
            "pg_dump",
            "-h",
            settings.postgres_host,
            "-p",
            str(settings.postgres_port),
            "-U",
            settings.postgres_user,
            "-d",
            settings.postgres_db,
            "--schema-only",
            "--table",
            table_name,
            "--no-owner",
            "--no-privileges",
            "-f",
            str(schema_file),
        ]

        env = {"PGPASSWORD": settings.postgres_password}
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info(f"✅ Exported schema for {table_name} to {schema_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error exporting schema for {table_name}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ pg_dump not found. Please install PostgreSQL client tools.")
        return False


def export_table_data(table_name: str, output_dir: Path) -> bool:
    """Export table data to CSV file."""
    data_file = output_dir / f"{table_name}_data.csv"

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

        # Export data using COPY command
        # Note: vector type needs special handling, we'll export as text
        with open(data_file, "w", encoding="utf-8") as f:
            cur.copy_expert(
                f"""
                COPY (
                    SELECT 
                        id,
                        text,
                        metadata_::text,
                        node_id,
                        embedding::text
                    FROM {table_name}
                ) TO STDOUT WITH CSV HEADER
                """,
                f,
            )

        cur.close()
        conn.close()

        logger.info(f"✅ Exported data for {table_name} to {data_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Error exporting data for {table_name}: {e}")
        return False


def export_table_full(table_name: str, output_dir: Path) -> bool:
    """Export table with both schema and data using pg_dump custom format."""
    dump_file = output_dir / f"{table_name}.dump"

    try:
        # Use pg_dump custom format for better compression and flexibility
        cmd = [
            "pg_dump",
            "-h",
            settings.postgres_host,
            "-p",
            str(settings.postgres_port),
            "-U",
            settings.postgres_user,
            "-d",
            settings.postgres_db,
            "--table",
            table_name,
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            "-f",
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

        logger.info(f"✅ Exported full table {table_name} to {dump_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error exporting table {table_name}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ pg_dump not found. Please install PostgreSQL client tools.")
        return False


def export_sequences(output_dir: Path) -> bool:
    """Export sequence information."""
    sequences_file = output_dir / "sequences.sql"

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

        # Get all sequences related to vector tables
        # Use pg_sequences view (PostgreSQL 10+) which includes last_value
        cur.execute(
            """
            SELECT 
                sequencename AS sequence_name,
                last_value
            FROM pg_sequences
            WHERE schemaname = 'public'
            AND sequencename LIKE '%vector%'
            ORDER BY sequencename;
            """
        )

        sequences = cur.fetchall()
        cur.close()
        conn.close()

        if sequences:
            with open(sequences_file, "w", encoding="utf-8") as f:
                f.write("-- Sequence information\n")
                f.write("-- These sequences are used for auto-incrementing IDs\n\n")
                for seq_name, last_value in sequences:
                    f.write(f"-- {seq_name}: last_value = {last_value}\n")
                    f.write(f"SELECT setval('{seq_name}', {last_value}, true);\n")

            logger.info(f"✅ Exported sequence information to {sequences_file}")
            return True
        else:
            logger.info("ℹ️  No sequences found")
            return True
    except Exception as e:
        logger.error(f"❌ Error exporting sequences: {e}")
        return False


def get_postgresql_version() -> dict:
    """Get PostgreSQL version information."""
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
        # Get PostgreSQL version
        cur.execute("SELECT version();")
        version_string = cur.fetchone()[0]

        # Extract version number (e.g., "PostgreSQL 14.19" -> "14.19")
        import re

        version_match = re.search(r"PostgreSQL (\d+\.\d+)", version_string)
        version_number = version_match.group(1) if version_match else None

        # Get server version number (numeric)
        cur.execute("SHOW server_version_num;")
        server_version_num = cur.fetchone()[0]

        cur.close()
        conn.close()

        return {
            "version_string": version_string,
            "version_number": version_number,
            "server_version_num": server_version_num,
        }
    except Exception as e:
        logger.warning(f"Error getting PostgreSQL version: {e}")
        return {
            "version_string": None,
            "version_number": None,
            "server_version_num": None,
            "error": str(e),
        }


def export_metadata(output_dir: Path) -> bool:
    """Export metadata about the export."""
    # Get PostgreSQL version information
    pg_version = get_postgresql_version()

    metadata = {
        "database": {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "database": settings.postgres_db,
            "user": settings.postgres_user,
            "postgresql_version": pg_version,
        },
        "vector_config": {
            "vector_dimension": settings.vector_dimension,
            "faq_vector_table_name": settings.faq_vector_table_name,
            "docs_vector_table_name": settings.docs_vector_table_name,
            "vector_table_name": settings.vector_table_name,
            "hnsw_enabled": settings.enable_hnsw,
            "hnsw_m": settings.hnsw_m,
            "hnsw_ef_construction": settings.hnsw_ef_construction,
            "hnsw_ef_search": settings.hnsw_ef_search,
            "hnsw_dist_method": settings.hnsw_dist_method,
        },
        "tables": {},
    }

    # Check which tables exist and get row counts
    for table_type, config_name, actual_name in get_table_names():
        if check_table_exists(actual_name):
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
                cur.execute(f"SELECT COUNT(*) FROM {actual_name};")
                count = cur.fetchone()[0]
                cur.close()
                conn.close()

                metadata["tables"][table_type] = {
                    "config_name": config_name,
                    "actual_name": actual_name,
                    "row_count": count,
                    "exists": True,
                }
            except Exception as e:
                logger.warning(f"Error getting count for {actual_name}: {e}")
                metadata["tables"][table_type] = {
                    "config_name": config_name,
                    "actual_name": actual_name,
                    "exists": True,
                    "error": str(e),
                }
        else:
            metadata["tables"][table_type] = {
                "config_name": config_name,
                "actual_name": actual_name,
                "exists": False,
            }

    # Try to read version info from existing dump files
    dump_files = list(output_dir.glob("*.dump"))
    if dump_files:
        metadata["dump_files"] = {}
        for dump_file in dump_files:
            dump_info = get_dump_version(dump_file)
            if dump_info:
                metadata["dump_files"][dump_file.name] = dump_info

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Exported metadata to {metadata_file}")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Export pgvector data from PostgreSQL to vector_data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vector_data",
        help="Output directory for exported data (default: vector_data)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["full", "separate"],
        default="full",
        help="Export format: 'full' uses pg_dump custom format, 'separate' exports schema and data separately (default: full)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting vector data export")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(
        f"Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    # Export metadata first
    logger.info("\n" + "=" * 80)
    logger.info("Exporting metadata...")
    logger.info("=" * 80)
    export_metadata(output_dir)

    # Export tables
    logger.info("\n" + "=" * 80)
    logger.info("Exporting tables...")
    logger.info("=" * 80)

    tables_exported = 0
    for table_type, config_name, actual_name in get_table_names():
        if not check_table_exists(actual_name):
            logger.info(
                f"ℹ️  Table {actual_name} ({table_type}) does not exist, skipping"
            )
            continue

        logger.info(f"Exporting {table_type} table: {actual_name}")

        if args.format == "full":
            success = export_table_full(actual_name, output_dir)
        else:
            schema_success = export_table_schema(actual_name, output_dir)
            data_success = export_table_data(actual_name, output_dir)
            success = schema_success and data_success

        if success:
            tables_exported += 1
        else:
            logger.warning(f"⚠️  Failed to export {actual_name}")

    # Export sequences
    logger.info("\n" + "=" * 80)
    logger.info("Exporting sequences...")
    logger.info("=" * 80)
    export_sequences(output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("Export completed!")
    logger.info(f"Exported {tables_exported} table(s)")
    logger.info(f"Data saved to: {output_dir.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

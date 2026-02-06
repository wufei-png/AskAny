#!/usr/bin/env python3
"""Check PostgreSQL version from dump files."""

import argparse
import subprocess
import sys
from pathlib import Path


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
        return {"error": f"Error reading dump file: {e.stderr}"}
    except FileNotFoundError:
        return {
            "error": "pg_restore not found. Please install PostgreSQL client tools."
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Check PostgreSQL version from dump files"
    )
    parser.add_argument(
        "dump_file",
        type=str,
        nargs="?",
        help="Path to dump file (or directory containing .dump files)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="vector_data",
        help="Directory to search for dump files (default: vector_data)",
    )

    args = parser.parse_args()

    dump_files = []
    if args.dump_file:
        dump_path = Path(args.dump_file)
        if dump_path.is_file():
            dump_files = [dump_path]
        elif dump_path.is_dir():
            dump_files = list(dump_path.glob("*.dump"))
    else:
        dump_dir = Path(args.dir)
        if dump_dir.exists():
            dump_files = list(dump_dir.glob("*.dump"))

    if not dump_files:
        print("❌ 未找到 dump 文件")
        if args.dump_file:
            print(f"   路径: {args.dump_file}")
        else:
            print(f"   目录: {args.dir}")
        sys.exit(1)

    print("=" * 80)
    print("PostgreSQL 版本信息")
    print("=" * 80)
    print()

    for dump_file in sorted(dump_files):
        print(f"📦 文件: {dump_file.name}")
        version_info = get_dump_version(dump_file)

        if "error" in version_info:
            print(f"   ❌ {version_info['error']}")
        else:
            if "database_version" in version_info:
                print(f"   🗄️  数据库版本: {version_info['database_version']}")
            if "version_number" in version_info:
                print(f"   📌 版本号: {version_info['version_number']}")
            if "pg_dump_version" in version_info:
                print(f"   🔧 pg_dump 版本: {version_info['pg_dump_version']}")
            if "archive_created_at" in version_info:
                print(f"   📅 创建时间: {version_info['archive_created_at']}")
            if "dump_version" in version_info:
                print(f"   📋 Dump 格式版本: {version_info['dump_version']}")

        print()

    print("=" * 80)


if __name__ == "__main__":
    main()

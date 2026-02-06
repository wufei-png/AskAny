#!/usr/bin/env python3
"""Query HNSW index structure and print layer information."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.config import settings


def query_hnsw_index_info(table_name: str, index_name: str = None):
    """Query HNSW index information from PostgreSQL.

    Args:
        table_name: Name of the table (e.g., 'data_askany_faq_vectors')
        index_name: Optional specific index name to query
    """
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not available. Please install: pip install psycopg2-binary")
        return

    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )
        cur = conn.cursor()

        print(f"\n{'=' * 80}")
        print(f"HNSW Index Information for Table: {table_name}")
        print(f"{'=' * 80}\n")

        # Query index basic information
        if index_name:
            index_filter = f"AND i.relname = '{index_name}'"
        else:
            index_filter = "AND i.relname LIKE '%embedding%'"

        cur.execute(f"""
            SELECT 
                i.relname as index_name,
                t.relname as table_name,
                pg_size_pretty(pg_relation_size(i.oid)) as index_size,
                pg_relation_size(i.oid) as index_size_bytes,
                am.amname as index_type,
                idx.indisunique as is_unique,
                idx.indisprimary as is_primary
            FROM pg_index idx
            JOIN pg_class i ON i.oid = idx.indexrelid
            JOIN pg_class t ON t.oid = idx.indrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE t.relname = '{table_name}'
              {index_filter}
            ORDER BY i.relname;
        """)

        indexes = cur.fetchall()
        if not indexes:
            print(f"❌ No HNSW indexes found for table '{table_name}'")
            cur.close()
            conn.close()
            return

        for idx_info in indexes:
            (
                idx_name,
                tbl_name,
                idx_size,
                idx_size_bytes,
                idx_type,
                is_unique,
                is_primary,
            ) = idx_info
            print(f"Index: {idx_name}")
            print(f"  Table: {tbl_name}")
            print(f"  Type: {idx_type}")
            print(f"  Size: {idx_size} ({idx_size_bytes:,} bytes)")
            print(f"  Unique: {is_unique}")
            print(f"  Primary: {is_primary}")

            # Query index options (HNSW parameters)
            cur.execute(f"""
                SELECT 
                    option_name,
                    option_value
                FROM pg_options_to_table(
                    (SELECT reloptions FROM pg_class WHERE relname = '{idx_name}')
                )
                ORDER BY option_name;
            """)

            options = cur.fetchall()
            if options:
                print("  HNSW Configuration:")
                for opt_name, opt_value in options:
                    print(f"    {opt_name}: {opt_value}")
            else:
                print("  HNSW Configuration: (using defaults)")

            # Query index statistics
            cur.execute(f"""
                SELECT 
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE indexrelid = (SELECT oid FROM pg_class WHERE relname = '{idx_name}');
            """)

            stats = cur.fetchone()
            if stats:
                idx_scan, idx_tup_read, idx_tup_fetch = stats
                print("  Statistics:")
                print(f"    Index Scans: {idx_scan:,}")
                print(f"    Tuples Read: {idx_tup_read:,}")
                print(f"    Tuples Fetched: {idx_tup_fetch:,}")

            # Query table row count for reference
            cur.execute(f"SELECT COUNT(*) FROM {tbl_name};")
            row_count = cur.fetchone()[0]
            print(f"  Table Row Count: {row_count:,}")

            print()

        # Note about HNSW layer structure
        print(f"{'=' * 80}")
        print("Note: HNSW Index Layer Structure")
        print(f"{'=' * 80}")
        print("""
pgvector's HNSW index internal structure (layer hierarchy and node connections) 
is stored in binary format within PostgreSQL's index structure and is not directly 
queryable via SQL.

To access HNSW layer structure, you would need to:
1. Use pgvector extension functions (if available in your version)
2. Access the index at a lower level through PostgreSQL's internal APIs
3. Use external tools that can parse pgvector's HNSW index format

Current limitations:
- PostgreSQL system catalogs don't expose HNSW layer information
- pgvector doesn't provide SQL functions to query layer structure
- Index structure is optimized for query performance, not introspection

However, you can infer some information:
- Index size indicates the overall structure complexity
- Configuration parameters (hnsw.m, hnsw.ef_construction) affect layer structure
- Higher 'm' values create more connections per node
- Higher 'ef_construction' values create better quality (potentially more layers)
        """)

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error querying HNSW index: {e}")
        import traceback

        traceback.print_exc()


def query_hnsw_layers_estimate(table_name: str, layer_n: int = None):
    """Estimate HNSW layer information based on index statistics.

    Args:
        table_name: Name of the table
        layer_n: Optional specific layer to query (not directly supported, shows estimate)
    """
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not available. Please install: pip install psycopg2-binary")
        return

    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )
        cur = conn.cursor()

        print(f"\n{'=' * 80}")
        print(f"HNSW Layer Estimation for Table: {table_name}")
        if layer_n is not None:
            print(f"Requested Layer: {layer_n}")
        print(f"{'=' * 80}\n")

        # Get index information
        cur.execute(f"""
            SELECT 
                i.relname as index_name,
                pg_relation_size(i.oid) as index_size_bytes,
                (SELECT option_value FROM pg_options_to_table(i.reloptions) WHERE option_name = 'hnsw.m') as hnsw_m,
                (SELECT option_value FROM pg_options_to_table(i.reloptions) WHERE option_name = 'hnsw.ef_construction') as hnsw_ef_construction
            FROM pg_index idx
            JOIN pg_class i ON i.oid = idx.indexrelid
            JOIN pg_class t ON t.oid = idx.indrelid
            WHERE t.relname = '{table_name}'
              AND i.relname LIKE '%embedding%'
            LIMIT 1;
        """)

        idx_info = cur.fetchone()
        if not idx_info:
            print(f"❌ No HNSW index found for table '{table_name}'")
            cur.close()
            conn.close()
            return

        idx_name, idx_size_bytes, hnsw_m, hnsw_ef_construction = idx_info

        # Get table row count
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cur.fetchone()[0]

        # Get vector dimension
        cur.execute(f"""
            SELECT vector_dims(embedding) as dim
            FROM {table_name}
            LIMIT 1;
        """)
        dim_result = cur.fetchone()
        vector_dim = dim_result[0] if dim_result else settings.vector_dimension

        print(f"Index: {idx_name}")
        print(f"Table Rows: {row_count:,}")
        print(f"Vector Dimension: {vector_dim}")
        print(
            f"Index Size: {idx_size_bytes:,} bytes ({idx_size_bytes / 1024 / 1024:.2f} MB)"
        )
        print(f"HNSW M: {hnsw_m or settings.hnsw_m}")
        print(
            f"HNSW EF Construction: {hnsw_ef_construction or settings.hnsw_ef_construction}"
        )

        # Estimate layer information
        # HNSW typically has log(N) layers on average, with top layer having very few nodes
        import math

        estimated_max_layers = max(1, int(math.log2(max(1, row_count))) + 1)

        print("\nEstimated Layer Information:")
        print(f"  Estimated Max Layers: ~{estimated_max_layers}")
        print("  (HNSW typically has log2(N) layers for N vectors)")

        if layer_n is not None:
            if layer_n < 0 or layer_n >= estimated_max_layers:
                print(
                    f"\n⚠️  Layer {layer_n} is outside estimated range [0, {estimated_max_layers - 1}]"
                )
            else:
                # Estimate nodes per layer (rough approximation)
                # Top layer (layer 0) has very few nodes, lower layers have more
                # This is a rough estimate based on HNSW properties
                estimated_nodes_layer_n = max(1, int(row_count / (2 ** (layer_n + 1))))
                print(f"\n  Layer {layer_n} Estimation:")
                print(f"    Estimated Nodes: ~{estimated_nodes_layer_n:,}")
                print("    (Note: This is a rough estimate, actual structure may vary)")

        print(f"\n{'=' * 80}")
        print("Limitation Notice:")
        print(f"{'=' * 80}")
        print("""
pgvector's HNSW index does not expose layer structure via SQL queries.
The actual layer hierarchy and node connections are stored in binary format
and require specialized tools or pgvector extension functions to access.

To get actual layer structure, you would need to:
1. Use pgvector's internal APIs (if available)
2. Parse the index binary structure directly
3. Use external tools designed for pgvector index introspection

The estimates above are based on HNSW algorithm properties and may not
reflect the actual structure of your specific index.
        """)

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error estimating HNSW layers: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function - CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Query HNSW index structure and layer information"
    )
    parser.add_argument(
        "--table",
        type=str,
        default=None,
        help="Table name (e.g., 'data_askany_faq_vectors' or 'data_askany_docs_vectors')",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Specific index name to query (optional)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer number to query (shows estimation, not actual structure)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Query all HNSW indexes (FAQ and Docs)",
    )

    args = parser.parse_args()

    if args.all:
        # Query both FAQ and Docs tables
        faq_table = f"data_{settings.faq_vector_table_name}"
        docs_table = f"data_{settings.docs_vector_table_name}"

        print("Querying FAQ HNSW Index:")
        query_hnsw_index_info(faq_table, args.index)
        if args.layer is not None:
            query_hnsw_layers_estimate(faq_table, args.layer)

        print("\n" + "=" * 80 + "\n")

        print("Querying Docs HNSW Index:")
        query_hnsw_index_info(docs_table, args.index)
        if args.layer is not None:
            query_hnsw_layers_estimate(docs_table, args.layer)
    elif args.table:
        query_hnsw_index_info(args.table, args.index)
        if args.layer is not None:
            query_hnsw_layers_estimate(args.table, args.layer)
    else:
        # Default: query both tables
        faq_table = f"data_{settings.faq_vector_table_name}"
        docs_table = f"data_{settings.docs_vector_table_name}"

        print("Querying FAQ HNSW Index:")
        query_hnsw_index_info(faq_table, args.index)
        if args.layer is not None:
            query_hnsw_layers_estimate(faq_table, args.layer)

        print("\n" + "=" * 80 + "\n")

        print("Querying Docs HNSW Index:")
        query_hnsw_index_info(docs_table, args.index)
        if args.layer is not None:
            query_hnsw_layers_estimate(docs_table, args.layer)


if __name__ == "__main__":
    main()

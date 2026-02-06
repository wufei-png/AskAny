#!/usr/bin/env python3
"""Check if documents are successfully ingested and print first n rows of data."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging to show INFO level messages
# This ensures logger.info() calls are visible when output is redirected
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",  # Simple format, just the message
    stream=sys.stderr,  # Output to stderr so it's captured by 2>&1
)

try:
    from llama_index.core.retrievers import VectorIndexRetriever

    LLAMA_INDEX_AVAILABLE = True
except (ImportError, SyntaxError) as e:
    print(f"⚠️  Warning: llama_index not available ({e})")
    print("Will use direct database queries instead.")
    LLAMA_INDEX_AVAILABLE = False

from askany.config import settings

if LLAMA_INDEX_AVAILABLE:
    from askany.ingest import VectorStoreManager
    from askany.main import initialize_llm


def check_faq_data(vector_store_manager, n: int = 5):
    """Check FAQ data and print first n rows."""
    print("\n" + "=" * 80)
    print("Checking FAQ Data")
    print("=" * 80)

    faq_index = vector_store_manager.get_faq_index()
    if faq_index is None:
        print("❌ FAQ index not found. FAQ data may not be ingested.")
        return

    # Create a retriever to get documents
    retriever = VectorIndexRetriever(
        index=faq_index,
        similarity_top_k=n,
    )

    # Use a dummy query to retrieve documents
    # We'll use an empty query or a generic query to get all documents
    from llama_index.core import QueryBundle

    query_bundle = QueryBundle("")

    try:
        nodes = retriever.retrieve(query_bundle)
        if not nodes:
            # Try with a more generic query
            query_bundle = QueryBundle("问题")
            nodes = retriever.retrieve(query_bundle)
    except Exception as e:
        print(f"⚠️  Error retrieving FAQ nodes: {e}")
        print("Trying to get nodes directly from index...")
        # Try to get nodes from docstore
        try:
            docstore = faq_index.storage_context.docstore
            all_doc_ids = list(docstore.docs.keys())
            print(f"Found {len(all_doc_ids)} documents in FAQ docstore")
            if all_doc_ids:
                nodes = [docstore.get_document(doc_id) for doc_id in all_doc_ids[:n]]
            else:
                nodes = []
        except Exception as e2:
            print(f"❌ Error accessing FAQ docstore: {e2}")
            return

    if not nodes:
        print("❌ No FAQ documents found in the index.")
        return

    print(
        f"✅ Found {len(nodes)} FAQ document(s) (showing first {min(n, len(nodes))}):\n"
    )

    for i, node in enumerate(nodes[:n], 1):
        print(f"--- FAQ Document {i} ---")
        print(f"Node ID: {node.node_id}")
        if hasattr(node, "ref_doc_id") and node.ref_doc_id:
            print(f"Ref Doc ID: {node.ref_doc_id}")

        # Print metadata
        if hasattr(node, "metadata") and node.metadata:
            print("Metadata:")
            for key, value in node.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")

        # Print text (truncated if too long)
        text = node.text if hasattr(node, "text") else str(node)
        if len(text) > 500:
            print(f"Text (truncated):\n{text[:500]}...")
        else:
            print(f"Text:\n{text}")
        print()


def check_keyword_data(vector_store_manager, n: int = 5):
    """Check Keyword index data and print keyword mappings."""
    print("\n" + "=" * 80)
    print("Checking Keyword Index Data")
    print("=" * 80)

    keyword_index = vector_store_manager.get_faq_keyword_index()
    if keyword_index is None:
        print("❌ Keyword index not found. Keyword data may not be ingested.")
        return

    try:
        # Get keyword table structure
        index_struct = keyword_index.index_struct
        if hasattr(index_struct, "table") and index_struct.table:
            # KeywordTableIndex stores keywords in index_struct.table
            # table is a dict mapping keywords to lists of node_ids
            keyword_table = index_struct.table
            print(f"✅ Found {len(keyword_table)} keywords in keyword table")

            # Get all keywords and sort them
            all_keywords = sorted(keyword_table.keys())
            print(
                f"\n--- Showing first {min(n, len(all_keywords))} keywords and their mappings ---\n"
            )

            # Get docstore for looking up node details
            docstore = keyword_index.storage_context.docstore

            for i, keyword in enumerate(all_keywords[:n], 1):
                node_ids = keyword_table[keyword]
                # Convert to list if it's a set (some versions of llama_index use sets)
                if isinstance(node_ids, set):
                    node_ids = list(node_ids)
                elif not isinstance(node_ids, (list, tuple)):
                    node_ids = list(node_ids) if node_ids else []

                print(f"Keyword {i}: '{keyword}'")
                # Show first 5 node IDs
                node_ids_preview = node_ids[:5] if len(node_ids) > 5 else node_ids
                preview_str = ", ".join(str(nid) for nid in node_ids_preview)
                if len(node_ids) > 5:
                    preview_str += "..."
                print(f"  Maps to {len(node_ids)} node(s): [{preview_str}]")

                # Get node text for first node if available
                if node_ids:
                    try:
                        first_node = docstore.get_document(node_ids[0])
                        if hasattr(first_node, "text"):
                            text_preview = (
                                first_node.text[:100]
                                if len(first_node.text) > 100
                                else first_node.text
                            )
                            print(f"  First node preview: {text_preview}...")
                    except Exception:
                        pass
                print()
        else:
            print("⚠️  Keyword table structure not found or empty")
            # Fallback: show nodes from docstore
            docstore = keyword_index.storage_context.docstore
            all_doc_ids = list(docstore.docs.keys())
            print(f"Found {len(all_doc_ids)} documents in Keyword docstore")

            if all_doc_ids:
                nodes = [docstore.get_document(doc_id) for doc_id in all_doc_ids[:n]]
                print(
                    f"\n--- Showing first {min(n, len(all_doc_ids))} nodes (fallback) ---\n"
                )

                for i, node in enumerate(nodes, 1):
                    print(f"--- Keyword Node {i} ---")
                    print(f"Node ID: {node.node_id}")
                    if hasattr(node, "ref_doc_id") and node.ref_doc_id:
                        print(f"Ref Doc ID: {node.ref_doc_id}")

                    # Print text (truncated if too long)
                    text = node.text if hasattr(node, "text") else str(node)
                    if len(text) > 200:
                        print(f"Text (truncated):\n{text[:200]}...")
                    else:
                        print(f"Text:\n{text}")
                    print()
    except Exception as e:
        print(f"❌ Error accessing Keyword index: {e}")
        import traceback

        traceback.print_exc()


def check_docs_data(vector_store_manager, n: int = 5):
    """Check Docs data and print first n rows."""
    print("\n" + "=" * 80)
    print("Checking Docs Data")
    print("=" * 80)

    docs_index = vector_store_manager.get_docs_index()
    if docs_index is None:
        print("❌ Docs index not found. Docs data may not be ingested.")
        return

    # Create a retriever to get documents
    retriever = VectorIndexRetriever(
        index=docs_index,
        similarity_top_k=n,
    )

    # Use a dummy query to retrieve documents
    from llama_index.core import QueryBundle

    query_bundle = QueryBundle("")

    try:
        nodes = retriever.retrieve(query_bundle)
        if not nodes:
            # Try with a more generic query
            query_bundle = QueryBundle("文档")
            nodes = retriever.retrieve(query_bundle)
    except Exception as e:
        print(f"⚠️  Error retrieving Docs nodes: {e}")
        print("Trying to get nodes directly from index...")
        # Try to get nodes from docstore
        try:
            docstore = docs_index.storage_context.docstore
            all_doc_ids = list(docstore.docs.keys())
            print(f"Found {len(all_doc_ids)} documents in Docs docstore")
            if all_doc_ids:
                nodes = [docstore.get_document(doc_id) for doc_id in all_doc_ids[:n]]
            else:
                nodes = []
        except Exception as e2:
            print(f"❌ Error accessing Docs docstore: {e2}")
            return

    if not nodes:
        print("❌ No Docs documents found in the index.")
        return

    print(
        f"✅ Found {len(nodes)} Docs document(s) (showing first {min(n, len(nodes))}):\n"
    )

    for i, node in enumerate(nodes[:n], 1):
        print(f"--- Docs Document {i} ---")
        print(f"Node ID: {node.node_id}")
        if hasattr(node, "ref_doc_id") and node.ref_doc_id:
            print(f"Ref Doc ID: {node.ref_doc_id}")

        # Print metadata
        if hasattr(node, "metadata") and node.metadata:
            print("Metadata:")
            for key, value in node.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")

        # Print text (truncated if too long)
        text = node.text if hasattr(node, "text") else str(node)
        if len(text) > 500:
            print(f"Text (truncated):\n{text[:500]}...")
        else:
            print(f"Text:\n{text}")
        print()


def check_database_tables(n: int = 5):
    """Check database tables directly using SQL and print first n rows."""
    print("\n" + "=" * 80)
    print("Checking Database Tables (Direct SQL Query)")
    print("=" * 80)

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

        # Helper function to check if table exists
        def table_exists(table_name):
            """Check if a table exists in the database."""
            try:
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
                return cur.fetchone()[0]
            except Exception:
                return False

        # Check FAQ table
        # PGVectorStore uses "data_{table_name}" as the actual table name
        faq_table = settings.faq_vector_table_name
        faq_actual_table = f"data_{faq_table}"

        # Check both possible table names
        faq_table_to_use = None
        if table_exists(faq_actual_table):
            faq_table_to_use = faq_actual_table
        elif table_exists(faq_table):
            faq_table_to_use = faq_table

        if faq_table_to_use is None:
            print(
                f"ℹ️  FAQ table '{faq_table}' or '{faq_actual_table}': not found "
                f"(table not created yet, run --ingest first)"
            )
        else:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {faq_table_to_use};")
                faq_count = cur.fetchone()[0]
                print(f"✅ FAQ table '{faq_table_to_use}': {faq_count} rows")

                if faq_count > 0:
                    print(f"\n--- First {min(n, faq_count)} rows from FAQ table ---")
                    # Get table structure first
                    cur.execute(
                        """
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = %s
                        ORDER BY ordinal_position;
                    """,
                        (faq_table_to_use,),
                    )
                    columns = [row[0] for row in cur.fetchall()]
                    print(f"Columns: {', '.join(columns)}")

                    # Query first n rows (embedding is vector type, use vector_dims to get dimension)
                    # Note: ref_doc_id is stored in metadata_ JSON field, not as a separate column
                    cur.execute(
                        f"""
                        SELECT id, node_id, metadata_->>'ref_doc_id' as ref_doc_id, text, metadata_, 
                               vector_dims(embedding) as embedding_dim
                        FROM {faq_table_to_use}
                        LIMIT %s;
                        """,
                        (n,),
                    )
                    rows = cur.fetchall()
                    for i, row in enumerate(rows, 1):
                        print(f"\n--- FAQ Row {i} ---")
                        print(f"ID: {row[0]}")
                        print(f"Node ID: {row[1]}")
                        print(f"Ref Doc ID: {row[2] if row[2] else '(None)'}")
                        if row[3]:  # text
                            text = row[3]
                            if len(text) > 500:
                                print(f"Text (truncated):\n{text[:500]}...")
                            else:
                                print(f"Text:\n{text}")
                        if row[4]:  # metadata_
                            try:
                                metadata = (
                                    row[4]
                                    if isinstance(row[4], dict)
                                    else json.loads(row[4])
                                )
                                print("Metadata:")
                                for key, value in metadata.items():
                                    if isinstance(value, (str, int, float, bool)):
                                        print(f"  {key}: {value}")
                                    else:
                                        print(f"  {key}: {type(value).__name__}")
                            except Exception:
                                print(f"Metadata: {row[4]}")
                        if row[5]:  # embedding_dim
                            print(f"Embedding dimension: {row[5]}")
            except Exception as e:
                print(f"⚠️  Error querying FAQ table: {e}")
                conn.rollback()  # Rollback to allow subsequent queries

        # Check Docs table
        # PGVectorStore uses "data_{table_name}" as the actual table name
        docs_table = settings.docs_vector_table_name
        docs_actual_table = f"data_{docs_table}"

        # Check both possible table names
        docs_table_to_use = None
        if table_exists(docs_actual_table):
            docs_table_to_use = docs_actual_table
        elif table_exists(docs_table):
            docs_table_to_use = docs_table

        if docs_table_to_use is None:
            print(
                f"\nℹ️  Docs table '{docs_table}' or '{docs_actual_table}': not found "
                f"(table not created yet, run --ingest first)"
            )
        else:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {docs_table_to_use};")
                docs_count = cur.fetchone()[0]
                print(f"\n✅ Docs table '{docs_table_to_use}': {docs_count} rows")

                if docs_count > 0:
                    print(f"\n--- First {min(n, docs_count)} rows from Docs table ---")
                    # Get table structure first
                    cur.execute(
                        """
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = %s
                        ORDER BY ordinal_position;
                    """,
                        (docs_table_to_use,),
                    )
                    columns = [row[0] for row in cur.fetchall()]
                    print(f"Columns: {', '.join(columns)}")

                    # Query first n rows (embedding is vector type, use vector_dims to get dimension)
                    # Note: ref_doc_id is stored in metadata_ JSON field, not as a separate column
                    cur.execute(
                        f"""
                        SELECT id, node_id, metadata_->>'ref_doc_id' as ref_doc_id, text, metadata_, 
                               vector_dims(embedding) as embedding_dim
                        FROM {docs_table_to_use}
                        LIMIT %s;
                        """,
                        (n,),
                    )
                    rows = cur.fetchall()
                    for i, row in enumerate(rows, 1):
                        print(f"\n--- Docs Row {i} ---")
                        print(f"ID: {row[0]}")
                        print(f"Node ID: {row[1]}")
                        print(f"Ref Doc ID: {row[2] if row[2] else '(None)'}")
                        if row[3]:  # text
                            text = row[3]
                            if len(text) > 500:
                                print(f"Text (truncated):\n{text[:500]}...")
                            else:
                                print(f"Text:\n{text}")
                        if row[4]:  # metadata_
                            try:
                                metadata = (
                                    row[4]
                                    if isinstance(row[4], dict)
                                    else json.loads(row[4])
                                )
                                print("Metadata:")
                                for key, value in metadata.items():
                                    if isinstance(value, (str, int, float, bool)):
                                        print(f"  {key}: {value}")
                                    else:
                                        print(f"  {key}: {type(value).__name__}")
                            except Exception:
                                print(f"Metadata: {row[4]}")
                        if row[5]:  # embedding_dim
                            print(f"Embedding dimension: {row[5]}")
            except Exception as e:
                print(f"⚠️  Error querying Docs table: {e}")
                conn.rollback()  # Rollback to allow subsequent queries

        # Check legacy table if exists
        legacy_table = settings.vector_table_name
        try:
            cur.execute(f"SELECT COUNT(*) FROM {legacy_table};")
            legacy_count = cur.fetchone()[0]
            print(f"\nℹ️  Legacy table '{legacy_table}': {legacy_count} rows")
        except Exception:
            print(f"\nℹ️  Legacy table '{legacy_table}': not found (this is OK)")

        cur.close()
        conn.close()

    except ImportError:
        print("❌ psycopg2 not available, cannot check database directly")
        print("Please install: pip install psycopg2-binary")
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback

        traceback.print_exc()


def check_ingested_data(n: int = 5, skip_db: bool = False, llm=None, embed_model=None):
    """Check ingested data - can be called directly with pre-initialized models.

    Args:
        n: Number of rows to print
        skip_db: Whether to skip database check
        llm: Pre-initialized LLM (optional)
        embed_model: Pre-initialized embedding model (optional)
    """

    # Always check database first (doesn't require llama_index)
    print("=" * 80)
    print("Step 1: Checking Database Tables (Direct SQL)")
    print("=" * 80)
    if not skip_db:
        check_database_tables(n=n)
    else:
        print("Skipping database check")

    # If llama_index is available, try to use it for more detailed checks
    if LLAMA_INDEX_AVAILABLE:
        print("\n" + "=" * 80)
        print("Step 2: Checking via llama_index (if available)")
        print("=" * 80)
        try:
            # Use provided models or initialize new ones
            if llm is None or embed_model is None:
                print("Initializing LLM and embedding model...")
                llm, embed_model = initialize_llm()
            else:
                print("Using provided LLM and embedding model...")

            print("Initializing vector store manager...")
            vector_store_manager = VectorStoreManager(embed_model, llm=llm)

            # Try to initialize indexes
            print("Loading FAQ index...")
            try:
                vector_store_manager.initialize_faq_index()
                print("✅ FAQ index loaded")
                # Check FAQ data
                check_faq_data(vector_store_manager, n=n)
            except Exception as e:
                print(f"⚠️  Could not load FAQ index: {e}")
                print("This is OK if database check above shows data exists.")

            print("Loading Keyword index...")
            try:
                keyword_index = vector_store_manager.get_faq_keyword_index()
                if keyword_index is not None:
                    print("✅ Keyword index loaded")
                    # Check Keyword data
                    check_keyword_data(vector_store_manager, n=n)
                else:
                    print("⚠️  Keyword index not found (may not be created yet)")
            except Exception as e:
                print(f"⚠️  Could not load Keyword index: {e}")
                print("This is OK if keyword index was not created.")

            print("Loading Docs index...")
            try:
                vector_store_manager.initialize_docs_index()
                print("✅ Docs index loaded")
                # Check Docs data
                check_docs_data(vector_store_manager, n=n)
            except Exception as e:
                print(f"⚠️  Could not load Docs index: {e}")
                print("This is OK if database check above shows data exists.")
        except Exception as e:
            print(f"⚠️  Error using llama_index: {e}")
            print("This is OK - database check above should show if data is ingested.")
            import traceback

            traceback.print_exc()
    else:
        print("\n" + "=" * 80)
        print("Step 2: Skipping llama_index checks (not available)")
        print("=" * 80)
        print("Database check above should confirm if data is ingested.")

    print("\n" + "=" * 80)
    print("Check completed!")
    print("=" * 80)


def main():
    """Main function - CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check if documents are successfully ingested"
    )
    parser.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=100,
        help="Number of rows to print (default: 5)",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip direct database table check",
    )

    args = parser.parse_args()
    check_ingested_data(n=args.num_rows, skip_db=args.skip_db)


if __name__ == "__main__":
    main()

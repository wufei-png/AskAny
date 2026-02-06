#!/usr/bin/env python3
"""Direct query from PGVectorStore - query nodes directly from vector store indexes."""

import sys
from pathlib import Path

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import QueryBundle
from llama_index.core.embeddings import BaseEmbedding

from askany.ingest import VectorStoreManager


def print_node(node, index: int, prefix: str = ""):
    """Print node information.

    Args:
        node: NodeWithScore object
        index: Node index number
        prefix: Optional prefix for node type (e.g., "FAQ", "DOCS")
    """
    node_label = f"{prefix} Node {index}" if prefix else f"Node {index}"
    print(f"{'=' * 80}")
    print(f"{node_label}:")
    print(
        f"Score: {node.score if hasattr(node, 'score') and node.score is not None else 'N/A'}"
    )
    print(f"Node ID: {node.node.node_id if hasattr(node.node, 'node_id') else 'N/A'}")
    print(f"Text: {node.node.text[:500] if hasattr(node.node, 'text') else 'N/A'}...")
    if hasattr(node.node, "metadata") and node.node.metadata:
        print(f"Metadata: {node.node.metadata}")
    print()


def query_from_vector_store(
    query_text: str,
    query_type: str,
    embed_model: BaseEmbedding,
    llm=None,
    similarity_top_k: int = 5,
):
    """Query nodes directly from PGVectorStore.

    Args:
        query_text: Query text to search
        query_type: Query type ("FAQ", "DOCS", or "AUTO")
        embed_model: Embedding model instance
        llm: Language model instance (optional)
        similarity_top_k: Number of top similar nodes to retrieve

    Returns:
        None (prints results directly)
    """
    # Initialize vector store manager
    vector_store_manager = VectorStoreManager(embed_model, llm=llm)

    # Try to initialize separate indexes (for new architecture)
    try:
        vector_store_manager.initialize_faq_index()
        vector_store_manager.initialize_docs_index()
        print("Using separate indexes for FAQ and docs")
    except Exception as e:
        # Fallback to legacy single index
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Separate indexes not available, using legacy index: {e}")
        vector_store_manager.initialize()

    # Parse query type
    query_type_upper = query_type.upper()

    print(f"\n{'=' * 80}")
    print(f"Query Type: {query_type_upper}")
    print(f"Query Text: {query_text}")
    print(f"{'=' * 80}\n")

    # Query based on query type
    if query_type_upper == "FAQ":
        # Query FAQ index
        faq_index = vector_store_manager.get_faq_index()
        if faq_index is None:
            print("Error: FAQ index not available. Please run --ingest first.")
            return

        # Create retriever from FAQ index
        retriever = faq_index.as_retriever(similarity_top_k=similarity_top_k)
        query_bundle = QueryBundle(query_text)
        nodes = retriever.retrieve(query_bundle)

        print(f"Found {len(nodes)} nodes from FAQ index:\n")
        for i, node in enumerate(nodes, 1):
            print_node(node, i)

    elif query_type_upper == "DOCS":
        # Query DOCS index
        docs_index = vector_store_manager.get_docs_index()
        if docs_index is None:
            print("Error: DOCS index not available. Please run --ingest first.")
            return

        # Create retriever from DOCS index
        retriever = docs_index.as_retriever(similarity_top_k=similarity_top_k)
        query_bundle = QueryBundle(query_text)
        nodes = retriever.retrieve(query_bundle)

        print(f"Found {len(nodes)} nodes from DOCS index:\n")
        for i, node in enumerate(nodes, 1):
            print_node(node, i)

    elif query_type_upper == "AUTO":
        # Try FAQ first, then DOCS
        faq_index = vector_store_manager.get_faq_index()
        docs_index = vector_store_manager.get_docs_index()

        query_bundle = QueryBundle(query_text)

        # Query FAQ if available
        if faq_index:
            print("Querying FAQ index...")
            faq_retriever = faq_index.as_retriever(similarity_top_k=similarity_top_k)
            faq_nodes = faq_retriever.retrieve(query_bundle)
            print(f"Found {len(faq_nodes)} nodes from FAQ index:\n")
            for i, node in enumerate(faq_nodes, 1):
                print_node(node, i, prefix="FAQ")

        # Query DOCS if available
        if docs_index:
            print("\nQuerying DOCS index...")
            docs_retriever = docs_index.as_retriever(similarity_top_k=similarity_top_k)
            docs_nodes = docs_retriever.retrieve(query_bundle)
            print(f"Found {len(docs_nodes)} nodes from DOCS index:\n")
            for i, node in enumerate(docs_nodes, 1):
                print_node(node, i, prefix="DOCS")

        if not faq_index and not docs_index:
            print("Error: No indexes available. Please run --ingest first.")
            return
    else:
        print(f"Error: Unknown query type: {query_type_upper}")
        print("Supported types: AUTO, FAQ, DOCS")
        return


if __name__ == "__main__":
    import argparse

    from askany.main import initialize_llm

    parser = argparse.ArgumentParser(
        description="Query nodes directly from PGVectorStore"
    )
    parser.add_argument(
        "--query-text",
        type=str,
        required=True,
        default="优化cassandra.yml配置",
        help="Query text to search",
    )
    parser.add_argument(
        "--query-type",
        type=str,
        default="DOCS",
        help="Query type (AUTO, FAQ, DOCS)",
    )
    parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=5,
        help="Number of top similar nodes to retrieve (default: 5)",
    )

    args = parser.parse_args()

    # Initialize LLM and embedding models
    llm, embed_model = initialize_llm()

    # Query from vector store
    query_from_vector_store(
        query_text=args.query_text,
        query_type=args.query_type,
        embed_model=embed_model,
        llm=llm,
        similarity_top_k=args.similarity_top_k,
    )

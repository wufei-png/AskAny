#!/usr/bin/env python3
"""Test script to verify LightRAG merge with LlamaIndex nodes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.main import initialize_llm
from askany.ingest import VectorStoreManager
from askany.rag.lightrag_adapter import LightRAGAdapter
from askany.rag.lightrag_merge import merge_lightrag_with_llamaindex
from askany.config import Settings
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from llama_index.core import QueryBundle


def main():
    query_text = "IPS fpach"
    query_type = "DOCS"

    print(f"Query: {query_text}")
    print("=" * 80)

    llm, embed_model = initialize_llm()
    device = Settings().device

    vector_store_manager = VectorStoreManager(embed_model, llm=llm)
    vector_store_manager.initialize_faq_index()
    vector_store_manager.initialize_docs_index()

    docs_index = vector_store_manager.get_docs_index()
    retriever = docs_index.as_retriever(similarity_top_k=5)
    query_bundle = QueryBundle(query_text)
    llama_nodes = retriever.retrieve(query_bundle)

    print(f"\n[LlamaIndex] Retrieved {len(llama_nodes)} nodes")

    lightrag_adapter = LightRAGAdapter()
    lightrag_nodes = lightrag_adapter.retrieve(
        query=query_text,
        top_k=5,
        mode="hybrid",
    )

    print(f"\n[LightRAG] Retrieved {len(lightrag_nodes)} nodes")

    print("\n" + "=" * 80)
    print("[MERGE] Calling merge_lightrag_with_llamaindex...")

    local_file_search = None

    merged_nodes = merge_lightrag_with_llamaindex(
        llama_nodes,
        lightrag_nodes,
        query=query_text,
        top_k=5,
        local_file_search=local_file_search,
    )

    print(f"\n[MERGED] Total nodes after merge: {len(merged_nodes)}")
    for i, node in enumerate(merged_nodes):
        meta = node.node.metadata or {}
        source_kind = meta.get("source_kind", "unknown")
        related = meta.get("related_lightrag_chunks", [])
        print(f"  Node {i} ({source_kind}): {node.node.text[:80]}...")
        if related:
            print(f"    -> Has {len(related)} related LightRAG chunks")


if __name__ == "__main__":
    main()

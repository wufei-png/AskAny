#!/usr/bin/env python3
"""Minimal test to debug vector retrieval."""

import logging
logging.basicConfig(level=logging.DEBUG)

from askany.config import settings
print(f"Table name from config: {settings.docs_vector_table_name}")

from askany.main import initialize_llm
llm, embed_model = initialize_llm()

from askany.ingest import VectorStoreManager
vsm = VectorStoreManager(embed_model, llm=llm)
vsm.initialize_docs_index()

# Check the actual table name being used
print(f"Docs vector store table: {vsm.docs_vector_store.table_name if hasattr(vsm.docs_vector_store, 'table_name') else 'unknown'}")

# Try a simple query
docs_index = vsm.get_docs_index()
print(f"Docs index type: {type(docs_index)}")

# Test retrieval
retriever = docs_index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("deploy")
print(f"Retrieved {len(nodes)} nodes for query 'deploy'")

#!/usr/bin/env python3
"""Direct database test to check vector retrieval."""

from askany.config import settings
from askany.main import initialize_llm
from askany.ingest import VectorStoreManager

# Initialize
llm, embed_model = initialize_llm()
vector_store_manager = VectorStoreManager(embed_model, llm=llm)

# Initialize docs index
vector_store_manager.initialize_docs_index()
docs_index = vector_store_manager.get_docs_index()

# Test query
query = "How to deploy?"
print(f"Query: {query}")
print(f"Embedding model: {embed_model}")

# Get query embedding
query_embedding = embed_model.get_query_embedding(query)
print(f"Query embedding dimension: {len(query_embedding)}")

# Try to retrieve
retriever = docs_index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve(query)
print(f"Retrieved {len(nodes)} nodes")

for i, node in enumerate(nodes[:3]):
    print(f"\nNode {i+1}:")
    print(f"  Score: {node.score}")
    print(f"  Text: {node.text[:100]}...")

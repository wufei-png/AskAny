#!/usr/bin/env python3
"""Debug script to test vector store query directly."""

from askany.config import settings
from askany.main import initialize_llm, get_device
from askany.ingest import VectorStoreManager
from llama_index.core import QueryBundle

def test_direct_query():
    """Test direct query from vector store."""
    print("Initializing components...")
    llm, embed_model = initialize_llm()
    device = get_device()
    
    # Initialize vector store manager
    vector_store_manager = VectorStoreManager(embed_model, llm=llm)
    
    # Initialize indexes
    print("\n=== Initializing FAQ index ===")
    vector_store_manager.initialize_faq_index()
    faq_index = vector_store_manager.get_faq_index()
    print(f"FAQ index: {faq_index}")
    
    print("\n=== Initializing Docs index ===")
    vector_store_manager.initialize_docs_index()
    docs_index = vector_store_manager.get_docs_index()
    print(f"Docs index: {docs_index}")
    
    # Test FAQ query
    print("\n=== Testing FAQ query ===")
    query_text = "acl error code = 507011"
    print(f"Query: {query_text}")
    
    if faq_index:
        retriever = faq_index.as_retriever(similarity_top_k=5)
        query_bundle = QueryBundle(query_text)
        nodes = retriever.retrieve(query_bundle)
        print(f"Retrieved {len(nodes)} nodes from FAQ")
        for i, node in enumerate(nodes[:3]):
            score = node.score if hasattr(node, 'score') and node.score else None
            print(f"  Node {i+1}: score={score}, text={node.node.text[:100]}...")
    
    # Test Docs query
    print("\n=== Testing Docs query ===")
    query_text = "方舟部署"
    print(f"Query: {query_text}")
    
    if docs_index:
        retriever = docs_index.as_retriever(similarity_top_k=5)
        query_bundle = QueryBundle(query_text)
        nodes = retriever.retrieve(query_bundle)
        print(f"Retrieved {len(nodes)} nodes from Docs")
        for i, node in enumerate(nodes[:3]):
            score = node.score if hasattr(node, 'score') and node.score else None
            print(f"  Node {i+1}: score={score}, text={node.node.text[:100]}...")

if __name__ == "__main__":
    test_direct_query()

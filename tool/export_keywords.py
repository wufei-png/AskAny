#!/usr/bin/env python3
"""Export keywords from keyword indices to word_freq.txt files."""

import sys
from pathlib import Path

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.ingest.vector_store import VectorStoreManager
from askany.main import initialize_llm
from tool.keyword_utils import export_keywords_to_word_freq


def main():
    """Main function to export keywords from keyword indices."""
    print("Initializing LLM and embedding models...")
    llm, embed_model = initialize_llm()

    print("Initializing VectorStoreManager...")
    vector_store_manager = VectorStoreManager(embed_model, llm=llm)

    print("Loading keyword indices...")
    # Load FAQ keyword index
    faq_keyword_index = vector_store_manager.load_faq_keyword_index()
    if faq_keyword_index:
        vector_store_manager.faq_keyword_index = faq_keyword_index
        print("✅ FAQ keyword index loaded")

    # Load Docs keyword index
    docs_keyword_index = vector_store_manager.load_docs_keyword_index()
    if docs_keyword_index:
        vector_store_manager.docs_keyword_index = docs_keyword_index
        print("✅ Docs keyword index loaded")

    print("\nExporting keywords to word_freq.txt files...")
    export_keywords_to_word_freq(vector_store_manager)

    print("\n✅ Export completed!")


if __name__ == "__main__":
    main()

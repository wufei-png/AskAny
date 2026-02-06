"""Document ingestion functions."""

import hashlib
import pickle
from logging import getLogger
from pathlib import Path

from llama_index.core import Document, Settings

from askany.config import settings

from .json_parser import JSONParser
from .markdown_parser import MarkdownParser
from .vector_store import VectorStoreManager

logger = getLogger(__name__)


def ingest_documents(embed_model, llm=None):
    """Ingest documents into separate vector stores for FAQ and docs.

    Args:
        embed_model: Embedding model for generating vectors
        llm: Language model for KeywordTableIndex (optional, uses Settings.llm if not provided)
    """

    # Initialize vector store manager
    vector_store_manager = VectorStoreManager(embed_model, llm=llm)

    # Separate FAQ and docs documents
    faq_docs = []
    docs_docs = []

    print("Parsing JSON files (FAQ)...")
    # Parse JSON files (FAQ)
    json_parser = JSONParser()
    json_dir = Path(settings.json_dir)
    if json_dir.exists():
        json_docs = json_parser.parse_directory(json_dir)
        print(f"Parsed {len(json_docs)} JSON FAQ documents")

        # Log detailed information about parsed documents
        logger.info(
            f"[ingest_documents] json_docs: count={len(json_docs)}, "
            f"types={[type(d).__name__ for d in json_docs[:5]]} (showing first 5)"
        )
        # Check for any non-Document items
        for idx, doc in enumerate(json_docs):
            if not isinstance(doc, Document):
                logger.error(
                    f"[ingest_documents] Non-Document at index {idx}: "
                    f"type={type(doc).__name__}, value={str(doc)[:200]}"
                )
            else:
                logger.debug(
                    f"[ingest_documents] Document {idx}: id_={getattr(doc, 'id_', 'None')}, "
                    f"type={type(doc).__name__}"
                )

        faq_docs = json_docs
        logger.info(
            f"[ingest_documents] faq_docs assigned: count={len(faq_docs)}, "
            f"all are Document instances: {all(isinstance(d, Document) for d in faq_docs)}"
        )

    # Initialize and add FAQ documents to separate FAQ vector store
    # if faq_docs:
    #     print("Initializing FAQ vector store...")
    #     vector_store_manager.initialize_faq_index()
    #     print("Adding FAQ documents to FAQ vector store...")

    #     # Log before calling add_faq_documents
    #     logger.info(
    #         f"[ingest_documents] Before add_faq_documents: count={len(faq_docs)}, "
    #         f"types={[type(d).__name__ for d in faq_docs[:10]]} (showing first 10)"
    #     )
    #     # Detailed check
    #     for idx, doc in enumerate(faq_docs[:10]):  # Check first 10
    #         logger.debug(
    #             f"[ingest_documents] faq_docs[{idx}]: type={type(doc).__name__}, "
    #             f"isinstance(Document)={isinstance(doc, Document)}, "
    #             f"has id_={hasattr(doc, 'id_') if isinstance(doc, Document) else 'N/A'}"
    #         )
    #     # Check for any non-Document items
    #     non_doc_indices = [
    #         idx for idx, doc in enumerate(faq_docs) if not isinstance(doc, Document)
    #     ]
    #     if non_doc_indices:
    #         logger.error(
    #             f"[ingest_documents] Found non-Document items at indices: {non_doc_indices[:20]}"
    #         )
    #         for idx in non_doc_indices[:5]:  # Show first 5 problematic items
    #             logger.error(
    #                 f"[ingest_documents] faq_docs[{idx}]: type={type(faq_docs[idx]).__name__}, "
    #                 f"value={str(faq_docs[idx])[:300]}"
    #             )

    #     vector_store_manager.add_faq_documents(faq_docs)
    #     print("Creating keyword index for FAQ...")
    #     # Use llm from Settings if available, otherwise use the passed llm
    #     keyword_llm = llm if llm else Settings.llm
    #     vector_store_manager.create_faq_keyword_index(faq_docs, llm=keyword_llm)
    #     print(f"Created FAQ keyword index with {len(faq_docs)} FAQ documents")

    print("Parsing Markdown files (Docs)...")
    # Parse Markdown files (Docs)
    markdown_dir = Path(settings.markdown_dir)
    if markdown_dir.exists():
        # Generate cache file path based on markdown_dir and split_mode
        # Use hash of absolute path + split_mode to create unique cache file
        cache_key = f"{markdown_dir.resolve()}_{settings.markdown_split_mode}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = Path(settings.storage_dir) / "docs_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"docs_nodes_{cache_hash}.pkl"

        # Try to load from cache
        if cache_file.exists():
            try:
                print(f"Loading cached docs nodes from {cache_file}...")
                with open(cache_file, "rb") as f:
                    docs_docs = pickle.load(f)
                print(f"Loaded {len(docs_docs)} cached Markdown nodes")
                logger.info(
                    f"[ingest_documents] Loaded {len(docs_docs)} docs nodes from cache: {cache_file}"
                )
            except Exception as e:
                logger.warning(
                    f"[ingest_documents] Failed to load cache from {cache_file}: {e}. "
                    "Will re-parse markdown files."
                )
                docs_docs = None
        else:
            docs_docs = None

        # Parse if cache doesn't exist or loading failed
        if docs_docs is None:
            markdown_parser = MarkdownParser(
                embed_model, split_mode=settings.markdown_split_mode
            )
            markdown_nodes = markdown_parser.parse_directory(markdown_dir)
            print(f"Parsed {len(markdown_nodes)} Markdown nodes")
            docs_docs = markdown_nodes

            # Save to cache
            try:
                print(f"Saving docs nodes to cache: {cache_file}...")
                with open(cache_file, "wb") as f:
                    pickle.dump(docs_docs, f)
                print(f"Saved {len(docs_docs)} nodes to cache")
                logger.info(
                    f"[ingest_documents] Saved {len(docs_docs)} docs nodes to cache: {cache_file}"
                )
            except Exception as e:
                logger.warning(
                    f"[ingest_documents] Failed to save cache to {cache_file}: {e}"
                )

    # Initialize and add docs documents to separate docs vector store
    if docs_docs:
        print("Initializing docs vector store...")
        vector_store_manager.initialize_docs_index()
        print("Adding documentation to docs vector store...")
        vector_store_manager.add_docs_nodes(docs_docs)
        print(f"Added {len(docs_docs)} nodes to docs vector store")
        # Only create keyword index if enabled in config
        if settings.using_docs_keyword_index:
            print("Creating keyword index for docs...")
            # Use llm from Settings if available, otherwise use the passed llm
            keyword_llm = llm if llm else Settings.llm
            if keyword_llm:
                vector_store_manager.create_docs_keyword_index(
                    docs_docs, llm=keyword_llm
                )
                print(f"Created docs keyword index with {len(docs_docs)} nodes")
            else:
                print(
                    "Warning: No LLM available, skipping docs keyword index creation. "
                    "Keyword index requires LLM for keyword extraction."
                )
        else:
            print(
                "Skipping docs keyword index creation (using_docs_keyword_index=False)"
            )

    print("Document ingestion completed!")
    return vector_store_manager

"""Vector store manager for PGVector."""

import json
import os
import tempfile
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psycopg2
from llama_index.core import (
    Document,
    KeywordTableIndex,
    VectorStoreIndex,
    load_index_from_storage,
    load_indices_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import BaseNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from tqdm import tqdm

from askany.config import settings
from askany.ingest.custom_keyword_index import (
    CustomKeywordTableIndex,
    get_global_keyword_extractor,
)

logger = getLogger(__name__)


def _get_keyword_index_class():
    """Get the appropriate KeywordTableIndex class based on configuration.

    Returns:
        CustomKeywordTableIndex if using_custom_keyword_index is True,
        otherwise KeywordTableIndex
    """
    if settings.using_custom_keyword_index:
        return CustomKeywordTableIndex
    return KeywordTableIndex


def _wrap_loaded_keyword_index(
    loaded_index: KeywordTableIndex,
) -> KeywordTableIndex:
    """Wrap a loaded KeywordTableIndex as CustomKeywordTableIndex if enabled.

    This function creates a CustomKeywordTableIndex instance using the same
    internal structure as the loaded index, allowing us to use custom
    keyword extraction methods. Only wraps if using_custom_keyword_index is True.

    Args:
        loaded_index: The loaded KeywordTableIndex instance

    Returns:
        CustomKeywordTableIndex instance if enabled, otherwise the original index
    """
    # Only wrap if custom keyword index is enabled
    if not settings.using_custom_keyword_index:
        return loaded_index

    # Get global keyword extractor if available
    keyword_extractor = get_global_keyword_extractor()

    # Create CustomKeywordTableIndex using the same internal structure
    custom_index = CustomKeywordTableIndex(
        nodes=None,  # Don't rebuild from nodes
        index_struct=loaded_index.index_struct,
        llm=loaded_index._llm,
        keyword_extract_template=loaded_index.keyword_extract_template,
        max_keywords_per_chunk=loaded_index.max_keywords_per_chunk,
        use_async=loaded_index._use_async,
        show_progress=False,
        keyword_extractor=keyword_extractor,
        storage_context=loaded_index.storage_context,
    )
    # Copy docstore and other necessary attributes
    custom_index._docstore = loaded_index._docstore
    custom_index._object_map = loaded_index._object_map
    # Copy index_id if it exists
    if hasattr(loaded_index, "_index_id"):
        custom_index._index_id = loaded_index._index_id
    return custom_index


class VectorStoreManager:
    """Manager for PGVector vector store.

    Supports separate indexes for FAQ and docs:
    - FAQ: VectorStoreIndex + KeywordTableIndex
    - Docs: VectorStoreIndex only
    """

    def __init__(self, embedding_model: BaseEmbedding, llm=None):
        """Initialize vector store manager.

        Args:
            embedding_model: Embedding model for generating vectors
            llm: Language model for KeywordTableIndex (optional)
        """
        self.embedding_model = embedding_model
        self.llm = llm
        # Legacy single index (for backward compatibility)
        self.vector_store: Optional[PGVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        self.keyword_index: Optional[KeywordTableIndex] = None

        # Separate indexes for FAQ and docs
        self.faq_vector_store: Optional[PGVectorStore] = None
        self.faq_index: Optional[VectorStoreIndex] = None
        self.faq_keyword_index: Optional[KeywordTableIndex] = None

        self.docs_vector_store: Optional[PGVectorStore] = None
        self.docs_index: Optional[VectorStoreIndex] = None
        self.docs_keyword_index: Optional[KeywordTableIndex] = None

    def _get_hnsw_kwargs(self) -> Optional[Dict[str, Any]]:
        """Get HNSW index configuration.

        Returns:
            Dictionary with HNSW parameters or None if HNSW is disabled
        """
        if not settings.enable_hnsw:
            return None

        return {
            "hnsw_m": settings.hnsw_m,
            "hnsw_ef_construction": settings.hnsw_ef_construction,
            "hnsw_ef_search": settings.hnsw_ef_search,
            "hnsw_dist_method": settings.hnsw_dist_method,
        }

    def _get_db_connection(self):
        """Get a database connection for direct SQL operations.

        Returns:
            psycopg2 connection object
        """
        return psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )

    def _wait_for_db_connection(
        self, max_retries: int = 5, retry_delay: float = 2.0
    ) -> bool:
        """Wait for database connection to be available.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if connection successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                conn = self._get_db_connection()
                conn.close()
                logger.debug(f"Database connection successful on attempt {attempt + 1}")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Database connection failed after {max_retries} attempts: {e}"
                    )
                    return False
        return False

    def _get_hnsw_index_name(self, table_name: str) -> str:
        """Get the HNSW index name for a given table.

        Args:
            table_name: Name of the vector table

        Returns:
            HNSW index name (format: {table_name}_embedding_idx)
        """
        return f"{table_name}_embedding_idx"

    def _drop_hnsw_index(self, vector_store: PGVectorStore) -> bool:
        """Drop HNSW index if it exists.

        Args:
            vector_store: PGVectorStore instance

        Returns:
            True if index was dropped, False if it didn't exist
        """
        if not settings.enable_hnsw:
            return False

        table_name = vector_store.table_name
        index_name = self._get_hnsw_index_name(table_name)

        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check if index exists
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE schemaname = 'public' 
                    AND indexname = %s
                )
                """,
                (index_name,),
            )
            index_exists = cursor.fetchone()[0]

            if index_exists:
                logger.info(
                    f"Dropping HNSW index '{index_name}' for table '{table_name}'..."
                )
                cursor.execute(f'DROP INDEX IF EXISTS "{index_name}"')
                conn.commit()
                logger.info(f"Successfully dropped HNSW index '{index_name}'")
                cursor.close()
                return True
            else:
                logger.debug(f"HNSW index '{index_name}' does not exist, skipping drop")
                cursor.close()
                return False
        except Exception as e:
            logger.error(f"Error dropping HNSW index '{index_name}': {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _create_hnsw_index(self, vector_store: PGVectorStore) -> None:
        """Create HNSW index for the vector store.

        Args:
            vector_store: PGVectorStore instance
        """
        if not settings.enable_hnsw:
            logger.warning("HNSW is disabled, skipping index creation")
            return

        table_name = vector_store.table_name
        index_name = self._get_hnsw_index_name(table_name)
        hnsw_kwargs = self._get_hnsw_kwargs()

        if not hnsw_kwargs:
            logger.warning("HNSW kwargs not available, skipping index creation")
            return

        # Get distance method operator class
        dist_method = hnsw_kwargs.get("hnsw_dist_method", "vector_cosine_ops")
        if dist_method == "vector_cosine_ops":
            opclass = "vector_cosine_ops"
        elif dist_method == "vector_l2_ops":
            opclass = "vector_l2_ops"
        else:
            opclass = "vector_cosine_ops"  # Default fallback

        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check if index already exists
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE schemaname = 'public' 
                    AND indexname = %s
                )
                """,
                (index_name,),
            )
            index_exists = cursor.fetchone()[0]

            if index_exists:
                logger.info(
                    f"HNSW index '{index_name}' already exists, skipping creation"
                )
                cursor.close()
                return

            # Temporarily increase maintenance_work_mem for faster index creation
            # This is a session-level setting that will be reset when connection closes
            maintenance_work_mem = settings.hnsw_maintenance_work_mem
            logger.info(
                f"Setting maintenance_work_mem to {maintenance_work_mem} for faster index creation..."
            )
            cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")
            # Note: No commit needed for SET commands, they're session-level

            # Build CREATE INDEX statement
            create_index_sql = f"""
                CREATE INDEX "{index_name}" 
                ON "{table_name}" 
                USING hnsw (embedding {opclass})
                WITH (
                    m = {hnsw_kwargs.get("hnsw_m", 16)},
                    ef_construction = {hnsw_kwargs.get("hnsw_ef_construction", 64)}
                )
            """

            logger.info(
                f"Creating HNSW index '{index_name}' for table '{table_name}' "
                f"with m={hnsw_kwargs.get('hnsw_m', 16)}, "
                f"ef_construction={hnsw_kwargs.get('hnsw_ef_construction', 64)}, "
                f"maintenance_work_mem={maintenance_work_mem}..."
            )
            cursor.execute(create_index_sql)
            conn.commit()
            logger.info(f"Successfully created HNSW index '{index_name}'")
            cursor.close()
        except Exception as e:
            logger.error(f"Error creating HNSW index '{index_name}': {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def initialize(self, table_name: Optional[str] = None) -> None:
        """Initialize the vector store connection.

        Args:
            table_name: Name of the vector table (defaults to settings)
        """
        table_name = table_name or settings.vector_table_name

        # Create vector store WITHOUT HNSW index configuration
        # Index will be created manually after all data is inserted for better performance
        self.vector_store = PGVectorStore.from_params(
            database=settings.postgres_db,
            host=settings.postgres_host,
            password=settings.postgres_password,
            port=settings.postgres_port,
            user=settings.postgres_user,
            table_name=table_name,
            embed_dim=settings.vector_dimension,
            hnsw_kwargs=self._get_hnsw_kwargs(),
            text_search_config=settings.text_search_config,
        )

        # Create index (storage_context is automatically created by from_vector_store)
        # Set insert_batch_size for optimal embedding batch processing
        insert_batch_size = settings.docs_insert_batch_size
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embedding_model,
            insert_batch_size=insert_batch_size,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        if self.index is None:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        # Add documents to index
        for doc in documents:
            self.index.insert(doc)

    def get_index(self) -> VectorStoreIndex:
        """Get the vector store index.

        Returns:
            VectorStoreIndex instance
        """
        if self.index is None:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        return self.index

    def get_keyword_index(self) -> Optional[KeywordTableIndex]:
        """Get the keyword table index.

        Returns:
            KeywordTableIndex instance or None
        """
        # Return FAQ keyword index if available, otherwise legacy keyword index
        return self.faq_keyword_index or self.keyword_index

    def initialize_faq_index(self, table_name: Optional[str] = None) -> None:
        """Initialize FAQ-specific vector store and index.

        Args:
            table_name: Name of the FAQ vector table (defaults to settings)
        """
        table_name = table_name or settings.faq_vector_table_name

        # Create FAQ vector store WITHOUT HNSW index configuration
        # Index will be created manually after all data is inserted for better performance
        self.faq_vector_store = PGVectorStore.from_params(
            database=settings.postgres_db,
            host=settings.postgres_host,
            password=settings.postgres_password,
            port=settings.postgres_port,
            user=settings.postgres_user,
            table_name=table_name,
            embed_dim=settings.vector_dimension,
            hnsw_kwargs=self._get_hnsw_kwargs(),
            text_search_config=settings.text_search_config,
            perform_setup=True,  # Explicitly enable table creation
        )

        # Create FAQ vector index (storage_context is automatically created by from_vector_store)
        # This will trigger _initialize() when data is inserted
        # Set insert_batch_size for optimal embedding batch processing
        # Use a reasonable default since FAQ batches are typically smaller
        insert_batch_size = getattr(
            settings, "faq_insert_batch_size", settings.docs_insert_batch_size
        )
        self.faq_index = VectorStoreIndex.from_vector_store(
            vector_store=self.faq_vector_store,
            embed_model=self.embedding_model,
            insert_batch_size=insert_batch_size,
        )

        # Explicitly ensure tables are created before use
        # _initialize() is called lazily, so we call it explicitly here
        if not self.faq_vector_store._is_initialized:
            logger.info("Initializing FAQ vector store (creating tables if needed)...")
            # Wait for database connection before initializing
            if not self._wait_for_db_connection():
                logger.warning(
                    "Database connection not available, but continuing. "
                    "Initialization will be retried on first use."
                )
            try:
                self.faq_vector_store._initialize()
                logger.info(
                    "FAQ vector store initialized, table '%s' should be created",
                    table_name,
                )
            except Exception as e:
                logger.warning(
                    f"FAQ vector store initialization failed: {e}. "
                    "This may be normal if tables already exist. Continuing..."
                )

        self.load_faq_keyword_index()

    def initialize_docs_index(self, table_name: Optional[str] = None) -> None:
        """Initialize docs-specific vector store and index.

        Args:
            table_name: Name of the docs vector table (defaults to settings)
        """
        table_name = table_name or settings.docs_vector_table_name

        # Create docs vector store WITHOUT HNSW index configuration
        # Index will be created manually after all data is inserted for better performance
        self.docs_vector_store = PGVectorStore.from_params(
            database=settings.postgres_db,
            host=settings.postgres_host,
            password=settings.postgres_password,
            port=settings.postgres_port,
            user=settings.postgres_user,
            table_name=table_name,
            embed_dim=settings.vector_dimension,
            hnsw_kwargs=self._get_hnsw_kwargs(),
            text_search_config=settings.text_search_config,
            perform_setup=True,  # Explicitly enable table creation
        )

        # Create docs vector index (storage_context is automatically created by from_vector_store)
        # This will trigger _initialize() when data is inserted
        # Set insert_batch_size to match docs_insert_batch_size for optimal performance
        insert_batch_size = settings.docs_insert_batch_size
        self.docs_index = VectorStoreIndex.from_vector_store(
            vector_store=self.docs_vector_store,
            embed_model=self.embedding_model,
            insert_batch_size=insert_batch_size,
        )

        # Explicitly ensure tables are created before use
        # _initialize() is called lazily, so we call it explicitly here
        if not self.docs_vector_store._is_initialized:
            logger.info("Initializing docs vector store (creating tables if needed)...")
            # Wait for database connection before initializing
            if not self._wait_for_db_connection():
                logger.warning(
                    "Database connection not available, but continuing. "
                    "Initialization will be retried on first use."
                )
            try:
                self.docs_vector_store._initialize()
                logger.info(
                    "Docs vector store initialized, table '%s' should be created",
                    table_name,
                )
            except Exception as e:
                logger.warning(
                    f"Docs vector store initialization failed: {e}. "
                    "This may be normal if tables already exist. Continuing..."
                )

        self.load_docs_keyword_index()

    def _documents_to_nodes(self, documents: List[Document]) -> List:
        """Convert Documents to Nodes without splitting.

        FAQ documents should not be split (one FAQ item = one Document = one Node).
        This ensures that each FAQ Document becomes exactly one Node.

        Args:
            documents: List of Documents to convert

        Returns:
            List of TextNode objects
        """
        from llama_index.core.schema import TextNode

        nodes = []
        for doc in documents:
            # Get document ID for ref_doc_id (required for delete_ref_doc to work)
            doc_id = None
            if hasattr(doc, "id_") and doc.id_:
                doc_id = doc.id_
            elif hasattr(doc, "metadata") and doc.metadata:
                doc_id = doc.metadata.get("id", "")

            # Create a TextNode directly from Document to prevent splitting
            # When faq_split_documents=False, one Document = one Node
            # So we can use doc_id as both node.id_ and node.ref_doc_id
            node = TextNode(
                text=doc.text,
                metadata=doc.metadata,
                id_=doc_id,  # Use doc_id as node.id_
            )
            # Note: ref_doc_id is a read-only property and cannot be set
            # We rely on id_ for deletion, which is set above and stored in metadata
            # The deletion logic in add_docs_nodes() and add_faq_documents() uses id_ as priority
            nodes.append(node)

        return nodes

    def add_faq_documents(self, documents: List[Document]) -> None:
        """Add FAQ documents to the FAQ vector store.

        Before inserting, checks if documents with the same ID already exist.
        If they exist, deletes them first to avoid duplicates (update instead of append).

        Args:
            documents: List of FAQ documents to add
        """
        if self.faq_index is None:
            raise ValueError(
                "FAQ vector store not initialized. Call initialize_faq_index() first."
            )

        # Process each document: check for existing ID and delete if found
        for doc in documents:
            # Get document ID for deletion
            # Note: For FAQ, Document.id_ is set by JSONParser when ID is provided
            # When Document is converted to Node (via insert() or _documents_to_nodes()),
            # Document.id_ is set as Node.ref_doc_id, which is what delete_ref_doc() uses
            # So using Document.id_ is correct for FAQ deletion
            doc_id = None
            if hasattr(doc, "id_") and doc.id_:
                doc_id = (
                    doc.id_
                )  # Priority 1: id_ (will be Node.ref_doc_id after conversion)
            elif hasattr(doc, "metadata") and doc.metadata:
                doc_id = doc.metadata.get("id", "")  # Priority 2: metadata (fallback)

            # If document has an ID, try to delete existing document with same ID
            if doc_id:
                try:
                    # Delete from vector index if exists
                    # delete_ref_doc() searches by ref_doc_id, which equals Document.id_ after conversion
                    self.faq_index.delete_ref_doc(doc_id, delete_from_docstore=True)
                    logger.debug(f"Deleted existing FAQ document with ID: {doc_id}")
                    # Also try to delete from keyword index if it exists
                    if self.faq_keyword_index:
                        try:
                            self.faq_keyword_index.delete_ref_doc(
                                doc_id, delete_from_docstore=True
                            )
                            logger.debug(
                                f"Deleted existing FAQ keyword index document with ID: {doc_id}"
                            )
                        except Exception:
                            # Keyword index deletion might fail, continue anyway
                            pass
                except Exception:
                    # If deletion fails, document doesn't exist yet, which is fine
                    logger.debug(
                        f"FAQ document with ID {doc_id} does not exist yet, will insert"
                    )

        # Check if splitting is enabled
        if settings.faq_split_documents:
            # Use insert() which may split documents via node_parser
            docs_iter = tqdm(
                documents, desc="Adding FAQ documents to vector store", unit="doc"
            )
            for doc in docs_iter:
                self.faq_index.insert(doc)
        else:
            # Convert Documents to Nodes manually to ensure one Document = one Node
            # This prevents automatic splitting by node_parser
            nodes = self._documents_to_nodes(documents)
            # Insert nodes directly (bypasses node_parser)
            # Note: insert_nodes() is a batch operation, so we show progress before insertion
            logger.info(f"Inserting {len(nodes)} FAQ nodes into vector store...")
            with tqdm(
                total=len(nodes), desc="Adding FAQ nodes to vector store", unit="node"
            ) as pbar:
                self.faq_index.insert_nodes(nodes)
                pbar.update(len(nodes))

    def add_docs_nodes(
        self,
        documents_or_nodes: List[Union[Document, BaseNode]],
        batch_size: Optional[int] = None,
        auto_create_index: bool = False,
    ) -> None:
        """Add docs documents or nodes to the docs vector store.

        Before inserting, checks if documents with the same ID already exist.
        If they exist, deletes them first to avoid duplicates (update instead of append).

        Args:
            documents_or_nodes: List of Document or Node objects to add
            batch_size: Batch size for inserting documents (default: from settings.docs_insert_batch_size)
            auto_create_index: If True, automatically create HNSW index after insertion.
                              If False (default), skip index creation (user must create manually).
        """
        if self.docs_index is None:
            raise ValueError(
                "Docs vector store not initialized. Call initialize_docs_index() first."
            )

        batch_size = batch_size or settings.docs_insert_batch_size

        # Convert documents to nodes if needed, or use nodes directly
        nodes = []
        for item in documents_or_nodes:
            if isinstance(item, Document):
                # Convert Document to Node
                from llama_index.core.schema import TextNode

                doc_id = None
                if hasattr(item, "id_") and item.id_:
                    doc_id = item.id_
                elif hasattr(item, "metadata") and item.metadata:
                    doc_id = item.metadata.get("id", "")

                node = TextNode(
                    text=item.text,
                    metadata=item.metadata,
                    id_=doc_id,
                )
                # Note: ref_doc_id is a read-only property and cannot be set
                # We rely on id_ for deletion, which is set above and stored in metadata
                # The deletion logic in add_docs_nodes() uses id_ as priority
                nodes.append(node)
            elif isinstance(item, BaseNode):
                # Already a node, use directly
                nodes.append(item)
            else:
                raise TypeError(
                    f"Expected Document or BaseNode, got {type(item).__name__}"
                )

        # Process each node: check for existing ID and delete if found
        # Use tqdm for deletion progress
        # delete_iter = tqdm(
        #     nodes, desc="Checking/Deleting existing docs", unit="node", leave=False
        # )
        # for node in delete_iter:
        #     # Get node ID for deletion
        #     # Since ref_doc_id cannot be modified (read-only property), we use id_ instead
        #     # id_ is set to our custom format ({base_id}_{idx}) and is stored in metadata
        #     # delete_ref_doc() searches by ref_doc_id, but we can use id_ if it matches
        #     # the stored ref_doc_id in the database
        #     node_id = None
        #     if hasattr(node, "id_") and node.id_:
        #         # Priority 1: Use id_ (our custom format, e.g., {base_id}_{idx})
        #         # This is what we set in markdown_parser.py
        #         node_id = node.id_
        #     elif hasattr(node, "metadata") and node.metadata:
        #         # Priority 2: Get from metadata (fallback)
        #         node_id = node.metadata.get("id", "")
        #     elif hasattr(node, "ref_doc_id") and node.ref_doc_id:
        #         # Priority 3: Use ref_doc_id (UUID from SemanticSplitterNodeParser)
        #         # This is a fallback, but may not match our custom id_ format
        #         node_id = node.ref_doc_id

        #     # If node has an ID, try to delete existing document with same ID
        #     if node_id:
        #         try:
        #             # Delete from vector index if exists
        #             # Note: delete_ref_doc() searches by ref_doc_id in the vector store
        #             # If ref_doc_id was set to our id_ format when inserted, this will work
        #             # Otherwise, we may need to use the stored ref_doc_id (UUID)
        #             self.docs_index.delete_ref_doc(node_id, delete_from_docstore=True)
        #             logger.debug(f"Deleted existing docs document with ID: {node_id}")
        #         except Exception as e:
        #             # If deletion fails, document doesn't exist yet, which is fine
        #             # Or the ref_doc_id in database doesn't match our id_
        #             logger.debug(
        #                 f"Docs document with ID {node_id} does not exist yet or "
        #                 f"ref_doc_id mismatch (will insert): {e}"
        #             )

        # Add nodes to docs index
        # Note: insert_nodes() already handles batch insertion efficiently in a single transaction
        # We only split into batches if there are many nodes to:
        # 1. Control transaction size (avoid very large transactions)
        # 2. Manage memory usage (avoid loading too many nodes at once)
        # 3. Show progress for large datasets
        total_nodes = len(nodes)

        # HNSW index optimization: always drop index before insert (if exists) for faster insertion
        # This applies to both first-time insertions and subsequent updates
        # Index will only be recreated if auto_create_index=True
        index_dropped = False
        # if settings.enable_hnsw:
        #     # Always try to drop index if it exists (for both first-time and subsequent insertions)
        #     # This ensures fast insertion regardless of whether index exists
        #     logger.info(
        #         f"Dropping HNSW index (if exists) before inserting {total_nodes} nodes for faster insertion..."
        #     )
        #     index_dropped = self._drop_hnsw_index(self.docs_vector_store)
        #     if not index_dropped:
        #         logger.info("No existing HNSW index found, will insert without index (faster)")

        try:
            # For small datasets, insert all at once (more efficient)
            # For large datasets, use batching to control transaction size and memory
            if (
                batch_size is None
                or batch_size == 0
                or batch_size == -1
                or total_nodes <= batch_size
            ):
                # Small dataset: insert all at once (single transaction, more efficient)
                logger.info(f"Inserting {total_nodes} docs nodes in a single batch")
                self.docs_index.insert_nodes(nodes)
            else:
                # Large dataset: split into batches to control transaction size
                num_batches = (
                    total_nodes + batch_size - 1
                ) // batch_size  # Ceiling division
                logger.info(
                    f"Inserting {total_nodes} docs nodes in {num_batches} batch(es) "
                    f"(batch_size={batch_size})"
                )

                with tqdm(
                    total=total_nodes,
                    desc="Adding docs nodes to vector store",
                    unit="node",
                ) as pbar:
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, total_nodes)
                        batch_nodes = nodes[start_idx:end_idx]

                        # Each batch is inserted in a separate transaction
                        # This helps with memory management and error recovery
                        logger.info(
                            f"Inserting batch {batch_idx + 1}/{num_batches} of {total_nodes} docs nodes"
                        )
                        self.docs_index.insert_nodes(batch_nodes)
                        logger.info(
                            f"Inserted batch  end {batch_idx + 1}/{num_batches} of {total_nodes} docs nodes"
                        )
                        pbar.update(len(batch_nodes))
                        pbar.set_postfix(batch=f"{batch_idx + 1}/{num_batches}")

            # Recreate HNSW index only if auto_create_index=True
            # By default, skip index creation to allow user to create it manually after all data is inserted
            if auto_create_index and settings.enable_hnsw:
                logger.info(
                    "Auto-creating HNSW index after insertion (auto_create_index=True)..."
                )
                # self._create_hnsw_index(self.docs_vector_store)
                logger.info("HNSW index created successfully")
            elif settings.enable_hnsw:
                logger.info(
                    "Skipping HNSW index creation (auto_create_index=False). "
                    "Please create index manually using create_docs_hnsw_index() after all data is inserted."
                )
        except Exception as e:
            # If insertion fails and we dropped the index, try to recreate it before re-raising
            # This helps restore the database to a consistent state
            if index_dropped and auto_create_index and settings.enable_hnsw:
                logger.warning(
                    f"Insertion failed, attempting to recreate HNSW index before re-raising error: {e}"
                )
                # try:
                # self._create_hnsw_index(self.docs_vector_store)
                # except Exception as recreate_error:
                #     logger.error(
                #         f"Failed to recreate HNSW index after insertion failure: {recreate_error}"
                #     )
            raise

    def create_docs_hnsw_index(self) -> None:
        """Manually create HNSW index for docs vector store.

        This method should be called after all data has been inserted to create
        the HNSW index for optimal query performance. Creating the index after
        all data is inserted is much faster than creating it incrementally during insertion.

        Raises:
            ValueError: If docs vector store is not initialized
        """
        if self.docs_vector_store is None:
            raise ValueError(
                "Docs vector store not initialized. Call initialize_docs_index() first."
            )

        if not settings.enable_hnsw:
            logger.warning("HNSW is disabled in settings, skipping index creation")
            return

        logger.info("Creating HNSW index for docs vector store...")
        self._create_hnsw_index(self.docs_vector_store)
        logger.info("HNSW index for docs vector store created successfully")

    def create_faq_hnsw_index(self) -> None:
        """Manually create HNSW index for FAQ vector store.

        This method should be called after all data has been inserted to create
        the HNSW index for optimal query performance. Creating the index after
        all data is inserted is much faster than creating it incrementally during insertion.

        Raises:
            ValueError: If FAQ vector store is not initialized
        """
        if self.faq_vector_store is None:
            raise ValueError(
                "FAQ vector store not initialized. Call initialize_faq_index() first."
            )

        if not settings.enable_hnsw:
            logger.warning("HNSW is disabled in settings, skipping index creation")
            return

        logger.info("Creating HNSW index for FAQ vector store...")
        self._create_hnsw_index(self.faq_vector_store)
        logger.info("HNSW index for FAQ vector store created successfully")

    def create_faq_keyword_index(
        self, documents: List[Document], persist: bool = True, llm=None
    ) -> KeywordTableIndex:
        """Create a keyword table index for FAQ documents.

        Before creating, checks if documents with the same ID already exist in the keyword index.
        If they exist, deletes them first to avoid duplicates (update instead of append).

        Args:
            documents: List of FAQ documents to index
            persist: Whether to persist the index to disk (default: True)
            llm: Language model for keyword extraction (optional, uses self.llm or Settings.llm)

        Returns:
            KeywordTableIndex instance
        """
        # Get LLM for keyword extraction
        from llama_index.core import Settings as LlamaIndexSettings

        keyword_llm = llm or self.llm or LlamaIndexSettings.llm
        if keyword_llm is None:
            raise ValueError(
                "LLM is required for KeywordTableIndex. "
                "Please provide llm parameter or set Settings.llm"
            )

        # Create persist directory path
        persist_dir = Path(settings.storage_dir) / settings.faq_keyword_storage_index
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Create storage context with persist directory
        # If directory doesn't exist or is empty, create a new storage context
        # Otherwise, load from existing persist directory
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        except (FileNotFoundError, OSError):
            # Directory exists but is empty or incomplete, create new storage context
            storage_context = StorageContext.from_defaults()

        # Try to load existing keyword index to check for duplicates
        existing_keyword_index = None
        try:
            # Try to load existing keyword index
            existing_keyword_index = load_index_from_storage(
                storage_context, index_id="faq_keyword_index"
            )
            if not isinstance(existing_keyword_index, KeywordTableIndex):
                existing_keyword_index = None
            elif settings.using_custom_keyword_index and not isinstance(
                existing_keyword_index, CustomKeywordTableIndex
            ):
                # Wrap as CustomKeywordTableIndex if enabled
                existing_keyword_index = _wrap_loaded_keyword_index(
                    existing_keyword_index
                )
        except (ValueError, KeyError, FileNotFoundError):
            # Index doesn't exist yet, which is fine
            pass

        # If existing keyword index found, delete documents with same IDs before creating new one
        if existing_keyword_index:
            for doc in documents:
                # Get document ID for deletion
                # Note: For FAQ, Document.id_ is set by JSONParser when ID is provided
                # When Document is converted to Node, Document.id_ is set as Node.ref_doc_id
                # So using Document.id_ is correct for keyword index deletion
                doc_id = None
                if hasattr(doc, "id_") and doc.id_:
                    doc_id = doc.id_
                elif hasattr(doc, "metadata") and doc.metadata:
                    doc_id = doc.metadata.get("id", "")

                # If document has an ID, try to delete existing document with same ID
                if doc_id:
                    try:
                        # delete_ref_doc() searches by ref_doc_id, which equals Document.id_ after conversion
                        existing_keyword_index.delete_ref_doc(
                            doc_id, delete_from_docstore=True
                        )
                        logger.debug(
                            f"Deleted existing keyword index document with ID: {doc_id}"
                        )
                    except Exception:
                        # If deletion fails, document doesn't exist yet, which is fine
                        logger.debug(
                            f"Keyword index document with ID {doc_id} does not exist yet"
                        )

        # Create keyword index with LLM
        # If existing index was found, insert new documents into existing index
        # Otherwise, create a new one
        if existing_keyword_index:
            # Insert new documents into existing keyword index
            docs_iter = tqdm(
                documents, desc="Adding documents to keyword index", unit="doc"
            )
            for doc in docs_iter:
                # Get document ID for logging
                doc_id = None
                if hasattr(doc, "id_") and doc.id_:
                    doc_id = doc.id_
                elif hasattr(doc, "metadata") and doc.metadata:
                    doc_id = doc.metadata.get("id", "")

                try:
                    existing_keyword_index.insert(doc)
                except Exception as e:
                    logger.warning(
                        f"Failed to insert document {doc_id or 'unknown'} into keyword index: {e}"
                    )
            keyword_index = existing_keyword_index
        else:
            # Create new keyword index with LLM
            # Use CustomKeywordTableIndex if enabled, otherwise use KeywordTableIndex
            keyword_index_class = _get_keyword_index_class()
            if settings.using_custom_keyword_index:
                # Get global keyword extractor if available
                keyword_extractor = get_global_keyword_extractor()
                keyword_index = keyword_index_class.from_documents(
                    documents,
                    storage_context=storage_context,
                    llm=keyword_llm,
                    show_progress=True,
                    keyword_extractor=keyword_extractor,
                )
            else:
                keyword_index = keyword_index_class.from_documents(
                    documents,
                    storage_context=storage_context,
                    llm=keyword_llm,
                    show_progress=True,
                )
            # Set a specific index_id for FAQ keyword index to avoid conflicts
            keyword_index.set_index_id("faq_keyword_index")

        # Persist to disk if requested
        if persist:
            storage_context.persist(persist_dir=str(persist_dir))

        self.faq_keyword_index = keyword_index
        return keyword_index

    def create_docs_keyword_index(
        self,
        documents_or_nodes: List[Union[Document, BaseNode]],
        persist: bool = True,
        llm=None,
    ) -> KeywordTableIndex:
        """Create a keyword table index for docs documents/nodes.

        Before creating, checks if documents with the same ID already exist in the keyword index.
        If they exist, deletes them first to avoid duplicates (update instead of append).

        Args:
            documents_or_nodes: List of Document or Node objects to index
            persist: Whether to persist the index to disk (default: True)
            llm: Language model for keyword extraction (optional, uses self.llm or Settings.llm)

        Returns:
            KeywordTableIndex instance
        """
        # Get LLM for keyword extraction
        from llama_index.core import Settings as LlamaIndexSettings

        keyword_llm = llm or self.llm or LlamaIndexSettings.llm
        if keyword_llm is None:
            raise ValueError(
                "LLM is required for KeywordTableIndex. "
                "Please provide llm parameter or set Settings.llm"
            )

        # Create persist directory path
        persist_dir = Path(settings.storage_dir) / settings.docs_keyword_storage_index
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Create storage context with persist directory
        # If directory doesn't exist or is empty, create a new storage context
        # Otherwise, load from existing persist directory
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        except (FileNotFoundError, OSError):
            # Directory exists but is empty or incomplete, create new storage context
            storage_context = StorageContext.from_defaults()

        # Try to load existing keyword index to check for duplicates
        existing_keyword_index = None
        try:
            # Try to load existing keyword index
            existing_keyword_index = load_index_from_storage(
                storage_context, index_id="docs_keyword_index"
            )
            if not isinstance(existing_keyword_index, KeywordTableIndex):
                existing_keyword_index = None
            elif settings.using_custom_keyword_index and not isinstance(
                existing_keyword_index, CustomKeywordTableIndex
            ):
                # Wrap as CustomKeywordTableIndex if enabled
                existing_keyword_index = _wrap_loaded_keyword_index(
                    existing_keyword_index
                )
        except (ValueError, KeyError, FileNotFoundError):
            # Index doesn't exist yet, which is fine
            pass

        # Convert nodes to documents if needed (KeywordTableIndex.from_documents requires Documents)
        documents = []
        for item in documents_or_nodes:
            if isinstance(item, Document):
                documents.append(item)
            elif isinstance(item, BaseNode):
                # Convert Node to Document
                doc = Document(
                    text=item.text if hasattr(item, "text") else "",
                    metadata=item.metadata if hasattr(item, "metadata") else {},
                    id_=item.id_ if hasattr(item, "id_") and item.id_ else None,
                )
                documents.append(doc)
            else:
                raise TypeError(
                    f"Expected Document or BaseNode, got {type(item).__name__}"
                )

        # If existing keyword index found, delete documents with same IDs before creating new one
        if existing_keyword_index:
            for doc in documents:
                # Get document ID for deletion
                doc_id = None
                if hasattr(doc, "id_") and doc.id_:
                    doc_id = doc.id_
                elif hasattr(doc, "metadata") and doc.metadata:
                    doc_id = doc.metadata.get("id", "")

                # If document has an ID, try to delete existing document with same ID
                if doc_id:
                    try:
                        # delete_ref_doc() searches by ref_doc_id, which equals Document.id_ after conversion
                        existing_keyword_index.delete_ref_doc(
                            doc_id, delete_from_docstore=True
                        )
                        logger.debug(
                            f"Deleted existing docs keyword index document with ID: {doc_id}"
                        )
                    except Exception:
                        # If deletion fails, document doesn't exist yet, which is fine
                        logger.debug(
                            f"Docs keyword index document with ID {doc_id} does not exist yet"
                        )

        # Create keyword index with LLM
        # If existing index was found, insert new documents into existing index
        # Otherwise, create a new one
        if existing_keyword_index:
            # Insert new documents into existing keyword index
            docs_iter = tqdm(
                documents, desc="Adding documents to docs keyword index", unit="doc"
            )
            for doc in docs_iter:
                # Get document ID for logging
                doc_id = None
                if hasattr(doc, "id_") and doc.id_:
                    doc_id = doc.id_
                elif hasattr(doc, "metadata") and doc.metadata:
                    doc_id = doc.metadata.get("id", "")

                try:
                    existing_keyword_index.insert(doc)
                except Exception as e:
                    logger.warning(
                        f"Failed to insert document {doc_id or 'unknown'} into docs keyword index: {e}"
                    )
            keyword_index = existing_keyword_index
        else:
            # Create new keyword index with LLM
            # Use CustomKeywordTableIndex if enabled, otherwise use KeywordTableIndex
            keyword_index_class = _get_keyword_index_class()
            if settings.using_custom_keyword_index:
                # Get global keyword extractor if available
                keyword_extractor = get_global_keyword_extractor()
                keyword_index = keyword_index_class.from_documents(
                    documents,
                    storage_context=storage_context,
                    llm=keyword_llm,
                    show_progress=True,
                    keyword_extractor=keyword_extractor,
                )
            else:
                keyword_index = keyword_index_class.from_documents(
                    documents,
                    storage_context=storage_context,
                    llm=keyword_llm,
                    show_progress=True,
                )
            # Set a specific index_id for docs keyword index to avoid conflicts
            keyword_index.set_index_id("docs_keyword_index")

        # Persist to disk if requested
        if persist:
            storage_context.persist(persist_dir=str(persist_dir))

        self.docs_keyword_index = keyword_index
        return keyword_index

    def load_docs_keyword_index(self) -> Optional[KeywordTableIndex]:
        """Load docs keyword index from persisted storage.

        Returns:
            KeywordTableIndex instance if found, None otherwise
        """
        persist_dir = Path(settings.storage_dir) / settings.docs_keyword_storage_index

        if not persist_dir.exists():
            logger.debug(
                "Docs keyword index persist directory does not exist: %s", persist_dir
            )
            return None

        try:
            # Load storage context from persist directory
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))

            # Load keyword index from storage
            # Try to load by specific index_id first (if we set it during creation)
            keyword_index = None
            try:
                keyword_index = load_index_from_storage(
                    storage_context, index_id="docs_keyword_index"
                )
                logger.info(
                    "Successfully loaded Docs KeywordTableIndex with index_id='docs_keyword_index'"
                )
            except (ValueError, KeyError) as e:
                logger.info("Could not load with index_id='docs_keyword_index': %s", e)
                # If index_id not found, try loading all indices and find KeywordTableIndex
                try:
                    # Load all indices and find the KeywordTableIndex
                    # load_indices_from_storage returns a list of indices
                    all_indices = load_indices_from_storage(storage_context)
                    logger.info(
                        "Found %d index(es) in storage, searching for Docs KeywordTableIndex",
                        len(all_indices),
                    )
                    # Find KeywordTableIndex from all loaded indices
                    for idx in all_indices:  # all_indices is a list, not a dict
                        if isinstance(idx, KeywordTableIndex):
                            keyword_index = idx
                            logger.info(
                                "Found Docs KeywordTableIndex in loaded indices"
                            )
                            break

                    if keyword_index is None:
                        logger.warning(
                            "No Docs KeywordTableIndex found in %d loaded index(es)",
                            len(all_indices),
                        )
                        # Try to get index_ids from index_store if available
                        try:
                            if hasattr(storage_context.index_store, "index_ids"):
                                index_ids = storage_context.index_store.index_ids()
                                logger.info("Available index_ids: %s", index_ids)
                                # Try loading each index_id
                                for idx_id in index_ids:
                                    try:
                                        test_idx = load_index_from_storage(
                                            storage_context, index_id=idx_id
                                        )
                                        if isinstance(test_idx, KeywordTableIndex):
                                            keyword_index = test_idx
                                            logger.info(
                                                "Found Docs KeywordTableIndex with index_id: %s",
                                                idx_id,
                                            )
                                            break
                                    except Exception:
                                        continue
                        except Exception:
                            pass
                except Exception as load_e:
                    logger.warning("Failed to load indices from storage: %s", load_e)
                    # Fallback: try loading without index_id (if only one index exists)
                    try:
                        keyword_index = load_index_from_storage(storage_context)
                        logger.info("Loaded single docs index without index_id")
                    except ValueError as e:
                        if "Expected to load a single index" in str(e):
                            logger.warning(
                                "Multiple indexes found but cannot determine which is Docs KeywordTableIndex"
                            )
                            return None
                        raise

            # Verify it's a KeywordTableIndex
            if not isinstance(keyword_index, KeywordTableIndex):
                logger.warning(
                    "Loaded index is not a KeywordTableIndex: %s", type(keyword_index)
                )
                return None

            # Wrap as CustomKeywordTableIndex if enabled and not already wrapped
            if settings.using_custom_keyword_index and not isinstance(
                keyword_index, CustomKeywordTableIndex
            ):
                keyword_index = _wrap_loaded_keyword_index(keyword_index)

            self.docs_keyword_index = keyword_index
            logger.info("Docs keyword index loaded successfully")
            return keyword_index
        except Exception as e:
            logger.warning(
                "Failed to load docs keyword index from %s: %s", persist_dir, e
            )
            return None

    def load_faq_keyword_index(self) -> Optional[KeywordTableIndex]:
        """Load FAQ keyword index from persisted storage.

        Returns:
            KeywordTableIndex instance if found, None otherwise
        """
        persist_dir = Path(settings.storage_dir) / settings.faq_keyword_storage_index

        if not persist_dir.exists():
            logger.debug(
                "Keyword index persist directory does not exist: %s", persist_dir
            )
            return None

        try:
            # Load storage context from persist directory
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))

            # Load keyword index from storage
            # Try to load by specific index_id first (if we set it during creation)
            keyword_index = None
            try:
                keyword_index = load_index_from_storage(
                    storage_context, index_id="faq_keyword_index"
                )
                logger.info(
                    "Successfully loaded KeywordTableIndex with index_id='faq_keyword_index'"
                )
            except (ValueError, KeyError) as e:
                logger.info("Could not load with index_id='faq_keyword_index': %s", e)
                # If index_id not found, try loading all indices and find KeywordTableIndex
                try:
                    # Load all indices and find the KeywordTableIndex
                    # load_indices_from_storage returns a list of indices
                    all_indices = load_indices_from_storage(storage_context)
                    logger.info(
                        "Found %d index(es) in storage, searching for KeywordTableIndex",
                        len(all_indices),
                    )
                    # Find KeywordTableIndex from all loaded indices
                    for idx in all_indices:  # all_indices is a list, not a dict
                        if isinstance(idx, KeywordTableIndex):
                            keyword_index = idx
                            logger.info("Found KeywordTableIndex in loaded indices")
                            break

                    if keyword_index is None:
                        logger.warning(
                            "No KeywordTableIndex found in %d loaded index(es)",
                            len(all_indices),
                        )
                        # Try to get index_ids from index_store if available
                        try:
                            if hasattr(storage_context.index_store, "index_ids"):
                                index_ids = storage_context.index_store.index_ids()
                                logger.info("Available index_ids: %s", index_ids)
                                # Try loading each index_id
                                for idx_id in index_ids:
                                    try:
                                        test_idx = load_index_from_storage(
                                            storage_context, index_id=idx_id
                                        )
                                        if isinstance(test_idx, KeywordTableIndex):
                                            keyword_index = test_idx
                                            logger.info(
                                                "Found KeywordTableIndex with index_id: %s",
                                                idx_id,
                                            )
                                            break
                                    except Exception:
                                        continue
                        except Exception:
                            pass
                except Exception as load_e:
                    logger.warning("Failed to load indices from storage: %s", load_e)
                    # Fallback: try loading without index_id (if only one index exists)
                    try:
                        keyword_index = load_index_from_storage(storage_context)
                        logger.info("Loaded single index without index_id")
                    except ValueError as e:
                        if "Expected to load a single index" in str(e):
                            logger.warning(
                                "Multiple indexes found but cannot determine which is KeywordTableIndex"
                            )
                            return None
                        raise

            # Verify it's a KeywordTableIndex
            if not isinstance(keyword_index, KeywordTableIndex):
                logger.warning(
                    "Loaded index is not a KeywordTableIndex: %s", type(keyword_index)
                )
                return None

            # Wrap as CustomKeywordTableIndex if enabled and not already wrapped
            if settings.using_custom_keyword_index and not isinstance(
                keyword_index, CustomKeywordTableIndex
            ):
                keyword_index = _wrap_loaded_keyword_index(keyword_index)

            self.faq_keyword_index = keyword_index
            logger.info("FAQ keyword index loaded successfully")
            return keyword_index
        except Exception as e:
            logger.warning(
                "Failed to load FAQ keyword index from %s: %s", persist_dir, e
            )
            return None

    def get_faq_index(self) -> Optional[VectorStoreIndex]:
        """Get the FAQ vector store index.

        Returns:
            FAQ VectorStoreIndex instance or None
        """
        return self.faq_index

    def get_docs_index(self) -> Optional[VectorStoreIndex]:
        """Get the docs vector store index.

        Returns:
            Docs VectorStoreIndex instance or None
        """
        return self.docs_index

    def get_faq_keyword_index(self) -> Optional[KeywordTableIndex]:
        """Get the FAQ keyword table index.

        Returns:
            FAQ KeywordTableIndex instance or None
        """
        return self.faq_keyword_index

    def get_docs_keyword_index(self) -> Optional[KeywordTableIndex]:
        """Get the docs keyword table index.

        Returns:
            Docs KeywordTableIndex instance or None
        """
        return self.docs_keyword_index

    def _rebuild_faq_keyword_index(
        self,
        new_documents: List[Document],
        json_dir: Optional[Path],
        json_parser,  # JSONParser instance
    ) -> None:
        """Rebuild FAQ keyword index with all FAQs (existing + new).

        Args:
            new_documents: New/updated FAQ documents
            json_dir: Optional directory to load existing FAQs from JSON files
            json_parser: JSONParser instance for parsing files
        """
        all_faq_docs = []

        # Load existing FAQs from JSON files if directory provided
        if json_dir and json_dir.exists():
            existing_docs = json_parser.parse_directory(json_dir)
            all_faq_docs.extend(existing_docs)

        # Add new/updated documents
        all_faq_docs.extend(new_documents)

        # Remove duplicates based on ID (keep the latest)
        seen_ids = {}
        deduplicated_docs = []
        for doc in reversed(all_faq_docs):  # Reverse to keep latest
            doc_id = doc.metadata.get("id", "")
            if not doc_id:
                # Fallback to question text if no ID
                doc_id = doc.text.split("\n")[0].replace("问题: ", "").strip()

            if doc_id and doc_id not in seen_ids:
                seen_ids[doc_id] = True
                deduplicated_docs.append(doc)
            elif not doc_id:
                # If no ID and no question, skip
                continue
        deduplicated_docs.reverse()  # Reverse back to original order

        # Recreate keyword index with all FAQs
        # Use self.llm or Settings.llm for keyword extraction
        from llama_index.core import Settings as LlamaIndexSettings

        keyword_llm = self.llm or LlamaIndexSettings.llm
        self.create_faq_keyword_index(deduplicated_docs, persist=True, llm=keyword_llm)

    def update_faqs(
        self, faq_items: List[Dict], json_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Update FAQs with new/updated items.

        This method supports hot updates:
        - If FAQ ID doesn't exist, insert it
        - If FAQ ID exists, update it (delete old, insert new)
        - Rebuilds KeywordTableIndex (doesn't support incremental updates)

        Args:
            faq_items: List of FAQ item dictionaries with keys: question, answer, id, etc.
            json_dir: Optional directory to load existing FAQs from JSON files

        Returns:
            Dictionary with keys: inserted_count, updated_count, errors, total_processed
        """
        from askany.ingest.json_parser import JSONParser

        if self.faq_index is None:
            raise ValueError(
                "FAQ vector store not initialized. Call initialize_faq_index() first."
            )

        inserted_count = 0
        updated_count = 0
        errors = []
        new_documents = []

        # Parse FAQs to documents using JSONParser
        json_parser = JSONParser()

        # Create a temporary JSON file with FAQ items and parse it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as temp_file:
            # Write FAQ items to temporary file
            json.dump(faq_items, temp_file, ensure_ascii=False, indent=2)
            temp_file_path = Path(temp_file.name)

        try:
            # Parse the temporary JSON file using json_parser
            parsed_docs = json_parser.parse_file(temp_file_path)
            new_documents = parsed_docs

            # Process each FAQ item to check for updates
            for item in faq_items:
                try:
                    question = item.get("question", "").strip()
                    if not question:
                        errors.append("Skipped item with empty question")
                        continue

                    # Check if FAQ exists by ID and delete if exists
                    doc_id = item.get("id", "")
                    if doc_id:
                        try:
                            # Try to delete existing document by ref_doc_id from vector index
                            # Note: delete_ref_doc() searches by metadata_["ref_doc_id"] in JSON field, not by node.id_
                            #       PGVectorStore.delete() uses: metadata_["ref_doc_id"].astext == ref_doc_id
                            #       This is why we need to set ref_doc_id when creating nodes
                            self.faq_index.delete_ref_doc(
                                doc_id, delete_from_docstore=True
                            )
                            # Also try to delete from keyword index if it exists
                            if self.faq_keyword_index:
                                try:
                                    self.faq_keyword_index.delete_ref_doc(
                                        doc_id, delete_from_docstore=True
                                    )
                                except Exception as del_e:
                                    logger.warning(
                                        "Error deleting from keyword index: %s", del_e
                                    )
                                    # Keyword index deletion might fail, continue anyway
                            updated_count += 1
                        except Exception:
                            # If deletion fails, it means it doesn't exist
                            inserted_count += 1
                    else:
                        # No ID provided, always insert (may create duplicates)
                        inserted_count += 1

                except Exception as e:
                    errors.append(f"Error processing FAQ item: {str(e)}")
                    continue
        finally:
            # Clean up temporary file
            try:
                if temp_file_path.exists():
                    os.unlink(temp_file_path)
            except Exception:
                pass

        if not new_documents:
            return {
                "inserted_count": 0,
                "updated_count": 0,
                "errors": errors,
                "total_processed": 0,
            }

        # Insert new/updated documents into vector index
        # Note: Both insert() and insert_nodes() automatically persist to PGVectorStore
        # Note: ref_doc_id is stored in metadata_ JSON field, not as a separate column
        #       node_to_metadata_dict() automatically stores node.ref_doc_id to metadata["ref_doc_id"]
        if settings.faq_split_documents:
            # Use insert() which may split documents via node_parser
            # insert() automatically sets ref_doc_id from Document.id_ when converting Document to Node
            # The ref_doc_id will be stored in metadata_ JSON field via node_to_metadata_dict()
            docs_iter = tqdm(
                new_documents, desc="Updating FAQ documents in vector store", unit="doc"
            )
            for doc in docs_iter:
                self.faq_index.insert(doc)
        else:
            # Convert to Nodes to ensure one Document = one Node (no splitting)
            # We manually set ref_doc_id in _documents_to_nodes() for delete_ref_doc to work
            # The ref_doc_id will be stored in metadata_ JSON field via node_to_metadata_dict()
            nodes = self._documents_to_nodes(new_documents)
            logger.info(f"Inserting {len(nodes)} FAQ nodes into vector store...")
            with tqdm(
                total=len(nodes), desc="Updating FAQ nodes in vector store", unit="node"
            ) as pbar:
                self.faq_index.insert_nodes(nodes)
                pbar.update(len(nodes))

        # Try incremental update for keyword index
        # KeywordTableIndex supports insert() method for incremental updates
        if self.faq_keyword_index:
            try:
                # Try incremental insert into keyword index
                docs_iter = tqdm(
                    new_documents, desc="Updating keyword index", unit="doc"
                )
                for doc in docs_iter:
                    self.faq_keyword_index.insert(doc)
                # Persist the updated keyword index
                persist_dir = (
                    Path(settings.storage_dir) / settings.faq_keyword_storage_index
                )
                if persist_dir.exists():
                    storage_context = self.faq_keyword_index.storage_context
                    if storage_context:
                        storage_context.persist(persist_dir=str(persist_dir))
            except Exception as e:
                # If incremental update fails, fall back to full rebuild
                logger.warning(
                    "Keyword index incremental update failed, rebuilding: %s", e
                )
                errors.append(
                    f"Keyword index incremental update failed, rebuilding: {str(e)}"
                )
                # Rebuild keyword index with all FAQs
                self._rebuild_faq_keyword_index(new_documents, json_dir, json_parser)
        else:
            logger.warning("No keyword index exists, creating it with all FAQs")
            # No keyword index exists, create it with all FAQs
            self._rebuild_faq_keyword_index(new_documents, json_dir, json_parser)

        return {
            "inserted_count": inserted_count,
            "updated_count": updated_count,
            "errors": errors,
            "total_processed": len(new_documents),
        }

    def delete_all(self) -> None:
        """Delete all vectors from the store.

        Note: PGVectorStore.delete() requires ref_doc_id parameter.
        To delete all vectors, use SQL directly:
        DELETE FROM {table_name};
        """
        raise NotImplementedError(
            "delete_all() requires SQL. Use: DELETE FROM {table_name}; "
            "where table_name is the vector table name."
        )

"""Markdown parser for documentation."""

# Set stdout/stderr encoding to UTF-8 FIRST, before any imports
# This ensures Chinese characters display correctly when SemanticSplitterNodeParser prints "> Adding chunk:" messages
import hashlib
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import MarkdownNodeParser, SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Disable the "> Adding chunk:" debug messages from llama_index.core.node_parser.node_utils
# These messages are printed via logger.debug() and can cause encoding issues
# We disable them at the module level to prevent encoding problems
# _llama_node_utils_logger = logging.getLogger("llama_index.core.node_parser.node_utils")
# _llama_node_utils_logger.setLevel(logging.INFO)  # Only show INFO and above, not DEBUG


def _ensure_utf8_encoding():
    """Ensure stdout/stderr use UTF-8 encoding.

    This function is called before SemanticSplitterNodeParser.get_nodes_from_documents()
    to ensure that "> Adding chunk:" messages are printed with correct encoding.
    """
    try:
        if hasattr(sys.stdout, "buffer") and (
            not hasattr(sys.stdout, "encoding") or sys.stdout.encoding != "utf-8"
        ):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        if hasattr(sys.stderr, "buffer") and (
            not hasattr(sys.stderr, "encoding") or sys.stderr.encoding != "utf-8"
        ):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
    except (AttributeError, OSError):
        # If stdout/stderr don't have buffer attribute or can't be wrapped, skip
        pass


class MarkdownParser:
    """Parser for Markdown documentation files."""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        split_mode: str = "hybrid",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
    ):
        """Initialize Markdown parser.

        Args:
            embedding_model: Embedding model for semantic splitting
            split_mode: Splitting mode - "markdown", "semantic", or "hybrid" (default "hybrid")
            chunk_size: Size of chunks in tokens (deprecated, not used by MarkdownNodeParser)
            chunk_overlap: Overlap between chunks (deprecated, not used by MarkdownNodeParser)
        """
        self.embedding_model = embedding_model
        self.split_mode = split_mode.lower()
        print(f"Split mode: {self.split_mode}")
        # Note: chunk_size and chunk_overlap are kept for backward compatibility
        # but MarkdownNodeParser doesn't support these parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize parsers based on mode
        if self.split_mode == "markdown":
            # MarkdownNodeParser splits based on Markdown structure (headers, etc.)
            # It doesn't support chunk_size/chunk_overlap parameters
            self.markdown_parser = MarkdownNodeParser.from_defaults()
            self.semantic_parser = None
        elif self.split_mode == "semantic":
            self.markdown_parser = None
            self.semantic_parser = SemanticSplitterNodeParser.from_defaults(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embedding_model,
            )
        elif self.split_mode == "hybrid":
            # Use both parsers in sequence
            # First: MarkdownNodeParser splits by structure
            # Then: SemanticSplitterNodeParser further splits semantically
            self.markdown_parser = MarkdownNodeParser.from_defaults()
            self.semantic_parser = SemanticSplitterNodeParser.from_defaults(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embedding_model,
            )
        else:
            raise ValueError(
                f"Invalid split_mode: {split_mode}. Must be 'markdown', 'semantic', or 'hybrid'"
            )

    def parse_file_return_documents(self, file_path: Path) -> List[Document]:
        """Parse a Markdown file into Documents.

        Args:
            file_path: Path to Markdown file

        Returns:
            List of Document objects
        """
        # Check if path is actually a file, not a directory
        if not file_path.is_file():
            if file_path.is_dir():
                logger.warning(f"Skipping directory (not a file): {file_path}")
                return []
            else:
                logger.warning(f"Path does not exist or is not a file: {file_path}")
                return []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Get file modification time for last_updated metadata
        last_updated = ""
        try:
            mtime = os.path.getmtime(file_path)
            last_updated = str(mtime)
        except OSError:
            pass

        metadata = {
            "source": str(file_path),
            "type": "markdown",
            "filename": file_path.name,
            "last_updated": last_updated,
        }

        doc = Document(
            text=content,
            metadata=metadata,
        )

        # Parse into documents based on split mode
        if self.split_mode == "markdown":
            # Use only MarkdownNodeParser
            nodes = self.markdown_parser.get_nodes_from_documents([doc])
            # Convert nodes to documents
            documents = [
                Document(
                    text=node.text,
                    metadata=node.metadata,
                )
                for node in nodes
            ]
        elif self.split_mode == "semantic":
            # Use only SemanticSplitterNodeParser
            # Ensure stdout encoding is UTF-8 before calling get_nodes_from_documents
            # This prevents encoding issues with "> Adding chunk:" messages
            _ensure_utf8_encoding()
            nodes = self.semantic_parser.get_nodes_from_documents([doc])
            # Convert nodes to documents
            documents = [
                Document(
                    text=node.text,
                    metadata=node.metadata,
                )
                for node in nodes
            ]
        elif self.split_mode == "hybrid":
            logger.debug(f"Parsing markdown file: {file_path} with hybrid mode")
            # First use MarkdownNodeParser to split by structure
            markdown_nodes = self.markdown_parser.get_nodes_from_documents([doc])
            # Then apply SemanticSplitterNodeParser to markdown nodes in batch
            # Batch processing is more efficient than processing one by one
            documents = []
            if markdown_nodes:
                # Convert all markdown nodes to documents at once
                print(f"Markdown node text: {markdown_nodes[0].text}")
                temp_docs = [
                    Document(
                        text=md_node.text,
                        metadata={**metadata, **md_node.metadata},
                    )
                    for md_node in markdown_nodes
                ]
                documents = temp_docs
                # Apply semantic splitting in batch (more efficient)
                # Ensure stdout encoding is UTF-8 before calling get_nodes_from_documents
                # This prevents encoding issues with "> Adding chunk:" messages
                # _ensure_utf8_encoding()
                semantic_nodes = self.semantic_parser.get_nodes_from_documents(
                    temp_docs
                )
                # Convert nodes back to documents
                documents = [
                    Document(
                        text=node.text,
                        metadata=node.metadata,
                    )
                    for node in semantic_nodes
                ]
            else:
                # If no markdown nodes, return empty list
                documents = []
        return documents

    def parse_file(self, file_path: Path) -> List[BaseNode]:
        """Parse a Markdown file into Nodes.

        Args:
            file_path: Path to Markdown file

        Returns:
            List of Node objects (ready for direct insertion into vector store)
        """
        # Check if path is actually a file, not a directory
        if not file_path.is_file():
            if file_path.is_dir():
                logger.warning(f"Skipping directory (not a file): {file_path}")
                return []
            else:
                logger.warning(f"Path does not exist or is not a file: {file_path}")
                return []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Get file modification time for last_updated metadata
        last_updated = ""
        try:
            mtime = os.path.getmtime(file_path)
            last_updated = str(mtime)
        except OSError:
            pass

        # Generate base ID from file path (consistent across runs)
        # Use MD5 hash of normalized file path for stable ID
        normalized_path = str(file_path.resolve())
        base_id = hashlib.md5(normalized_path.encode()).hexdigest()[:16]

        metadata = {
            "source": str(file_path),
            "type": "markdown",
            "filename": file_path.name,
            "last_updated": last_updated,
        }

        doc = Document(
            text=content,
            metadata=metadata,
        )

        # Parse into nodes based on split mode
        if self.split_mode == "markdown":
            # Use only MarkdownNodeParser
            nodes = self.markdown_parser.get_nodes_from_documents([doc])
            print(f"Parsed {len(nodes)} Markdown nodes")
            logger.info(f"Parsed {len(nodes)} Markdown nodes")
        elif self.split_mode == "semantic":
            # Use only SemanticSplitterNodeParser
            # Ensure stdout encoding is UTF-8 before calling get_nodes_from_documents
            # This prevents encoding issues with "> Adding chunk:" messages
            _ensure_utf8_encoding()
            nodes = self.semantic_parser.get_nodes_from_documents([doc])
        elif self.split_mode == "hybrid":
            # raise Exception("Hybrid mode is not supported")
            logger.debug(f"Parsing markdown file: {file_path} with hybrid mode")
            # First use MarkdownNodeParser to split by structure
            markdown_nodes = self.markdown_parser.get_nodes_from_documents([doc])
            # Then apply SemanticSplitterNodeParser to markdown nodes in batch
            # Batch processing is more efficient than processing one by one
            nodes = []
            if markdown_nodes:
                # for node in markdown_nodes:
                #     print("markdown node text: ", node.text)
                # Convert all markdown nodes to documents at once
                temp_docs = [
                    Document(
                        text=md_node.text,
                        metadata={**metadata, **md_node.metadata},
                    )
                    for md_node in markdown_nodes
                ]
                # Apply semantic splitting in batch (more efficient)
                # Ensure stdout encoding is UTF-8 before calling get_nodes_from_documents
                # This prevents encoding issues with "> Adding chunk:" messages
                _ensure_utf8_encoding()
                semantic_nodes = self.semantic_parser.get_nodes_from_documents(
                    temp_docs
                )
                # for node in semantic_nodes:
                #     print("semantic node text: ", node.text)
                # raise Exception("Stop here")
                nodes.extend(semantic_nodes)

        # Set node IDs and metadata directly (no need to convert to Document)
        # This avoids redundant conversion: nodes -> documents -> nodes
        for idx, node in enumerate(nodes):
            # Generate unique ID for each node
            # Always use {base_id}_{idx} format for consistency, even for single node files
            # Single node files will have {base_id}_0
            node_id = f"{base_id}_{idx}"

            # Merge metadata: base metadata + node metadata + id
            node.metadata = {
                **metadata,
                **node.metadata,
                "id": node_id,  # Add id field to metadata (for vector_store compatibility)
            }

            # Set node ID (required for delete_ref_doc to work)
            # Note: ref_doc_id is a read-only property and cannot be modified
            # We rely on id_ for deletion, which is stored in metadata and can be used
            # to identify documents for deletion
            node.id_ = node_id
            # Store id_ in metadata as well for easier lookup
            # This ensures we can find and delete documents even if ref_doc_id doesn't match
            if "id" not in node.metadata:
                node.metadata["id"] = node_id
        return nodes

    def parse_directory_return_documents(self, directory: Path) -> List[Document]:
        """Parse all Markdown files in a directory (recursively).

        Args:
            directory: Directory containing Markdown files

        Returns:
            List of Node objects (ready for direct insertion into vector store)
        """
        documents = []
        markdown_files = []

        # Try rglob first (most efficient)
        try:
            md_files = list(directory.rglob("*.md"))
            markdown_files.extend(md_files)
            markdown_files.extend(directory.rglob("*.markdown"))
            logger.debug(f"rglob found {len(markdown_files)} files")
        except Exception as e:
            logger.warning(f"rglob failed: {e}, trying manual traversal")

        # If rglob didn't find files, manually traverse subdirectories
        # This handles cases where rglob might not work from the root directory
        if not markdown_files:
            logger.debug("rglob found no files, manually traversing subdirectories")
            for subdir in directory.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    try:
                        subdir_files = list(subdir.rglob("*.md"))
                        subdir_files.extend(subdir.rglob("*.markdown"))
                        markdown_files.extend(subdir_files)
                        logger.debug(
                            f"Found {len(subdir_files)} files in {subdir.name}"
                        )
                    except Exception as e:
                        logger.warning(f"Error traversing {subdir}: {e}")

        # Filter out files in .git directories and other hidden directories
        # Also filter out directories that match .md/.markdown pattern (some systems allow dirs with these extensions)
        markdown_files = [
            f
            for f in markdown_files
            if f.is_file()  # Ensure it's actually a file, not a directory
            and ".git" not in f.parts
            and not any(part.startswith(".") for part in f.parts[1:])
        ]

        # Remove duplicates (in case both patterns match the same file)
        markdown_files = list(set(markdown_files))
        # Sort for consistent ordering
        markdown_files.sort()

        logger.info(f"Found {len(markdown_files)} markdown file(s) in {directory}")
        if markdown_files:
            logger.debug(
                f"Markdown files (first 10): {[str(f.relative_to(directory)) for f in markdown_files[:10]]}"
                + (
                    f" (showing first 10 of {len(markdown_files)})"
                    if len(markdown_files) > 10
                    else ""
                )
            )
        else:
            logger.warning(
                f"No markdown files found in {directory}. "
                f"Directory exists: {directory.exists()}, "
                f"Is directory: {directory.is_dir() if directory.exists() else 'N/A'}, "
                f"Subdirectories: {[p.name for p in directory.iterdir() if p.is_dir()][:5]}"
            )

        # Use tqdm to show progress for files
        files_iter = tqdm(markdown_files, desc="Parsing Markdown files", unit="file")
        for md_file in files_iter:
            try:
                # Show current file name in progress bar
                files_iter.set_postfix(file=md_file.name)
                logger.debug(f"Parsing markdown file: {md_file}")
                file_nodes = self.parse_file_return_documents(md_file)
                documents.extend(file_nodes)
                files_iter.set_postfix(
                    file=md_file.name, nodes=len(file_nodes), total=len(documents)
                )
                logger.debug(f"Parsed {md_file}: {len(file_nodes)} node(s) created")
            except Exception as e:
                logger.error(f"Error parsing {md_file}: {e}", exc_info=True)
                files_iter.set_postfix(file=f"{md_file.name} (ERROR)")
                print(f"Error parsing {md_file}: {e}")

        logger.info(
            f"Total nodes created from {len(markdown_files)} markdown file(s): {len(documents)}"
        )
        return documents

    def parse_directory(self, directory: Path) -> List[BaseNode]:
        """Parse all Markdown files in a directory (recursively).

        Args:
            directory: Directory containing Markdown files

        Returns:
            List of Node objects (ready for direct insertion into vector store)
        """
        nodes = []
        markdown_files = []

        # Try rglob first (most efficient)
        try:
            md_files = list(directory.rglob("*.md"))
            markdown_files.extend(md_files)
            markdown_files.extend(directory.rglob("*.markdown"))
            logger.debug(f"rglob found {len(markdown_files)} files")
        except Exception as e:
            logger.warning(f"rglob failed: {e}, trying manual traversal")

        # If rglob didn't find files, manually traverse subdirectories
        # This handles cases where rglob might not work from the root directory
        if not markdown_files:
            logger.debug("rglob found no files, manually traversing subdirectories")
            for subdir in directory.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    try:
                        subdir_files = list(subdir.rglob("*.md"))
                        subdir_files.extend(subdir.rglob("*.markdown"))
                        markdown_files.extend(subdir_files)
                        logger.debug(
                            f"Found {len(subdir_files)} files in {subdir.name}"
                        )
                    except Exception as e:
                        logger.warning(f"Error traversing {subdir}: {e}")

        # Filter out files in .git directories and other hidden directories
        # Also filter out directories that match .md/.markdown pattern (some systems allow dirs with these extensions)
        markdown_files = [
            f
            for f in markdown_files
            if f.is_file()  # Ensure it's actually a file, not a directory
            and ".git" not in f.parts
            and not any(part.startswith(".") for part in f.parts[1:])
        ]

        # Remove duplicates (in case both patterns match the same file)
        markdown_files = list(set(markdown_files))
        # Sort for consistent ordering
        markdown_files.sort()

        logger.info(f"Found {len(markdown_files)} markdown file(s) in {directory}")
        if markdown_files:
            logger.debug(
                f"Markdown files (first 10): {[str(f.relative_to(directory)) for f in markdown_files[:10]]}"
                + (
                    f" (showing first 10 of {len(markdown_files)})"
                    if len(markdown_files) > 10
                    else ""
                )
            )
        else:
            logger.warning(
                f"No markdown files found in {directory}. "
                f"Directory exists: {directory.exists()}, "
                f"Is directory: {directory.is_dir() if directory.exists() else 'N/A'}, "
                f"Subdirectories: {[p.name for p in directory.iterdir() if p.is_dir()][:5]}"
            )

        # Use tqdm to show progress for files
        files_iter = tqdm(markdown_files, desc="Parsing Markdown files", unit="file")
        for md_file in files_iter:
            try:
                # Show current file name in progress bar
                files_iter.set_postfix(file=md_file.name)
                logger.debug(f"Parsing markdown file: {md_file}")
                file_nodes = self.parse_file(md_file)
                nodes.extend(file_nodes)
                files_iter.set_postfix(
                    file=md_file.name, nodes=len(file_nodes), total=len(nodes)
                )
                logger.debug(f"Parsed {md_file}: {len(file_nodes)} node(s) created")
            except Exception as e:
                logger.error(f"Error parsing {md_file}: {e}", exc_info=True)
                files_iter.set_postfix(file=f"{md_file.name} (ERROR)")
                print(f"Error parsing {md_file}: {e}")

        logger.info(
            f"Total nodes created from {len(markdown_files)} markdown file(s): {len(nodes)}"
        )
        return nodes

    def delete_file_documents(
        self, file_path: Path, vector_store_manager
    ) -> Dict[str, Any]:
        """Delete all documents associated with a Markdown file from vector store.

        This method identifies all document IDs that were generated from the given file
        and deletes them from the vector store. Since a file may be split into multiple
        nodes (with IDs like {base_id}_0, {base_id}_1, etc.), this method needs to
        delete all related documents.

        Args:
            file_path: Path to Markdown file whose documents should be deleted
            vector_store_manager: VectorStoreManager instance for deleting documents

        Returns:
            Dictionary with keys:
                - deleted_count: Number of documents deleted
                - errors: List of error messages (if any)

        TODO:
            - Implement ID generation logic to match parse_file() behavior
            - Query vector store to find all documents with matching base_id pattern
            - Delete documents using vector_store_manager.docs_index.delete_ref_doc()
            - Handle edge cases (file not found, no documents to delete, etc.)
            - Add logging for deletion operations
            - Consider batch deletion for performance
        """
        # TODO: Implement deletion logic
        # 1. Generate base_id from file_path (same logic as parse_file)
        # 2. Query vector store to find all documents with IDs matching {base_id}_*
        # 3. Delete each document using vector_store_manager.docs_index.delete_ref_doc()
        # 4. Return deletion results

        raise NotImplementedError(
            "delete_file_documents() is not yet implemented. "
            "See TODO in method docstring and README.md roadmap."
        )

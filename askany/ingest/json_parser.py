"""JSON parser for FAQ data."""

import json
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from tqdm import tqdm

logger = getLogger(__name__)


class JSONParser:
    """Parser for JSON FAQ files."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize JSON parser.

        Args:
            chunk_size: Size of chunks in tokens (default 512, FAQ typically 84-135 tokens)
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def parse_file(
        self, file_path: Path, show_progress: bool = False
    ) -> List[Document]:
        """Parse a JSON FAQ file into Documents.

        Args:
            file_path: Path to JSON file
            show_progress: Whether to show progress bar (default: False)

        Returns:
            List of Document objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        if isinstance(data, list):
            items_iter = tqdm(
                data,
                desc=f"Parsing {file_path.name}",
                disable=not show_progress,
                leave=False,
            )
            for item in items_iter:
                # Skip non-dict items (e.g., nested lists, strings, etc.)
                if not isinstance(item, dict):
                    logger.warning(
                        f"Skipping non-dict item in {file_path}: {type(item).__name__}"
                    )
                    continue
                try:
                    doc = self._item_to_document(item, file_path)
                    if doc is not None:
                        documents.append(doc)
                except Exception as e:
                    logger.error(
                        f"Error converting item to document in {file_path}: {e}"
                    )
                    continue
        elif isinstance(data, dict):
            try:
                doc = self._item_to_document(data, file_path)
                if doc is not None:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error converting dict to document in {file_path}: {e}")
        else:
            logger.warning(
                f"Unexpected data type in {file_path}: {type(data).__name__}"
            )

        return documents

    def _item_to_document(self, item: Dict[str, Any], source_path: Path) -> Document:
        """Convert a FAQ item to a Document.

        Args:
            item: FAQ item dictionary
            source_path: Source file path

        Returns:
            Document object

        Raises:
            TypeError: If item is not a dictionary
        """
        # Validate input type
        if not isinstance(item, dict):
            raise TypeError(f"Expected dict, got {type(item).__name__}")

        # Build text content from FAQ item
        question = item.get("question", "")
        answer = item.get("answer", "")
        text = f"问题: {question}\n答案: {answer}"

        # Build metadata
        metadata = {
            "source": str(source_path),
            "type": "faq",
            "id": item.get("id", ""),
            "hardware": item.get("hardware", ""),
            "lang": item.get("lang", "zh-CN"),
            "tags": (
                ",".join(item.get("tags", []))
                if isinstance(item.get("tags"), list)
                else str(item.get("tags", ""))
            ),
            "last_updated": item.get("last_updated", ""),
        }

        # Add custom metadata if present
        if "metadata" in item and isinstance(item["metadata"], dict):
            for key, value in item["metadata"].items():
                metadata[f"meta_{key}"] = str(value)

        # Get document ID from item (used for delete_ref_doc)
        doc_id = item.get("id", "")

        # Create Document with id_ parameter if ID is provided
        # This ensures delete_ref_doc() can find and delete the document correctly
        if doc_id:
            return Document(
                text=text,
                metadata=metadata,
                id_=doc_id,  # Set Document's id_ for delete_ref_doc to work
            )
        else:
            return Document(
                text=text,
                metadata=metadata,
            )

    def parse_directory(
        self, directory: Path, show_progress: bool = True
    ) -> List[Document]:
        """Parse all JSON files in a directory.

        Args:
            directory: Directory containing JSON files
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of Document objects
        """
        documents = []
        json_files = list(directory.glob("*.json"))

        if not json_files:
            logger.info(f"No JSON files found in {directory}")
            return documents

        logger.info(f"Found {len(json_files)} JSON file(s) in {directory}")

        # Use tqdm to show progress for files
        files_iter = tqdm(
            json_files, desc="Parsing JSON files", disable=not show_progress
        )
        for json_file in files_iter:
            try:
                # Show current file name in progress bar
                files_iter.set_postfix(file=json_file.name)
                docs = self.parse_file(json_file, show_progress=False)
                documents.extend(docs)
                files_iter.set_postfix(
                    file=json_file.name, docs=len(docs), total=len(documents)
                )
            except Exception as e:
                logger.error(f"Error parsing {json_file}: {e}", exc_info=True)
                files_iter.set_postfix(file=f"{json_file.name} (ERROR)")

        logger.info(
            f"Total documents created from {len(json_files)} JSON file(s): {len(documents)}"
        )
        return documents

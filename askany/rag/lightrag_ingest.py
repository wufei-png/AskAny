"""
LightRAG ingestion helper.

Use this to build / update the LightRAG knowledge graph independently of
the main AskAny ingestion pipeline.

CLI usage
---------
    # Ingest all markdown files from data/markdown
    python -m askany.rag.lightrag_ingest --ingest-markdown

    # Ingest all JSON FAQ files from data/json
    python -m askany.rag.lightrag_ingest --ingest-json

    # Ingest a single file
    python -m askany.rag.lightrag_ingest --file path/to/doc.md

    # Run both
    python -m askany.rag.lightrag_ingest --ingest-markdown --ingest-json

Programmatic usage
------------------
    from askany.rag.lightrag_ingest import LightRAGIngestor

    async with LightRAGIngestor() as ingestor:
        await ingestor.ingest_directory("data/markdown", glob="**/*.md")
        await ingestor.ingest_json_faqs("data/json")
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class LightRAGIngestor:
    """
    Async context manager that drives LightRAG document ingestion.

    Example
    -------
    ::

        async with LightRAGIngestor() as ingestor:
            await ingestor.ingest_directory("data/markdown")
            await ingestor.ingest_json_faqs("data/json")
    """

    def __init__(self, batch_size: int = 10) -> None:
        from askany.rag.lightrag_adapter import LightRAGAdapter

        self._adapter = LightRAGAdapter()
        self._batch_size = batch_size

    async def __aenter__(self) -> "LightRAGIngestor":
        await self._adapter.initialize()
        return self

    async def __aexit__(self, *_) -> None:
        await self._adapter.finalize()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def ingest_file(self, path: str | Path) -> None:
        """Ingest a single file (Markdown or plain text)."""
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            logger.warning("Skipping empty file: %s", path)
            return
        await self._adapter.insert_async(
            text, file_paths=str(path), split_by_character="\n## "
        )
        logger.info("Ingested: %s", path)

    async def ingest_directory(
        self,
        directory: str | Path,
        glob: str = "**/*.md",
    ) -> int:
        """Recursively ingest all matching files in *directory*.

        Returns the number of files processed.
        """
        directory = Path(directory)
        files = sorted(directory.glob(glob))
        if not files:
            logger.warning("No files found matching %s in %s", glob, directory)
            return 0

        logger.info("Found %d files to ingest from %s", len(files), directory)
        count = 0
        for i in range(0, len(files), self._batch_size):
            batch = files[i : i + self._batch_size]
            texts, paths = _read_batch(batch)
            if texts:
                await self._adapter.insert_async(
                    texts, file_paths=paths, split_by_character="\n## "
                )
                count += len(texts)
                logger.info(
                    "Ingested batch %d/%d (%d files)",
                    i // self._batch_size + 1,
                    (len(files) + self._batch_size - 1) // self._batch_size,
                    len(texts),
                )
        return count

    async def ingest_json_faqs(self, directory: str | Path) -> int:
        """Ingest JSON FAQ files.

        Each JSON file must be either:
        * a single object with ``question`` + ``answer`` keys, or
        * a list of such objects.

        Returns the number of FAQ entries processed.
        """
        directory = Path(directory)
        json_files = sorted(directory.glob("**/*.json"))
        if not json_files:
            logger.warning("No JSON files found in %s", directory)
            return 0

        texts: List[str] = []
        paths: List[str] = []
        count = 0

        for fpath in json_files:
            entries = _parse_json_faq(fpath)
            for entry_text in entries:
                texts.append(entry_text)
                paths.append(str(fpath))
                count += 1
                if len(texts) >= self._batch_size:
                    await self._adapter.insert_async(texts, file_paths=paths)
                    logger.info("Ingested FAQ batch (%d entries)", len(texts))
                    texts, paths = [], []

        if texts:
            await self._adapter.insert_async(texts, file_paths=paths)
            logger.info("Ingested final FAQ batch (%d entries)", len(texts))

        logger.info("Total FAQ entries ingested: %d", count)
        return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_batch(files: List[Path]) -> Tuple[List[str], List[str]]:
    texts, paths = [], []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                texts.append(text)
                paths.append(str(f))
        except Exception as exc:
            logger.warning("Could not read %s: %s", f, exc)
    return texts, paths


def _parse_json_faq(path: Path) -> List[str]:
    """Return formatted FAQ strings from a JSON FAQ file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse JSON file %s: %s", path, exc)
        return []

    entries = raw if isinstance(raw, list) else [raw]
    results = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            results.append(f"问题: {q}\n答案: {a}")
        elif q:
            results.append(q)
        elif a:
            results.append(a)
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


async def _main(args: argparse.Namespace) -> None:
    async with LightRAGIngestor(batch_size=args.batch_size) as ingestor:
        if args.file:
            await ingestor.ingest_file(args.file)

        if args.ingest_markdown:
            from askany.config import settings

            md_dir = args.markdown_dir or settings.markdown_dir
            count = await ingestor.ingest_directory(md_dir)
            print(f"Markdown ingestion complete: {count} files")

        if args.ingest_json:
            from askany.config import settings

            json_dir = args.json_dir or settings.json_dir
            count = await ingestor.ingest_json_faqs(json_dir)
            print(f"JSON FAQ ingestion complete: {count} entries")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Ingest documents into LightRAG knowledge graph"
    )
    parser.add_argument("--file", help="Ingest a single file")
    parser.add_argument(
        "--ingest-markdown", action="store_true", help="Ingest all markdown docs"
    )
    parser.add_argument(
        "--ingest-json", action="store_true", help="Ingest all JSON FAQs"
    )
    parser.add_argument("--markdown-dir", help="Override markdown directory")
    parser.add_argument("--json-dir", help="Override JSON directory")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size (default: 10)"
    )
    args = parser.parse_args()

    if not any([args.file, args.ingest_markdown, args.ingest_json]):
        parser.print_help()
    else:
        asyncio.run(_main(args))

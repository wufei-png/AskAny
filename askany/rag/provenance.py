"""Shared provenance helpers for cross-retriever chunk alignment."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg2
from psycopg2.extras import execute_batch

from askany.config import settings

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

PROVENANCE_TABLE = "askany_retrieval_provenance"


def normalize_text(text: str) -> str:
    """Return a deterministic normalized representation for hashing/matching."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def canonicalize_path(file_path: str) -> str:
    """Normalize a path so both pipelines share the same document identity."""
    if not file_path:
        return ""

    path = Path(file_path)
    try:
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        return str(path)
    except Exception:
        return str(path)


def compute_source_doc_id(file_path: str) -> str:
    canonical_path = canonicalize_path(file_path)
    return hashlib.md5(canonical_path.encode("utf-8")).hexdigest()


def compute_source_unit_id(
    *,
    retrieval_origin: str,
    origin_id: str,
    canonical_path: str,
    start_line: int | None,
    end_line: int | None,
    text_hash: str,
) -> str:
    payload = "|".join(
        [
            retrieval_origin,
            origin_id or "",
            canonical_path or "",
            str(start_line or ""),
            str(end_line or ""),
            text_hash,
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ProvenanceRecord:
    retrieval_origin: str
    source_kind: str
    origin_id: str
    canonical_path: str
    source_doc_id: str
    source_unit_id: str
    start_line: int | None
    end_line: int | None
    text_hash: str
    content_length: int

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "retrieval_origin": self.retrieval_origin,
            "source_kind": self.source_kind,
            "origin_id": self.origin_id,
            "canonical_path": self.canonical_path,
            "source_doc_id": self.source_doc_id,
            "source_unit_id": self.source_unit_id,
            "text_hash": self.text_hash,
            "content_length": self.content_length,
        }
        if self.start_line is not None:
            metadata["start_line"] = self.start_line
        if self.end_line is not None:
            metadata["end_line"] = self.end_line
        return metadata


class ProvenanceRepository:
    """Simple Postgres-backed sidecar for chunk provenance."""

    def __init__(self) -> None:
        self._dsn = {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "user": settings.postgres_user,
            "password": settings.postgres_password.get_secret_value(),
            "database": settings.postgres_db,
        }
        self._table_ensured = False

    def _connect(self):
        return psycopg2.connect(**self._dsn)

    def ensure_table(self) -> None:
        if self._table_ensured:
            return

        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {PROVENANCE_TABLE} (
                            retrieval_origin TEXT NOT NULL,
                            source_kind TEXT NOT NULL,
                            origin_id TEXT NOT NULL,
                            canonical_path TEXT NOT NULL,
                            source_doc_id TEXT NOT NULL,
                            source_unit_id TEXT,
                            start_line INTEGER,
                            end_line INTEGER,
                            line_range int4range,
                            text_hash TEXT,
                            content_length INTEGER,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            PRIMARY KEY (retrieval_origin, source_kind, origin_id)
                        );
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{PROVENANCE_TABLE}_source_doc
                        ON {PROVENANCE_TABLE} (source_doc_id);
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{PROVENANCE_TABLE}_canonical_path
                        ON {PROVENANCE_TABLE} (canonical_path);
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{PROVENANCE_TABLE}_text_hash
                        ON {PROVENANCE_TABLE} (text_hash);
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{PROVENANCE_TABLE}_line_range
                        ON {PROVENANCE_TABLE} USING GIST (line_range);
                        """
                    )
        finally:
            conn.close()

        self._table_ensured = True

    def upsert_records(self, records: Iterable[ProvenanceRecord]) -> None:
        rows = list(records)
        if not rows:
            return

        self.ensure_table()
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    execute_batch(
                        cur,
                        f"""
                        INSERT INTO {PROVENANCE_TABLE} (
                            retrieval_origin,
                            source_kind,
                            origin_id,
                            canonical_path,
                            source_doc_id,
                            source_unit_id,
                            start_line,
                            end_line,
                            line_range,
                            text_hash,
                            content_length
                        )
                        VALUES (
                            %(retrieval_origin)s,
                            %(source_kind)s,
                            %(origin_id)s,
                            %(canonical_path)s,
                            %(source_doc_id)s,
                            %(source_unit_id)s,
                            %(start_line)s,
                            %(end_line)s,
                            CASE
                                WHEN %(start_line)s IS NOT NULL AND %(end_line)s IS NOT NULL
                                THEN int4range(%(start_line)s, %(end_plus_one)s, '[)')
                                ELSE NULL
                            END,
                            %(text_hash)s,
                            %(content_length)s
                        )
                        ON CONFLICT (retrieval_origin, source_kind, origin_id)
                        DO UPDATE SET
                            canonical_path = EXCLUDED.canonical_path,
                            source_doc_id = EXCLUDED.source_doc_id,
                            source_unit_id = EXCLUDED.source_unit_id,
                            start_line = EXCLUDED.start_line,
                            end_line = EXCLUDED.end_line,
                            line_range = EXCLUDED.line_range,
                            text_hash = EXCLUDED.text_hash,
                            content_length = EXCLUDED.content_length;
                        """,
                        [
                            {
                                "retrieval_origin": row.retrieval_origin,
                                "source_kind": row.source_kind,
                                "origin_id": row.origin_id,
                                "canonical_path": row.canonical_path,
                                "source_doc_id": row.source_doc_id,
                                "source_unit_id": row.source_unit_id,
                                "start_line": row.start_line,
                                "end_line": row.end_line,
                                "end_plus_one": row.end_line + 1
                                if row.end_line is not None
                                else None,
                                "text_hash": row.text_hash,
                                "content_length": row.content_length,
                            }
                            for row in rows
                        ],
                    )
        finally:
            conn.close()

    def get_record(
        self, retrieval_origin: str, source_kind: str, origin_id: str
    ) -> dict[str, Any] | None:
        self.ensure_table()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT retrieval_origin, source_kind, origin_id, canonical_path,
                           source_doc_id, source_unit_id, start_line, end_line,
                           text_hash, content_length
                    FROM {PROVENANCE_TABLE}
                    WHERE retrieval_origin = %s
                      AND source_kind = %s
                      AND origin_id = %s
                    """,
                    (retrieval_origin, source_kind, origin_id),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return None

        keys = [
            "retrieval_origin",
            "source_kind",
            "origin_id",
            "canonical_path",
            "source_doc_id",
            "source_unit_id",
            "start_line",
            "end_line",
            "text_hash",
            "content_length",
        ]
        return dict(zip(keys, row, strict=False))


@lru_cache(maxsize=64)
def _read_file_content(canonical_path: str) -> str:
    path = Path(canonical_path)
    if not canonical_path or not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _char_index_to_line_number(text: str, index: int) -> int:
    return text.count("\n", 0, max(index, 0)) + 1


def recover_line_range(
    text: str, file_path: str, *, hint_start_line: int = 1
) -> tuple[int | None, int | None]:
    """Recover [start_line, end_line] by locating chunk text inside the source file."""
    canonical_path = canonicalize_path(file_path)
    file_content = _read_file_content(canonical_path)
    if not text or not file_content:
        return (None, None)

    stripped_text = text.strip()
    if not stripped_text:
        return (None, None)

    line_hint_index = 0
    if hint_start_line > 1:
        line_hint_index = max(0, _line_start_offset(file_content, hint_start_line))

    start_idx = file_content.find(stripped_text, line_hint_index)
    if start_idx == -1 and line_hint_index > 0:
        start_idx = file_content.find(stripped_text)

    if start_idx != -1:
        end_idx = start_idx + len(stripped_text) - 1
        return (
            _char_index_to_line_number(file_content, start_idx),
            _char_index_to_line_number(file_content, end_idx),
        )

    return _recover_line_range_by_lines(
        stripped_text, file_content, hint_start_line=hint_start_line
    )


def _line_start_offset(text: str, line_number: int) -> int:
    if line_number <= 1:
        return 0
    current_line = 1
    for idx, char in enumerate(text):
        if char == "\n":
            current_line += 1
            if current_line == line_number:
                return idx + 1
    return len(text)


def _recover_line_range_by_lines(
    text: str, file_content: str, *, hint_start_line: int
) -> tuple[int | None, int | None]:
    chunk_lines = [line.strip() for line in text.splitlines() if line.strip()]
    file_lines = file_content.splitlines()
    if not chunk_lines or not file_lines:
        return (None, None)

    first_line = chunk_lines[0]
    last_line = chunk_lines[-1]

    start_candidates = []
    for idx in range(max(0, hint_start_line - 1), len(file_lines)):
        file_line = file_lines[idx].strip()
        if file_line and (first_line == file_line or first_line in file_line):
            start_candidates.append(idx)

    if not start_candidates:
        for idx, raw_line in enumerate(file_lines):
            file_line = raw_line.strip()
            if file_line and (first_line == file_line or first_line in file_line):
                start_candidates.append(idx)

    for start_idx in start_candidates:
        end_idx = start_idx
        for idx in range(len(file_lines) - 1, start_idx - 1, -1):
            file_line = file_lines[idx].strip()
            if file_line and (last_line == file_line or last_line in file_line):
                end_idx = idx
                if end_idx >= start_idx:
                    return (start_idx + 1, end_idx + 1)

    return (None, None)


def build_provenance_record(
    *,
    retrieval_origin: str,
    source_kind: str,
    origin_id: str,
    file_path: str,
    text: str,
    source_unit_id: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    hint_start_line: int = 1,
) -> ProvenanceRecord:
    canonical_path = canonicalize_path(file_path)
    if start_line is None or end_line is None:
        start_line, end_line = recover_line_range(
            text, canonical_path, hint_start_line=hint_start_line
        )
    text_hash = hash_text(text)
    source_doc_id = compute_source_doc_id(canonical_path)
    computed_source_unit_id = source_unit_id or compute_source_unit_id(
        retrieval_origin=retrieval_origin,
        origin_id=origin_id,
        canonical_path=canonical_path,
        start_line=start_line,
        end_line=end_line,
        text_hash=text_hash,
    )
    return ProvenanceRecord(
        retrieval_origin=retrieval_origin,
        source_kind=source_kind,
        origin_id=origin_id,
        canonical_path=canonical_path,
        source_doc_id=source_doc_id,
        source_unit_id=computed_source_unit_id,
        start_line=start_line,
        end_line=end_line,
        text_hash=text_hash,
        content_length=len(text or ""),
    )


def enrich_nodes_with_provenance(
    nodes: list[NodeWithScore],
    retrieval_origin: str = "llamaindex",
    source_kind: str = "docs_chunk",
) -> list[NodeWithScore]:
    if not nodes:
        return nodes

    repo = ProvenanceRepository()
    repo.ensure_table()

    for node in nodes:
        metadata = node.node.metadata if hasattr(node.node, "metadata") else {}

        if metadata.get("source_doc_id") and metadata.get("canonical_path"):
            continue

        origin_id = metadata.get("id") or metadata.get("origin_id")
        if not origin_id:
            continue

        record = repo.get_record(retrieval_origin, source_kind, origin_id)
        if record:
            metadata["source_doc_id"] = record.get("source_doc_id")
            metadata["canonical_path"] = record.get("canonical_path")
            if record.get("start_line") is not None:
                metadata["start_line"] = record.get("start_line")
            if record.get("end_line") is not None:
                metadata["end_line"] = record.get("end_line")

            node.node.metadata = metadata

    return nodes

#!/usr/bin/env python3
"""Opt-in LightRAG retrieval integration test.

Verifies that the LightRAGAdapter can initialise, query the knowledge graph,
and return well-formed NodeWithScore objects.

Prerequisites
-------------
* PostgreSQL running with LightRAG tables populated (run lightrag_ingest first).
* Environment variables / .env configured for DB + LLM access.

Usage
-----
    python -m pytest test/test_lightrag_retrieval.py -v -s
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import psycopg2
import pytest
import pytest_asyncio
from lightrag_question_loader import load_lightrag_questions
from lightrag_test_support import (
    is_lightrag_prerequisite_error,
    prerequisite_error_reason,
    skip_if_no_lightrag_data,
)

from askany.config import settings
from askany.rag.lightrag_adapter import LightRAGAdapter, get_lightrag_adapter

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.lightrag]


def _check_database() -> None:
    """Skip with a precise reason when the configured PostgreSQL is unavailable."""
    try:
        connection = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password.get_secret_value(),
            database=settings.postgres_db,
            connect_timeout=3,
        )
    except (
        psycopg2.OperationalError,
        psycopg2.InterfaceError,
        ConnectionError,
        OSError,
        TimeoutError,
    ) as exc:
        pytest.skip(
            "LightRAG retrieval skipped: PostgreSQL is unavailable "
            f"({prerequisite_error_reason(exc)})."
        )
    else:
        connection.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def lightrag_questions() -> list[str]:
    """Load questions during test execution so prerequisite skips exit cleanly."""
    try:
        questions = load_lightrag_questions()
    except FileNotFoundError as exc:
        pytest.skip(
            "LightRAG retrieval skipped: question file is missing "
            f"({exc.filename}); provide ASKANY_LIGHTRAG_QUESTIONS_FILE or create the "
            "gitignored local fixture."
        )
    if not questions:
        pytest.skip(
            "LightRAG retrieval skipped: the question file contains no questions."
        )
    if os.environ.get("ASKANY_RUN_LIGHTRAG_INTEGRATION") != "1":
        pytest.skip(
            "LightRAG retrieval skipped: opt-in integration; set "
            "ASKANY_RUN_LIGHTRAG_INTEGRATION=1 to run it."
        )
    return questions


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def adapter(lightrag_questions: list[str]) -> LightRAGAdapter:
    """Reuse one adapter on one event loop for the whole module."""
    if importlib.util.find_spec("lightrag") is None:
        pytest.skip(
            "LightRAG retrieval skipped: the optional lightrag-hku dependency is not installed."
        )
    _check_database()
    try:
        adapter = LightRAGAdapter()
    except Exception as exc:
        if not is_lightrag_prerequisite_error(exc):
            raise
        pytest.skip(
            "LightRAG retrieval skipped: the embedding model or adapter could not "
            f"be initialized ({prerequisite_error_reason(exc)})."
        )
    if adapter._rag is None:
        pytest.skip(
            "LightRAG retrieval skipped: LightRAG dependency is unavailable at runtime."
        )
    try:
        await adapter.initialize()
        probe_nodes = await adapter.retrieve_async(
            lightrag_questions[0], raise_on_error=True
        )
    except Exception as exc:
        if not is_lightrag_prerequisite_error(exc):
            raise
        pytest.skip(
            "LightRAG retrieval skipped: database tables or model endpoint are "
            f"unavailable ({prerequisite_error_reason(exc)})."
        )
    skip_if_no_lightrag_data(probe_nodes)
    yield adapter
    if adapter._initialized:
        await adapter.finalize()


# Use pytest-asyncio's built-in loop management instead of deprecated
# manual event_loop fixture.  Configure via pyproject.toml or pytest.ini:
#   [tool.pytest.ini_options]
#   asyncio_mode = "auto"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLightRAGAdapter:
    """Tests for LightRAGAdapter initialisation and retrieval."""

    def test_adapter_instantiation(self, adapter: LightRAGAdapter):
        """Adapter should instantiate without error."""
        assert adapter is not None
        assert adapter._rag is not None, (
            "LightRAG instance is None — is lightrag-hku installed?"
        )

    def test_singleton_identity(self):
        """get_lightrag_adapter() should return the same instance."""
        singleton_before = get_lightrag_adapter()
        assert get_lightrag_adapter() is singleton_before

    @pytest.mark.asyncio(loop_scope="module")
    async def test_initialize(self, adapter: LightRAGAdapter):
        """initialize() should succeed (idempotent)."""
        await adapter.initialize()
        assert adapter._initialized is True

    @pytest.mark.asyncio(loop_scope="module")
    async def test_retrieve_returns_nodes(
        self, adapter: LightRAGAdapter, lightrag_questions: list[str]
    ):
        """A retrieval query should return a non-empty list of NodeWithScore."""
        await adapter.initialize()
        test_query = lightrag_questions[0]
        nodes = await adapter.retrieve_async(test_query)

        assert isinstance(nodes, list), f"Expected list, got {type(nodes)}"
        assert len(nodes) > 0, (
            f"Expected non-empty results for query: {test_query!r}. "
            "Has viper-v5.5 data been ingested?"
        )

        # Verify each node's structure
        for node_with_score in nodes:
            assert hasattr(node_with_score, "score"), "Missing 'score' attribute"
            assert hasattr(node_with_score, "node"), "Missing 'node' attribute"
            assert isinstance(node_with_score.score, (int, float))
            assert node_with_score.node.get_content(), "Node text should be non-empty"

            metadata = node_with_score.node.metadata
            assert "type" in metadata, f"Missing 'type' in metadata: {metadata}"
            assert metadata["type"] in (
                "lightrag_chunk",
                "lightrag_entity",
                "lightrag_relation",
                "lightrag_context",
            ), f"Unexpected type: {metadata['type']}"

    @pytest.mark.asyncio(loop_scope="module")
    async def test_retrieve_node_metadata_keys(
        self, adapter: LightRAGAdapter, lightrag_questions: list[str]
    ):
        """Chunk nodes should have expected metadata keys."""
        await adapter.initialize()
        nodes = await adapter.retrieve_async(lightrag_questions[0])

        chunk_nodes = [
            n for n in nodes if n.node.metadata.get("type") == "lightrag_chunk"
        ]
        entity_nodes = [
            n for n in nodes if n.node.metadata.get("type") == "lightrag_entity"
        ]
        relation_nodes = [
            n for n in nodes if n.node.metadata.get("type") == "lightrag_relation"
        ]

        # At least one category should be non-empty
        assert chunk_nodes or entity_nodes or relation_nodes, (
            "Expected at least one chunk, entity, or relation node"
        )

        for chunk in chunk_nodes:
            meta = chunk.node.metadata
            assert "source" in meta
            assert "file_path" in meta
            assert "lightrag_ref_id" in meta or "chunk_id" in meta

        for entity in entity_nodes:
            meta = entity.node.metadata
            assert "entity_name" in meta
            assert "entity_type" in meta

        for rel in relation_nodes:
            meta = rel.node.metadata
            assert "src_id" in meta
            assert "tgt_id" in meta

    @pytest.mark.asyncio(loop_scope="module")
    async def test_retrieve_sync_wrapper(
        self, adapter: LightRAGAdapter, lightrag_questions: list[str]
    ):
        """The synchronous retrieve() wrapper should also work."""
        await adapter.initialize()
        nodes = adapter.retrieve(lightrag_questions[0])

        assert isinstance(nodes, list)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_retrieve_multiple_questions(
        self, adapter: LightRAGAdapter, lightrag_questions: list[str]
    ):
        """Run a few lightrag_questions and print results for manual inspection."""
        await adapter.initialize()

        # Test first 3 questions
        sample_questions = lightrag_questions[:3]
        for q in sample_questions:
            nodes = await adapter.retrieve_async(q)
            print(f"\n{'=' * 80}")
            print(f"Query: {q}")
            print(f"Results: {len(nodes)} nodes")

            for i, n in enumerate(nodes[:5]):  # show top 5
                node_type = n.node.metadata.get("type", "unknown")
                text_preview = n.node.get_content()[:200].replace("\n", " ")
                print(
                    f"  [{i + 1}] ({node_type}, score={n.score:.2f}) {text_preview}..."
                )

            assert isinstance(nodes, list)
            # Not asserting non-empty — some questions may not have hits
            # depending on what's been ingested.

    @pytest.mark.asyncio(loop_scope="module")
    async def test_finalize(self, adapter: LightRAGAdapter):
        """finalize() should close connections cleanly."""
        await adapter.finalize()
        assert adapter._initialized is False

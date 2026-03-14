#!/usr/bin/env python3
"""Standalone LightRAG retrieval test.

Verifies that the LightRAGAdapter can initialise, query the knowledge graph,
and return well-formed NodeWithScore objects.

Prerequisites
-------------
* PostgreSQL running with LightRAG tables populated (run lightrag_ingest first).
* Environment variables / .env configured for DB + LLM access.

Usage
-----
    python -m pytest test/test_lightrag_retrieval.py -v -s
    # or directly:
    python test/test_lightrag_retrieval.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio

from askany.rag.lightrag_adapter import LightRAGAdapter, get_lightrag_adapter

# Use a small subset of lightrag_questions for quick validation
from askany.workflow.question import lightrag_questions

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_QUERY = lightrag_questions[0]  # cross-component data flow question


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def adapter() -> LightRAGAdapter:
    """Reuse one adapter on one event loop for the whole module."""
    adapter = LightRAGAdapter()
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
    async def test_retrieve_returns_nodes(self, adapter: LightRAGAdapter):
        """A retrieval query should return a non-empty list of NodeWithScore."""
        await adapter.initialize()
        nodes = await adapter.retrieve_async(TEST_QUERY)

        assert isinstance(nodes, list), f"Expected list, got {type(nodes)}"
        assert len(nodes) > 0, (
            f"Expected non-empty results for query: {TEST_QUERY!r}. "
            "Has viper-v5.5 data been ingested?"
        )

        # Verify each node's structure
        for node_with_score in nodes:
            assert hasattr(node_with_score, "score"), "Missing 'score' attribute"
            assert hasattr(node_with_score, "node"), "Missing 'node' attribute"
            assert isinstance(node_with_score.score, (int, float))
            assert node_with_score.node.text, "Node text should be non-empty"

            metadata = node_with_score.node.metadata
            assert "type" in metadata, f"Missing 'type' in metadata: {metadata}"
            assert metadata["type"] in (
                "lightrag_chunk",
                "lightrag_entity",
                "lightrag_relation",
                "lightrag_context",
            ), f"Unexpected type: {metadata['type']}"

    @pytest.mark.asyncio(loop_scope="module")
    async def test_retrieve_node_metadata_keys(self, adapter: LightRAGAdapter):
        """Chunk nodes should have expected metadata keys."""
        await adapter.initialize()
        nodes = await adapter.retrieve_async(TEST_QUERY)

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
    async def test_retrieve_sync_wrapper(self, adapter: LightRAGAdapter):
        """The synchronous retrieve() wrapper should also work."""
        await adapter.initialize()
        nodes = adapter.retrieve(TEST_QUERY)

        assert isinstance(nodes, list)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_retrieve_multiple_questions(self, adapter: LightRAGAdapter):
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
                text_preview = n.node.text[:200].replace("\n", " ")
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


# ---------------------------------------------------------------------------
# CLI runner (for running outside pytest)
# ---------------------------------------------------------------------------


async def _run_manual():
    """Manual test runner with verbose output."""
    print("=" * 80)
    print("LightRAG Standalone Retrieval Test")
    print("=" * 80)

    adapter = get_lightrag_adapter()
    print(f"\n✓ Adapter created (LightRAG available: {adapter._rag is not None})")

    if adapter._rag is None:
        print("✗ LightRAG not available — install lightrag-hku")
        return False

    await adapter.initialize()
    print("✓ Adapter initialised")

    success = True
    for i, question in enumerate(lightrag_questions):
        print(f"\n{'─' * 80}")
        print(f"Question [{i + 1}/{len(lightrag_questions)}]: {question}")

        try:
            nodes = await adapter.retrieve_async(question)
            print(f"  → {len(nodes)} nodes returned")

            chunks = [
                n for n in nodes if n.node.metadata.get("type") == "lightrag_chunk"
            ]
            entities = [
                n for n in nodes if n.node.metadata.get("type") == "lightrag_entity"
            ]
            rels = [
                n for n in nodes if n.node.metadata.get("type") == "lightrag_relation"
            ]
            print(
                f"    Chunks: {len(chunks)}, Entities: {len(entities)}, Relations: {len(rels)}"
            )

            if not nodes:
                print("  ⚠ No results (data may not be ingested)")
            else:
                # Show top 3 nodes
                for j, n in enumerate(nodes[:3]):
                    text_preview = n.node.text[:150].replace("\n", " ")
                    print(
                        f"    [{j + 1}] ({n.node.metadata.get('type')}, score={n.score:.2f}) {text_preview}"
                    )

        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            success = False

    await adapter.finalize()
    print(f"\n{'=' * 80}")
    print(f"✓ Test complete ({'PASS' if success else 'FAIL'})")
    return success


if __name__ == "__main__":
    result = asyncio.run(_run_manual())
    sys.exit(0 if result else 1)

from llama_index.core.schema import NodeWithScore, TextNode

from askany.rag.lightrag_merge import merge_lightrag_with_llamaindex


class DummyRepo:
    def ensure_table(self):
        return None

    def get_record(self, retrieval_origin, source_kind, origin_id):
        return None

    def upsert_records(self, records):
        return None


class DummyLocalFileSearch:
    def find_text_line_range(self, text: str, file_path: str):
        return None


def _make_node(text: str, score: float, metadata: dict) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, metadata=metadata), score=score)


def test_many_to_many_overlap_keeps_llama_order_and_primary_flag():
    local_search = DummyLocalFileSearch()
    repo = DummyRepo()

    llama_1 = _make_node(
        "A\nB\nC",
        0.9,
        {
            "type": "markdown",
            "file_path": "/tmp/doc.md",
            "source": "/tmp/doc.md",
            "retrieval_origin": "llamaindex",
            "source_kind": "docs_chunk",
            "origin_id": "llama-1",
            "source_doc_id": "doc-1",
            "start_line": 1,
            "end_line": 3,
            "text_hash": "t1",
        },
    )
    llama_2 = _make_node(
        "D\nE\nF",
        0.8,
        {
            "type": "markdown",
            "file_path": "/tmp/doc.md",
            "source": "/tmp/doc.md",
            "retrieval_origin": "llamaindex",
            "source_kind": "docs_chunk",
            "origin_id": "llama-2",
            "source_doc_id": "doc-1",
            "start_line": 4,
            "end_line": 6,
            "text_hash": "t2",
        },
    )
    llama_3 = _make_node(
        "G\nH\nI",
        0.7,
        {
            "type": "markdown",
            "file_path": "/tmp/doc.md",
            "source": "/tmp/doc.md",
            "retrieval_origin": "llamaindex",
            "source_kind": "docs_chunk",
            "origin_id": "llama-3",
            "source_doc_id": "doc-1",
            "start_line": 7,
            "end_line": 9,
            "text_hash": "t3",
        },
    )
    lightrag_chunk = _make_node(
        "B\nC\nD\nE",
        0.75,
        {
            "type": "lightrag_chunk",
            "file_path": "/tmp/doc.md",
            "source": "/tmp/doc.md",
            "retrieval_origin": "lightrag",
            "source_kind": "lightrag_chunk",
            "origin_id": "chunk-1",
            "chunk_id": "chunk-1",
            "source_doc_id": "doc-1",
            "start_line": 2,
            "end_line": 5,
            "text_hash": "tc",
        },
    )

    merged = merge_lightrag_with_llamaindex(
        [llama_1, llama_2, llama_3],
        [lightrag_chunk],
        query="question",
        top_k=3,
        local_file_search=local_search,
        reranker=None,
        provenance_repo=repo,
    )

    assert [node.node.metadata["origin_id"] for node in merged[:2]] == [
        "llama-1",
        "llama-2",
    ]
    assert merged[0].node.metadata["related_lightrag_chunks"][0]["primary_overlap"] is True
    assert merged[1].node.metadata["related_lightrag_chunks"][0]["primary_overlap"] is False


def test_overlap_block_short_circuits_top_k():
    local_search = DummyLocalFileSearch()
    repo = DummyRepo()

    llama_1 = _make_node(
        "A\nB",
        0.9,
        {
            "type": "markdown",
            "file_path": "/tmp/doc.md",
            "source": "/tmp/doc.md",
            "retrieval_origin": "llamaindex",
            "source_kind": "docs_chunk",
            "origin_id": "llama-1",
            "source_doc_id": "doc-1",
            "start_line": 1,
            "end_line": 2,
            "text_hash": "t1",
        },
    )
    llama_2 = _make_node(
        "C\nD",
        0.8,
        {
            "type": "markdown",
            "file_path": "/tmp/doc.md",
            "source": "/tmp/doc.md",
            "retrieval_origin": "llamaindex",
            "source_kind": "docs_chunk",
            "origin_id": "llama-2",
            "source_doc_id": "doc-1",
            "start_line": 3,
            "end_line": 4,
            "text_hash": "t2",
        },
    )
    lightrag_1 = _make_node(
        "A\nB",
        0.7,
        {
            "type": "lightrag_chunk",
            "file_path": "/tmp/doc.md",
            "source": "/tmp/doc.md",
            "retrieval_origin": "lightrag",
            "source_kind": "lightrag_chunk",
            "origin_id": "chunk-1",
            "chunk_id": "chunk-1",
            "source_doc_id": "doc-1",
            "start_line": 1,
            "end_line": 2,
            "text_hash": "lt1",
        },
    )
    lightrag_2 = _make_node(
        "C\nD",
        0.7,
        {
            "type": "lightrag_chunk",
            "file_path": "/tmp/doc.md",
            "source": "/tmp/doc.md",
            "retrieval_origin": "lightrag",
            "source_kind": "lightrag_chunk",
            "origin_id": "chunk-2",
            "chunk_id": "chunk-2",
            "source_doc_id": "doc-1",
            "start_line": 3,
            "end_line": 4,
            "text_hash": "lt2",
        },
    )
    unmatched_chunk = _make_node(
        "X\nY",
        0.95,
        {
            "type": "lightrag_chunk",
            "file_path": "/tmp/other.md",
            "source": "/tmp/other.md",
            "retrieval_origin": "lightrag",
            "source_kind": "lightrag_chunk",
            "origin_id": "chunk-3",
            "chunk_id": "chunk-3",
            "source_doc_id": "doc-2",
            "start_line": 1,
            "end_line": 2,
            "text_hash": "lt3",
        },
    )

    merged = merge_lightrag_with_llamaindex(
        [llama_1, llama_2],
        [lightrag_1, lightrag_2, unmatched_chunk],
        query="question",
        top_k=2,
        local_file_search=local_search,
        reranker=None,
        provenance_repo=repo,
    )

    assert [node.node.metadata["origin_id"] for node in merged] == ["llama-1", "llama-2"]

"""Provenance-aware merge of LlamaIndex docs chunks and LightRAG results."""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Any, Optional

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore

from askany.rag.provenance import (
    ProvenanceRepository,
    build_provenance_record,
    canonicalize_path,
    compute_source_doc_id,
)

logger = logging.getLogger(__name__)


def render_node_with_enrichment(
    node: NodeWithScore, *, max_related_items: int = 3
) -> str:
    """Render node text plus primary LightRAG enrichments for LLM context/output."""
    content = (
        node.node.get_content() if hasattr(node.node, "get_content") else node.node.text
    )
    metadata = node.node.metadata if hasattr(node.node, "metadata") else {}
    parts = [content]

    related_chunks = _primary_only(metadata.get("related_lightrag_chunks", []))
    if related_chunks:
        chunk_lines = []
        for item in related_chunks[:max_related_items]:
            text = item.get("content", "").strip()
            if text:
                chunk_lines.append(text)
        if chunk_lines:
            parts.append("[LightRAG Related Chunks]\n" + "\n\n".join(chunk_lines))

    related_entities = _primary_only(metadata.get("related_lightrag_entities", []))
    if related_entities:
        entity_lines = []
        for item in related_entities[:max_related_items]:
            name = item.get("entity_name", "").strip()
            description = item.get("description", "").strip()
            if name and description:
                entity_lines.append(f"- {name}: {description}")
            elif name:
                entity_lines.append(f"- {name}")
            elif description:
                entity_lines.append(f"- {description}")
        if entity_lines:
            parts.append("[LightRAG Related Entities]\n" + "\n".join(entity_lines))

    related_relations = _primary_only(metadata.get("related_lightrag_relations", []))
    if related_relations:
        relation_lines = []
        for item in related_relations[:max_related_items]:
            src = item.get("src_id", "").strip()
            tgt = item.get("tgt_id", "").strip()
            description = item.get("description", "").strip()
            if src or tgt:
                relation_lines.append(f"- {src} -> {tgt}: {description}")
            elif description:
                relation_lines.append(f"- {description}")
        if relation_lines:
            parts.append("[LightRAG Related Relations]\n" + "\n".join(relation_lines))

    return "\n\n".join(part for part in parts if part.strip())


def merge_lightrag_with_llamaindex(
    llama_nodes: list[NodeWithScore],
    lightrag_nodes: list[NodeWithScore],
    *,
    query: str,
    top_k: int,
    local_file_search,
    reranker=None,
    provenance_repo: Optional[ProvenanceRepository] = None,
) -> list[NodeWithScore]:
    """Keep overlapped LlamaIndex chunks first and rerank the remainder pool."""
    if not lightrag_nodes:
        return llama_nodes
    if top_k <= 0:
        top_k = max(len(llama_nodes), len(lightrag_nodes))

    repo = provenance_repo or ProvenanceRepository()
    repo.ensure_table()

    prepared_llama_nodes = [
        _ensure_node_provenance(
            node,
            repo=repo,
            local_file_search=local_file_search,
            retrieval_origin_hint=node.node.metadata.get(
                "retrieval_origin", "llamaindex"
            )
            if hasattr(node.node, "metadata")
            else "llamaindex",
            source_kind_hint=_infer_source_kind(node),
        )
        for node in llama_nodes
    ]
    prepared_lightrag_nodes = [
        _ensure_node_provenance(
            node,
            repo=repo,
            local_file_search=local_file_search,
            retrieval_origin_hint="lightrag",
            source_kind_hint=_infer_source_kind(node),
        )
        for node in lightrag_nodes
    ]

    survivor_candidates = [
        idx for idx, node in enumerate(prepared_llama_nodes) if _is_docs_survivor(node)
    ]
    if not survivor_candidates:
        logger.debug("No docs survivor candidates found; skipping LightRAG/docs merge")
        return llama_nodes

    lightrag_chunks = [
        node
        for node in prepared_lightrag_nodes
        if _metadata(node).get("source_kind") == "lightrag_chunk"
    ]
    lightrag_entities = [
        node
        for node in prepared_lightrag_nodes
        if _metadata(node).get("source_kind") == "lightrag_entity"
    ]
    lightrag_relations = [
        node
        for node in prepared_lightrag_nodes
        if _metadata(node).get("source_kind") == "lightrag_relation"
    ]

    survivors = [copy.deepcopy(node) for node in prepared_llama_nodes]
    matched_lightrag_chunk_ids: set[str] = set()
    matched_survivor_indexes: set[int] = set()
    doc_to_survivor_indexes: dict[str, list[int]] = defaultdict(list)
    doc_to_primary_indexes: dict[str, list[int]] = defaultdict(list)

    for chunk in lightrag_chunks:
        matches = []
        for survivor_index in survivor_candidates:
            overlap = _calculate_overlap(
                survivors[survivor_index], chunk, local_file_search=local_file_search
            )
            if overlap is not None:
                matches.append((survivor_index, overlap))

        if not matches:
            continue

        matched_lightrag_chunk_ids.add(_origin_id(chunk))
        matched_indexes = [index for index, _ in matches]
        matched_survivor_indexes.update(matched_indexes)
        primary_survivor_index = _choose_primary_match(matches, survivors)

        for survivor_index, overlap in matches:
            is_primary = survivor_index == primary_survivor_index
            _attach_related_chunk(
                survivors[survivor_index],
                chunk,
                overlap=overlap,
                primary_overlap=is_primary,
            )
            doc_id = _doc_identity(survivors[survivor_index])
            if doc_id:
                if survivor_index not in doc_to_survivor_indexes[doc_id]:
                    doc_to_survivor_indexes[doc_id].append(survivor_index)
                if is_primary and survivor_index not in doc_to_primary_indexes[doc_id]:
                    doc_to_primary_indexes[doc_id].append(survivor_index)

    _attach_secondary_context(
        survivors,
        lightrag_entities,
        key="related_lightrag_entities",
        doc_to_survivor_indexes=doc_to_survivor_indexes,
        doc_to_primary_indexes=doc_to_primary_indexes,
    )
    _attach_secondary_context(
        survivors,
        lightrag_relations,
        key="related_lightrag_relations",
        doc_to_survivor_indexes=doc_to_survivor_indexes,
        doc_to_primary_indexes=doc_to_primary_indexes,
    )

    overlapped_survivors = [
        survivors[idx]
        for idx in range(len(survivors))
        if idx in matched_survivor_indexes
    ]
    if len(overlapped_survivors) >= top_k:
        return overlapped_survivors[:top_k]

    remainder_pool = [
        survivors[idx]
        for idx in range(len(survivors))
        if idx not in matched_survivor_indexes
    ]
    remainder_pool.extend(
        node
        for node in lightrag_chunks
        if _origin_id(node) not in matched_lightrag_chunk_ids
    )
    remainder_pool.extend(
        node
        for node in lightrag_entities
        if _doc_identity(node) not in doc_to_survivor_indexes
    )
    remainder_pool.extend(
        node
        for node in lightrag_relations
        if _doc_identity(node) not in doc_to_survivor_indexes
    )

    remainder_pool.sort(key=lambda node: node.score or 0.0, reverse=True)
    if remainder_pool and reranker:
        try:
            remainder_pool = reranker.postprocess_nodes(
                remainder_pool, query_bundle=QueryBundle(query)
            )
        except Exception as exc:
            logger.warning("Reranking remainder pool failed: %s", exc)

    remaining_slots = max(top_k - len(overlapped_survivors), 0)
    return overlapped_survivors + remainder_pool[:remaining_slots]


def _attach_secondary_context(
    survivors: list[NodeWithScore],
    lightrag_nodes: list[NodeWithScore],
    *,
    key: str,
    doc_to_survivor_indexes: dict[str, list[int]],
    doc_to_primary_indexes: dict[str, list[int]],
) -> None:
    for node in lightrag_nodes:
        doc_id = _doc_identity(node)
        if not doc_id or doc_id not in doc_to_survivor_indexes:
            continue
        primary_indexes = set(doc_to_primary_indexes.get(doc_id, []))
        for survivor_index in doc_to_survivor_indexes[doc_id]:
            payload = _secondary_payload(node)
            payload["primary_overlap"] = survivor_index in primary_indexes
            _metadata(survivors[survivor_index]).setdefault(key, []).append(payload)


def _attach_related_chunk(
    survivor: NodeWithScore,
    chunk: NodeWithScore,
    *,
    overlap: dict[str, Any],
    primary_overlap: bool,
) -> None:
    metadata = _metadata(survivor)
    metadata.setdefault("merge_trace", [])
    metadata.setdefault("related_lightrag_chunks", [])

    chunk_payload = {
        "origin_id": _origin_id(chunk),
        "file_path": _metadata(chunk).get("file_path")
        or _metadata(chunk).get("source"),
        "content": (
            chunk.node.get_content()
            if hasattr(chunk.node, "get_content")
            else chunk.node.text
        ),
        "primary_overlap": primary_overlap,
        **overlap,
    }

    metadata["related_lightrag_chunks"].append(chunk_payload)
    metadata["related_lightrag_chunks"].sort(
        key=lambda item: item.get("overlap_line_count", 0), reverse=True
    )
    metadata["merge_trace"].append(
        {
            "type": "lightrag_chunk_overlap",
            "origin_id": _origin_id(chunk),
            "primary_overlap": primary_overlap,
            **overlap,
        }
    )


def _secondary_payload(node: NodeWithScore) -> dict[str, Any]:
    metadata = _metadata(node)
    payload = {
        "origin_id": _origin_id(node),
        "file_path": metadata.get("file_path") or metadata.get("source"),
        "primary_overlap": False,
    }
    if metadata.get("source_kind") == "lightrag_entity":
        payload.update(
            {
                "entity_name": metadata.get("entity_name", ""),
                "description": (
                    node.node.get_content()
                    if hasattr(node.node, "get_content")
                    else node.node.text
                ),
            }
        )
    else:
        payload.update(
            {
                "src_id": metadata.get("src_id", ""),
                "tgt_id": metadata.get("tgt_id", ""),
                "description": (
                    node.node.get_content()
                    if hasattr(node.node, "get_content")
                    else node.node.text
                ),
            }
        )
    return payload


def _choose_primary_match(
    matches: list[tuple[int, dict[str, Any]]], survivors: list[NodeWithScore]
) -> int:
    sorted_matches = sorted(
        matches,
        key=lambda item: (
            item[1].get("overlap_line_count", 0),
            -item[0],
        ),
        reverse=True,
    )
    return sorted_matches[0][0]


def _calculate_overlap(
    survivor: NodeWithScore, chunk: NodeWithScore, *, local_file_search
) -> Optional[dict[str, Any]]:
    survivor_meta = _metadata(survivor)
    chunk_meta = _metadata(chunk)

    if not _same_document(survivor_meta, chunk_meta):
        return None

    survivor_start = survivor_meta.get("start_line")
    survivor_end = survivor_meta.get("end_line")
    chunk_start = chunk_meta.get("start_line")
    chunk_end = chunk_meta.get("end_line")

    if None not in (survivor_start, survivor_end, chunk_start, chunk_end):
        if survivor_start <= chunk_end and survivor_end >= chunk_start:
            overlap_start = max(survivor_start, chunk_start)
            overlap_end = min(survivor_end, chunk_end)
            return _build_overlap_metrics(
                survivor_start,
                survivor_end,
                chunk_start,
                chunk_end,
                overlap_start,
                overlap_end,
            )
        return None

    if survivor_meta.get("text_hash") and survivor_meta.get(
        "text_hash"
    ) == chunk_meta.get("text_hash"):
        return {
            "overlap_start_line": survivor_start,
            "overlap_end_line": survivor_end,
            "overlap_line_count": max(
                (survivor_end or 0) - (survivor_start or 0) + 1, 0
            ),
            "overlap_ratio_against_llama": 1.0,
            "overlap_ratio_against_lightrag": 1.0,
            "match_reason": "text_hash",
        }

    if not _needs_fallback(survivor_meta, chunk_meta):
        return None

    survivor_range = _resolve_range_with_local_search(survivor, local_file_search)
    chunk_range = _resolve_range_with_local_search(chunk, local_file_search)
    if not survivor_range or not chunk_range:
        return None

    survivor_start, survivor_end = survivor_range
    chunk_start, chunk_end = chunk_range
    if survivor_start <= chunk_end and survivor_end >= chunk_start:
        overlap_start = max(survivor_start, chunk_start)
        overlap_end = min(survivor_end, chunk_end)
        _metadata(survivor)["start_line"] = survivor_start
        _metadata(survivor)["end_line"] = survivor_end
        _metadata(chunk)["start_line"] = chunk_start
        _metadata(chunk)["end_line"] = chunk_end
        overlap = _build_overlap_metrics(
            survivor_start,
            survivor_end,
            chunk_start,
            chunk_end,
            overlap_start,
            overlap_end,
        )
        overlap["match_reason"] = "fallback_line_search"
        return overlap
    return None


def _build_overlap_metrics(
    survivor_start: int,
    survivor_end: int,
    chunk_start: int,
    chunk_end: int,
    overlap_start: int,
    overlap_end: int,
) -> dict[str, Any]:
    overlap_line_count = max(overlap_end - overlap_start + 1, 0)
    llama_span = max(survivor_end - survivor_start + 1, 1)
    lightrag_span = max(chunk_end - chunk_start + 1, 1)
    return {
        "overlap_start_line": overlap_start,
        "overlap_end_line": overlap_end,
        "overlap_line_count": overlap_line_count,
        "overlap_ratio_against_llama": overlap_line_count / llama_span,
        "overlap_ratio_against_lightrag": overlap_line_count / lightrag_span,
        "match_reason": "line_range",
    }


def _resolve_range_with_local_search(
    node: NodeWithScore, local_file_search
) -> Optional[tuple[int, int]]:
    metadata = _metadata(node)
    file_path = metadata.get("file_path") or metadata.get("source")
    if not file_path:
        return None
    content = (
        node.node.get_content() if hasattr(node.node, "get_content") else node.node.text
    )
    resolved = local_file_search.find_text_line_range(file_path=file_path, text=content)
    if resolved:
        return resolved
    return None


def _needs_fallback(a_meta: dict[str, Any], b_meta: dict[str, Any]) -> bool:
    fields = ("source_doc_id", "canonical_path", "start_line", "end_line")
    return any(not a_meta.get(field) or not b_meta.get(field) for field in fields)


def _same_document(a_meta: dict[str, Any], b_meta: dict[str, Any]) -> bool:
    if a_meta.get("source_doc_id") and b_meta.get("source_doc_id"):
        return a_meta["source_doc_id"] == b_meta["source_doc_id"]
    if a_meta.get("canonical_path") and b_meta.get("canonical_path"):
        return a_meta["canonical_path"] == b_meta["canonical_path"]
    return False


def _doc_identity(node: NodeWithScore) -> str:
    metadata = _metadata(node)
    return metadata.get("source_doc_id") or metadata.get("canonical_path", "")


def _origin_id(node: NodeWithScore) -> str:
    metadata = _metadata(node)
    return (
        metadata.get("origin_id")
        or metadata.get("chunk_id")
        or getattr(node.node, "node_id", "")
        or getattr(node.node, "id_", "")
    )


def _infer_source_kind(node: NodeWithScore) -> str:
    metadata = _metadata(node)
    node_type = metadata.get("type", "")
    if node_type == "lightrag_chunk":
        return "lightrag_chunk"
    if node_type == "lightrag_entity":
        return "lightrag_entity"
    if node_type == "lightrag_relation":
        return "lightrag_relation"
    if node_type == "faq":
        return "faq"
    return "docs_chunk"


def _is_docs_survivor(node: NodeWithScore) -> bool:
    metadata = _metadata(node)
    return _infer_source_kind(node) == "docs_chunk"


def _metadata(node: NodeWithScore) -> dict[str, Any]:
    return node.node.metadata if hasattr(node.node, "metadata") else {}


def _primary_only(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []
    primary_items = [item for item in items if item.get("primary_overlap", False)]
    if primary_items:
        return _dedupe_by_origin(primary_items)
    return _dedupe_by_origin(items)


def _dedupe_by_origin(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        origin_id = (
            item.get("origin_id")
            or item.get("entity_name")
            or (
                item.get("src_id"),
                item.get("tgt_id"),
            )
        )
        if origin_id in seen:
            continue
        seen.add(origin_id)
        deduped.append(item)
    return deduped


def _ensure_node_provenance(
    node: NodeWithScore,
    *,
    repo: ProvenanceRepository,
    local_file_search,
    retrieval_origin_hint: str,
    source_kind_hint: str,
) -> NodeWithScore:
    metadata = _metadata(node)
    origin_id = (
        metadata.get("origin_id")
        or metadata.get("chunk_id")
        or getattr(node.node, "node_id", "")
        or getattr(node.node, "id_", "")
    )
    if not origin_id:
        return node

    retrieval_origin = metadata.get("retrieval_origin", retrieval_origin_hint)
    source_kind = metadata.get("source_kind", source_kind_hint)
    metadata["retrieval_origin"] = retrieval_origin
    metadata["source_kind"] = source_kind
    metadata["origin_id"] = origin_id

    record = repo.get_record(retrieval_origin, source_kind, origin_id)
    if record:
        metadata.update({k: v for k, v in record.items() if v is not None})
        return node

    file_path = metadata.get("file_path") or metadata.get("source")
    if not file_path:
        return node

    content = (
        node.node.get_content() if hasattr(node.node, "get_content") else node.node.text
    )
    start_line = metadata.get("start_line")
    end_line = metadata.get("end_line")
    if start_line is None or end_line is None:
        if local_file_search is not None:
            resolved = local_file_search.find_text_line_range(
                text=content, file_path=file_path
            )
            if resolved:
                start_line, end_line = resolved

    record = build_provenance_record(
        retrieval_origin=retrieval_origin,
        source_kind=source_kind,
        origin_id=origin_id,
        file_path=file_path,
        text=content,
        source_unit_id=metadata.get("source_unit_id") or origin_id,
        start_line=start_line,
        end_line=end_line,
    )
    metadata.update(record.to_metadata())
    repo.upsert_records([record])
    return node

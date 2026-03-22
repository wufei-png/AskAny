"""Centralized Prometheus metrics registry for AskAny."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

if TYPE_CHECKING:
    from prometheus_client import HistogramBase


# Default histogram buckets for latency metrics (in seconds)
DEFAULT_LATENCY_BUCKETS: list[float] = [
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    700.0,
]

# LLM-specific latency buckets
LLM_LATENCY_BUCKETS: list[float] = [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    700.0,
]


class MetricsRegistry:
    """Centralized registry for all Prometheus metrics."""

    def __init__(self) -> None:
        # API Layer Metrics
        self.askany_http_requests_total = Counter(
            "askany_http_requests_total",
            "Total HTTP requests",
            ["endpoint", "method", "status_code"],
        )
        self.askany_http_request_duration_seconds = Histogram(
            "askany_http_request_duration_seconds",
            "HTTP request duration",
            ["endpoint", "method"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_http_streaming_duration_seconds = Histogram(
            "askany_http_streaming_duration_seconds",
            "Streaming response duration",
            ["endpoint", "model"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_active_streams = Gauge(
            "askany_active_streams",
            "Active streaming connections",
            ["endpoint"],
        )
        self.askany_workflow_selected_total = Counter(
            "askany_workflow_selected_total",
            "Workflow selection count",
            ["workflow_type"],
        )
        self.askany_query_type_total = Counter(
            "askany_query_type_total",
            "Query type routing count",
            ["query_type"],
        )
        self.askany_health_status = Gauge(
            "askany_health_status",
            "Component health status",
            ["component"],
        )

        # LLM API Metrics
        self.askany_llm_requests_total = Counter(
            "askany_llm_requests_total",
            "LLM API requests",
            ["model", "status", "error_type"],
        )
        self.askany_llm_request_duration_seconds = Histogram(
            "askany_llm_request_duration_seconds",
            "LLM request latency",
            ["model", "operation"],
            buckets=LLM_LATENCY_BUCKETS,
        )
        self.askany_llm_tokens_total = Counter(
            "askany_llm_tokens_total",
            "LLM token usage",
            ["model", "token_type"],
        )
        self.askany_llm_timeout_total = Counter(
            "askany_llm_timeout_total",
            "LLM timeout count",
            ["model"],
        )
        self.askany_llm_404_retries_total = Counter(
            "askany_llm_404_retries_total",
            "LLM 404 retry count",
            ["model"],
        )
        self.askany_llm_rate_limit_wait_seconds = Histogram(
            "askany_llm_rate_limit_wait_seconds",
            "Rate limit wait time",
            ["model"],
            buckets=LLM_LATENCY_BUCKETS,
        )
        self.askany_llm_stream_chunks_total = Counter(
            "askany_llm_stream_chunks_total",
            "LLM streaming chunks",
            ["model"],
        )
        self.askany_llm_connection_errors_total = Counter(
            "askany_llm_connection_errors_total",
            "LLM connection errors",
            ["model", "error_type"],
        )

        # Database Metrics
        self.askany_db_connections_active = Gauge(
            "askany_db_connections_active",
            "Active DB connections",
            ["pool"],
        )
        self.askany_db_connection_errors_total = Counter(
            "askany_db_connection_errors_total",
            "DB connection errors",
            ["error_type"],
        )
        self.askany_db_connection_wait_time_seconds = Histogram(
            "askany_db_connection_wait_time_seconds",
            "DB connection wait time",
            ["pool"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_db_query_duration_seconds = Histogram(
            "askany_db_query_duration_seconds",
            "DB query latency",
            ["table", "operation"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_vector_search_duration_seconds = Histogram(
            "askany_vector_search_duration_seconds",
            "Vector search latency",
            ["index_type"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_vector_search_results_count = Histogram(
            "askany_vector_search_results_count",
            "Vector search results count",
            ["index_type"],
        )
        self.askany_keyword_search_duration_seconds = Histogram(
            "askany_keyword_search_duration_seconds",
            "Keyword search latency",
            ["index_name"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_hnsw_index_operations_total = Counter(
            "askany_hnsw_index_operations_total",
            "HNSW index operations",
            ["operation"],
        )
        self.askany_faq_update_total = Counter(
            "askany_faq_update_total",
            "FAQ update count",
            ["status"],
        )

        # RAG Retrieval Metrics
        self.askany_rag_query_total = Counter(
            "askany_rag_query_total",
            "RAG query count",
            ["engine", "query_type"],
        )
        self.askany_rag_query_duration_seconds = Histogram(
            "askany_rag_query_duration_seconds",
            "RAG query duration",
            ["engine", "query_type"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_rag_retrieval_latency_seconds = Histogram(
            "askany_rag_retrieval_latency_seconds",
            "RAG retrieval latency by stage",
            ["stage"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_rag_nodes_retrieved = Histogram(
            "askany_rag_nodes_retrieved",
            "RAG nodes retrieved",
            ["engine"],
        )
        self.askany_rerank_requests_total = Counter(
            "askany_rerank_requests_total",
            "Rerank requests",
            ["model", "status"],
        )
        self.askany_rerank_duration_seconds = Histogram(
            "askany_rerank_duration_seconds",
            "Rerank latency",
            ["model"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_rerank_input_nodes = Histogram(
            "askany_rerank_input_nodes",
            "Rerank input nodes",
            ["model"],
        )
        self.askany_rerank_output_nodes = Histogram(
            "askany_rerank_output_nodes",
            "Rerank output nodes",
            ["model"],
        )
        self.askany_rerank_fallback_total = Counter(
            "askany_rerank_fallback_total",
            "Rerank fallback count",
            ["reason"],
        )
        self.askany_relevance_judgment_total = Counter(
            "askany_relevance_judgment_total",
            "Relevance judgment count",
            ["result"],
        )

        # Workflow Metrics
        self.askany_workflow_execution_total = Counter(
            "askany_workflow_execution_total",
            "Workflow execution count",
            ["workflow_type", "status"],
        )
        self.askany_workflow_execution_duration_seconds = Histogram(
            "askany_workflow_execution_duration_seconds",
            "Workflow execution duration",
            ["workflow_type"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_workflow_node_execution_total = Counter(
            "askany_workflow_node_execution_total",
            "Workflow node execution count",
            ["node_name", "status"],
        )
        self.askany_workflow_node_duration_seconds = Histogram(
            "askany_workflow_node_duration_seconds",
            "Workflow node duration",
            ["node_name"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_relevance_iterations = Histogram(
            "askany_relevance_iterations",
            "Relevance iterations per query",
            ["query_type"],
        )
        self.askany_subproblem_generation_total = Counter(
            "askany_subproblem_generation_total",
            "Subproblem generation count",
            ["status"],
        )
        self.askany_subproblem_count = Histogram(
            "askany_subproblem_count",
            "Subproblem count per query",
            ["query_complexity"],
        )
        self.askany_lightrag_retrieval_total = Counter(
            "askany_lightrag_retrieval_total",
            "LightRAG retrieval count",
            ["mode", "status"],
        )
        self.askany_lightrag_retrieval_duration_seconds = Histogram(
            "askany_lightrag_retrieval_duration_seconds",
            "LightRAG retrieval duration",
            ["mode"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )

        # Web Search Metrics
        self.askany_websearch_requests_total = Counter(
            "askany_websearch_requests_total",
            "Web search requests",
            ["engine", "status"],
        )
        self.askany_websearch_duration_seconds = Histogram(
            "askany_websearch_duration_seconds",
            "Web search latency",
            ["engine"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_websearch_timeout_total = Counter(
            "askany_websearch_timeout_total",
            "Web search timeout count",
            ["engine"],
        )
        self.askany_websearch_results_count = Histogram(
            "askany_websearch_results_count",
            "Web search results count",
            ["engine"],
        )
        self.askany_websearch_tokens_total = Counter(
            "askany_websearch_tokens_total",
            "Web search token usage",
            ["token_type"],
        )
        self.askany_websearch_connection_error_total = Counter(
            "askany_websearch_connection_error_total",
            "Web search connection errors",
            ["engine"],
        )

        # Embedding Metrics
        self.askany_embedding_requests_total = Counter(
            "askany_embedding_requests_total",
            "Embedding requests",
            ["model", "status"],
        )
        self.askany_embedding_duration_seconds = Histogram(
            "askany_embedding_duration_seconds",
            "Embedding latency",
            ["model"],
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.askany_embedding_batch_size = Histogram(
            "askany_embedding_batch_size",
            "Embedding batch size",
            ["model"],
        )
        self.askany_embedding_tokens_total = Counter(
            "askany_embedding_tokens_total",
            "Embedding token usage",
            ["model"],
        )

        # Mem0 Metrics
        self.askany_mem0_search_total = Counter(
            "askany_mem0_search_total",
            "Mem0 search count",
            ["status"],
        )
        self.askany_mem0_search_duration_seconds = Histogram(
            "askany_mem0_search_duration_seconds",
            "Mem0 search latency",
        )
        self.askany_mem0_memories_retrieved = Histogram(
            "askany_mem0_memories_retrieved",
            "Memories retrieved per search",
        )
        self.askany_mem0_save_total = Counter(
            "askany_mem0_save_total",
            "Mem0 save count",
            ["status"],
        )
        self.askany_mem0_save_duration_seconds = Histogram(
            "askany_mem0_save_duration_seconds",
            "Mem0 save latency",
        )

        # Resource Metrics
        self.askany_background_tasks_pending = Gauge(
            "askany_background_tasks_pending",
            "Pending background tasks",
            ["task_type"],
        )
        self.askany_cache_hit_ratio = Gauge(
            "askany_cache_hit_ratio",
            "Cache hit ratio",
            ["cache_type"],
        )

    def get_histogram(self, name: str) -> HistogramBase | None:
        """Get a histogram by attribute name for external timing support."""
        return getattr(self, name, None)


# Global singleton instance
_registry: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Get the global MetricsRegistry singleton instance."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry

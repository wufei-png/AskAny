"""Unit tests for Prometheus metrics instrumentation."""


def test_metrics_registry_singleton():
    """Test that MetricsRegistry is a singleton."""
    from askany.metrics import get_metrics

    m1 = get_metrics()
    m2 = get_metrics()
    assert m1 is m2, "MetricsRegistry should be singleton"


def test_counter_increment():
    """Test counter increment."""
    from askany.metrics import get_metrics

    metrics = get_metrics()
    # Test that we can increment a counter
    counter = metrics.askany_llm_requests_total.labels(
        model="test_model", status="success", error_type="none"
    )
    initial = counter._value.get()
    counter.inc()
    assert counter._value.get() == initial + 1


def test_histogram_observe():
    """Test histogram observe records values to Prometheus registry."""
    from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry

    registry = CollectorRegistry()
    test_histogram = Histogram(
        "test_duration_seconds", "Test histogram", ["model"], registry=registry
    )
    child = test_histogram.labels(model="test")

    child.observe(0.5)
    child.observe(1.0)

    output = generate_latest(registry).decode("utf-8")
    assert "test_duration_seconds" in output
    assert 'model="test"' in output


def test_gauge_set():
    """Test gauge set."""
    from askany.metrics import get_metrics

    metrics = get_metrics()
    gauge = metrics.askany_health_status.labels(component="test")
    gauge.set(1)
    assert gauge._value.get() == 1
    gauge.set(0)
    assert gauge._value.get() == 0


def test_timing_context_manager():
    """Test Timer records elapsed time correctly when used with explicit enter/exit."""
    import time

    from askany.metrics import get_timer

    timer = get_timer(
        "askany_llm_request_duration_seconds", model="test", operation="chat"
    )
    timer.__enter__()
    time.sleep(0.05)
    elapsed_before_exit = timer.elapsed
    assert elapsed_before_exit is not None, (
        "Timer should track elapsed during execution"
    )
    assert elapsed_before_exit >= 0.04, (
        f"Elapsed {elapsed_before_exit:.3f}s should be >= 40ms"
    )

    timer.__exit__(None, None, None)
    elapsed_after_exit = timer.elapsed
    assert elapsed_after_exit is not None, "Timer should record elapsed after exit"
    assert elapsed_after_exit >= 0.04, (
        f"Final elapsed {elapsed_after_exit:.3f}s should be >= 40ms"
    )


def test_all_metrics_defined():
    """Test that all required metrics are defined."""
    from askany.metrics import get_metrics

    metrics = get_metrics()

    expected_metrics = [
        # API Layer
        "askany_http_requests_total",
        "askany_http_request_duration_seconds",
        "askany_http_streaming_duration_seconds",
        "askany_active_streams",
        "askany_workflow_selected_total",
        "askany_query_type_total",
        "askany_health_status",
        # LLM API
        "askany_llm_requests_total",
        "askany_llm_request_duration_seconds",
        "askany_llm_tokens_total",
        "askany_llm_timeout_total",
        "askany_llm_404_retries_total",
        "askany_llm_stream_chunks_total",
        "askany_llm_connection_errors_total",
        # Database
        "askany_db_connections_active",
        "askany_db_connection_errors_total",
        "askany_db_query_duration_seconds",
        "askany_vector_search_duration_seconds",
        "askany_keyword_search_duration_seconds",
        "askany_hnsw_index_operations_total",
        "askany_faq_update_total",
        # RAG
        "askany_rag_query_total",
        "askany_rag_query_duration_seconds",
        "askany_rag_retrieval_latency_seconds",
        "askany_rag_nodes_retrieved",
        "askany_rerank_requests_total",
        "askany_rerank_duration_seconds",
        "askany_rerank_fallback_total",
        "askany_relevance_judgment_total",
        # Workflow
        "askany_workflow_execution_total",
        "askany_workflow_execution_duration_seconds",
        "askany_workflow_node_execution_total",
        "askany_workflow_node_duration_seconds",
        # Web Search
        "askany_websearch_requests_total",
        "askany_websearch_duration_seconds",
        "askany_websearch_timeout_total",
        "askany_websearch_results_count",
        # Mem0
        "askany_mem0_search_total",
        "askany_mem0_save_total",
    ]

    for metric_name in expected_metrics:
        assert hasattr(metrics, metric_name), f"Missing metric: {metric_name}"


def test_metrics_endpoint():
    """Test that /metrics endpoint returns valid Prometheus format."""
    from fastapi import FastAPI
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.testclient import TestClient

    from askany.metrics import get_metrics

    # Create minimal app for testing
    app = FastAPI()

    class TestMetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path == "/metrics":
                return await call_next(request)

            metrics = get_metrics()
            metrics.askany_http_requests_total.labels(
                endpoint=request.url.path, method=request.method, status_code=200
            ).inc()
            return await call_next(request)

    app.add_middleware(TestMetricsMiddleware)

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    content = response.text
    # Should contain metric families
    assert (
        "askany_http_requests_total" in content
        or "askany_llm_requests_total" in content
    )


def test_http_middleware_records_metrics():
    """Test that HTTP middleware records request metrics."""
    from fastapi import FastAPI
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.testclient import TestClient

    from askany.metrics import get_metrics

    # Create minimal app for testing
    app = FastAPI()

    class TestMetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path == "/metrics":
                return await call_next(request)

            metrics = get_metrics()
            metrics.askany_http_requests_total.labels(
                endpoint=request.url.path, method=request.method, status_code=200
            ).inc()
            return await call_next(request)

    app.add_middleware(TestMetricsMiddleware)

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    client = TestClient(app)

    # Make a request to a non-metrics endpoint
    response = client.get("/health")
    assert response.status_code == 200

    # Check /metrics endpoint has recorded the request
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics_content = metrics_response.text

    # Should have recorded the health request
    assert "askany_http_requests_total" in metrics_content

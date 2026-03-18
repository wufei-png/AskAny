"""FastAPI server with OpenAI-compatible interface."""

import asyncio
import base64
import json
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel
from starlette.requests import Request

from askany.config import settings
from askany.ingest import VectorStoreManager
from askany.memory.mem0_adapter import get_mem0_adapter
from askany.rag import FAQQueryEngine
from askany.rag.router import QueryRouter, QueryType
from askany.workflow.workflow_filter import WorkflowFilter
from askany.workflow.workflow_langgraph import AgentWorkflow, process_parallel_group

logger = getLogger(__name__)

# Global device variable (set during initialization)
_device: str | None = None

_background_tasks: set[asyncio.Task[Any]] = set()  # prevent GC of fire-and-forget tasks


def set_device(device: str) -> None:
    """Set the global device for reranker models.

    Args:
        device: Device string ("cuda" or "cpu")
    """
    global _device
    _device = device


def get_device() -> str:
    """Get the global device, or auto-detect if not set.

    Returns:
        Device string ("cuda" or "cpu")
    """
    global _device
    if _device is not None:
        return _device

    # Auto-detect if not set and cache the result
    # This matches the logic in main.get_device() but avoids circular import
    try:
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"

    return _device


# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: list[ChatMessage]
    temperature: float = settings.temperature
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict[str, int]


class OpenAPISchema(BaseModel):
    """OpenAPI schema for open-webui compatibility."""

    openapi: str = "3.0.0"
    info: dict[str, Any]
    servers: list[dict[str, str]]
    paths: dict[str, Any]
    components: dict[str, Any]


# Global variables (will be initialized in startup)
router: QueryRouter | None = None
vector_store_manager: VectorStoreManager | None = None
llm = None
embed_model = None
agent_workflow_global: AgentWorkflow | None = None
simple_agent_global = None  # Simple agent (min_langchain_agent) instance
update_lock = threading.Lock()  # Lock for thread-safe updates


class UpdateFAQsRequest(BaseModel):
    """Request model for updating FAQs."""

    json_base64: str  # Base64 encoded JSON content


class UpdateFAQsResponse(BaseModel):
    """Response model for updating FAQs."""

    success: bool
    message: str
    inserted: int  # Number of new FAQs inserted
    updated: int  # Number of existing FAQs updated
    errors: list[str]  # List of error messages


def _make_sse_chunk(
    model: str,
    content: str | None = None,
    finish_reason: str | None = None,
    chunk_id: str | None = None,
) -> str:
    """Build a single SSE data line in OpenAI chat.completion.chunk format.

    Args:
        model: Model name echoed back to the client.
        content: Token text for ``delta.content`` (omit for empty delta).
        finish_reason: If set, signals the end of this choice.
        chunk_id: Optional chunk id; auto-generated if not provided.

    Returns:
        A complete SSE line ``data: {json}\n\n``.
    """
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    chunk = {
        "id": chunk_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _sse_generator(
    model: str,
    stream: AsyncGenerator[str, None],
    *,
    mem0_adapter=None,
    user_id: str | None = None,
    user_query: str | None = None,
) -> AsyncGenerator[str, None]:
    """Wrap an async content stream into OpenAI-compatible SSE events.

    Yields ``data: {json}\n\n`` lines for each content chunk, a final
    ``finish_reason='stop'`` chunk, and the terminating ``data: [DONE]\n\n``.

    Args:
        model: Model name to echo in each chunk.
        stream: Async generator yielding content strings.
        mem0_adapter: Optional Mem0Adapter for saving conversation to memory.
        user_id: Optional user ID for mem0.
        user_query: Optional user query for mem0.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    first = True
    collected_content: list[str] = []
    try:
        async for content in stream:
            if not content:
                continue
            collected_content.append(content)
            if first:
                role_chunk: dict[str, Any] = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"
                first = False
            yield _make_sse_chunk(model, content=content, chunk_id=chunk_id)
    except Exception as e:
        logger.exception("Error during SSE streaming")
        yield _make_sse_chunk(
            model,
            content=f"[Error: {type(e).__name__}] {str(e)}",
            chunk_id=chunk_id,
        )
    finally:
        yield _make_sse_chunk(model, finish_reason="stop", chunk_id=chunk_id)
        yield "data: [DONE]\n\n"
        if mem0_adapter and user_id and user_query and collected_content:
            response_text = "".join(collected_content)
            asyncio.create_task(
                mem0_adapter.save_turn_async(user_query, response_text, user_id)
            )


def create_app(
    query_router: QueryRouter,
    vstore_manager: VectorStoreManager | None = None,
    _llm=None,
    _embed_model=None,
    agent_workflow: AgentWorkflow | None = None,
    workflow_filter: WorkflowFilter | None = None,
    simple_agent=None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        query_router: Initialized query router
        vstore_manager: Vector store manager for hot updates
        _llm: LLM instance for recreating query engines
        _embed_model: Embedding model instance
        agent_workflow: AgentWorkflow instance for running workflows (deep search)
        workflow_filter: WorkflowFilter instance for filtering queries
        simple_agent: Simple agent instance (min_langchain_agent) for fast queries
    Returns:
        FastAPI application instance
    """
    global \
        router, \
        vector_store_manager, \
        llm, \
        embed_model, \
        agent_workflow_global, \
        workflow_filter_global, \
        simple_agent_global
    router = query_router
    vector_store_manager = vstore_manager
    llm = _llm
    embed_model = _embed_model
    agent_workflow_global = agent_workflow
    workflow_filter_global = workflow_filter
    simple_agent_global = simple_agent

    # ── Lifespan: shutdown hook for Langfuse and RAGAS flush ───────────────────
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield
        # Shutdown: flush Langfuse and RAGAS events
        try:
            from askany.observability import shutdown_langfuse, shutdown_ragas

            shutdown_langfuse()
            shutdown_ragas()
        except ImportError:
            pass
        except Exception:
            logger.exception("Error during observability shutdown")

    app = FastAPI(
        title="AskAny API",
        description="OpenAI-compatible API for AskAny RAG system",
        version="0.1.0",
        lifespan=_lifespan,
    )
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request size limit middleware
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB

    @app.middleware("http")
    async def limit_request_size(request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large. Maximum size is 10MB."},
            )
        return await call_next(request)

    # Note: We use AgentWorkflow (LangGraph) directly instead of WorkflowServer/WorkflowClient
    # AgentWorkflow runs in-process and doesn't require a separate HTTP server

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        health_status = {
            "status": "ok",
            "workflow": "unknown",
            "simple_agent": "unknown",
        }

        # Check AgentWorkflow availability (for deep search)
        if agent_workflow_global is not None:
            health_status["workflow"] = "ready"
        else:
            health_status["workflow"] = "unavailable"
            health_status["status"] = "degraded"

        # Check simple agent availability (for fast queries)
        if simple_agent_global is not None:
            health_status["simple_agent"] = "ready"
        else:
            health_status["simple_agent"] = "unavailable"
            if health_status["status"] == "ok":
                health_status["status"] = "degraded"

        status_code = 200 if health_status["status"] == "ok" else 503
        return Response(
            content=json.dumps(health_status),
            media_type="application/json",
            status_code=status_code,
        )

    def _fallback_models_response() -> JSONResponse:
        """Return /v1/models response using configured openai_model when upstream fails."""
        model_id = settings.openai_model
        models_data = {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model"},
                {"id": f"{model_id}-deepsearch", "object": "model"},
            ],
        }
        return JSONResponse(content=models_data)

    @app.get("/v1/models")
    async def list_models():
        """List available models by forwarding to the configured OpenAI API base.

        Additionally, creates -deepsearch variants for each model to enable
        deep search workflow selection. On upstream error or missing config,
        returns the configured openai_model (and its -deepsearch variant).
        """
        if not settings.openai_api_base:
            logger.warning(
                "OpenAI API base not configured, returning configured openai_model"
            )
            return _fallback_models_response()

        target_url = f"{settings.openai_api_base}/models"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(target_url)
                response.raise_for_status()
                models_data = response.json()
                original_models = models_data.get("data") or []
                if not original_models:
                    logger.warning(
                        "Upstream /models returned no data, using openai_model fallback"
                    )
                    return _fallback_models_response()

                # Add -deepsearch variants for each model
                deepsearch_models = []
                for model in original_models:
                    model_id = model.get("id", "")
                    if model_id and not model_id.endswith("-deepsearch"):
                        deepsearch_model = model.copy()
                        deepsearch_model["id"] = f"{model_id}-deepsearch"
                        deepsearch_models.append(deepsearch_model)

                models_data["data"] = original_models + deepsearch_models
                return JSONResponse(content=models_data)
        except httpx.HTTPError as e:
            logger.warning(
                f"Upstream /models failed ({target_url}): {e}, using openai_model fallback"
            )
            return _fallback_models_response()

    @app.get("/openapi.json")
    async def openapi_json():
        """OpenAPI schema endpoint for open-webui compatibility."""

        schema = {
            "openapi": "3.0.0",
            "info": {
                "title": "AskAny API",
                "version": "0.1.0",
                "description": "OpenAI-compatible API for AskAny RAG system",
            },
            "servers": [
                {
                    "url": f"http://{settings.api_host}:{settings.api_port}",
                    "description": "AskAny API Server",
                }
            ],
            "paths": {
                "/v1/chat/completions": {
                    "post": {
                        "summary": "Create a chat completion",
                        "operationId": "createChatCompletion",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model": {"type": "string"},
                                            "messages": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "role": {"type": "string"},
                                                        "content": {"type": "string"},
                                                    },
                                                },
                                            },
                                            "temperature": {"type": "number"},
                                            "max_tokens": {"type": "integer"},
                                            "stream": {"type": "boolean"},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Chat completion response",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "components": {
                "schemas": {},
            },
        }

        return JSONResponse(content=schema)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        """Chat completions endpoint (OpenAI-compatible).

        Selects workflow based on model name suffix:
        - Models ending with '-deepsearch' use AgentWorkflow (complex workflow)
        - Other models use simple_agent (min_langchain_agent)
        """
        # Determine which workflow to use based on model name suffix
        use_deepsearch = request.model.endswith("-deepsearch")
        # Extract user message
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        user_query = user_messages[-1].content
        logger.info(
            "Chat request model=%s (use_deepsearch=%s) query=%s",
            request.model,
            use_deepsearch,
            user_query,
        )

        # ── Mem0: extract user identity & retrieve memories ──
        user_id: str | None = raw_request.headers.get("x-openwebui-user-id")
        mem0_adapter = get_mem0_adapter()
        memory_system_text: str | None = None
        if user_id and mem0_adapter:
            memories = mem0_adapter.search(user_query, user_id)
            if memories:
                memory_system_text = mem0_adapter.format_memories_as_system_text(
                    memories
                )
                logger.debug(
                    "Mem0 injecting %d memories for user_id=%s",
                    len(memories),
                    user_id,
                )

        # ── Streaming path: return SSE StreamingResponse early ──
        if request.stream:
            # Pre-validate required globals before creating the StreamingResponse.
            # HTTPException raised inside an async generator won't be caught by
            # FastAPI's exception middleware (headers are already sent), so we
            # validate eagerly here while we can still return a proper error response.
            if use_deepsearch and len(user_messages) == 1:
                if agent_workflow_global is None:
                    raise HTTPException(
                        status_code=500, detail="AgentWorkflow not initialized"
                    )
            else:
                if simple_agent_global is None:
                    raise HTTPException(
                        status_code=500, detail="Simple agent not initialized"
                    )

            from askany.workflow.min_langchain_agent import astream_agent_response

            async def _build_content_stream() -> AsyncGenerator[str, None]:
                """Build the content-level async generator for the chosen workflow."""
                # Prepare mem0_qa_context for workflow path
                mem0_qa_context: list[dict[str, str]] = []
                if memory_system_text:
                    mem0_qa_context = [
                        {"query": "user_memory_context", "answer": memory_system_text}
                    ]

                if use_deepsearch and len(user_messages) == 1:
                    # Deepsearch single-query streaming
                    query_type = QueryType.AUTO
                    system_messages = [
                        msg for msg in request.messages if msg.role == "system"
                    ]
                    if system_messages:
                        system_content = system_messages[-1].content.lower()
                        if "faq" in system_content:
                            query_type = QueryType.FAQ
                        elif "docs" in system_content:
                            query_type = QueryType.DOCS
                        elif "code" in system_content:
                            query_type = QueryType.CODE
                    # Pass mem0_qa_context to workflow
                    async for chunk in agent_workflow_global.astream_final_answer(
                        user_query, query_type, mem0_qa_context=mem0_qa_context
                    ):
                        yield chunk
                else:
                    # Simple agent streaming
                    mem0_messages: list[dict[str, str]] = []
                    if memory_system_text:
                        mem0_messages = [
                            {"role": "system", "content": memory_system_text}
                        ]
                    messages_input = {
                        "messages": mem0_messages
                        + [
                            {"role": msg.role, "content": msg.content}
                            for msg in request.messages
                        ]
                    }
                    async for chunk in astream_agent_response(
                        simple_agent_global, messages_input
                    ):
                        yield chunk

            return StreamingResponse(
                _sse_generator(
                    request.model,
                    _build_content_stream(),
                    mem0_adapter=mem0_adapter if user_id else None,
                    user_id=user_id,
                    user_query=user_query,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        if use_deepsearch and len(user_messages) == 1:  # 二次问答直接给simple agent
            # Use complex workflow (AgentWorkflow)
            if agent_workflow_global is None:
                raise HTTPException(
                    status_code=500, detail="AgentWorkflow not initialized"
                )

            # Get query type from system message or default to AUTO
            query_type = QueryType.AUTO
            system_messages = [msg for msg in request.messages if msg.role == "system"]
            if system_messages:
                system_content = system_messages[
                    -1
                ].content.lower()  # 取最后一个system message作为query type.
                if "faq" in system_content:
                    query_type = QueryType.FAQ
                elif "docs" in system_content:
                    query_type = QueryType.DOCS
                elif "code" in system_content:
                    query_type = QueryType.CODE

            # Use process_query_with_subproblems to handle sub-problem decomposition
            # This function handles:
            # - Single question: direct workflow execution
            # - Multiple unrelated questions: parallel processing
            # - Multiple related questions: serial processing with context accumulation
            # Inject Mem0 memory context into workflow as previous QA context
            mem0_qa_context: list[dict[str, str]] = []
            if memory_system_text:
                mem0_qa_context = [
                    {"query": "user_memory_context", "answer": memory_system_text}
                ]

            retrieved_nodes: list[NodeWithScore] = []
            retrieved_contexts: list[str] = []

            try:
                response_text, retrieved_nodes = await process_query_with_subproblems(
                    agent_workflow_global,
                    workflow_filter_global,
                    user_query,
                    query_type,
                    mem0_qa_context=mem0_qa_context,
                )
                retrieved_contexts = [
                    node.node.get_content()
                    if hasattr(node.node, "get_content")
                    else str(node.node)
                    for node in retrieved_nodes
                ]
            except Exception as e:
                # Handle workflow errors gracefully
                error_msg = str(e)
                logger.error(f"Workflow error: {error_msg}", exc_info=True)

                # For workflow errors, return 500 with error details
                raise HTTPException(
                    status_code=500, detail=f"Workflow execution failed: {error_msg}"
                ) from e
        else:
            # Use simple agent (min_langchain_agent)
            if simple_agent_global is None:
                raise HTTPException(
                    status_code=500, detail="Simple agent not initialized"
                )

            try:
                # Convert messages to format expected by simple agent
                # Simple agent expects messages in format: [{"role": "user", "content": "..."}]
                # Inject Mem0 memory context as a system message at the front
                mem0_messages = []
                if memory_system_text:
                    mem0_messages = [{"role": "system", "content": memory_system_text}]

                messages_input = {
                    "messages": mem0_messages
                    + [
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ]
                }

                # Invoke simple agent
                from askany.workflow.min_langchain_agent import (
                    extract_and_format_response,
                    extract_references_from_result,
                    invoke_with_retry,
                )

                result = invoke_with_retry(simple_agent_global, messages_input)
                response_text = extract_and_format_response(result)
                retrieved_contexts = extract_references_from_result(result)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Simple agent error: {error_msg}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Simple agent execution failed: {error_msg}",
                ) from e
        logger.info(
            "Chat response model=%s characters=%d\n%s",
            request.model,
            len(response_text),
            response_text,
        )

        # Build response

        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": len(user_query.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_query.split()) + len(response_text.split()),
            },
        )

        # ── Mem0: fire-and-forget save ──
        if user_id and mem0_adapter:
            task = asyncio.create_task(
                mem0_adapter.save_turn_async(user_query, response_text, user_id)
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        # ── RAGAS: fire-and-forget evaluation ──
        try:
            from askany.observability import evaluate_rag_response
            from askany.observability.langfuse_setup import (
                get_langfuse_callback_handler,
            )

            _lf_handler = get_langfuse_callback_handler()
            _trace_id = None
            if _lf_handler is not None:
                try:
                    _trace_id = _lf_handler.get_trace_id()
                    if _trace_id:
                        logger.debug(
                            "Retrieved trace_id from Langfuse handler: %s", _trace_id
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to get trace_id from Langfuse handler: %s", str(e)
                    )

            # Convert retrieved nodes to text for RAGAS evaluation
            ragas_task = asyncio.create_task(
                evaluate_rag_response(
                    trace_id=_trace_id,
                    user_input=user_query,
                    response=response_text,
                    retrieved_contexts=retrieved_contexts,
                )
            )
            _background_tasks.add(ragas_task)
            ragas_task.add_done_callback(_background_tasks.discard)
        except ImportError:
            pass  # observability package not installed
        except Exception:
            logger.debug("RAGAS evaluation task creation failed", exc_info=True)

        return response

    @app.post("/v1/update_faqs", response_model=UpdateFAQsResponse)
    async def update_faqs(request: UpdateFAQsRequest):
        """Update FAQs with base64 encoded JSON content.

        This endpoint supports hot updates:
        - If question doesn't exist, insert it
        - If question exists, update it
        - Server continues running without restart
        """
        if vector_store_manager is None:
            raise HTTPException(
                status_code=500, detail="Vector store manager not initialized"
            )

        if llm is None:
            raise HTTPException(status_code=500, detail="LLM not initialized")

        # Use lock to ensure thread-safe updates
        with update_lock:
            try:
                # Decode base64 JSON content
                try:
                    json_bytes = base64.b64decode(request.json_base64)
                    json_str = json_bytes.decode("utf-8")
                    faq_data = json.loads(json_str)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to decode base64 JSON: {str(e)}",
                    ) from e

                # Ensure faq_data is a list
                if isinstance(faq_data, dict):
                    faq_data = [faq_data]
                elif not isinstance(faq_data, list):
                    raise HTTPException(
                        status_code=400, detail="JSON must be a list or object"
                    )

                # Use vector_store_manager to update FAQs
                json_dir = Path(settings.json_dir)
                result = vector_store_manager.update_faqs(faq_data, json_dir=json_dir)

                # Update router's FAQ query engine
                # Need to recreate the entire FAQQueryEngine because it creates
                # retrievers and query_engine internally based on the indexes during __init__
                # Simply updating index attributes won't update the internal retrievers
                faq_vector_index = vector_store_manager.get_faq_index()
                faq_keyword_index = vector_store_manager.get_faq_keyword_index()

                if faq_vector_index and faq_keyword_index:
                    # Use global device (already detected during initialization)
                    device = get_device()

                    # Recreate FAQ query engine with updated indexes
                    router.faq_query_engine = FAQQueryEngine(
                        vector_index=faq_vector_index,
                        keyword_index=faq_keyword_index,
                        llm=llm,
                        similarity_top_k=settings.faq_similarity_top_k,
                        ensemble_weights=settings.faq_ensemble_weights,
                        device=device,
                    )

                return UpdateFAQsResponse(
                    success=True,
                    message=f"Successfully processed {result['total_processed']} FAQ(s)",
                    inserted=result["inserted_count"],
                    updated=result["updated_count"],
                    errors=result["errors"] if result["errors"] else [],
                )

            except HTTPException:
                raise
            except Exception as e:
                return UpdateFAQsResponse(
                    success=False,
                    message=f"Failed to update FAQs: {str(e)}",
                    inserted=0,
                    updated=0,
                    errors=[str(e)],
                )

    return app


async def process_query_with_subproblems(
    agent_workflow: AgentWorkflow,
    workflow_filter: WorkflowFilter,
    user_query: str,
    query_type: QueryType,
    *,
    mem0_qa_context: list[dict[str, str]] | None = None,
) -> tuple[str, list["NodeWithScore"]]:
    """处理用户查询，支持子问题分解和并行/串行处理。

    首先使用 WorkflowFilter 尝试直接回答或通过网络搜索回答。
    如果 WorkflowFilter 返回 have_result=False，则进行子问题分解。

    根据 SubProblemGenerator 的结果：
    - 如果只有一个问题，直接进行后续workflow
    - 如果有多个问题且问题不相关，并行处理每个问题
    - 如果有多个问题且相关，串行处理，将上一个问题的答案附加到下一个问题

    Args:
        agent_workflow: AgentWorkflow 实例
        user_query: 用户查询字符串
        query_type: 查询类型
        mem0_qa_context: Mem0 memory context as QA pairs (optional)

    Returns:
        Tuple of (处理结果字符串, 检索到的节点列表)
    """
    import asyncio

    _mem0_ctx = mem0_qa_context or []

    # TODO 二轮问答这里跳过filter处理.
    filter_result = workflow_filter.process(user_query)
    if filter_result.have_result:
        logger.debug("工作流过滤器成功生成答案，直接返回")
        result_text = filter_result.result or ""
        return result_text, []

    # 如果 WorkflowFilter 返回 have_result=False，继续执行子问题提取
    logger.debug("工作流过滤器未生成答案，进行子问题提取")
    try:
        sub_problem_structure = agent_workflow.sub_problem_generator.generate(
            user_query
        )
        logger.debug(
            "子问题分解完成 - 并行组数: %d",
            len(sub_problem_structure.parallel_groups),
        )
    except Exception as e:
        logger.error("子问题分解失败: %s", str(e), exc_info=True)
        # 如果分解失败，直接处理原始查询
        logger.debug("子问题分解失败，直接处理原始查询")
        # 直接调用 AgentWorkflow 的异步方法
        # 使用 workflow_filter 返回的 need_web_search 和 need_rag_search 作为缓存结果
        initial_state = {
            "query": user_query,
            "query_type": query_type,
            "can_direct_answer": False,
            "need_web_search": filter_result.need_web_search,
            "need_rag_search": filter_result.need_rag_search,
            "web_or_rag_result_cached": True,  # 标记结果已缓存
            "nodes": [],
            "keywords": [],
            "analysis": None,
            "iteration": 0,
            "no_relevant_result": None,
            "current_sub_query": None,
            "inner_previous_qa_context": [],
            "outer_previous_qa_context": _mem0_ctx,
            "is_inner_sub_query_workflow": False,
            "is_outer_sub_query_workflow": False,
            "result": None,
        }
        result = await agent_workflow.graph.ainvoke(initial_state)
        answer = result.get("result", "抱歉，无法生成答案。")
        nodes = result.get("nodes", [])
        return answer, nodes

    # 如果只有一个问题组，且该组只有一个问题，直接处理
    if (
        len(sub_problem_structure.parallel_groups) == 1
        and len(sub_problem_structure.parallel_groups[0]) == 1
    ):
        logger.debug("只有一个问题，直接处理")
        # 直接调用 AgentWorkflow 的异步方法
        # 使用 workflow_filter 返回的 need_web_search 和 need_rag_search 作为缓存结果
        initial_state = {
            "query": sub_problem_structure.parallel_groups[0][0],
            "query_type": query_type,
            "can_direct_answer": False,
            "need_web_search": filter_result.need_web_search,
            "need_rag_search": filter_result.need_rag_search,
            "web_or_rag_result_cached": True,  # 标记结果已缓存
            "nodes": [],
            "keywords": [],
            "analysis": None,
            "iteration": 0,
            "no_relevant_result": None,
            "current_sub_query": None,
            "inner_previous_qa_context": [],
            "outer_previous_qa_context": _mem0_ctx,
            "is_inner_sub_query_workflow": False,
            "is_outer_sub_query_workflow": False,
            "result": None,
        }
        result = await agent_workflow.graph.ainvoke(initial_state)
        answer = result.get("result", "抱歉，无法生成答案。")
        nodes = result.get("nodes", [])
        return answer, nodes

    # 处理所有并行组
    if len(sub_problem_structure.parallel_groups) == 1:
        # 只有一个并行组，直接处理（可能是单个问题或多个相关问题）
        logger.debug("只有一个并行组，直接处理")
        result, nodes = await process_parallel_group(
            agent_workflow,
            sub_problem_structure.parallel_groups[0],
            query_type,
            mem0_qa_context=_mem0_ctx,
        )
        return result, nodes
    else:
        # 多个并行组，并行处理每个组
        logger.debug(
            "有 %d 个并行组，并行处理", len(sub_problem_structure.parallel_groups)
        )
        # 创建并行任务
        tasks = [
            process_parallel_group(
                agent_workflow, group, query_type, mem0_qa_context=_mem0_ctx
            )
            for group in sub_problem_structure.parallel_groups
        ]
        # 等待所有并行任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 组合所有结果
        all_results = []
        all_nodes = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("并行组 %d 处理失败: %s", idx + 1, str(result))
                all_results.append(
                    f"问题组 {idx + 1}: 抱歉，处理问题时发生错误: {str(result)}"
                )
            else:
                result_text, result_nodes = result
                all_results.append(result_text)
                if result_nodes:
                    all_nodes.extend(result_nodes)

        # 组合所有结果
        final_answer = "\n\n".join(all_results)
        logger.debug("所有并行组处理完成，生成最终答案")
        return final_answer, all_nodes


# Create default app instance (will be initialized with router)
app = FastAPI()


def run_server(
    query_router: QueryRouter,
    vector_store_manager: VectorStoreManager | None = None,
    llm=None,
    embed_model=None,
    agent_workflow: AgentWorkflow | None = None,
    workflow_filter: WorkflowFilter | None = None,
    simple_agent=None,
):
    """Run the API server.

    Args:
        query_router: Initialized query router
        vector_store_manager: Vector store manager for hot updates
        llm: LLM instance for recreating query engines
        embed_model: Embedding model instance
        agent_workflow: AgentWorkflow instance for running workflows (deep search)
        workflow_filter: WorkflowFilter instance for filtering queries
        simple_agent: Simple agent instance (min_langchain_agent) for fast queries
    """
    global app
    app = create_app(
        query_router,
        vector_store_manager,
        llm,
        embed_model,
        agent_workflow,
        workflow_filter,
        simple_agent,
    )

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )

# Langfuse & RAGAS Improvements

## Current Status

The AskAny project has a well-structured Langfuse and RAGAS integration:

### Implemented Features

1. **Langfuse Tracing** (`askany/observability/langfuse_setup.py`)
   - Environment variable propagation for LightRAG
   - Shared Langfuse client for score pushing
   - LangChain CallbackHandler for OpenAI calls
   - LlamaIndex OpenTelemetry instrumentation

2. **RAGAS Evaluation** (`askany/observability/ragas_eval.py`)
   - Async, non-blocking evaluation
   - Support for multiple metrics (faithfulness, response_relevancy, context_precision)
   - Sampling rate control
   - Automatic score push to Langfuse

3. **LightRAG Integration** (`askany/rag/lightrag_adapter.py`)
   - Langfuse attribute propagation
   - Flush/shutdown helpers

4. **Configuration** (`askany/config.py`)
   - All settings configurable via environment variables
   - Graceful fallback when disabled

## Issues Found & Fixed

### Fixed

1. **Missing `shutdown_ragas()` call** (server.py)
   - Issue: Only `shutdown_langfuse()` was called during app shutdown
   - Fix: Added `shutdown_ragas()` to lifespan handler
   - Commit: `73e21c2`

### Known Limitations

1. **`retrieved_contexts=[]` hardcoded** (server.py:731)
   - **Impact**: RAGAS cannot calculate context-dependent metrics (faithfulness, context_precision)
   - **Current behavior**: Only `response_relevancy` metric works (doesn't require contexts)
   - **Root cause**: The workflow functions (`process_query_with_subproblems`, `invoke_with_retry`) don't return retrieved contexts
   - **Fix required**: Modify workflow to return contexts along with response

2. **Trace ID retrieval may fail silently** (server.py:718-720)
   - **Impact**: RAGAS scores may not be associated with Langfuse traces
   - **Current behavior**: Silently falls back to None
   - **Fix required**: Improve trace_id retrieval reliability

## Recommended Improvements

### High Priority

1. **Plumb retrieved contexts to RAGAS**
   - Modify `process_query_with_subproblems` to return contexts
   - Modify `simple_agent` path to return contexts
   - Pass contexts to `evaluate_rag_response`

2. **Improve trace_id retrieval**
   - Add proper error handling/logging
   - Consider alternative approaches (thread-local storage, context variables)

### Medium Priority

3. **Add LangChain agent callback injection**
   - Currently, LangChain agents may not use the callback handler
   - Verify `get_langfuse_callback_handler()` is properly passed to ChatOpenAI instances

4. **Add end-to-end tests**
   - Test with mock Langfuse server
   - Test RAGAS evaluation flow
   - Test trace ID association

5. **Add metrics dashboard**
   - Track RAGAS scores over time
   - Alert on degraded quality

### Low Priority

6. **Support additional RAGAS metrics**
   - `context_recall` (requires reference answer)
   - `answer_similarity` (requires reference answer)

7. **Add custom Langfuse metrics**
   - Latency per stage
   - Token usage
   - Retrieval precision

## Testing

Current test coverage:
- Unit tests for Langfuse setup: ✅ 60 tests passing
- Unit tests for RAGAS: ✅ Included in above
- Integration tests: ❌ Not implemented
- E2E tests: ❌ Not implemented

## Usage

To enable Langfuse tracing:

```bash
# Set environment variables
export ENABLE_LANGFUSE=true
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or self-hosted

# For RAGAS evaluation (optional)
export ENABLE_RAGAS=true
export RAGAS_SAMPLE_RATE=0.1  # evaluate 10% of requests
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        AskAny API                            │
│                    (askany/api/server.py)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Observability Layer                            │
│           (askany/observability/)                           │
│  ┌─────────────────┐    ┌─────────────────────────────┐   │
│  │ langfuse_setup  │    │       ragas_eval            │   │
│  │ - initialize    │    │ - initialize_ragas          │   │
│  │ - shutdown      │    │ - evaluate_rag_response     │   │
│  │ - get_handler   │    │ - shutdown_ragas            │   │
│  └─────────────────┘    └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
           │                                   │
           ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    Langfuse Cloud                            │
│  - Traces (LLM calls, retrieval)                            │
│  - Scores (RAGAS metrics)                                   │
└─────────────────────────────────────────────────────────────┘
```

## Conclusion

The Langfuse integration is well-designed and follows best practices. The main limitation is the lack of retrieved contexts for RAGAS evaluation. With the suggested improvements, the observability stack can provide valuable insights into RAG quality.

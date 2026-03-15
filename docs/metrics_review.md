# Metrics Observability - Review & Improvement Suggestions

## Current Implementation Overview

The metrics functionality in AskAny consists of:
1. **Langfuse** (`langfuse_setup.py`) - Tracing and observability
2. **RAGAS** (`ragas_eval.py`) - RAG evaluation metrics

### Components
- `askany/observability/langfuse_setup.py` - Langfuse initialization/shutdown
- `askany/observability/ragas_eval.py` - RAGAS metrics evaluation
- `askany/observability/__init__.py` - Package exports
- `test/test_observability.py` - Comprehensive test suite (59 tests)

---

## Bug Fixes Applied

### 1. shutdown_langfuse() not resetting singletons (FIXED)
**Issue**: After calling `shutdown_langfuse()`, the module-level singletons 
(`_langfuse_client`, `_langfuse_callback_handler`, `_llamaindex_instrumentor`) 
were not reset to `None`. This caused stale object references to be returned by 
getter functions.

**Fix**: Added reset logic in `shutdown_langfuse()` to set all singletons to `None`.

### 2. langfuse_setup overwriting existing env vars (FIXED)
**Issue**: `initialize_langfuse()` was unconditionally overwriting `LANGFUSE_*` 
environment variables, ignoring any pre-existing values set by the user.

**Fix**: Added checks to only set env vars if not already present:
```python
if "LANGFUSE_PUBLIC_KEY" not in os.environ:
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
```

### 3. Missing shutdown_ragas() function (FIXED)
**Issue**: No cleanup function existed for RAGAS module-level state.

**Fix**: Added `shutdown_ragas()` function in `ragas_eval.py` and exported it in `__init__.py`.

---

## Identified Issues & Improvements

### 1. Empty retrieved_contexts in server.py (KNOWN LIMITATION)

**Location**: `askany/api/server.py:706`

**Issue**: The RAGAS evaluation is called with `retrieved_contexts=[]` (hardcoded empty):
```python
ragas_task = asyncio.create_task(
    evaluate_rag_response(
        trace_id=_trace_id,
        user_query=user_query,
        response=response_text,
        retrieved_contexts=[],  # <-- Always empty!
    )
)
```

**Impact**: 
- `faithfulness` metric requires retrieved contexts to evaluate if the response is grounded in the context
- `context_precision` metric also requires contexts
- Only `response_relevancy` works (doesn't need contexts)

**Suggestion**: Plumb retrieved_contexts from the workflow/agent response. This requires:
1. Modifying the workflow to return retrieved contexts
2. Passing contexts through to the RAGAS evaluation call

---

### 2. Missing RAGAS Shutdown Function (RESOLVED)

**Issue**: There's no `shutdown_ragas()` function to clean up RAGAS resources.

**Status**: ✅ RESOLVED - Added `shutdown_ragas()` function in `ragas_eval.py`

---

### 3. No End-to-End Integration Tests

**Current State**: All tests are unit tests with mocks.

**Suggestion**: Add integration tests that:
1. Start the server with Langfuse/RAGAS enabled
2. Make a request to `/v1/chat/completions`
3. Verify metrics are recorded in Langfuse

---

### 4. Metrics Configuration Could Be More Flexible

**Current**: Metrics are configured via `ragas_metrics` in settings.

**Suggestion**: Consider supporting custom metrics beyond the built-in RAGAS metrics.

---

### 5. Error Handling in RAGAS Evaluation

**Current**: All errors are caught silently in `evaluate_rag_response()`.

**Suggestion**: Consider adding optional error logging or metrics for monitoring evaluation failures.

---

### 6. Sample Rate Applies Per-Request, Not Per-Metric

**Current**: `_sample_rate` applies to the entire evaluation.

**Suggestion**: Consider allowing per-metric sampling rates for more granular control.

---

## Summary

### What's Working Well
- Comprehensive unit test coverage (59 tests)
- Graceful handling of missing dependencies
- Fire-and-forget evaluation pattern doesn't block API responses
- Proper initialization/idempotency checks
- Langfuse score pushing works correctly

### Immediate Actions Recommended
1. ✅ Bug fix: Reset singletons in shutdown (COMPLETED)
2. ✅ Bug fix: Preserve existing env vars in langfuse_setup (COMPLETED)
3. ✅ Bug fix: Add shutdown_ragas() function (COMPLETED)
4. 🔲 Plumb retrieved_contexts from workflow (requires significant changes)
5. 🔲 Add integration tests (medium priority)

---

*Generated: 2026-03-16*
*Updated: 2026-03-16*
*Review by: Sisyphus Agent*

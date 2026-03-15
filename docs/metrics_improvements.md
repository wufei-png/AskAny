# Metrics/Observability Improvement Suggestions

## Current Implementation Overview

The project has Langfuse tracing and RAGAS evaluation metrics in `askany/observability/`:

1. **langfuse_setup.py** - Langfuse client and callback handler initialization
2. **ragas_eval.py** - RAGAS metrics evaluation (faithfulness, response_relevancy, context_precision)

## Known Issues

### 1. Empty retrieved_contexts (Critical)
**Location**: `askany/api/server.py` line 706
**Problem**: `retrieved_contexts=[]` is hardcoded, meaning RAGAS cannot compute context-dependent metrics
**Impact**: Only `response_relevancy` works; `faithfulness` and `context_precision` always fail
**Status**: Acknowledged with TODO comment

### 2. Workflow Doesn't Return Contexts
**Location**: `askany/workflow/workflow_langgraph.py` and `askany/workflow/min_langchain_agent.py`
**Problem**: Workflow functions only return `response_text`, not retrieved contexts
**Impact**: Cannot pass actual contexts to RAGAS without significant refactoring

## Improvement Suggestions

### Priority 1: Fix retrieved_contexts (High Impact)

**Option A: Modify workflow to return contexts**
- Update `process_query_with_subproblems` to return `(response_text, retrieved_contexts)`
- Update server.py to unpack and pass contexts to RAGAS
- Pros: Complete solution
- Cons: Requires changes to multiple workflow files

**Option B: Add context capture in RAG engine**
- Capture contexts during retrieval in FAQQueryEngine/RAGQueryEngine
- Store in a thread-local or request-scoped context
- Pros: Less invasive
- Cons: Need thread-safety considerations

### Priority 2: Add Metrics Dashboard (Medium Impact)

**Suggestion**: Expose RAGAS metrics via API endpoint
```python
@app.get("/v1/metrics")
async def get_metrics_summary():
    """Return aggregated RAGAS metrics."""
    # Return cached/aggregated metrics
```

### Priority 3: Improve Error Handling (Low Impact)

**Current**: Errors are silently caught with try/except
**Suggestion**: Add error logging with specific error types to help debugging

### Priority 4: Add More RAGAS Metrics (Low Impact)

**Currently supported**:
- faithfulness
- response_relevancy
- context_precision

**Could add** (requires reference):
- answer_similarity
- answer_correctness

### Priority 5: Sampling Configuration

**Current**: `ragas_sample_rate` controls sampling
**Suggestion**: Add per-metric sampling rates for more granular control

## Usage Analysis

### Current Flow
```
User Query → Server → Workflow → RAG Engine → Response
                              ↓
                         RAGAS Eval (empty contexts!)
                              ↓
                         Langfuse (partial scores)
```

### Issues with Current Usage

1. **No actual evaluation happens** - Contexts are empty
2. **No user feedback** - Users don't know metrics aren't working
3. **No fallback** - When Langfuse is down, no way to know

### Recommendations

1. Add warning log when contexts are empty
2. Add metrics endpoint to check evaluation status
3. Consider storing RAGAS scores in local database as backup

## Testing Gaps

### Current Coverage
- Unit tests: ✓ Comprehensive
- Integration tests: ✗ No end-to-end tests
- Mock-based: ✓ All tests use mocks

### Recommended Tests to Add
1. End-to-end test with real workflow (requires DB setup)
2. Test for context capture in RAG engine
3. Test for metrics aggregation

## Summary

The observability module is well-structured with good test coverage. The main issue is the missing integration point for retrieved contexts. Fixing this would enable the full RAGAS evaluation functionality.

---

*Generated: 2026-03-16*

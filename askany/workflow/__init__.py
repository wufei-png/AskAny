"""Workflow module for agent-based RAG refinement."""

from askany.workflow.AnalysisRelated import (
    RelevantResult,
    analyze_relevance_and_completeness,
)
from askany.workflow.FinalSummaryLlm import (
    extract_docs_references,
    format_docs_references,
    generate_final_answer,
)
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from askany.workflow.SubProblemGenerator import SubProblemGenerator, SubProblemStructure
from askany.workflow.workflow_langgraph import AgentWorkflow

# from askany.workflow.workflow_llamaindex import AgentWorkflowLlama

__all__ = [
    # "AgentWorkflowLlama",
    # "AgentWorkflowLlamaIndex",
    "AgentWorkflow",
    "RelevantResult",
    "analyze_relevance_and_completeness",
    "extract_docs_references",
    "format_docs_references",
    "generate_final_answer",
    "LocalFileSearchTool",
    "SubProblemGenerator",
    "SubProblemStructure",
]

# -*- coding: utf-8 -*-
# coding=utf-8
"""Main entry point for AskAny application."""

# Set stdout/stderr encoding to UTF-8 FIRST, before any imports
# This must be done before importing any modules that might print to stdout/stderr
# This ensures Chinese characters and other Unicode characters display correctly in log files
import io
import sys

# Disable ANSI color codes when output is redirected to files
# This prevents escape sequences from appearing in log files
# if not sys.stdout.isatty():
#     os.environ.setdefault("NO_COLOR", "1")
#     os.environ.setdefault("TERM", "dumb")
#     # Disable tqdm progress bars when not in terminal
#     os.environ.setdefault("TQDM_DISABLE", "1")

try:
    # Set encoding for stdout/stderr to UTF-8
    # This is critical for preventing encoding issues when output is redirected to files
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
except (AttributeError, OSError):
    # If stdout/stderr don't have buffer attribute or can't be wrapped, skip
    # The PYTHONIOENCODING environment variable should handle it
    pass

import logging
import multiprocessing
import os
from typing import Optional

# Configure logging for OpenAI client to reduce verbose output
# Set OpenAI client logging to WARNING level to reduce noise
# logging.getLogger("openai").setLevel(logging.WARNING)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

try:
    import torch
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    TORCH_AVAILABLE = False
    torch = None

from logging import getLogger

from askany.api.server import run_server
from askany.config import settings
from askany.ingest import VectorStoreManager, ingest_documents
from askany.rag import create_query_router
from askany.vllm.vllm import AutoRetryVLLM
from askany.workflow.workflow_langgraph import AgentWorkflow
from askany.workflow.workflow_filter import WorkflowFilter
logger = getLogger(__name__)


def limit_cpu_cores(num_cores: Optional[int] = None) -> None:
    """Limit CPU cores used by the process.
    
    This function:
    1. Sets process CPU affinity (if supported)
    2. Sets environment variables for underlying libraries (OMP, MKL, etc.)
    3. Sets PyTorch thread count (if available)
    
    Args:
        num_cores: Number of CPU cores to use (None = use all available)
    """
    if num_cores is None:
        num_cores = settings.cpu_cores
    
    if num_cores is None:
        # No limit specified, use all cores
        return
    
    # Get total available cores
    total_cores = multiprocessing.cpu_count()
    if num_cores > total_cores:
        logger.warning(
            f"Requested {num_cores} cores but only {total_cores} available. "
            f"Using {total_cores} cores."
        )
        num_cores = total_cores
    
    if num_cores <= 0:
        logger.warning(f"Invalid num_cores={num_cores}, skipping CPU limit")
        return
    
    logger.info(f"Limiting CPU usage to {num_cores} core(s) out of {total_cores} available")
    
    # 1. Set process CPU affinity (Linux/Unix only)
    try:
        import os
        if hasattr(os, 'sched_setaffinity'):
            # Get current process ID
            pid = os.getpid()
            # Create CPU set with cores 0 to num_cores-1
            cpu_set = set(range(num_cores))
            os.sched_setaffinity(pid, cpu_set)
            logger.info(f"Set CPU affinity to cores: {sorted(cpu_set)}")
        else:
            logger.debug("os.sched_setaffinity not available on this platform")
    except Exception as e:
        logger.warning(f"Failed to set CPU affinity: {e}")
    
    # 2. Set environment variables for underlying libraries
    # These affect libraries like NumPy, SciPy, OpenBLAS, MKL, etc.
    env_vars = {
        "OMP_NUM_THREADS": str(num_cores),  # OpenMP
        "MKL_NUM_THREADS": str(num_cores),  # Intel MKL
        "NUMEXPR_NUM_THREADS": str(num_cores),  # NumExpr
        "OPENBLAS_NUM_THREADS": str(num_cores),  # OpenBLAS
        "VECLIB_MAXIMUM_THREADS": str(num_cores),  # macOS Accelerate
        "NUMBA_NUM_THREADS": str(num_cores),  # Numba
    }
    
    for var_name, var_value in env_vars.items():
        os.environ[var_name] = var_value
        logger.debug(f"Set {var_name}={var_value}")
    
    # 3. Set PyTorch thread count (if available)
    if TORCH_AVAILABLE:
        try:
            torch.set_num_threads(num_cores)
            logger.info(f"Set PyTorch threads to {num_cores}")
        except Exception as e:
            logger.warning(f"Failed to set PyTorch threads: {e}")
    
    logger.info(f"CPU core limit applied: {num_cores} core(s)")


def get_device() -> str:
    """Detect and return the appropriate device (cuda or cpu).

    Returns:
        Device string: "cuda" if CUDA is available, "cpu" otherwise
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available, using GPU")
    else:
        device = "cpu"
        logger.info("Using CPU")
    settings.device = device
    return device


class SentenceTransformerEmbedding(BaseEmbedding):
    """Custom embedding class using sentence-transformers for BGE models."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 512, **kwargs):
        """Initialize SentenceTransformer embedding.

        Args:
            model_name: HuggingFace model name (e.g., "BAAI/bge-large-zh-v1.5") or 
                       local path to model directory (e.g., "/path/to/bge-m3").
                       If local path is provided, model will be loaded offline.
            device: Device to use ("cpu" or "cuda")
            batch_size: Batch size for encoding multiple texts (default: 512)
        """
        super().__init__(**kwargs)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        # Use object.__setattr__ to bypass Pydantic field validation if needed
        # Store model as private attribute to avoid Pydantic validation issues
        # Check if local_files_only should be used (from kwargs or auto-detect from path)
        local_files_only = kwargs.get("local_files_only", False)
        # Auto-detect: if model_name is an absolute path, use local_files_only
        import os
        if os.path.isabs(model_name) and os.path.exists(model_name):
            local_files_only = True
        model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        object.__setattr__(self, "_sentence_transformer_model", model)
        object.__setattr__(self, "_model_name", model_name)
        object.__setattr__(self, "_batch_size", batch_size)
        # Get embedding dimension from model
        object.__setattr__(self, "_dimension", model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._sentence_transformer_model.encode(
            query, normalize_embeddings=True
        ).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        return self._sentence_transformer_model.encode(
            text, normalize_embeddings=True
        ).tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get multiple text embeddings with batch processing."""
        batch_size = self._batch_size
        embeddings = self._sentence_transformer_model.encode(
            texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
        )
        return embeddings.tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get query embedding (async)."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Get text embedding (async)."""
        return self._get_text_embedding(text)


def initialize_llm():
    """Initialize LLM and embedding models.

    Supports:
    - OpenAI API: Set openai_api_key and openai_api_base=None
    - vLLM: Set openai_api_base to vLLM endpoint (api_key can be None or any value)
    - Local LLM: Set openai_api_base to local LLM endpoint (e.g., http://localhost:8000/v1)

    Returns:
        Tuple of (llm, embed_model)
    """
    # For vLLM/local LLM: api_key can be None if api_base is set
    # For OpenAI: api_key is required
    api_key = settings.openai_api_key if settings.openai_api_key else None

    # Initialize embedding model
    # Support both OpenAI and SentenceTransformer (BGE) models
    embedding_model_type = getattr(settings, "embedding_model_type", "openai")

    if (
        embedding_model_type == "sentence_transformer"
        and SENTENCE_TRANSFORMERS_AVAILABLE
    ):
        # Use SentenceTransformer for BGE models
        # BGE models work better for Chinese text
        try:
            # Auto-detect CUDA availability
            device = get_device()
            logger.info("Using %s for embedding model", device)

            embed_model = SentenceTransformerEmbedding(
                model_name=settings.embedding_model,
                device=device,
                batch_size=getattr(settings, "embedding_batch_size", 512),
                local_files_only=getattr(settings, "embedding_local_files_only", False),
            )
            logger.info(
                "Using SentenceTransformer embedding model: %s (dimension: %d)",
                settings.embedding_model,
                embed_model.dimension,
            )
            # Update vector dimension if it doesn't match
            if settings.vector_dimension != embed_model.dimension:
                logger.warning(
                    "Vector dimension mismatch: config has %d, model has %d. "
                    "Update vector_dimension in config.py",
                    settings.vector_dimension,
                    embed_model.dimension,
                )
        except Exception as e:
            logger.error(
                "Failed to load SentenceTransformer model %s: %s. Falling back to OpenAI.",
                settings.embedding_model,
                e,
            )
            # Fallback to OpenAI
            embed_model = OpenAIEmbedding(
                api_key=api_key,
                api_base=settings.openai_api_base,
                model="text-embedding-ada-002",  # Default OpenAI model
            )
            logger.info("Using OpenAI embedding model (fallback)")
    else:
        # Use OpenAI embedding (default or fallback)
        embed_model = OpenAIEmbedding(
            api_key=api_key,
            api_base=settings.openai_api_base,
            model=settings.embedding_model,
        )
        logger.info("Using OpenAI embedding model: %s", settings.embedding_model)

    # Initialize LLM
    # For vLLM/local LLM: Set api_base to the endpoint URL
    # For OpenAI: Set api_base=None (uses default OpenAI endpoint)
    # Note: model parameter is still required to specify which model to use
    # Use VLLMOpenAI wrapper for vLLM to handle custom model paths
    if (
        settings.openai_api_base
        and settings.openai_api_base != "https://api.openai.com/v1"
    ):
        # vLLM or custom endpoint - use wrapper to handle unknown model names
        model_name = settings.openai_model
        llm = AutoRetryVLLM(model_name, settings.openai_api_base, api_key)
    else:
        # Standard OpenAI API
        llm = OpenAI(
            api_key=api_key,
            api_base=settings.openai_api_base,  # None for OpenAI
            model=settings.openai_model,
            temperature=settings.temperature,
        )

    # Set global settings
    # If llm is AutoRetryVLLM wrapper, use the underlying LLM for Settings
    # (Settings.llm expects an LLM instance, not a wrapper)
    if isinstance(llm, AutoRetryVLLM):
        Settings.llm = llm._llm
    else:
        Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model


def main():
    """Main function."""

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",  # Simple format, just the message
        stream=sys.stderr,  # Output to stderr so it's captured by 2>&1
    )

    import argparse

    parser = argparse.ArgumentParser(description="AskAny RAG System")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents into vector store",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Query test",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        default="",
        help="Query text to search",
    )
    parser.add_argument(
        "--query-type",
        type=str,
        default="AUTO",
        help="Query type (AUTO, FAQ, DOCS)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start API server",
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Check database and print ingested data",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create HNSW index for vector stores (docs and FAQ). "
             "Use this after ingesting all documents for optimal performance.",
    )
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=None,
        help="Limit CPU cores used by the process (overrides config.cpu_cores)",
    )

    args = parser.parse_args()
    
    # Limit CPU cores if configured (should be done early, before loading models)
    # Command line argument takes precedence over config
    limit_cpu_cores(args.cpu_cores if args.cpu_cores is not None else None)
    
    device = get_device()
    print(f"Using device: {device}")
    # Initialize LLM and embedding models once
    llm, embed_model = initialize_llm()
    # After ingestion, check database if requested
    if args.check_db:
        from tool.ingest_check import check_ingested_data

        check_ingested_data(n=5, skip_db=False, llm=llm, embed_model=embed_model)

    elif args.ingest:
        ingest_documents(embed_model, llm=llm)
    
    elif args.create_index:
        # Create HNSW indexes for vector stores
        print("Creating HNSW indexes for vector stores...")
        vector_store_manager = VectorStoreManager(embed_model, llm=llm)
        
        # Initialize indexes first
        try:
            vector_store_manager.initialize_faq_index()
            print("FAQ vector store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize FAQ vector store: {e}")
        
        try:
            vector_store_manager.initialize_docs_index()
            print("Docs vector store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize docs vector store: {e}")
        
        # Create indexes
        try:
            vector_store_manager.create_faq_hnsw_index()
            print("FAQ HNSW index created successfully")
        except Exception as e:
            logger.error(f"Failed to create FAQ HNSW index: {e}")
        
        try:
            vector_store_manager.create_docs_hnsw_index()
            print("Docs HNSW index created successfully")
        except Exception as e:
            logger.error(f"Failed to create docs HNSW index: {e}")
        
        print("Index creation completed!")

    elif args.query:
        # Query nodes from vector store
        if not args.query_text:
            print("Error: --query-text is required when using --query")
            parser.print_help()
            return

        # Import and call query function from test file
        from tool.query_direct_from_vector_test import query_from_vector_store

        query_from_vector_store(
            query_text=args.query_text,
            query_type=args.query_type,
            embed_model=embed_model,
            llm=llm,
        )

    elif args.serve:
        # Initialize vector store manager (assumes already ingested)
        vector_store_manager = VectorStoreManager(embed_model, llm=llm)

        # Try to initialize separate indexes (for new architecture)
        try:
            vector_store_manager.initialize_faq_index()
            vector_store_manager.initialize_docs_index()
            print("Using separate indexes for FAQ and docs")
        except Exception as e:
            # Fallback to legacy single index
            logger.error(f"Separate indexes not available, using legacy index: {e}")
            vector_store_manager.initialize()

        # Create router (pass initialized LLM and embed_model)
        router = create_query_router(vector_store_manager, llm, embed_model, device)

        # Initialize shared tools to reduce resource usage and enable caching
        from askany.workflow.WebSearchTool import WebSearchTool
        from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
        from askany.ingest.keyword_extract_wrapper import KeywordExtractorWrapper
        from askany.ingest.custom_keyword_index import (
            get_global_keyword_extractor,
            set_global_keyword_extractor,
        )
        
        # Initialize keyword extractor (shared)
        global_extractor = get_global_keyword_extractor()
        if global_extractor is None:
            keyword_extractor = KeywordExtractorWrapper(
                priority=settings.keyword_extractor_priority
            )
            set_global_keyword_extractor(keyword_extractor)
        else:
            keyword_extractor = global_extractor
        
        # Initialize shared web search tool
        try:
            shared_web_search_tool = WebSearchTool()
            print("Shared WebSearchTool initialized")
        except ImportError:
            logger.warning("WebSearchTool not available, web search will be disabled")
            shared_web_search_tool = None
        
        # Initialize shared local file search tool
        base_path = settings.local_file_search_dir if settings.local_file_search_dir else None
        shared_local_file_search = LocalFileSearchTool(
            base_path=base_path,
            keyword_extractor=keyword_extractor.GetKeywordExtractorFromTFIDF()
        )
        print("Shared LocalFileSearchTool initialized")

        # Create AgentWorkflow (LangGraph version) with shared tools
        # Use Settings.llm which is the underlying LLM instance (unwrapped if AutoRetryVLLM)
        # Settings.llm is set in initialize_llm() above
        workflow_llm = getattr(Settings, "llm", llm)
        agent_workflow = AgentWorkflow(
            router=router,
            llm=workflow_llm,
            web_search_tool=shared_web_search_tool,
            local_file_search=shared_local_file_search,
        )
        workflow_filter = WorkflowFilter(
            direct_answer_generator=agent_workflow.direct_answer_generator,
            web_or_rag_generator=agent_workflow.web_or_rag_generator,
            final_answer_generator=agent_workflow.final_answer_generator,
            web_search_tool=agent_workflow.web_search_tool,
            reranker=agent_workflow.reranker,
        )
        print("AgentWorkflow (deep search) initialized")
        
        # Create simple agent (min_langchain_agent) with shared tools
        from askany.workflow.min_langchain_agent import create_agent_with_tools
        simple_agent = create_agent_with_tools(
            router=router,
            web_search_tool=shared_web_search_tool,
            local_file_search=shared_local_file_search,
            llm_instance=llm,
            keyword_extractor=keyword_extractor,
        )
        print("Simple agent initialized")
        
        # Start server with vector_store_manager for hot updates
        print(f"Starting API server on {settings.api_host}:{settings.api_port}")
        print(
            f"OpenAPI schema available at: http://{settings.api_host}:{settings.api_port}/openapi.json"
        )
        print(
            f"FAQ hot update endpoint: http://{settings.api_host}:{settings.api_port}/v1/update_faqs"
        )
        print("Model selection:")
        print("  - Models with '-deepsearch' suffix: Use AgentWorkflow (complex workflow)")
        print("  - Other models: Use simple agent (fast workflow)")
        run_server(
            router,
            vector_store_manager,
            llm,
            embed_model,
            agent_workflow,
            workflow_filter,
            simple_agent,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

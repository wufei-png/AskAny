"""Configuration management for AskAny."""

import re
from typing import List, Literal, Optional

from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Language setting for prompts
    # Options: "cn" (Chinese), "en" (English)
    language: Literal["cn", "en"] = "cn"

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, v: str) -> str:
        """Normalize language from locale format (e.g., 'en_US:en' -> 'en', 'zh_CN:zh' -> 'cn')."""
        if not isinstance(v, str):
            return v

        # Handle locale format like "en_US:en" or "zh_CN:zh"
        # Extract the language code before the colon or underscore
        if ":" in v:
            v = v.split(":")[0]
        if "_" in v:
            lang_code = v.split("_")[0].lower()
        else:
            lang_code = v.lower()

        # Map common language codes to our format
        if lang_code.startswith("zh") or lang_code == "cn":
            return "cn"
        elif lang_code.startswith("en"):
            return "en"

        # If it's already "cn" or "en", return as is
        if lang_code in ("cn", "en"):
            return lang_code

        # Default to "cn" if unrecognized
        return "cn"

    device: str = "cuda"
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "wufei"
    postgres_password: str = "123456"
    postgres_db: str = "askany"
    log_level: str = "DEBUG"
    query_fusion_num_queries: int = 1
    # OpenAI/LLM
    openai_api_key: Optional[str] = ""
    openai_api_base: Optional[str] = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"  # For vLLM compatibility
    )
    # For vLLM: model name is still required to specify which model to use
    openai_model: str = "qwen-plus"

    num_concurrent_runs: int = 1  # agent workflow concurrent runs

    enable_web_search: bool = True
    web_search_api_url: str = (
        "http://localhost:8800/search"  # Web search API endpoint URL
    )

    temperature: float = 0.2
    top_p: float = 0.8
    llm_max_tokens: int = 40000
    web_chat_tokens: int = llm_max_tokens // 2
    output_tokens: int = llm_max_tokens // 10
    llm_timeout: int = 700  # LLM timeout in seconds
    # Embedding Model
    # Options:
    # - "text-embedding-ada-002" (OpenAI, English-focused, 1536 dim)
    # - "BAAI/bge-m3" (Multilingual, 1024 dim, supports Chinese + English, recommended)
    # - "BAAI/bge-large-zh-v1.5" (Chinese-focused, 1024 dim, best quality for Chinese only, alternative)
    # - "BAAI/bge-base-zh-v1.5" (Chinese, 1024 dim, faster, good quality)
    # - "BAAI/bge-small-zh-v1.5" (Chinese, 512 dim, fastest, lower quality)
    # - Local path: Use absolute path to local model directory (e.g., "/path/to/bge-m3")
    #   For offline loading, ensure model files are complete (config.json, model files, tokenizer, etc.)
    embedding_model: str = "BAAI/bge-m3"  # Multilingual, recommended (alternative: BAAI/bge-large-zh-v1.5 for Chinese-only)  #! TODO 稀疏还是稠密向量
    embedding_model_type: str = (
        "sentence_transformer"  # Options: "openai", "sentence_transformer"
    )
    embedding_local_files_only: bool = False  # If True, only load from local files (no HuggingFace download). Set to True for offline mode.

    # Vector Store
    vector_table_name: str = (
        "askany_vectors"  # Default/legacy table name, currently not used
    )

    faq_vector_table_name: str = "askany_faq_vectors"  # FAQ-specific vector table
    docs_vector_table_name: str = "askany3_docs_vectors"  # Docs-specific vector table

    text_search_config: str = "english"  # Text search configuration for PostgreSQL
    # Vector dimension depends on embedding model:
    # - OpenAI ada-002: 1536
    # - BGE models: 1024 (large/base/m3) or 512 (small)
    vector_dimension: int = (
        1024  # BGE-large-zh-v1.5 dimension (update if using different model)
    )

    # HNSW Index Configuration for PGVector
    # HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor index
    # that significantly speeds up vector similarity search in PostgreSQL
    # Recommended values:
    # - hnsw_m: 16 (default, good balance between speed and accuracy)
    # - hnsw_ef_construction: 64 (default, higher = better quality but slower build)
    # - hnsw_ef_search: 40 (default, higher = better recall but slower queries)
    # - hnsw_dist_method: "vector_cosine_ops" (for cosine similarity, default)
    #                     or "vector_l2_ops" (for L2 distance)
    hnsw_m: int = 16  # Number of bi-directional links for each node
    hnsw_ef_construction: int = 128  # Size of the candidate list during construction
    hnsw_ef_search: int = 40  # Size of the candidate list during search
    hnsw_dist_method: str = (
        "vector_cosine_ops"  # Distance method: "vector_cosine_ops" or "vector_l2_ops"
    )
    enable_hnsw: bool = (
        True  # Whether to enable HNSW index (recommended for production)
    )

    # Reranker
    # Options for Chinese FAQ:
    # - "BAAI/bge-reranker-v2-m3" (multilingual, good for mixed content, recommended)
    # - "BAAI/bge-reranker-base" (Chinese-focused, balanced performance)
    # - "BAAI/bge-reranker-large" (better accuracy, slower)
    # - "openbmb/MiniCPM-Reranker" (alternative Chinese reranker)
    # - "cross-encoder/stsb-distilroberta-base" (English-focused, not recommended for Chinese)
    # - Local path: Use absolute path to local model directory (e.g., "/path/to/bge-reranker-v2-m3")
    #   For offline loading, ensure model files are complete (config.json, model files, tokenizer, etc.)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # Default reranker model (multilingual, matches bge-m3 embedding)
    reranker_type: str = (
        "sentence_transformer"  # Options: "sentence_transformer", "flag_embedding"
    )
    reranker_local_files_only: bool = False  # If True, only load from local files (no HuggingFace download). Set to True for offline mode.

    # Data paths
    data_dir: str = "data"
    json_dir: str = "data/json"
    markdown_dir: str = "data/markdown"
    local_file_search_dir: str = "data/markdown"
    stopwords_dir: str = "data/stopwords"
    # Storage paths
    storage_dir: str = "key_word_storage"  # Directory for persisting KeywordTableIndex
    keyword_storage_index: str = (
        "keyword_index"  # Directory for persisting KeywordTableIndex
    )
    faq_keyword_storage_index: str = (
        "faq_keyword_index"  # Directory for persisting FAQ KeywordTableIndex
    )
    docs_keyword_storage_index: str = (
        "docs_keyword_index"  # Directory for persisting Docs KeywordTableIndex
    )
    using_docs_keyword_index: bool = False  # Whether to use keyword index
    # faq_keyword_storage_index: str = (
    #     "hybrid_faq_keyword_index"  # Directory for persisting FAQ KeywordTableIndex
    # )
    # docs_keyword_storage_index: str = (
    #     "hybrid_docs_keyword_index"  # Directory for persisting Docs KeywordTableIndex
    # )
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    inner_server_host: str = "0.0.0.0"
    inner_server_port: int = 8001
    # Auto mode settings
    faq_score_threshold: float = (
        0.75  # Minimum score threshold for FAQ to be considered sufficient
    )
    faq_second_score_threshold: float = (
        0.6  # Minimum score threshold for FAQ to be added as low-confidence context
    )

    # Query engine settings
    faq_similarity_top_k: int = (
        3  # Number of top similar documents to retrieve (final output)
    )
    # Rerank settings: retriever should retrieve more candidates for reranker to select from
    # This allows reranker to truly filter and select the best results
    faq_rerank_candidate_k: int = 10  # Number of candidates to retrieve before reranking (should be > similarity_top_k)
    max_keywords_for_faq: int = 3  # Maximum number of keywords to extract for FAQ
    # Ensemble weights for FAQ query engine [keyword_weight, vector_weight]
    # Options:
    # - [0.5, 0.5] (balanced, default)
    # - [0.7, 0.3] (keyword-focused, better for exact matches)
    # - [0.3, 0.7] (vector-focused, better for semantic understanding)
    faq_ensemble_weights: List[float] = [0.5, 0.5]
    docs_similarity_top_k: int = 5  # Similarity top k for docs query engine
    max_nodes_to_llm: int = 30  # Maximum number of nodes to send to LLM
    max_keywords_for_docs: int = 3  # Maximum number of keywords to extract for docs
    docs_rerank_candidate_k: int = 10  # Number of candidates to retrieve before reranking for docs (should be > docs_similarity_top_k)
    docs_similarity_threshold: float = 0.6  # Similarity threshold for docs query engine

    custom_keyword_similarity_threshold: float = (
        0.8  # Similarity threshold for sense keyword
    )

    if using_docs_keyword_index == False:
        docs_similarity_threshold -= 0.05
    # Ensemble weights for docs query engine [keyword_weight, vector_weight]
    # Options:
    # - [0.5, 0.5] (balanced, default)
    # - [0.7, 0.3] (keyword-focused, better for exact matches)
    # - [0.3, 0.7] (vector-focused, better for semantic understanding)
    docs_ensemble_weights: List[float] = [0.5, 0.5]
    faq_vector_only_similarity_top_k: int = (
        3  # Similarity top k when FAQ has no keyword index
    )

    # FAQ document splitting settings
    faq_split_documents: bool = True  # Whether to split FAQ documents using node_parser
    # If True: uses insert() which may split documents via node_parser
    # If False: uses insert_nodes() directly, ensuring one Document = one Node (no splitting)

    # Markdown document splitting settings
    # Options:
    # - "markdown": Use only MarkdownNodeParser (structure-aware splitting)
    # - "semantic": Use only SemanticSplitterNodeParser (semantic-aware splitting)
    # - "hybrid": Use MarkdownNodeParser first, then SemanticSplitterNodeParser (default, best of both)
    markdown_split_mode: str = "markdown"

    # Batch processing settings for vector store insertion
    docs_insert_batch_size: int = (
        1000  # Batch size for inserting docs documents (default: 1000)
    )
    # Embedding batch size for sentence-transformers
    # Larger batch sizes improve throughput but use more GPU/CPU memory
    # Recommended: 32-512 for GPU, 16-128 for CPU
    embedding_batch_size: int = 512  # Batch size for embedding model when encoding multiple texts (default: 512)
    # HNSW index optimization for bulk insert
    # When inserting large batches (>10k nodes), it's faster to:
    # 1. Drop HNSW index before insertion
    # 2. Insert all data
    # 3. Recreate HNSW index
    # This avoids index maintenance overhead during insertion
    enable_hnsw_bulk_insert_optimization: bool = (
        True  # Whether to enable HNSW index optimization for bulk insert
    )
    hnsw_bulk_insert_threshold: int = (
        10000  # Minimum number of nodes to trigger index drop/recreate optimization
    )
    # PostgreSQL maintenance_work_mem for HNSW index creation
    # Larger values speed up index creation but use more memory
    # Recommended: 256MB-1GB for 10k-100k vectors, adjust based on available RAM
    hnsw_maintenance_work_mem: str = (
        "512MB"  # Memory allocation for index creation (e.g., "256MB", "512MB", "1GB")
    )

    ##  for agent
    min_tfidf_score: float = 0.0  # Minimum TF-IDF score for keywords
    agent_max_iterations: int = 3  # Maximum number of iterations for the agent
    reserve_keywords_old_nodes: bool = True  # Whether to reserve old nodes
    reserve_expanded_old_nodes: bool = False  # Whether to reserve old keywords
    expand_context_mode: str = "ratio"  # Whether to expand context by ratio or lines
    expand_context_ratio: float = 1.0  # Ratio of context to expand
    keyword_expand_ratio: float = 1.0  # Ratio of context to expand
    one_keyword_max_file_num: int = 40  # Maximum number of files for one keyword
    one_keyword_max_matches_num: int = 500  # Maximum number of matches for one keyword
    query_rewrite_bool: bool = True  # Whether to rewrite query
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from environment variables
    )

    recursive_search_time_for_expand_or_norelevant: int = (
        2  # Recursive search time for expand or no relevant
    )
    # TODO 这个参数需要最后再调整
    freq_in_rag_threshold: int = 80  # Frequency threshold for keywords in RAG

    using_custom_keyword_index: bool = True  # Whether to use custom keyword index
    keyword_extractor_priority: str = "tfidf"  # Keyword extractor priority
    expand_node_force_end_line: bool = (
        True  # Whether to force end workflow after expand node
    )

    return_middle_result: bool = True  # Whether to return middle result
    inner_sub_problems: bool = False  # Whether to return inner sub problems
    max_content_length: int = -1  # Maximum content length for each node

    # Question-Answer Cache settings
    enable_qa_cache: bool = True  # Whether to enable question-answer caching
    qa_cache_use_similarity: bool = False  # Whether to use semantic similarity matching for cache (requires embedding model)
    qa_cache_similarity_threshold: float = 0.95  # Similarity threshold for cache matching (0-1, only used when qa_cache_use_similarity=True)

    # CPU resource limits
    cpu_cores: Optional[int] = None  # Limit CPU cores (None = use all available cores)

    # HanLP Tokenizer settings
    # Options:
    # - None or empty string: Use default pretrained model (hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    # - Local path: Use absolute path to local HanLP model directory or zip file
    #   Example: "/root/.hanlp/tok/coarse_electra_small_20220616_012050.zip"
    #   or extracted directory: "/workspace/models/hanlp/coarse_electra_small"
    #   or extracted directory: "/workspace/models/hanlp/coarse_electra_small"
    # hanlp_tokenizer_path: Optional[str] = "/workspace/hanlp/tok/coarse_electra_small_20220616_012050/coarse_electra_small_20220616_012050"  # Local path to HanLP tokenizer model (None = use default pretrained)
    hanlp_tokenizer_path: Optional[str] = None
    # HanLP home directory (where models and dependencies are cached)
    # Default: ~/.hanlp (user's home directory)
    # Example: "/workspace/hanlp" (for Docker containers with mounted volumes)
    # hanlp_home: Optional[str] = "/workspace/hanlp"  # HanLP home directory (None = use default ~/.hanlp)
    hanlp_home: Optional[str] = None


settings = Settings()

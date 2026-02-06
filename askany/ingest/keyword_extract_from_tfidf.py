"""Keyword extraction using TF-IDF with HanLP tokenization."""

import logging
import pickle
import sys
import zipfile
import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add parent directory to path to import askany modules
# This allows the script to be run directly: python askany/ingest/keyword_extract.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from askany.config import settings  # noqa: E402

# Set HanLP home directory to use local cache
# This helps HanLP find local files without downloading
# Priority: 1. settings.hanlp_home, 2. HANLP_HOME env var, 3. default ~/.hanlp
if settings.hanlp_home:
    os.environ["HANLP_HOME"] = settings.hanlp_home
elif "HANLP_HOME" not in os.environ:
    os.environ["HANLP_HOME"] = str(Path.home() / ".hanlp")

import hanlp
from sklearn.feature_extraction.text import TfidfVectorizer

from askany.config import settings
from tool.keyword_utils import (
    load_keywords_and_frequency_from_txt,
    load_keywords_from_txt,
)
from cachetools import LRUCache, cachedmethod

logger = logging.getLogger(__name__)


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle module path issues when loading from different entry points.

    This fixes the issue where pickle tries to load _whitespace_tokenizer from
    the wrong module (e.g., askany.main instead of askany.ingest.keyword_extract)
    when running with -m flag.

    This is a compatibility fix that works for both:
    - Direct execution: python askany/ingest/keyword_extract.py
    - Module execution: python -m askany.main
    """

    def find_class(self, module, name):
        """Override find_class to redirect to correct module for _whitespace_tokenizer."""
        # If trying to load _whitespace_tokenizer from wrong module, redirect to correct one
        if (
            name == "_whitespace_tokenizer"
            and module != "askany.ingest.keyword_extract"
        ):
            # Return the function from the current module (this module)
            # This avoids circular import issues and works regardless of entry point
            return _whitespace_tokenizer
        # For all other cases, use default behavior
        try:
            return super().find_class(module, name)
        except AttributeError:
            # If class not found, try alternative module paths
            if name == "_whitespace_tokenizer":
                return _whitespace_tokenizer
            raise


def _whitespace_tokenizer(text: str) -> List[str]:
    """Tokenize text by splitting on whitespace.

    This is a module-level function that can be pickled, used as a tokenizer
    for TfidfVectorizer. The text should already be tokenized and space-separated.

    Args:
        text: Space-separated tokenized text.

    Returns:
        List of tokens.
    """
    return text.split()


class KeywordExtractorFromTFIDF:
    """Keyword extractor using TF-IDF with HanLP tokenization.

    This class:
    1. Traverses markdown files in a directory (similar to markdown_parser.py:431-471)
    2. Tokenizes documents using HanLP
    3. Trains a TF-IDF model on the tokenized documents
    4. Extracts keywords from domain-specific queries
    """

    def __init__(
        self,
        custom_dict_path: Optional[str] = None,
        max_features: Optional[int] = None,
        min_df: int = 1,
        max_df: float = 0.7,
    ):
        """Initialize the keyword extractor.

        Args:
            custom_dict_path: Path to custom dictionary file (word_freq.txt format).
                             If None, uses default from settings.
            max_features: Maximum number of features (terms) in TF-IDF vectorizer.
            min_df: Minimum document frequency for a term to be included.
            max_df: Maximum document frequency threshold (0.95 means ignore terms
                   that appear in more than 95% of documents).
        """
        # Initialize HanLP tokenizer
        logger.info("Loading HanLP tokenizer...")
        # Support loading from local path if configured
        if settings.hanlp_tokenizer_path:
            # Convert to absolute path to handle relative paths correctly
            model_path = Path(settings.hanlp_tokenizer_path).resolve()
            if not model_path.exists():
                logger.warning(
                    f"HanLP tokenizer path configured but not found: {model_path}. "
                    "Falling back to default pretrained model."
                )
                self.tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
                return

            logger.info(f"Loading HanLP tokenizer from local path: {model_path}")

            # If path is a zip file, extract it first
            if model_path.suffix == ".zip":
                logger.info(f"HanLP tokenizer path is a zip file: {model_path}")
                # Extract to a directory next to the zip file
                extract_dir = model_path.parent / model_path.stem

                # Extract if directory doesn't exist or is empty
                if not extract_dir.exists() or not any(extract_dir.iterdir()):
                    logger.info(f"Extracting HanLP tokenizer zip to: {extract_dir}")
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(model_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    logger.info(f"HanLP tokenizer extracted successfully")

                # Find the actual model directory inside extracted files
                # Usually it's a subdirectory with the same name as the zip (without .zip)
                model_dir = extract_dir / model_path.stem
                if not model_dir.exists():
                    # Try to find any subdirectory
                    subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
                    if subdirs:
                        model_dir = subdirs[0]
                        logger.info(f"Using model directory: {model_dir}")
                    else:
                        # Use extract_dir itself if no subdirectory found
                        model_dir = extract_dir

                logger.info(
                    f"Loading HanLP tokenizer from extracted directory: {model_dir}"
                )
                self.tok = hanlp.load(str(model_dir))
            else:
                # Path is a directory, load directly
                self.tok = hanlp.load(str(model_path))
        else:
            # Use default pretrained model
            self.tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

        # Load custom dictionary
        if custom_dict_path is None:
            word_freq_file = (
                Path(settings.storage_dir)
                / settings.docs_keyword_storage_index
                / "word_freq.txt"
            )
            custom_dict_path = str(word_freq_file)

        if Path(custom_dict_path).exists():
            custom_dict = load_keywords_from_txt(custom_dict_path)
            self.tok.dict_combine = custom_dict
            logger.info(f"Loaded {len(custom_dict)} keywords from custom dictionary")
        else:
            logger.warning(f"Custom dictionary not found: {custom_dict_path}")

        # Create HanLP pipeline for sentence splitting and tokenization
        self.hanlp_pipeline = (
            hanlp.pipeline().append(hanlp.utils.rules.split_sentence).append(self.tok)
        )

        # Initialize TF-IDF vectorizer
        # We'll use space-separated tokenized text, so we use a simple tokenizer
        # that splits by whitespace
        # Note: Using module-level function instead of lambda for pickle compatibility
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            tokenizer=_whitespace_tokenizer,  # Module-level function (pickle-compatible)
            token_pattern=None,  # Disable regex pattern since we use custom tokenizer
            lowercase=False,  # Keep original case for Chinese
            sublinear_tf=True,
            # 默认情况下 TF 是线性的。对于关键词提取，词频为 10 的词并不比词频为 1 的词重要 10 倍。通常建议开启 sublinear_tf=True（使用 $1 + \log(TF)$），这样能削弱超高频词的统治力。
        )

        # Load stopwords from configuration directory
        self.stop_words = self._load_stopwords()

        # Store trained documents and their tokenized versions
        self.documents: List[str] = []
        self.tokenized_documents: List[str] = []
        self.tfidf_matrix = None
        self.feature_names: List[str] = []

        # Load domain keywords from word_freq.txt for filtering
        self.domain_keywords: Dict[str, int] = {}
        if Path(custom_dict_path).exists():
            self.domain_keywords = load_keywords_and_frequency_from_txt(
                custom_dict_path
            )
            logger.info(
                f"Loaded {len(self.domain_keywords)} domain keywords for filtering"
            )

        # Try to load persisted model if it exists
        self._load_persisted_model()
        self.cache_set = LRUCache(maxsize=500)
        self.cache_list = LRUCache(maxsize=500)

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from all txt files in the stopwords directory.

        Reads all .txt files from the stopwords directory specified in settings,
        extracts stopwords (one per line), and returns a deduplicated set.

        Returns:
            Set of stopwords.
        """
        stopwords = set()
        stopwords_dir = Path(settings.stopwords_dir)

        if not stopwords_dir.exists():
            logger.warning(
                f"Stopwords directory not found: {stopwords_dir}. "
                "No stopwords will be loaded."
            )
            return stopwords

        # Find all .txt files in the stopwords directory
        txt_files = list(stopwords_dir.glob("*.txt"))

        if not txt_files:
            logger.warning(
                f"No .txt files found in stopwords directory: {stopwords_dir}"
            )
            return stopwords

        logger.info(f"Loading stopwords from {len(txt_files)} file(s)...")

        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        word = line.strip()
                        if word:  # Skip empty lines
                            stopwords.add(word)
                logger.debug(f"Loaded stopwords from {txt_file.name}")
            except Exception as e:
                logger.error(f"Error loading stopwords from {txt_file}: {e}")
        logger.info(f"Loaded {len(stopwords)} unique stopwords")
        return stopwords

    def _get_persist_dir(self) -> Path:
        """Get the directory for persisting the model.

        Returns:
            Path to the persistence directory.
        """
        return Path(settings.storage_dir) / settings.docs_keyword_storage_index

    def _get_model_file(self) -> Path:
        """Get the path to the persisted model file.

        Returns:
            Path to the model pickle file.
        """
        return self._get_persist_dir() / "tfidf_model.pkl"

    def _load_persisted_model(self) -> bool:
        """Load persisted TF-IDF model if it exists.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        model_file = self._get_model_file()
        if not model_file.exists():
            logger.info("No persisted model found. Will train new model.")
            return False

        try:
            logger.info(f"Loading persisted model from {model_file}...")
            with open(model_file, "rb") as f:
                # Use custom unpickler to handle module path issues
                unpickler = CustomUnpickler(f)
                data = unpickler.load()

            self.vectorizer = data["vectorizer"]
            self.feature_names = data["feature_names"]
            self.tfidf_matrix = data.get("tfidf_matrix")  # Optional, may be None

            logger.info(
                f"✅ Loaded persisted model with {len(self.feature_names)} features"
            )
            if self.tfidf_matrix is not None:
                logger.info(
                    f"Loaded TF-IDF matrix with shape {self.tfidf_matrix.shape}"
                )
            return True
        except Exception as e:
            logger.error(f"Error loading persisted model: {e}")
            logger.info("Will train new model instead.")
            return False

    def persist(self) -> None:
        """Persist the trained TF-IDF model to disk.

        Saves the vectorizer, feature names, and TF-IDF matrix to the storage directory.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not trained. Call train() first.")

        persist_dir = self._get_persist_dir()
        persist_dir.mkdir(parents=True, exist_ok=True)

        model_file = self._get_model_file()

        try:
            logger.info(f"Persisting model to {model_file}...")
            data = {
                "vectorizer": self.vectorizer,
                "feature_names": self.feature_names,
                "tfidf_matrix": self.tfidf_matrix,
            }

            with open(model_file, "wb") as f:
                pickle.dump(data, f)

            logger.info(
                f"✅ Model persisted successfully: {len(self.feature_names)} features, "
                f"matrix shape {self.tfidf_matrix.shape}"
            )
        except Exception as e:
            logger.error(f"Error persisting model: {e}")
            raise

    def _tokenize_text(self, text: str, filter_stopwords: bool = True) -> List[str]:
        """Tokenize a text using HanLP pipeline.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens (words).
        """
        try:
            # Use HanLP pipeline to split sentences and tokenize
            # Pipeline returns a list of lists (each sentence is a list of tokens)
            sentences = self.hanlp_pipeline(text)
            # Flatten the list of sentences into a single list of tokens
            tokens = []
            for sentence in sentences:
                if isinstance(sentence, list):
                    # Filter out empty strings, whitespace, and stopwords
                    tokens.extend(
                        [
                            token
                            for token in sentence
                            if token.strip()
                            and (
                                token not in self.stop_words
                                if filter_stopwords
                                else True
                            )
                        ]
                    )
                elif isinstance(sentence, str) and sentence.strip():
                    # Filter out stopwords for string tokens
                    if sentence not in self.stop_words:
                        tokens.append(sentence)
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []

    def _find_markdown_files(self, directory: Path) -> List[Path]:
        """Find all markdown files in a directory (recursively).

        This method uses the same logic as markdown_parser.py:431-471.

        Args:
            directory: Directory to search for markdown files.

        Returns:
            List of Path objects for markdown files.
        """
        markdown_files = []

        # Try rglob first (most efficient)
        try:
            md_files = list(directory.rglob("*.md"))
            markdown_files.extend(md_files)
            markdown_files.extend(directory.rglob("*.markdown"))
            logger.debug(f"rglob found {len(markdown_files)} files")
        except Exception as e:
            logger.warning(f"rglob failed: {e}, trying manual traversal")

        # If rglob didn't find files, manually traverse subdirectories
        # This handles cases where rglob might not work from the root directory
        if not markdown_files:
            logger.debug("rglob found no files, manually traversing subdirectories")
            for subdir in directory.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    try:
                        subdir_files = list(subdir.rglob("*.md"))
                        subdir_files.extend(subdir.rglob("*.markdown"))
                        markdown_files.extend(subdir_files)
                        logger.debug(
                            f"Found {len(subdir_files)} files in {subdir.name}"
                        )
                    except Exception as e:
                        logger.warning(f"Error traversing {subdir}: {e}")

        # Filter out files in .git directories and other hidden directories
        # Also filter out directories that match .md/.markdown pattern
        markdown_files = [
            f
            for f in markdown_files
            if f.is_file()  # Ensure it's actually a file, not a directory
            and ".git" not in f.parts
            and not any(part.startswith(".") for part in f.parts[1:])
        ]

        # Remove duplicates (in case both patterns match the same file)
        markdown_files = list(set(markdown_files))
        # Sort for consistent ordering
        markdown_files.sort()

        logger.info(f"Found {len(markdown_files)} markdown file(s) in {directory}")
        return markdown_files

    def load_documents_from_directory(self, directory: Path) -> None:
        """Load and tokenize all markdown documents from a directory.

        Args:
            directory: Directory containing markdown files.
        """
        markdown_files = self._find_markdown_files(directory)

        self.documents = []
        self.tokenized_documents = []

        logger.info(f"Loading {len(markdown_files)} markdown files...")
        for md_file in markdown_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.documents.append(content)
                    # Tokenize the document
                    tokens = self._tokenize_text(content)
                    # Join tokens with space for TF-IDF vectorizer
                    tokenized_text = " ".join(tokens)
                    self.tokenized_documents.append(tokenized_text)
                    logger.debug(f"Loaded and tokenized: {md_file.name}")
            except Exception as e:
                logger.error(f"Error loading {md_file}: {e}")

        logger.info(
            f"Loaded {len(self.documents)} documents, "
            f"{len(self.tokenized_documents)} tokenized documents"
        )

    def train(self) -> None:
        """Train the TF-IDF model on loaded documents."""
        if not self.tokenized_documents:
            raise ValueError(
                "No documents loaded. Call load_documents_from_directory() first."
            )

        logger.info("Training TF-IDF model...")
        # Fit the vectorizer on tokenized documents
        self.tfidf_matrix = self.vectorizer.fit_transform(self.tokenized_documents)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        logger.info(
            f"TF-IDF model trained on {len(self.documents)} documents, "
            f"{len(self.feature_names)} features"
        )

    def tokenize_text(self, text: str, filter_stopwords: bool = False) -> List[str]:
        """Tokenize text using HanLP pipeline."""
        return self._tokenize_text(text, filter_stopwords)

    @cachedmethod(lambda self: self.cache_list)
    def extract_keywords(
        self,
        query: str,
        top_k: int = 3,
        min_tfidf_score: float = 0.0,
        filter_by_domain: bool = True,
    ) -> List[tuple]:
        """Extract keywords from a query using the trained TF-IDF model.

        Args:
            query: Input query text.
            top_k: Number of top keywords to return.
            min_tfidf_score: Minimum TF-IDF score threshold.
            filter_by_domain: If True, only return keywords that exist in word_freq.txt
                            (domain-specific keywords). If False, return all keywords
                            found in training documents.

        Returns:
            List of tuples (keyword, tfidf_score) sorted by score (descending).
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF model not trained. Call train() first.")

        # Tokenize the query
        query_tokens = self._tokenize_text(query)
        query_text = " ".join(query_tokens)

        # Transform query using the trained vectorizer
        query_vector = self.vectorizer.transform([query_text])

        # Get TF-IDF scores for query terms
        feature_index = query_vector.indices
        scores = query_vector.data

        # Create list of (keyword, score) pairs
        keyword_scores = [
            (self.feature_names[idx], score)
            for idx, score in zip(feature_index, scores)
            if score >= min_tfidf_score
        ]

        # Filter by domain keywords if enabled
        if filter_by_domain and self.domain_keywords:
            keyword_scores = [
                (keyword, score)
                for keyword, score in keyword_scores
                if keyword in self.domain_keywords
            ]

        # Sort by score (descending)
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k keywords
        return keyword_scores[:top_k]

    @cachedmethod(lambda self: self.cache_set)
    def extract_keywords_set(
        self,
        query: str,
        top_k: int = 3,
        min_tfidf_score: float = 0.0,
        filter_by_domain: bool = True,
    ) -> Set[str]:
        """Extract keywords from a query and return as a set.

        Args:
            query: Input query text.
            top_k: Number of top keywords to return.
            min_tfidf_score: Minimum TF-IDF score threshold.
            filter_by_domain: If True, only return keywords that exist in word_freq.txt
                            (domain-specific keywords). If False, return all keywords
                            found in training documents.

        Returns:
            Set of keywords (without scores).
        """
        keyword_scores = self.extract_keywords(
            query, top_k, min_tfidf_score, filter_by_domain
        )
        return {keyword for keyword, _ in keyword_scores}

    def get_document_keywords(self, doc_index: int, top_k: int = 20) -> List[tuple]:
        """Get top keywords for a specific document.

        Args:
            doc_index: Index of the document in the loaded documents.
            top_k: Number of top keywords to return.

        Returns:
            List of tuples (keyword, tfidf_score) sorted by score (descending).
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF model not trained. Call train() first.")

        if doc_index >= len(self.documents):
            raise ValueError(
                f"Document index {doc_index} out of range "
                f"(total documents: {len(self.documents)})"
            )

        # Get TF-IDF scores for this document
        doc_vector = self.tfidf_matrix[doc_index]
        feature_index = doc_vector.indices
        scores = doc_vector.data

        # Create list of (keyword, score) pairs
        keyword_scores = [
            (self.feature_names[idx], score)
            for idx, score in zip(feature_index, scores)
        ]

        # Sort by score (descending)
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k keywords
        return keyword_scores[:top_k]

    def get_frequency_in_freqfile(self, keyword: str) -> int:
        """Get frequency of a keyword in word_freq.txt.

        Args:
            keyword: Keyword to get frequency for.

        Returns:
            Frequency of the keyword in word_freq.txt. Returns 0 if keyword not found.
        """
        return self.domain_keywords.get(keyword, 0)


if __name__ == "__main__":
    import logging

    # Configure logging to see the process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("KeywordExtractorFromTFIDF 使用示例")
    print("=" * 80)

    # ============================================================
    # 步骤 1: 初始化 KeywordExtractor
    # ============================================================
    print("\n[步骤 1] 初始化 KeywordExtractorFromTFIDF...")
    print("-" * 80)
    keyword_extractor = KeywordExtractorFromTFIDF()

    # 检查是否已有持久化的模型
    if keyword_extractor.tfidf_matrix is not None:
        print("✅ 检测到已持久化的模型，已自动加载！")
        print(f"   - 特征数量: {len(keyword_extractor.feature_names)}")
        print("\n[步骤 2-3] 跳过训练和持久化（模型已存在）")
        print("-" * 80)
    else:
        print("ℹ️  未找到持久化的模型，需要重新训练")

        # ============================================================
        # 步骤 2: 加载文档并训练模型
        # ============================================================
        print("\n[步骤 2] 加载文档并训练模型...")
        print("-" * 80)
        markdown_dir = Path(settings.markdown_dir)
        print(f"📂 文档目录: {markdown_dir}")

        keyword_extractor.load_documents_from_directory(markdown_dir)
        print(f"✅ 已加载 {len(keyword_extractor.documents)} 个文档")

        print("\n🔧 开始训练 TF-IDF 模型...")
        keyword_extractor.train()
        print(f"✅ 训练完成！特征数量: {len(keyword_extractor.feature_names)}")

        # ============================================================
        # 步骤 3: 持久化模型
        # ============================================================
        print("\n[步骤 3] 持久化模型...")
        print("-" * 80)
        keyword_extractor.persist()
        print("✅ 模型已保存到持久化目录")

    # ============================================================
    # 步骤 4: 重新加载模型（演示）
    # ============================================================
    print("\n[步骤 4] 重新加载模型（演示）...")
    print("-" * 80)
    print("创建新的 KeywordExtractor 实例，应该会自动加载持久化的模型...")
    new_extractor = KeywordExtractorFromTFIDF()

    if new_extractor.tfidf_matrix is not None:
        print("✅ 成功从持久化文件加载模型！")
        print(f"   - 特征数量: {len(new_extractor.feature_names)}")
    else:
        print("❌ 未能加载持久化的模型")

    # ============================================================
    # 步骤 5: 使用模型提取关键词
    # ============================================================
    print("\n[步骤 5] 使用模型提取关键词...")
    print("-" * 80)

    # 使用新加载的模型进行关键词提取
    test_queries = [
        """
初始选择：你面前有三扇关闭的门 (门 1, 门 2, 门 3)，一扇后面是汽车，另外两扇后面是山羊。你随机选择一扇门（例如门 1），但先不打开。
主持人操作：主持人知道汽车在哪。他会打开你没选的另外两扇门中（门 2 或门 3）的一扇，且这扇门后一定是山羊。
关键抉择：主持人问你：“要不要换到剩下那扇未打开的门？”""",
        "如何重启解析的多模态applet 服务?",
    ]
    # tokenize the test_queries
    for query in test_queries:
        print(f"\n📝 查询: {query}")
        tokens = new_extractor.tokenize_text(query)
        print("   分词结果:")
        for token in tokens:
            print(f"     - {token}")
        print("-" * 80)
    # raise Exception("Stop here")
    for query in test_queries:
        print(f"\n📝 查询: {query}")
        keywords = new_extractor.extract_keywords(query, top_k=5)
        print("   关键词:")
        for keyword, score in keywords:
            print(f"     - {keyword}: {score:.4f}")

        # 也可以获取关键词集合
        keyword_set = new_extractor.extract_keywords_set(query, top_k=5)
        print(f"   关键词集合: {keyword_set}")

    # ============================================================
    # 步骤 6: 获取文档的关键词（可选）
    # ============================================================
    if len(new_extractor.documents) > 0:
        print("\n[步骤 6] 获取文档的关键词（示例）...")
        print("-" * 80)
        doc_keywords = new_extractor.get_document_keywords(0, top_k=10)
        print("文档 0 的 top 10 关键词:")
        for keyword, score in doc_keywords:
            print(f"  - {keyword}: {score:.4f}")

    print("\n" + "=" * 80)
    print("✅ 完整流程演示完成！")
    print("=" * 80)

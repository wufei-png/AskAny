"""Security and round-trip tests for the safe TF-IDF persistence format."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.config import settings
from askany.ingest.keyword_extract_from_tfidf import (
    KeywordExtractorFromTFIDF,
    _whitespace_tokenizer,
)


def _bare_extractor(
    vectorizer: TfidfVectorizer, matrix: object, feature_names: list[str]
) -> KeywordExtractorFromTFIDF:
    extractor = object.__new__(KeywordExtractorFromTFIDF)
    extractor.vectorizer = vectorizer
    extractor.tfidf_matrix = matrix
    extractor.feature_names = feature_names
    return extractor


def _configure_storage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "docs_keyword_storage_index", "tfidf")


def test_safe_json_npz_round_trip(monkeypatch, tmp_path: Path) -> None:
    _configure_storage(monkeypatch, tmp_path)
    vectorizer = TfidfVectorizer(
        tokenizer=_whitespace_tokenizer,
        token_pattern=None,
        lowercase=False,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(["alpha beta", "beta gamma"])
    feature_names = vectorizer.get_feature_names_out().tolist()
    extractor = _bare_extractor(vectorizer, matrix, feature_names)

    extractor.persist()
    assert extractor._get_model_file().suffix == ".json"
    assert extractor._get_model_file().exists()
    assert extractor._get_matrix_file().exists()
    assert not (extractor._get_persist_dir() / "tfidf_model.pkl").exists()

    metadata = json.loads(extractor._get_model_file().read_text(encoding="utf-8"))
    assert set(metadata) == {"version", "vocabulary", "feature_names", "idf"}

    loaded = object.__new__(KeywordExtractorFromTFIDF)
    assert loaded._load_persisted_model() is True
    np.testing.assert_allclose(
        loaded.vectorizer.transform(["alpha gamma"]).toarray(),
        vectorizer.transform(["alpha gamma"]).toarray(),
    )
    np.testing.assert_allclose(loaded.tfidf_matrix.toarray(), matrix.toarray())
    assert loaded.feature_names == feature_names


def test_old_pickle_cache_is_ignored_without_execution(
    monkeypatch, tmp_path: Path
) -> None:
    _configure_storage(monkeypatch, tmp_path)
    persist_dir = tmp_path / "tfidf"
    persist_dir.mkdir()
    marker = tmp_path / "pickle-executed"
    malicious_pickle = (
        b"cos\nsystem\n(S'echo executed > " + str(marker).encode() + b"'\ntR."
    )
    (persist_dir / "tfidf_model.pkl").write_bytes(malicious_pickle)

    extractor = object.__new__(KeywordExtractorFromTFIDF)
    assert extractor._load_persisted_model() is False
    assert not marker.exists()


def test_corrupt_json_cache_fails_closed(monkeypatch, tmp_path: Path) -> None:
    _configure_storage(monkeypatch, tmp_path)
    persist_dir = tmp_path / "tfidf"
    persist_dir.mkdir()
    (persist_dir / "tfidf_model.json").write_text("{not-json", encoding="utf-8")
    sparse.save_npz(persist_dir / "tfidf_matrix.npz", sparse.csr_matrix((1, 0)))

    extractor = object.__new__(KeywordExtractorFromTFIDF)
    assert extractor._load_persisted_model() is False

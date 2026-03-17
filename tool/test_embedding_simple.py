#!/usr/bin/env python3
"""Simple test for BGE embedding model."""

import sys
from pathlib import Path

# Add parent directory to path before importing askany
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from askany.config import settings

    model_name = settings.embedding_model
except ImportError:
    # Fallback to default model if can't import config
    model_name = "BAAI/bge-m3"
    print("⚠️  Could not import askany.config, using default model:", model_name)

print("=" * 60)
print("Testing BGE Embedding Model Loading")
print("=" * 60)

try:
    import torch
    from sentence_transformers import SentenceTransformer

    print("\n1. Checking dependencies...")
    print("   ✅ sentence-transformers: OK")
    print("   ✅ torch: OK")

    print("\n2. Checking CUDA...")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("   ℹ️  Using CPU")

    print("\n3. Loading model (this may download if not cached)...")
    print(f"   Model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print("   ✅ Model loaded successfully!")

    print("\n4. Checking model properties...")
    dim = model.get_sentence_embedding_dimension()
    print(f"   Dimension: {dim}")
    assert dim == 1024, f"Expected 1024, got {dim}"
    print("   ✅ Dimension correct (1024)")

    print("\n5. Testing encoding (multilingual support)...")
    test_texts = [
        "这是一个中文测试文本",  # Chinese
        "This is an English test text",  # English
        "这是一个混合测试: This is a mixed test",  # Mixed
    ]

    for test_text in test_texts:
        embedding = model.encode(test_text, normalize_embeddings=True)
        print(f"   Input: '{test_text[:50]}...'")
        print(f"   Output shape: {embedding.shape}")
        print(f"   Output dtype: {embedding.dtype}")
        assert embedding.shape == (1024,), f"Expected (1024,), got {embedding.shape}"

    print("   ✅ Encoding successful for all languages!")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Model is ready to use.")
    print("=" * 60)
    print("\n💡 Note: Model is automatically downloaded from HuggingFace")
    print("   Cache location: ~/.cache/huggingface/hub/")
    print("   Model size: ~1.3GB")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("   Please install: pip install sentence-transformers torch")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

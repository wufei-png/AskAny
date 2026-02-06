#!/usr/bin/env python3
"""Test script to verify BGE embedding model loading."""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    import torch

    print("✅ sentence-transformers and torch are available")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install sentence-transformers torch")
    sys.exit(1)

# Check CUDA
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ CUDA available, using GPU (device: {torch.cuda.get_device_name(0)})")
else:
    device = "cpu"
    print("ℹ️  CUDA not available, using CPU")

# Test model loading
model_name = "BAAI/bge-large-zh-v1.5"
print(f"\n📥 Loading model: {model_name}")
print("   (First time will download from HuggingFace, this may take a few minutes...)")

try:
    start_time = time.time()
    model = SentenceTransformer(model_name, device=device)
    load_time = time.time() - start_time

    print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
    print(f"   Dimension: {model.get_sentence_embedding_dimension()}")

    # Test encoding
    print("\n🧪 Testing encoding...")
    test_texts = ["测试文本", "Hello world", "这是一个中文测试"]
    start_time = time.time()
    embeddings = model.encode(test_texts, normalize_embeddings=True)
    encode_time = time.time() - start_time

    print(f"✅ Encoding successful in {encode_time:.3f} seconds")
    print(f"   Encoded {len(test_texts)} texts")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Embedding dtype: {embeddings.dtype}")

    # Verify dimension
    expected_dim = 1024
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim == expected_dim:
        print(f"✅ Dimension matches expected: {actual_dim}")
    else:
        print(f"⚠️  Dimension mismatch: expected {expected_dim}, got {actual_dim}")

    print("\n✅ All tests passed! Model is ready to use.")

except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback

    traceback.print_exc()
    print("\n💡 Tips:")
    print("   1. Check internet connection (model downloads from HuggingFace)")
    print("   2. If download fails, you can manually download:")
    print(
        f"      python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\""
    )
    print("   3. Model will be cached in ~/.cache/huggingface/")
    sys.exit(1)

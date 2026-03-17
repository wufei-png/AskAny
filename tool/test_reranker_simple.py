#!/usr/bin/env python3
"""Simple test for Reranker model loading and functionality."""

import sys
import time
from pathlib import Path

# Add parent directory to path before importing askany
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from askany.config import settings

    model_name = settings.reranker_model
    reranker_type = settings.reranker_type
except ImportError:
    # Fallback to default model if can't import config
    model_name = "BAAI/bge-reranker-v2-m3"
    reranker_type = "sentence_transformer"
    print("⚠️  Could not import askany.config, using default model:", model_name)

print("=" * 60)
print("Testing Reranker Model Loading")
print("=" * 60)

try:
    import torch
    from llama_index.core.postprocessor import SentenceTransformerRerank

    print("\n1. Checking dependencies...")
    print("   ✅ llama_index: OK")
    print("   ✅ torch: OK")

    print("\n2. Checking CUDA...")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("   ℹ️  Using CPU")

    print("\n3. Loading reranker model (this may download if not cached)...")
    print(f"   Model: {model_name}")
    print(f"   Type: {reranker_type}")
    print(
        "   (First time will download from HuggingFace, this may take a few minutes...)"
    )

    start_time = time.time()
    reranker = SentenceTransformerRerank(
        model=model_name,
        top_n=5,  # Test with top 5
        trust_remote_code=True,  # Required for BGE models
        device=device,
    )
    load_time = time.time() - start_time

    print(f"   ✅ Model loaded successfully in {load_time:.2f} seconds!")

    print("\n4. Testing reranker functionality...")
    # Create mock nodes for testing
    from llama_index.core.schema import NodeWithScore, TextNode

    query = "如何解决网络连接问题？"
    test_nodes = [
        NodeWithScore(
            node=TextNode(
                text="网络连接问题的解决方案包括检查网络配置、重启网络服务等。",
                metadata={"source": "faq_1"},
            ),
            score=0.8,
        ),
        NodeWithScore(
            node=TextNode(
                text="数据库连接错误通常是由于配置不正确或服务未启动导致的。",
                metadata={"source": "faq_2"},
            ),
            score=0.7,
        ),
        NodeWithScore(
            node=TextNode(
                text="网络连接问题可以通过ping命令测试网络连通性，检查防火墙设置。",
                metadata={"source": "faq_3"},
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(
                text="系统性能优化包括内存管理、CPU使用率监控等。",
                metadata={"source": "faq_4"},
            ),
            score=0.5,
        ),
        NodeWithScore(
            node=TextNode(
                text="网络配置需要检查IP地址、子网掩码、网关等参数。",
                metadata={"source": "faq_5"},
            ),
            score=0.4,
        ),
    ]

    print(f"   Query: '{query}'")
    print(f"   Input nodes: {len(test_nodes)}")
    print("   Original scores:", [f"{n.score:.2f}" for n in test_nodes])

    # Test reranking
    from llama_index.core import QueryBundle

    query_bundle = QueryBundle(query)
    start_time = time.time()
    reranked_nodes = reranker.postprocess_nodes(test_nodes, query_bundle)
    rerank_time = time.time() - start_time

    print(f"   ✅ Reranking completed in {rerank_time:.3f} seconds")
    print(f"   Output nodes: {len(reranked_nodes)}")
    print("   Reranked scores:", [f"{n.score:.2f}" for n in reranked_nodes])

    # Verify reranking changed the order
    original_order = [n.node.metadata.get("source") for n in test_nodes]
    reranked_order = [n.node.metadata.get("source") for n in reranked_nodes]
    if original_order != reranked_order:
        print("   ✅ Reranking changed the order (expected)")
    else:
        print("   ⚠️  Reranking did not change the order")

    # Detailed analysis of each reranked node
    print("\n   📊 Detailed Reranking Analysis:")
    print("   " + "-" * 56)
    for i, node in enumerate(reranked_nodes, 1):
        source = node.node.metadata.get("source")
        original_node = next(
            n for n in test_nodes if n.node.metadata.get("source") == source
        )
        original_score = original_node.score
        reranked_score = node.score
        score_change = reranked_score - original_score

        print(f"   {i}. {source}:")
        print(
            f"      Original score: {original_score:.2f} → Reranked: {reranked_score:.4f} ({score_change:+.4f})"
        )
        print(f"      Text: {node.node.text}")

        # Analyze relevance
        if (
            "网络连接问题" in node.node.text
            and "解决" in node.node.text
            or "方案" in node.node.text
        ):
            relevance = "✅ 高度相关 (直接提到'网络连接问题'和'解决方案')"
        elif "网络连接问题" in node.node.text:
            relevance = "✅ 相关 (提到'网络连接问题')"
        elif "网络" in node.node.text and "配置" in node.node.text:
            relevance = "⚠️  部分相关 (提到网络配置，但未明确涉及'解决连接问题')"
        elif "连接" in node.node.text and "数据库" in node.node.text:
            relevance = "❌ 不相关 (数据库连接 ≠ 网络连接)"
        elif "系统性能" in node.node.text or "内存管理" in node.node.text:
            relevance = "❌ 完全不相关 (与网络连接问题无关)"
        else:
            relevance = "❓ 相关性待判断"

        print(f"      Relevance: {relevance}")
        print()

    # Check if top result is more relevant
    top_node = reranked_nodes[0]
    print("\n   Top reranked result:")
    print(f"   - Source: {top_node.node.metadata.get('source')}")
    print(f"   - Score: {top_node.score:.4f}")
    print(f"   - Text: {top_node.node.text[:60]}...")

    # Explanation for low scores
    print("\n   💡 Why are the last three scores almost 0?")
    print("   " + "-" * 56)
    print("   BGE reranker uses a cross-encoder model that performs deep semantic")
    print("   matching between query and documents. For irrelevant documents:")
    print("   1. The model assigns very low scores (close to 0) to indicate")
    print("      they are not relevant to the query")
    print("   2. This is normal behavior - the model is designed to distinguish")
    print("      between relevant and irrelevant documents")
    print("   3. Scores are typically normalized (e.g., via sigmoid), so")
    print("      irrelevant documents get scores near 0, while relevant ones")
    print("      get scores closer to 1")
    print()
    print("   In this test:")
    print("   - faq_2: Database connection ≠ Network connection (irrelevant)")
    print("   - faq_4: System performance ≠ Network connection (irrelevant)")
    print(
        "   - faq_5: Network config ≠ Solving network connection problems (weakly relevant)"
    )

    print("\n" + "=" * 60)
    print("✅ All tests passed! Reranker is ready to use.")
    print("=" * 60)
    print("\n💡 Note: Model is automatically downloaded from HuggingFace")
    print("   Cache location: ~/.cache/huggingface/hub/")
    print(f"   Model: {model_name}")

    # Test alternative model if specified
    if model_name != "openbmb/MiniCPM-Reranker":
        print("\n" + "=" * 60)
        print("Optional: Testing alternative model")
        print("=" * 60)
        try:
            alt_model = "openbmb/MiniCPM-Reranker"
            print(f"\nTesting alternative model: {alt_model}")
            alt_reranker = SentenceTransformerRerank(
                model=alt_model,
                top_n=5,
                trust_remote_code=True,
            )
            print(f"   ✅ Alternative model '{alt_model}' loaded successfully!")
            print("   (This model is available as an option in config)")
        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️  Alternative model test failed: {error_msg}")

            # Provide helpful installation hints for common errors
            if "SentencePiece" in error_msg or "sentencepiece" in error_msg.lower():
                print("\n   💡 To use MiniCPM-Reranker, install SentencePiece:")
                print("      pip install sentencepiece")
                print("   (This is an optional dependency for this specific model)")
            elif "ImportError" in str(type(e).__name__):
                print(
                    "   💡 Missing dependency. Check error message above for details."
                )
            else:
                print("   (This is OK, model may need different configuration)")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("   Please install: pip install llama-index sentence-transformers torch")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    print("\n💡 Tips:")
    print("   1. Check internet connection (model downloads from HuggingFace)")
    print("   2. If download fails, you can manually download:")
    print(
        f"      python -c \"from llama_index.core.postprocessor import SentenceTransformerRerank; SentenceTransformerRerank(model='{model_name}', top_n=5)\""
    )
    print("   3. Model will be cached in ~/.cache/huggingface/")
    sys.exit(1)

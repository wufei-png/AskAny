# Rerank 模型选择指南

## 针对中文FAQ的Rerank推荐

### 当前配置
- **模型**: `BAAI/bge-reranker-base`
- **类型**: `SentenceTransformerRerank`
- **适用场景**: 中文FAQ检索

### 推荐选项

#### 1. BAAI/bge-reranker-base ⭐ (当前使用)
- **优点**:
  - 专门针对中文优化
  - 性能与速度平衡
  - 模型较小，推理速度快
  - 在中文检索任务上表现优秀
- **缺点**:
  - 对英文支持相对较弱
- **适用**: 纯中文FAQ，需要快速响应

#### 2. BAAI/bge-reranker-large
- **优点**:
  - 更高的准确率
  - 更好的语义理解
- **缺点**:
  - 模型更大，推理速度较慢
  - 内存占用更高
- **适用**: 对准确率要求高，可以接受较慢速度

#### 3. BAAI/bge-reranker-v2-m3
- **优点**:
  - 多语言支持（中英文混合）
  - 适合国际化场景
- **缺点**:
  - 在纯中文场景可能不如base版本
- **适用**: 中英文混合内容

#### 4. Qwen Reranker

#### 5. cross-encoder/stsb-distilroberta-base (不推荐)
- **优点**:
  - 通用模型
- **缺点**:
  - 主要针对英文优化
  - 中文支持较差
- **适用**: 仅英文FAQ

### FAQ数据特点分析

根据 `data/json/faq.json` 的数据特点：
- ✅ 中文技术FAQ
- ✅ 问题通常较短（1-2句话）
- ✅ 答案包含代码、命令、URL等技术内容
- ✅ 需要精确匹配（错误代码、硬件型号等）

**结论**: `BAAI/bge-reranker-base` 是当前最佳选择

### 配置方式

在 `.env` 文件中配置：

```bash
# 使用BGE reranker (推荐)
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_TYPE=sentence_transformer

# 或使用更大的模型（更准确但更慢）
# RERANKER_MODEL=BAAI/bge-reranker-large

# 或使用多语言模型
# RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 性能对比

| 模型 | 中文准确率 | 速度 | 内存占用 | 推荐度 |
|------|-----------|------|---------|--------|
| bge-reranker-base | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| bge-reranker-large | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| bge-reranker-v2-m3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| stsb-distilroberta | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |

### 使用建议

1. **默认场景**: 使用 `BAAI/bge-reranker-base`
2. **高精度需求**: 使用 `BAAI/bge-reranker-large`
3. **多语言场景**: 使用 `BAAI/bge-reranker-v2-m3`
4. **测试对比**: 可以在不同模型间切换测试效果


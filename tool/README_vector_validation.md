# 向量数据导出导入验证流程

本文档说明如何验证向量数据的导出和导入功能是否正常工作。通过备份原表、导入数据、比较数据等步骤，确保导出和导入过程不会丢失或损坏数据。

## 工具说明

### 1. `export_vector_data.py`
导出 PostgreSQL 中的向量数据到文件系统。

**功能：**
- 导出表结构（schema）
- 导出表数据（使用 pg_dump 自定义格式）
- 导出序列信息
- 生成元数据文件（metadata.json）

**用法：**
```bash
python tool/export_vector_data.py [--output-dir vector_data] [--format full|separate]
```

**参数：**
- `--output-dir`: 输出目录（默认：`vector_data`）
- `--format`: 导出格式，`full` 使用 pg_dump 自定义格式（默认），`separate` 分别导出 schema 和数据

### 2. `backup_and_clear.py`
备份原始表到 test 后缀的表，并清空原始表。

**功能：**
- 将原表（如 `data_askany_faq_vectors`）复制到 test 表（如 `data_askany_faq_vectors_test`）
- 复制表结构、数据和序列信息
- 清空原表数据（保留表结构）

**用法：**
```bash
python tool/backup_and_clear.py
```

**注意：**
- 如果 test 表已存在，会被删除并重新创建
- 原表数据会被清空，但表结构保持不变

### 3. `import_vector_data.py`
从文件系统导入向量数据到 PostgreSQL。

**功能：**
- 检查/创建 pgvector 扩展
- 从 dump 文件恢复表结构
- 导入表数据
- 恢复序列信息
- 验证导入结果

**用法：**
```bash
python tool/import_vector_data.py [--input-dir vector_data] [--drop-existing] [--skip-sequences]
```

**参数：**
- `--input-dir`: 输入目录（默认：`vector_data`）
- `--drop-existing`: 导入前删除已存在的表
- `--skip-sequences`: 跳过序列导入

### 4. `compare_table_data.py`
比较原始表和 test 表的数据一致性。

**功能：**
- 检查两表是否存在
- 比较行数
- 逐行比较所有数据（包括 id、text、metadata、node_id、embedding）
- 报告差异详情

**用法：**
```bash
python tool/compare_table_data.py
```

## 完整验证流程

### 步骤 1: 导出数据

首先导出当前数据库中的向量数据：

```bash
# 激活虚拟环境（如果需要）
source .venv/bin/activate

# 导出数据
python tool/export_vector_data.py --output-dir vector_data
```

**输出：**
- `vector_data/data_askany_faq_vectors.dump` - FAQ 表数据
- `vector_data/data_askany_docs_vectors.dump` - Docs 表数据
- `vector_data/sequences.sql` - 序列信息
- `vector_data/metadata.json` - 元数据

### 步骤 2: 备份原表并清空

备份原始表到 test 表，并清空原表数据：

```bash
python tool/backup_and_clear.py
```

**执行的操作：**
1. 备份 `data_askany_faq_vectors` → `data_askany_faq_vectors_test`
2. 备份 `data_askany_docs_vectors` → `data_askany_docs_vectors_test`
3. 清空 `data_askany_faq_vectors` 的数据
4. 清空 `data_askany_docs_vectors` 的数据

**重要提示：**
- 原表数据会被清空，但表结构保留
- test 表包含完整的备份数据

### 步骤 3: 导入数据

从导出文件重新导入数据到原表：

```bash
python tool/import_vector_data.py --input-dir vector_data
```

**执行的操作：**
1. 检查 pgvector 扩展
2. 从 dump 文件恢复表结构（如果不存在）
3. 导入表数据
4. 恢复序列信息
5. 验证导入结果

**注意：**
- 如果表已存在，可能会有一些警告（因为表结构已存在），但数据会正常导入
- 导入完成后会显示每个表的行数

### 步骤 4: 比较数据

比较原表和 test 表的数据，验证导入是否正确：

```bash
python tool/compare_table_data.py
```

**比较内容：**
- 行数是否一致
- 每行的所有字段是否完全匹配（id、text、metadata、node_id、embedding）

**预期结果：**
```
✅ FAQ tables match!
✅ Docs tables match!
✅ All tables match perfectly!
```

## 验证场景

### 场景 1: 验证导出导入功能

**目的：** 确保导出和导入过程不会丢失或损坏数据。

**步骤：**
1. 导出数据 → `export_vector_data.py`
2. 备份原表 → `backup_and_clear.py`
3. 导入数据 → `import_vector_data.py`
4. 比较数据 → `compare_table_data.py`

**成功标准：** 所有表的数据完全匹配

### 场景 2: 数据迁移验证

**目的：** 验证从一个数据库迁移到另一个数据库时数据完整性。

**步骤：**
1. 在源数据库导出数据
2. 在目标数据库导入数据
3. 使用 `compare_table_data.py` 比较（需要手动指定两个数据库的表）

### 场景 3: 备份恢复验证

**目的：** 验证备份和恢复流程。

**步骤：**
1. 导出数据作为备份
2. 模拟数据丢失（清空表）
3. 从备份恢复数据
4. 比较恢复前后的数据

## 故障排查

### 问题 1: 导入时表已存在错误

**症状：**
```
ERROR: relation "data_askany_faq_vectors" already exists
```

**解决方案：**
- 使用 `--drop-existing` 参数删除已存在的表
- 或者先手动清空表数据（使用 `backup_and_clear.py`）

### 问题 2: 数据不匹配

**症状：**
```
❌ Found X differences
```

**排查步骤：**
1. 检查导出文件是否完整
2. 检查导入过程是否有错误
3. 查看差异详情，确定哪些字段不匹配
4. 检查是否有并发写入操作

### 问题 3: 序列值不匹配

**症状：**
```
Error executing sequence command: relation "xxx_id_seq" does not exist
```

**解决方案：**
- 检查 `sequences.sql` 文件中的序列名是否正确
- 确保序列在导入时正确创建

## 注意事项

1. **数据安全：**
   - 在生产环境操作前，请先备份数据库
   - `backup_and_clear.py` 会清空原表数据，请谨慎使用

2. **性能考虑：**
   - 大表（数万行以上）的导出导入可能需要较长时间
   - 比较大量数据时可能需要几分钟

3. **依赖要求：**
   - 需要 PostgreSQL 客户端工具（pg_dump, pg_restore）
   - 需要 psycopg2 Python 库
   - 需要 pgvector 扩展

4. **表名约定：**
   - 原表：`data_askany_faq_vectors`, `data_askany_docs_vectors`
   - test 表：`data_askany_faq_vectors_test`, `data_askany_docs_vectors_test`

## 示例输出

### 导出数据
```
2025-11-24 01:05:20 - INFO - ✅ Exported full table data_askany_faq_vectors to vector_data/data_askany_faq_vectors.dump
2025-11-24 01:05:20 - INFO - ✅ Exported full table data_askany_docs_vectors to vector_data/data_askany_docs_vectors.dump
2025-11-24 01:05:20 - INFO - ✅ Exported sequence information to vector_data/sequences.sql
```

### 备份和清空
```
2025-11-24 01:15:12 - INFO - ✅ Copied 6 rows from data_askany_faq_vectors to data_askany_faq_vectors_test
2025-11-24 01:15:15 - INFO - ✅ Copied 2519 rows from data_askany_docs_vectors to data_askany_docs_vectors_test
2025-11-24 01:15:15 - INFO - ✅ Cleared 6 rows from data_askany_faq_vectors (now has 0 rows)
2025-11-24 01:15:15 - INFO - ✅ Cleared 2519 rows from data_askany_docs_vectors (now has 0 rows)
```

### 导入数据
```
2025-11-24 01:15:42 - INFO - ✅ Imported table data_askany_faq_vectors from vector_data/data_askany_faq_vectors.dump
2025-11-24 01:15:45 - INFO - ✅ Imported table data_askany_docs_vectors from vector_data/data_askany_docs_vectors.dump
2025-11-24 01:15:45 - INFO - ✅ data_askany_faq_vectors: 6 rows
2025-11-24 01:15:45 - INFO - ✅ data_askany_docs_vectors: 2519 rows
```

### 比较数据
```
2025-11-24 01:15:53 - INFO - 📊 data_askany_faq_vectors: 6 rows
2025-11-24 01:15:53 - INFO - 📊 data_askany_faq_vectors_test: 6 rows
2025-11-24 01:15:53 - INFO - ✅ All 6 rows match perfectly!
2025-11-24 01:15:53 - INFO - ✅ FAQ tables match!
2025-11-24 01:15:53 - INFO - 📊 data_askany_docs_vectors: 2519 rows
2025-11-24 01:15:53 - INFO - 📊 data_askany_docs_vectors_test: 2519 rows
2025-11-24 01:15:53 - INFO - ✅ All 2519 rows match perfectly!
2025-11-24 01:15:53 - INFO - ✅ Docs tables match!
2025-11-24 01:15:53 - INFO - ✅ All tables match perfectly!
```

## 相关文件

- `export_vector_data.py` - 导出工具
- `backup_and_clear.py` - 备份和清空工具
- `import_vector_data.py` - 导入工具
- `compare_table_data.py` - 数据比较工具
- `README_vector_data.md` - 导出导入功能说明


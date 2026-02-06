# Vector Data Export/Import Scripts

这两个脚本用于导出和导入PostgreSQL中的pgvector向量数据。

## 脚本说明

### 1. export_vector_data.py

将PostgreSQL中的向量表数据导出到 `vector_data` 目录。

**功能：**
- 导出所有向量表（FAQ、Docs、Legacy）的表结构和数据
- 导出序列（sequence）信息
- 导出元数据（配置信息、表统计等）
- 支持两种格式：
  - `full`: 使用pg_dump自定义格式（推荐，压缩更好）
  - `separate`: 分别导出schema和data文件

**使用方法：**

```bash
# 使用默认设置（导出到vector_data目录，使用full格式）
python tool/export_vector_data.py

# 指定输出目录
python tool/export_vector_data.py --output-dir /path/to/vector_data

# 使用separate格式（分别导出schema和data）
python tool/export_vector_data.py --format separate
```

**导出文件：**
- `metadata.json`: 元数据信息（数据库配置、表统计等）
- `data_askany_faq_vectors.dump` 或 `data_askany_faq_vectors_schema.sql` + `data_askany_faq_vectors_data.csv`: FAQ表数据
- `data_askany_docs_vectors.dump` 或 `data_askany_docs_vectors_schema.sql` + `data_askany_docs_vectors_data.csv`: Docs表数据
- `data_askany_vectors.dump` 或 `data_askany_vectors_schema.sql` + `data_askany_vectors_data.csv`: Legacy表数据（如果存在）
- `sequences.sql`: 序列信息

### 2. import_vector_data.py

从 `vector_data` 目录导入向量数据到PostgreSQL数据库。

**功能：**
- 自动检查/创建pgvector扩展
- 导入表结构和数据
- 恢复序列值
- 支持删除现有表后导入（--drop-existing）

**使用方法：**

```bash
# 使用默认设置（从vector_data目录导入）
python tool/import_vector_data.py

# 指定输入目录
python tool/import_vector_data.py --input-dir /path/to/vector_data

# 删除现有表后导入（谨慎使用！）
python tool/import_vector_data.py --drop-existing

# 跳过序列导入
python tool/import_vector_data.py --skip-sequences
```

## 完整工作流程

### 导出数据（在源机器上）

```bash
# 1. 导出所有向量数据
python tool/export_vector_data.py

# 2. 检查导出结果
ls -lh vector_data/

# 3. 打包（可选）
tar -czf vector_data_backup.tar.gz vector_data/
```

### 导入数据（在目标机器上）

```bash
# 1. 解压（如果打包了）
tar -xzf vector_data_backup.tar.gz

# 2. 确保PostgreSQL已安装并运行
# 3. 确保pgvector扩展可用（脚本会自动创建）

# 4. 导入数据
python tool/import_vector_data.py --input-dir vector_data

# 5. 验证导入
python tool/ingest_check.py
```

## 注意事项

1. **数据库连接**：脚本使用 `askany/config.py` 中的数据库配置，确保配置正确。

2. **pgvector扩展**：导入脚本会自动创建pgvector扩展，但需要确保PostgreSQL已安装pgvector。

3. **表名格式**：PGVectorStore使用 `data_{table_name}` 作为实际表名，脚本会自动处理。

4. **向量数据格式**：
   - `full`格式使用pg_dump自定义格式，保留完整的表结构和索引信息
   - `separate`格式将向量导出为文本，导入时需要重新解析

5. **权限要求**：
   - 导出：需要SELECT权限
   - 导入：需要CREATE、INSERT权限

6. **数据大小**：向量数据可能很大，确保有足够的磁盘空间。

7. **备份建议**：在生产环境使用前，建议先备份数据库。

## 故障排除

### 导出失败

- 检查PostgreSQL连接配置
- 确保有足够的磁盘空间
- 检查pg_dump是否安装

### 导入失败

- 检查PostgreSQL是否运行
- 检查pgvector扩展是否可用
- 检查表是否已存在（使用--drop-existing删除）
- 检查文件权限

### 向量维度不匹配

如果导入后查询失败，检查：
- `vector_dimension`配置是否与导出时一致
- embedding模型是否相同


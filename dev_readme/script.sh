# 入库
# Set encoding environment variables to ensure UTF-8 encoding
# PYTHONIOENCODING: Sets Python's stdout/stderr encoding
# LC_ALL: Sets locale to UTF-8 for all categories
# NO_COLOR: Disables ANSI color codes in output
# TERM: Set to dumb to disable terminal-specific features (only for Python commands)
# This prevents encoding issues with Chinese characters and ANSI codes in log files
# Note: Environment variables are set inline with commands to avoid affecting the shell session

# Ingest documents
PYTHONIOENCODING=utf-8 LC_ALL=en_US.UTF-8 NO_COLOR=1 TERM=dumb python -u -m askany.main --ingest > ingest.log 2>&1 

# Start server
PYTHONIOENCODING=utf-8 LC_ALL=en_US.UTF-8 NO_COLOR=1 TERM=dumb python -u -m askany.main --serve > serve.log 2>&1 
 PYTHONIOENCODING=utf-8  python askany/main.py --serve > serve.log 2>&1
python tool/query_test.py > query.log 2>&1
# 检查入库数据
python tool/ingest_check.py > ingest_check.log 2>&1
#检查pqvector数据量

cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT 'data_askany_faq_vectors' as table_name, COUNT(*) as row_count FROM data_askany_faq_vectors UNION ALL SELECT 'data_askany_docs_vectors', COUNT(*) FROM data_askany_docs_vectors;"


cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT 'data_askany_faq_vectors' as table_name, COUNT(*) as row_count FROM data_askany_faq_vectors UNION ALL SELECT 'data_hybrid_askany_docs_vectors', COUNT(*) FROM data_hybrid_askany_docs_vectors;"
        table_name        | row_count 
--------------------------+-----------
 data_askany_faq_vectors  |         6
 data_askany_docs_vectors |      2519
(2 rows)
#查看数据
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT id, LEFT(text, 200) as text_preview, metadata_, node_id FROM data_askany_docs_vectors LIMIT 1;"

#查看表和索引表大小：
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT pg_size_pretty(pg_relation_size('data_askany3_docs_vectors'::regclass)) as table_size, pg_size_pretty(pg_relation_size('data_askany3_docs_vectors_embedding_idx'::regclass)) as index_size;"
 table_size | index_size 
------------+------------
 24 kB      | 272 kB
(1 row)


cd AskAny && \
PGPASSWORD=123456 psql -h localhost -U root -d askany -c "
SELECT *
FROM data_askany_docs_vectors
WHERE text ILIKE '%xxx%';
"

# 查看 HNSW 索引信息
echo "=== HNSW 索引基本信息 ==="
cd AskAny && \
PGPASSWORD=123456 psql -h localhost -U root -d askany -c "
SELECT 
    i.relname as index_name,
    t.relname as table_name,
    pg_size_pretty(pg_relation_size(i.oid)) as index_size,
    am.amname as index_type,
    idx.indisunique as is_unique,
    idx.indisprimary as is_primary
FROM pg_index idx
JOIN pg_class i ON i.oid = idx.indexrelid
JOIN pg_class t ON t.oid = idx.indrelid
JOIN pg_am am ON am.oid = i.relam
WHERE t.relname IN ('data_askany_faq_vectors', 'data_askany_docs_vectors')
  AND i.relname LIKE '%embedding%'
ORDER BY t.relname, i.relname;
"

# 查看 HNSW 索引配置参数
echo "=== HNSW 索引配置参数 ==="
cd AskAny && \
PGPASSWORD=123456 psql -h localhost -U root -d askany -c "
SELECT 
    i.relname as index_name,
    t.relname as table_name,
    pg_indexes_size(t.oid) as total_indexes_size,
    pg_relation_size(i.oid) as index_size_bytes,
    (SELECT option_value FROM pg_options_to_table(i.reloptions) WHERE option_name = 'hnsw.m') as hnsw_m,
    (SELECT option_value FROM pg_options_to_table(i.reloptions) WHERE option_name = 'hnsw.ef_construction') as hnsw_ef_construction,
    (SELECT option_value FROM pg_options_to_table(i.reloptions) WHERE option_name = 'hnsw.ef_search') as hnsw_ef_search,
    (SELECT option_value FROM pg_options_to_table(i.reloptions) WHERE option_name = 'hnsw.dist_method') as hnsw_dist_method
FROM pg_index idx
JOIN pg_class i ON i.oid = idx.indexrelid
JOIN pg_class t ON t.oid = idx.indrelid
WHERE t.relname IN ('data_askany_faq_vectors', 'data_askany_docs_vectors')
  AND i.relname LIKE '%embedding%'
ORDER BY t.relname, i.relname;
"

# 查看 HNSW 索引统计信息
echo "=== HNSW 索引统计信息 ==="
cd AskAny && \
PGPASSWORD=123456 psql -h localhost -U root -d askany -c "
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE tablename IN ('data_askany_faq_vectors', 'data_askany_docs_vectors')
  AND indexname LIKE '%embedding%'
ORDER BY tablename, indexname;
"

# 查看索引的详细选项（包括 HNSW 参数）
echo "=== HNSW 索引详细选项 ==="
cd AskAny && \
PGPASSWORD=123456 psql -h localhost -U root -d askany -c "
SELECT 
    i.relname as index_name,
    t.relname as table_name,
    array_to_string(i.reloptions, ', ') as index_options
FROM pg_index idx
JOIN pg_class i ON i.oid = idx.indexrelid
JOIN pg_class t ON t.oid = idx.indrelid
WHERE t.relname IN ('data_askany_faq_vectors', 'data_askany_docs_vectors')
  AND i.relname LIKE '%embedding%'
ORDER BY t.relname, i.relname;
"

# 查询指定表的 HNSW 索引层级信息（通过查询索引的元数据）
# 注意：pgvector 的 HNSW 索引内部结构不直接暴露，但可以通过索引大小和配置来推断
echo "=== HNSW 索引层级估算（基于索引大小和配置） ==="
cd AskAny && \
PGPASSWORD=123456 psql -h localhost -U root -d askany -c "
WITH index_info AS (
    SELECT 
        i.relname as index_name,
        t.relname as table_name,
        pg_relation_size(i.oid) as index_size_bytes,
        (SELECT COUNT(*) FROM pg_attribute WHERE attrelid = t.oid AND attname = 'embedding') as has_embedding_col,
        (SELECT COUNT(*)::bigint FROM pg_stat_user_tables WHERE relname = t.relname) as row_count_estimate
    FROM pg_index idx
    JOIN pg_class i ON i.oid = idx.indexrelid
    JOIN pg_class t ON t.oid = idx.indrelid
    WHERE t.relname IN ('data_askany_faq_vectors', 'data_askany_docs_vectors')
      AND i.relname LIKE '%embedding%'
)
SELECT 
    index_name,
    table_name,
    pg_size_pretty(index_size_bytes) as index_size,
    index_size_bytes,
    CASE 
        WHEN index_size_bytes > 0 THEN 
            ROUND(index_size_bytes::numeric / 1024.0 / 1024.0, 2)
        ELSE 0
    END as index_size_mb,
    'HNSW 索引层级结构无法直接查询，需要通过 pgvector 扩展函数或应用层访问' as note
FROM index_info
ORDER BY table_name, index_name;
"

# 使用 Python 脚本查询 HNSW 索引结构和层级信息
echo "=== 使用 Python 脚本查询 HNSW 索引结构 ==="
cd AskAny && \
PYTHONIOENCODING=utf-8 LC_ALL=en_US.UTF-8 NO_COLOR=1 TERM=dumb python -u tool/query_hnsw_structure.py --all

# 查询指定层级的 HNSW 索引结构（示例：查询第 0 层）
echo "=== 查询 HNSW 索引第 0 层结构（估算） ==="
cd AskAny && \
PYTHONIOENCODING=utf-8 LC_ALL=en_US.UTF-8 NO_COLOR=1 TERM=dumb python -u tool/query_hnsw_structure.py --all --layer 0

# 查询指定层级的 HNSW 索引结构（示例：查询第 1 层）
echo "=== 查询 HNSW 索引第 1 层结构（估算） ==="
cd AskAny && \
PYTHONIOENCODING=utf-8 LC_ALL=en_US.UTF-8 NO_COLOR=1 TERM=dumb python -u tool/query_hnsw_structure.py --all --layer 1


cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"

# 删除 askany2_docs_vectors 表的数据
# 先查看删除前的数据量
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT COUNT(*) as row_count_before_delete FROM data_askany3_docs_vectors;"
# 执行删除
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "DELETE FROM data_askany3_docs_vectors;"
# 验证删除后的数据量
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT COUNT(*) as row_count_after_delete FROM data_askany3_docs_vectors;"
# 清理索引空间（重建索引以释放空间）
# 方案1：重建索引（推荐，保留索引结构，释放空间）
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "REINDEX TABLE data_askany3_docs_vectors;"
# 方案2：如果不再使用该表，可以删除索引（取消注释下面的命令）
# cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "DROP INDEX IF EXISTS data_askany3_docs_vectors_embedding_idx, askany2_docs_vectors_idx_1;"
# 查看清理后的索引大小
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT i.relname as index_name, pg_size_pretty(pg_relation_size(i.oid)) as index_size FROM pg_index idx JOIN pg_class i ON i.oid = idx.indexrelid JOIN pg_class t ON t.oid = idx.indrelid WHERE t.relname = 'data_askany3_docs_vectors' ORDER BY pg_relation_size(i.oid) DESC;"
# 查看HNSW索引配置（m和ef_construction参数）- 指定表
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT i.relname as index_name, pg_get_indexdef(i.oid) as index_definition FROM pg_index idx JOIN pg_class i ON i.oid = idx.indexrelid JOIN pg_class t ON t.oid = idx.indrelid WHERE t.relname = 'data_askany3_docs_vectors' AND i.relname LIKE '%embedding%' ORDER BY i.relname;"
# 查看所有HNSW索引配置（m和ef_construction参数）- 所有表
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT t.relname as table_name, i.relname as index_name, pg_get_indexdef(i.oid) as index_definition FROM pg_index idx JOIN pg_class i ON i.oid = idx.indexrelid JOIN pg_class t ON t.oid = idx.indrelid WHERE idx.indisvalid = true AND pg_get_indexdef(i.oid) LIKE '%hnsw%' ORDER BY t.relname, i.relname;"
# 重建data_askany3_docs_vectors表的HNSW索引（使用config.py中的配置：m=16, ef_construction=128）
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "DROP INDEX IF EXISTS data_askany3_docs_vectors_embedding_idx;"
cd AskAny && PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SET maintenance_work_mem = '512MB'; CREATE INDEX data_askany3_docs_vectors_embedding_idx ON public.data_askany3_docs_vectors USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);"
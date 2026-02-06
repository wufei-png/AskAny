-- Delete migrated data from data_askany_docs_vectors (MOVE operation)
-- This script deletes data where id > 3720 that has already been migrated to data_hybrid_askany_docs_vectors
-- WARNING: This will permanently delete data from data_askany_docs_vectors
-- Only execute this after verifying the migration was successful

-- Step 1: Verify data exists in target table before deletion
SELECT 
    'Target table (data_hybrid_askany_docs_vectors)' as table_name,
    COUNT(*) as row_count,
    MIN(id) as min_id,
    MAX(id) as max_id
FROM data_hybrid_askany_docs_vectors
WHERE id > 3720;

-- Step 2: Check how many rows will be deleted from source table
SELECT 
    'Source table (data_askany_docs_vectors)' as table_name,
    COUNT(*) as rows_to_delete,
    MIN(id) as min_id,
    MAX(id) as max_id
FROM data_askany_docs_vectors
WHERE id > 3720;

-- Step 3: Show sample rows that will be deleted
SELECT 
    id,
    (metadata_->>'last_updated')::float as last_updated_timestamp,
    to_timestamp((metadata_->>'last_updated')::float) as last_updated_date,
    LEFT(text, 100) as text_preview
FROM data_askany_docs_vectors
WHERE id > 3720
LIMIT 10;

-- Step 4: Delete migrated data from source table
-- UNCOMMENT THE FOLLOWING LINE TO EXECUTE THE DELETION:
-- DELETE FROM data_askany_docs_vectors WHERE id > 3720;

-- Step 5: Verify deletion (after executing DELETE)
-- SELECT 
--     'Source table (after deletion)' as table_name,
--     COUNT(*) as row_count
-- FROM data_askany_docs_vectors
-- UNION ALL
-- SELECT 
--     'Target table' as table_name,
--     COUNT(*) as row_count
-- FROM data_hybrid_askany_docs_vectors;

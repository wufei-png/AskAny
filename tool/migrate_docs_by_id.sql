-- Migrate docs vectors data from data_askany_docs_vectors to data_hybrid_askany_docs_vectors
-- This script migrates data where id > 3720 (newly added data after the initial 2519 rows)

-- Step 1: Check how many rows will be migrated
SELECT 
    COUNT(*) as rows_to_migrate,
    MIN(id) as min_id,
    MAX(id) as max_id
FROM data_askany_docs_vectors
WHERE id > 3720;

-- Step 2: Show sample rows that will be migrated
SELECT 
    id,
    (metadata_->>'last_updated')::float as last_updated_timestamp,
    to_timestamp((metadata_->>'last_updated')::float) as last_updated_date,
    LEFT(text, 100) as text_preview
FROM data_askany_docs_vectors
WHERE id > 3720
LIMIT 10;

-- Step 3: Create the new table (drop if exists)
DROP TABLE IF EXISTS data_hybrid_askany_docs_vectors CASCADE;

-- Step 4: Create table with same structure as source
CREATE TABLE data_hybrid_askany_docs_vectors (LIKE data_askany_docs_vectors INCLUDING ALL);

-- Step 5: Migrate data (rows with id > 3720)
INSERT INTO data_hybrid_askany_docs_vectors
SELECT * 
FROM data_askany_docs_vectors
WHERE id > 3720;

-- Step 6: Verify migration
SELECT 
    'Source table (all)' as table_name,
    COUNT(*) as row_count
FROM data_askany_docs_vectors
UNION ALL
SELECT 
    'Source table (id <= 3720)' as table_name,
    COUNT(*) as row_count
FROM data_askany_docs_vectors
WHERE id <= 3720
UNION ALL
SELECT 
    'Target table (id > 3720)' as table_name,
    COUNT(*) as row_count
FROM data_hybrid_askany_docs_vectors;

-- Step 7: Create sequence for target table
DO $$
DECLARE
    max_id bigint;
    seq_name text := 'data_hybrid_askany_docs_vectors_id_seq';
BEGIN
    SELECT COALESCE(MAX(id), 0) INTO max_id FROM data_hybrid_askany_docs_vectors;
    
    -- Check if sequence exists
    IF EXISTS (SELECT 1 FROM pg_sequences WHERE sequencename = seq_name) THEN
        -- Update existing sequence
        PERFORM setval(seq_name, max_id + 1, true);
        RAISE NOTICE 'Updated sequence % to start from %', seq_name, max_id + 1;
    ELSE
        -- Create new sequence
        EXECUTE format('CREATE SEQUENCE %I START WITH %s', seq_name, max_id + 1);
        RAISE NOTICE 'Created sequence % starting from %', seq_name, max_id + 1;
    END IF;
END $$;

-- Step 8: Delete migrated data from source table (MOVE operation)
-- WARNING: This will permanently delete data from data_askany_docs_vectors
-- Only execute this after verifying the migration was successful
DELETE FROM data_askany_docs_vectors
WHERE id > 3720;

-- Step 9: Verify deletion
SELECT 
    'Source table (after deletion)' as table_name,
    COUNT(*) as row_count
FROM data_askany_docs_vectors
UNION ALL
SELECT 
    'Target table' as table_name,
    COUNT(*) as row_count
FROM data_hybrid_askany_docs_vectors;

-- Step 10: Show final statistics
SELECT 
    'Migration Summary' as summary,
    (SELECT COUNT(*) FROM data_hybrid_askany_docs_vectors) as rows_in_target,
    (SELECT COUNT(*) FROM data_askany_docs_vectors) as rows_remaining_in_source;




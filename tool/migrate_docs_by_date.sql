-- Migrate docs vectors data from data_askany_docs_vectors to data_hybrid_askany_docs_vectors
-- This script migrates data where last_updated >= December 1, 2024 (timestamp: 1732982400)

-- Step 1: Check how many rows will be migrated
SELECT 
    COUNT(*) as rows_to_migrate,
    MIN((metadata_->>'last_updated')::float) as min_timestamp,
    MAX((metadata_->>'last_updated')::float) as max_timestamp
FROM data_askany_docs_vectors
WHERE (metadata_->>'last_updated')::float >= 1732982400.0;

-- Step 2: Show sample rows that will be migrated
SELECT 
    id,
    (metadata_->>'last_updated')::float as last_updated_timestamp,
    to_timestamp((metadata_->>'last_updated')::float) as last_updated_date,
    LEFT(text, 100) as text_preview
FROM data_askany_docs_vectors
WHERE (metadata_->>'last_updated')::float >= 1732982400.0
LIMIT 10;

-- Step 3: Create the new table (drop if exists)
DROP TABLE IF EXISTS data_hybrid_askany_docs_vectors CASCADE;

-- Step 4: Create table with same structure as source
CREATE TABLE data_hybrid_askany_docs_vectors (LIKE data_askany_docs_vectors INCLUDING ALL);

-- Step 5: Migrate data (rows with last_updated >= 2024-12-01)
INSERT INTO data_hybrid_askany_docs_vectors
SELECT * 
FROM data_askany_docs_vectors
WHERE (metadata_->>'last_updated')::float >= 1732982400.0;

-- Step 6: Verify migration
SELECT 
    'Source table' as table_name,
    COUNT(*) as row_count
FROM data_askany_docs_vectors
UNION ALL
SELECT 
    'Target table' as table_name,
    COUNT(*) as row_count
FROM data_hybrid_askany_docs_vectors;

-- Step 7: Create sequence for target table (if needed)
-- Get the max ID from migrated data
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

-- Step 8: Show final statistics
SELECT 
    'Migration Summary' as summary,
    (SELECT COUNT(*) FROM data_askany_docs_vectors WHERE (metadata_->>'last_updated')::float >= 1732982400.0) as rows_migrated,
    (SELECT COUNT(*) FROM data_hybrid_askany_docs_vectors) as rows_in_target,
    (SELECT COUNT(*) FROM data_askany_docs_vectors WHERE (metadata_->>'last_updated')::float < 1732982400.0) as rows_remaining_in_source;




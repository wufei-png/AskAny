#!/bin/bash
# Check insertion progress for docs vector store
# Usage: ./check_insert_progress.sh [--monitor] [--interval SECONDS]

TABLE_NAME="data_askany2_docs_vectors"
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-root}"
PGPASSWORD="${PGPASSWORD:-123456}"
PGDATABASE="${PGDATABASE:-askany}"

export PGPASSWORD

check_progress() {
    echo ""
    echo "============================================================"
    echo "Insertion Progress for $TABLE_NAME"
    echo "============================================================"
    
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "
        SELECT 
            COUNT(*) as current_rows,
            MAX(id) as max_id,
            pg_size_pretty(pg_relation_size('$TABLE_NAME'::regclass)) as table_size
        FROM $TABLE_NAME;
    " 2>/dev/null
    
    # Check index size if exists
    INDEX_NAME="${TABLE_NAME}_embedding_idx"
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "
        SELECT pg_size_pretty(pg_relation_size('$INDEX_NAME'::regclass)) as index_size;
    " 2>/dev/null || echo "Index size: N/A"
    
    echo "============================================================"
    echo ""
}

monitor_progress() {
    local interval=${1:-10}
    echo "Monitoring insertion progress (checking every $interval seconds)..."
    echo "Press Ctrl+C to stop"
    echo ""
    
    local prev_rows=0
    local start_time=$(date +%s)
    
    while true; do
        check_progress
        
        # Get current row count
        local current_rows=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -t -c "SELECT COUNT(*) FROM $TABLE_NAME;" 2>/dev/null | tr -d ' ')
        
        if [ -n "$current_rows" ] && [ "$prev_rows" -gt 0 ]; then
            local rows_diff=$((current_rows - prev_rows))
            local elapsed=$(($(date +%s) - start_time))
            
            if [ "$rows_diff" -gt 0 ]; then
                local rate=$(echo "scale=2; $rows_diff / $interval" | bc)
                echo "Progress: +$rows_diff rows in last ${interval}s ($rate rows/sec)"
            else
                echo "Progress: No new rows detected (insertion may be stuck or completed)"
            fi
        fi
        
        prev_rows=$current_rows
        sleep "$interval"
    done
}

# Parse arguments
if [ "$1" = "--monitor" ] || [ "$1" = "-m" ]; then
    interval=${2:-10}
    monitor_progress "$interval"
else
    check_progress
fi

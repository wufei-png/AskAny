#!/usr/bin/env python3
"""Test direct database query to verify data exists."""

import psycopg2
from askany.config import settings

def test_db_query():
    """Test direct database query."""
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
    )
    
    cursor = conn.cursor()
    
    # Check FAQ table
    print("=== FAQ Table ===")
    cursor.execute(f"SELECT COUNT(*) FROM {settings.faq_vector_table_name}")
    faq_count = cursor.fetchone()[0]
    print(f"FAQ count: {faq_count}")
    
    if faq_count > 0:
        cursor.execute(f"SELECT id, text, embedding IS NOT NULL as has_embedding FROM {settings.faq_vector_table_name} LIMIT 3")
        rows = cursor.fetchall()
        for row in rows:
            print(f"  ID: {row[0]}, Text: {row[1][:50]}..., Has embedding: {row[2]}")
    
    # Check Docs table
    print("\n=== Docs Table ===")
    cursor.execute(f"SELECT COUNT(*) FROM {settings.docs_vector_table_name}")
    docs_count = cursor.fetchone()[0]
    print(f"Docs count: {docs_count}")
    
    if docs_count > 0:
        cursor.execute(f"SELECT id, text, embedding IS NOT NULL as has_embedding FROM {settings.docs_vector_table_name} LIMIT 3")
        rows = cursor.fetchall()
        for row in rows:
            print(f"  ID: {row[0]}, Text: {row[1][:50]}..., Has embedding: {row[2]}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    test_db_query()

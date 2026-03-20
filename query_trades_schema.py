
import sqlite3
import os

db_path = "data/trades.db"
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get column names
    cursor.execute("PRAGMA table_info(trades)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Columns: {columns}")
    
    # Query last 10 trades
    # We want invested (volume/lot * price?) and return (profit)
    # Let's see what columns we have first
    cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    
    conn.close()

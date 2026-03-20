
import sqlite3
import os

db_path = "data/trades.db"
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    # Use DictCursor-like row factory
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query last 10 closed trades
    cursor.execute("""
        SELECT ticket, symbol, volume, entry_price, exit_price, profit, time_closed 
        FROM trades 
        WHERE status = 'closed' 
        ORDER BY time_closed DESC 
        LIMIT 10
    """)
    rows = cursor.fetchall()
    
    print("--- LAST 10 CLOSED TRADES ---")
    for row in rows:
        # invested = volume * entry_price (for display purposes, or just volume)
        # return = profit
        print(f"Ticket: {row['ticket']}, Symbol: {row['symbol']}, Volume: {row['volume']}, Entry: {row['entry_price']}, Exit: {row['exit_price']}, Profit: {row['profit']}, Closed: {row['time_closed']}")
    
    conn.close()

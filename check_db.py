
import sqlite3
import os

db_path = "data/trades.db"
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("--- OPEN TRADES IN DATABASE ---")
    cursor.execute("SELECT ticket, symbol, status, stop_loss, take_profit FROM trades WHERE status = 'open'")
    rows = cursor.fetchall()
    for row in rows:
        print(f"Ticket: {row[0]}, Symbol: {row[1]}, Status: {row[2]}, SL: {row[3]}, TP: {row[4]}")
        
    print("\n--- RECENT CLOSED TRADES ---")
    cursor.execute("SELECT ticket, symbol, status, stop_loss, take_profit FROM trades WHERE status = 'closed' ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(f"Ticket: {row[0]}, Symbol: {row[1]}, Status: {row[2]}, SL: {row[3]}, TP: {row[4]}")

    conn.close()

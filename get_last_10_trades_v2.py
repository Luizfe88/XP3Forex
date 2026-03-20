
import sqlite3
import os

db_path = "data/trades.db"
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query last 10 closed trades
    cursor.execute("""
        SELECT ticket, symbol, volume, entry_price, exit_price, profit, exit_time 
        FROM trades 
        WHERE status = 'closed' 
        ORDER BY exit_time DESC 
        LIMIT 10
    """)
    rows = cursor.fetchall()
    
    print("--- ULTIMAS 10 TRADES FECHADAS ---")
    for row in rows:
        # Invested logic: usually volume * contract_size * entry_price / leverage
        # Since I don't have contract size or leverage here, I'll show Volume and Entry Price
        # and maybe calculate a "Value" as volume * entry_price if it makes sense (e.g. for Crypto)
        # For Forex, it's more complex, but I'll provide what's in the DB.
        print(f"Ticket: {row['ticket']} | Ativo: {row['symbol']} | Volume: {row['volume']} | Entrada: {row['entry_price']:.5f} | Saída: {row['exit_price']:.5f} | Retorno: {row['profit']:.2f} | Data Fim: {row['exit_time']}")
    
    conn.close()


import sqlite3
import os

db_path = "data/trades.db"
output_path = "last_10_trades_report.txt"

if not os.path.exists(db_path):
    with open(output_path, "w") as f:
        f.write(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticket, symbol, volume, entry_price, exit_price, profit, exit_time 
        FROM trades 
        WHERE status = 'closed' 
        ORDER BY exit_time DESC 
        LIMIT 10
    """)
    rows = cursor.fetchall()
    
    with open(output_path, "w") as f:
        f.write(f"{'Ticket':<12} | {'Ativo':<8} | {'Volume':<6} | {'Entrada':<10} | {'Saída':<10} | {'Retorno':<10} | {'Data Fim'}\n")
        f.write("-" * 90 + "\n")
        for row in rows:
            line = f"{row['ticket']:<12} | {row['symbol']:<8} | {row['volume']:<6} | {row['entry_price']:<10.5f} | {row['exit_price']:<10.5f} | {row['profit']:<10.2f} | {row['exit_time']}\n"
            f.write(line)
    
    conn.close()
    print(f"Report saved to {output_path}")

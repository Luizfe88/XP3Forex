"""
🔄 RECOVERY SCRIPT: XP3 DB SYNC
Sincroniza trades marcados como 'open' no banco de dados que já foram fechados no MT5.
"""

import MetaTrader5 as mt5
import sqlite3
import os
from datetime import datetime
from xp3_forex.core.settings import settings
from xp3_forex.utils.data_utils import update_trade_exit, update_performance_table

def run_recovery():
    print("Iniciando Recupeacao Retroativa do Banco de Dados...")
    
    # 1. Conectar ao MT5 usando a sessao ja logada no terminal
    if not mt5.initialize():
        print(f"Falha CRITICAL ao conectar ao MetaTrader 5: {mt5.last_error()}")
        return

    # 2. Buscar trades 'open' no DB
    db_path = "data/trades.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT ticket, symbol FROM trades WHERE status = 'open'")
    open_trades = cursor.fetchall()
    conn.close()
    
    print(f"Encontrados {len(open_trades)} trades com status 'open' no banco de dados.")
    if len(open_trades) > 0:
        print(f"DEBUG: Primeiros tickets para analise: {[t[0] for t in open_trades[:5]]}")
    
    updated_count = 0
    
    for ticket, symbol in open_trades:
        if not ticket: continue
        
        # 3. Buscar histórico no MT5
        # Janela de tempo muito grande (desde 2024 para garantir)
        from_date = datetime(2024, 1, 1)
        to_date = datetime.now()
        
        history_deals = mt5.history_deals_get(from_date, to_date, position=ticket)
        
        if history_deals is None:
            print(f"DEBUG: history_deals_get retornou None para ticket {ticket} ({symbol}). Erro: {mt5.last_error()}")
            continue
            
        if len(history_deals) > 0:
            print(f"DEBUG: Encontrados {len(history_deals)} deals para ticket {ticket}")
            # Encontrou histórico -> Posição está fechada (ou pelo menos teve deals de fechamento)
            # Filtra apenas deals de saída
            exit_deals = [d for d in history_deals if d.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]]
            
            if exit_deals:
                # O último deal de saída é o que define o fechamento final
                last_deal = exit_deals[-1]
                
                # Calcula lucro total (profit + swap + commission) de todos os deals desse ticket
                total_profit = sum(d.profit + d.swap + d.commission for d in history_deals)
                
                exit_data = {
                    'exit_price': last_deal.price,
                    'exit_time': datetime.fromtimestamp(last_deal.time).strftime('%Y-%m-%d %H:%M:%S'),
                    'profit': total_profit,
                    'pips': 0,
                    'comment': f"Recovery:{last_deal.comment}"
                }
                
                if update_trade_exit(ticket, exit_data, db_path):
                    print(f"Trade {ticket} ({symbol}) atualizada: Profit ${total_profit:.2f}")
                    updated_count += 1
            else:
                # Se tem deals mas nenhum de saída, a posição pode ainda estar aberta no MT5
                # Vamos verificar se ela existe nas posições abertas atuais do MT5
                current_positions = mt5.positions_get(ticket=ticket)
                if not current_positions:
                    # Não está nas abertas e não tem deal de saída? Estranho, mas vamos marcar como closed se tiver Lucro
                    # Isso pode acontecer em remoções abruptas ou expiração
                    last_deal = history_deals[-1]
                    total_profit = sum(d.profit + d.swap + d.commission for d in history_deals)
                    exit_data = {
                        'exit_price': last_deal.price,
                        'exit_time': datetime.fromtimestamp(last_deal.time).strftime('%Y-%m-%d %H:%M:%S'),
                        'profit': total_profit,
                        'pips': 0,
                        'comment': "Recovery:Closed"
                    }
                    update_trade_exit(ticket, exit_data, db_path)
                    updated_count += 1
        else:
            # Se não há histórico nenhum, pode ser um ticket muito antigo ou inválido
            # ou a posição ainda está aberta no MT5.
            pass

    print(f"Sincronizacao concluida! {updated_count} trades atualizados.")
    
    # Atualizar tabela de performance final
    update_performance_table(db_path)
    print("Tabela de performance recalculada.")
    
    mt5.shutdown()

if __name__ == "__main__":
    run_recovery()

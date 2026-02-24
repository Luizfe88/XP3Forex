"""Data utilities for XP3 PRO FOREX"""

import json
import csv
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from threading import Lock

logger = logging.getLogger(__name__)

def save_json_data(data: Dict[str, Any], filename: str, directory: str = "data") -> bool:
    """Salva dados em formato JSON"""
    try:
        Path(directory).mkdir(exist_ok=True)
        filepath = Path(directory) / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
            
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar JSON {filename}: {e}")
        return False

# ===========================
# DAILY ANALYSIS LOGGER
# ===========================
REJECTION_CATEGORIES = {
    "EMA200": "Rejeitado: Contra tendência Macro (EMA 200)",
    "DRAWDOWN": "Rejeitado: Circuit Breaker Ativo (Risco Diário)",
    "ROLLOVER": "Rejeitado: Período de Rollover Bancário",
    "ADX_LOW": "Rejeitado: Mercado Lateral (ADX < 20)",
    "RSI_NO_REVERSAL": "Rejeitado: RSI sem reversão confirmada",
    "SPREAD": "Rejeitado: Spread Alto",
    "NEWS": "Rejeitado: Notícia Alto Impacto",
    "COOLDOWN": "Rejeitado: Cooldown ativo",
    "LIMIT": "Rejeitado: Limite de ordens atingido",
    "VOLUME": "Rejeitado: Volume insuficiente",
    "VETO_TECH": "Rejeitado: Veto técnico",
}

class DailyAnalysisLogger:
    """
    Logger que cria um arquivo TXT novo para cada dia
    registrando todas as análises de sinais
    """
    
    def __init__(self, log_dir: str = "analysis_logs", rotation_hours: int = 3, compress_after_hours: int = 24, retention_days: int = 7, max_files: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.rotation_hours = max(1, int(rotation_hours))
        self.compress_after_hours = max(1, int(compress_after_hours))
        self.retention_days = max(1, int(retention_days))
        self.max_files = max(1, int(max_files))
        self.current_window_start = None
        self.current_file = None
        self.lock = Lock()
        
    def _get_log_filename(self) -> Path:
        now = datetime.now()
        window_hour = (now.hour // self.rotation_hours) * self.rotation_hours
        window_dt = now.replace(hour=window_hour, minute=0, second=0, microsecond=0)
        self.current_window_start = window_dt
        date_part = window_dt.strftime("%Y-%m-%d")
        time_part = window_dt.strftime("%H%M")
        return self.log_dir / f"analysis_log_{date_part}_{time_part}.txt"
    
    def _check_date_rollover(self):
        now = datetime.now()
        window_hour = (now.hour // self.rotation_hours) * self.rotation_hours
        window_dt = now.replace(hour=window_hour, minute=0, second=0, microsecond=0)
        if self.current_window_start != window_dt or self.current_file is None:
            self.current_file = self._get_log_filename()
            if not self.current_file.exists():
                with open(self.current_file, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"📊 XP3 PRO FOREX - LOG DE ANÁLISES\n")
                    f.write(f"📅 Janela: {window_dt.strftime('%d/%m/%Y %H:%M')}\n")
                    f.write("="*80 + "\n\n")
            try:
                self._compress_old_logs()
                self._enforce_retention_policy()
            except Exception:
                pass
    
    def log_trade(self, symbol: str, order_type: str, volume: float, entry_price: float, 
                  stop_loss: float, take_profit: float, ticket: int):
        """Registra uma trade executada"""
        with self.lock:
            self._check_date_rollover()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] ✅ TRADE EXECUTADA\n"
            log_entry += f"Símbolo: {symbol} | Tipo: {order_type} | Volume: {volume}\n"
            log_entry += f"Entrada: {entry_price} | SL: {stop_loss} | TP: {take_profit}\n"
            log_entry += f"Ticket: {ticket}\n"
            log_entry += "-" * 60 + "\n\n"
            
            with open(self.current_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
    
    def log_analysis(self, symbol: str, signal: str, strategy: str, score: float,
                     rejected: bool, reason: str, indicators: dict, ml_score: float = 0.0,
                     is_baseline: bool = False, user: Optional[str] = None, 
                     context: Optional[dict] = None):
        """Registra uma análise de sinal"""
        with self.lock:
            self._check_date_rollover()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            status = "❌ REJEITADO" if rejected else "✅ APROVADO"
            
            log_entry = f"[{timestamp}] {status} | {symbol} | {signal}\n"
            log_entry += f"Estratégia: {strategy} | Score: {score:.2f} | ML: {ml_score:.2f}\n"
            
            if rejected:
                log_entry += f"Motivo: {reason}\n"
            
            if indicators:
                log_entry += "Indicadores: " + ", ".join([f"{k}={v}" for k, v in indicators.items() if v is not None]) + "\n"
            
            if context:
                log_entry += f"Contexto: {context}\n"
            
            log_entry += "-" * 60 + "\n\n"
            
            with open(self.current_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
    
    def _compress_old_logs(self):
        """Comprime logs antigos (placeholder para implementação futura)"""
        pass
    
    def _enforce_retention_policy(self):
        """Remove logs antigos (placeholder para implementação futura)"""
        pass

# Global instance for easy access
daily_logger = DailyAnalysisLogger()

def load_json_data(filename: str, directory: str = "data") -> Optional[Dict[str, Any]]:
    """Carrega dados de arquivo JSON"""
    try:
        filepath = Path(directory) / filename
        
        if not filepath.exists():
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Erro ao carregar JSON {filename}: {e}")
        return None

def save_csv_data(data: List[List[Any]], filename: str, directory: str = "data", headers: Optional[List[str]] = None) -> bool:
    """Salva dados em formato CSV"""
    try:
        Path(directory).mkdir(exist_ok=True)
        filepath = Path(directory) / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if headers:
                writer.writerow(headers)
                
            writer.writerows(data)
            
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar CSV {filename}: {e}")
        return False

def load_csv_data(filename: str, directory: str = "data") -> Optional[List[List[Any]]]:
    """Carrega dados de arquivo CSV"""
    try:
        filepath = Path(directory) / filename
        
        if not filepath.exists():
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return list(reader)
            
    except Exception as e:
        logger.error(f"Erro ao carregar CSV {filename}: {e}")
        return None

def init_database(db_path: str = "data/trades.db") -> bool:
    """Inicializa banco de dados SQLite"""
    try:
        Path(db_path).parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Criar tabela de trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,
                volume REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                profit REAL,
                pips REAL,
                status TEXT DEFAULT 'open',
                magic_number INTEGER,
                comment TEXT
            )
        ''')
        
        # Criar tabela de performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_profit REAL,
                total_pips REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")
        return False

def save_trade(trade_data: Dict[str, Any], db_path: str = "data/trades.db") -> bool:
    """Salva trade no banco de dados"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                symbol, order_type, volume, entry_price, exit_price, 
                stop_loss, take_profit, entry_time, exit_time, 
                profit, pips, status, magic_number, comment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('symbol'),
            trade_data.get('order_type'),
            trade_data.get('volume'),
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('stop_loss'),
            trade_data.get('take_profit'),
            trade_data.get('entry_time'),
            trade_data.get('exit_time'),
            trade_data.get('profit'),
            trade_data.get('pips'),
            trade_data.get('status', 'open'),
            trade_data.get('magic_number'),
            trade_data.get('comment')
        ))
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar trade: {e}")
        return False

def get_trade_history(symbol: Optional[str] = None, limit: int = 100, db_path: str = "data/trades.db") -> List[Dict[str, Any]]:
    """Obtém histórico de trades"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute('''
                SELECT * FROM trades 
                WHERE symbol = ? 
                ORDER BY entry_time DESC 
                LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT * FROM trades 
                ORDER BY entry_time DESC 
                LIMIT ?
            ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        trades = []
        
        for row in cursor.fetchall():
            trade = dict(zip(columns, row))
            trades.append(trade)
        
        conn.close()
        return trades
        
    except Exception as e:
        logger.error(f"Erro ao obter histórico de trades: {e}")
        return []

def save_performance_data(performance_data: Dict[str, Any], db_path: str = "data/trades.db") -> bool:
    """Salva dados de performance"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance (
                date, symbol, total_trades, wins, losses, win_rate,
                profit_factor, sharpe_ratio, max_drawdown, total_profit, total_pips
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance_data.get('date'),
            performance_data.get('symbol'),
            performance_data.get('total_trades'),
            performance_data.get('wins'),
            performance_data.get('losses'),
            performance_data.get('win_rate'),
            performance_data.get('profit_factor'),
            performance_data.get('sharpe_ratio'),
            performance_data.get('max_drawdown'),
            performance_data.get('total_profit'),
            performance_data.get('total_pips')
        ))
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar dados de performance: {e}")
        return False
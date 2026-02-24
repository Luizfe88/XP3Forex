"""Data utilities for XP3 PRO FOREX"""

import json
import csv
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

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
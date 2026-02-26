"""Calculation utilities for XP3 PRO FOREX"""

import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple, Optional, Dict, Any
from ..core.settings import ELITE_CONFIG, SYMBOL_MAP

def get_pip_size(symbol: str) -> float:
    """Obtém o tamanho do pip para o símbolo"""
    try:
        if symbol.startswith("XAU") or symbol.startswith("GOLD"):
            return 0.01
        elif symbol.startswith("XAG") or symbol.startswith("SILVER"):
            return 0.001
        elif "JPY" in symbol or "JPY" in symbol.replace("USD", "").replace("EUR", "").replace("GBP", "").replace("AUD", "").replace("CAD", "").replace("CHF", "").replace("NZD", "").replace("CNH", "").replace("TRY", "").replace("ZAR", "").replace("MXN", "").replace("RUB", "").replace("PLN", "").replace("HUF", "").replace("CZK", "").replace("DKK", "").replace("NOK", "").replace("SEK", ""):
            return 0.01
        else:
            return 0.0001
    except:
        return 0.0001

def get_tick_value(symbol: str, lot_size: float = 1.0) -> float:
    """Obtém o valor do tick para o símbolo"""
    try:
        if symbol in ELITE_CONFIG:
            return lot_size * ELITE_CONFIG[symbol].get("tick_value", 1.0)
        return lot_size
    except:
        return lot_size

def calculate_lot_size(symbol: str, risk_amount: float, stop_loss_pips: float) -> float:
    """Calcula o tamanho do lote baseado no risco"""
    try:
        pip_size = get_pip_size(symbol)
        tick_value = get_tick_value(symbol)
        
        if stop_loss_pips <= 0:
            return 0.01
            
        lot_size = risk_amount / (stop_loss_pips * pip_size * tick_value)
        
        # Aplicar multiplicador de lote se configurado
        if symbol in ELITE_CONFIG and "lot_size_multiplier" in ELITE_CONFIG[symbol]:
            lot_size *= ELITE_CONFIG[symbol]["lot_size_multiplier"]
        
        # Arredondar para o passo mais próximo (0.01)
        lot_size = round(lot_size, 2)
        
        # Limites de volume
        lot_size = max(0.01, min(lot_size, 100.0))
        
        return lot_size
    except Exception as e:
        return 0.01

def calculate_sl_tp(symbol: str, entry_price: float, order_type: str, atr_value: float) -> Tuple[float, float]:
    """Calcula Stop Loss e Take Profit baseado em ATR"""
    try:
        if symbol not in ELITE_CONFIG:
            return 0, 0
            
        config = ELITE_CONFIG[symbol]
        atr_multiplier = config.get("atr_multiplier", 2.0)
        
        # ATR-based SL/TP
        sl_distance = atr_value * atr_multiplier
        tp_distance = atr_value * atr_multiplier * 1.5  # TP maior que SL
        
        pip_size = get_pip_size(symbol)
        sl_pips = sl_distance / pip_size
        tp_pips = tp_distance / pip_size
        
        if order_type.upper() == "BUY":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
            
        return sl, tp
    except Exception as e:
        return 0, 0

def calculate_position_size(account_balance: float, risk_percent: float, stop_loss_pips: float, pip_value: float) -> float:
    """Calcula o tamanho da posição baseado no risco"""
    try:
        risk_amount = account_balance * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Arredondar para o passo mais próximo (0.01)
        position_size = round(position_size, 2)
        
        # Limites de volume
        position_size = max(0.01, min(position_size, 100.0))
        
        return position_size
    except:
        return 0.01

def calculate_margin_requirement(symbol: str, lot_size: float, leverage: float = 100) -> float:
    """Calcula requisito de margem"""
    try:
        contract_size = 100000  # Tamanho padrão do contrato
        
        if symbol.startswith("XAU") or symbol.startswith("GOLD"):
            contract_size = 100  # Ouro: 100 oz
        elif symbol.startswith("XAG") or symbol.startswith("SILVER"):
            contract_size = 5000  # Prata: 5000 oz
            
        margin = (lot_size * contract_size) / leverage
        return margin
    except:
        return 0

def calculate_profit_factor(profits: list, losses: list) -> float:
    """Calcula Profit Factor"""
    try:
        total_profits = sum(profits) if profits else 0
        total_losses = abs(sum(losses)) if losses else 0
        
        if total_losses == 0:
            return float('inf') if total_profits > 0 else 0
            
        return total_profits / total_losses
    except:
        return 0

def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.02) -> float:
    """Calcula Sharpe Ratio"""
    try:
        if not returns or len(returns) < 2:
            return 0
            
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
            
        return (avg_return - risk_free_rate) / std_return
    except:
        return 0

def calculate_max_drawdown(equity_curve: list) -> float:
    """Calcula Maximum Drawdown"""
    try:
        if not equity_curve or len(equity_curve) < 2:
            return 0
            
        peak = equity_curve[0]
        max_dd = 0
        
        for current in equity_curve:
            if current > peak:
                peak = current
            else:
                dd = (peak - current) / peak
                if dd > max_dd:
                    max_dd = dd
                    
        return max_dd * 100  # Retorna em porcentagem
    except:
        return 0

def calculate_win_rate(wins: int, losses: int) -> float:
    """Calcula Win Rate"""
    try:
        total = wins + losses
        if total == 0:
            return 0
        return (wins / total) * 100
    except:
        return 0
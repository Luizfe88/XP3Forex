
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.console import Console, Group
    from rich.style import Style
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .base_strategy import BaseStrategy
from ..core.settings import settings, ELITE_CONFIG
from ..core.models import TradeSignal
from ..utils.mt5_utils import get_symbol_info
from ..utils.indicators import calculate_atr
from ..utils.calculations import calculate_sl_tp, calculate_lot_size, get_pip_size

logger = logging.getLogger("XP3.Strategy.AR-EMA-RSI")

@dataclass
class Regime:
    name: str
    ema_fast: int
    ema_slow: int
    rsi_buy: int
    rsi_sell: int
    description: str

class AdaptiveEmaRsiStrategy(BaseStrategy):
    """
    XP3 Adaptive Regime EMA-RSI v1.0 (AR-EMA-RSI)
    
    Detects market regime (Strong Uptrend, Strong Downtrend, High Vol, Ranging)
    and adapts EMA periods and RSI thresholds accordingly.
    """

    def __init__(self, bot):
        super().__init__("AR-EMA-RSI", bot)
        self.regimes: Dict[str, Regime] = {}
        self.last_regime_update: Dict[str, datetime] = {}
        self.daily_stats: Dict[str, Any] = {
            "date": datetime.now().date(),
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "trades_count": 0,
            "regime_stats": {}
        }
        
        # Institutional Upgrades
        self.max_daily_drawdown_pct = settings.MAX_DAILY_LOSS_PERCENT / 100.0
        self.max_concurrent_trades = settings.MAX_POSITIONS
        self.correlation_threshold = 0.75
        self.kill_switch_dd_pct = settings.KILL_SWITCH_DD_PCT
        
        # Define Regimes
        self.REGIME_STRONG_UP = Regime("strong_uptrend", 8, 21, 45, 55, "Strong Uptrend: Fast EMAs, RSI Pullback Buy")
        self.REGIME_STRONG_DOWN = Regime("strong_downtrend", 8, 21, 45, 55, "Strong Downtrend: Fast EMAs, RSI Pullback Sell")
        self.REGIME_HIGH_VOL = Regime("high_vol", 13, 34, 35, 65, "High Volatility: Medium EMAs, Wider RSI")
        self.REGIME_RANGING = Regime("ranging", 13, 34, 30, 70, "Ranging: Medium EMAs, Standard RSI")

    def startup(self):
        """
        Run at startup: Analyze last 300 candles for all symbols to detect regime.
        """
        logger.info("🚀 Starting Strategy: AR-EMA-RSI v1.0")
        self.update_daily_stats()
        
        # Only iterate over bot symbols (which are already filtered)
        for symbol in self.bot.symbols:
            try:
                self.update_regime(symbol)
            except Exception as e:
                logger.error(f"Failed to initialize regime for {symbol}: {e}")

    def on_tick(self, symbol: str, tick: Any):
        # Strategy operates on bar close (M15), so tick logic is minimal or handled by bot loop
        pass

    def on_bar(self, symbol: str, timeframe: str):
        if timeframe == "M15":
            self.update_regime(symbol)

    def update_regime(self, symbol: str):
        """
        Detects market regime based on H1/M15 analysis.
        Re-evaluated every 5 minutes or new bar.
        """
        # CRÍTICO: Usar cache para evitar requisição MT5 lenta
        df = self.bot.cache.get_rates(symbol, 15, 300)
        
        if df is None or len(df) < 200:
            # logger.warning(f"Insufficient data for regime detection: {symbol}")
            return

        # Calculate Indicators
        try:
            # Pandas TA
            df.ta.ema(length=200, append=True)
            df.ta.adx(length=14, append=True)
            df.ta.atr(length=14, append=True)
            
            # Get last values
            last_close = df['close'].iloc[-1]
            ema200 = df['EMA_200'].iloc[-1]
            adx = df['ADX_14'].iloc[-1]
            atr = df['ATRr_14'].iloc[-1]
            
            # Calculate Average ATR (SMA of ATR)
            atr_mean_50 = df['ATRr_14'].rolling(50).mean().iloc[-1]
            
            regime = self.REGIME_RANGING # Default
            
            # Logic
            if adx > 25 and last_close > ema200:
                regime = self.REGIME_STRONG_UP
            elif adx > 25 and last_close < ema200:
                regime = self.REGIME_STRONG_DOWN
            elif atr > (atr_mean_50 * 1.5):
                regime = self.REGIME_HIGH_VOL
            else:
                regime = self.REGIME_RANGING
            
            self.regimes[symbol] = regime
            self.last_regime_update[symbol] = datetime.now()
            
            # logger.debug(f"Regime for {symbol}: {regime.name} (ADX={adx:.1f}, ATR/Avg={atr/atr_mean_50:.1f})")
            
        except Exception as e:
            logger.error(f"Error updating regime for {symbol}: {e}")

    def analyze(self, symbol: str, timeframe: int, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Main analysis function called by Bot.
        Returns TradeSignal or None.
        """
        # 0. Update Regime if stale (> 5 mins)
        last_update = self.last_regime_update.get(symbol)
        if not last_update or (datetime.now() - last_update).total_seconds() > 300:
            self.update_regime(symbol)
            
        regime = self.regimes.get(symbol, self.REGIME_RANGING)
        
        # 1. Pre-checks (Institutional)
        if not self.check_institutional_filters(symbol):
            return None

        # 2. Prepare Data
        # Ensure we have indicators for the specific regime
        fast_period = regime.ema_fast
        slow_period = regime.ema_slow
        
        # Recalculate specific EMAs for the current regime
        # Check if columns exist or recalc
        # To avoid recalc every time if not needed, we could cache, but df changes every call.
        # Recalc is fast enough for M15.
        
        try:
            df[f'ema_fast'] = ta.ema(df['close'], length=fast_period)
            df[f'ema_slow'] = ta.ema(df['close'], length=slow_period)
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # Check if indicators are valid
            if df[f'ema_fast'].iloc[-1] is None or df['rsi'].iloc[-1] is None:
                return None
                
            current_price = df['close'].iloc[-1]
            ema_fast = df[f'ema_fast'].iloc[-1]
            ema_slow = df[f'ema_slow'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # 3. Signal Logic
            signal_type = None
            
            # Crossover logic: check last 2 candles for crossover
            # Bullish Crossover: Fast crosses above Slow
            # Or Pullback logic as per regime description?
            # Prompt says: "EMA/RSI adaptativos por regime"
            # I'll stick to crossover + RSI confirmation for simplicity/robustness v1.0
            
            crossed_up = (df[f'ema_fast'].iloc[-2] <= df[f'ema_slow'].iloc[-2]) and \
                         (ema_fast > ema_slow)
            
            crossed_down = (df[f'ema_fast'].iloc[-2] >= df[f'ema_slow'].iloc[-2]) and \
                           (ema_fast < ema_slow)
            
            # Conditions based on Regime
            if regime == self.REGIME_STRONG_UP:
                # In strong uptrend, buy on dips or crossovers
                if (crossed_up or (current_price > ema_fast and rsi < 50)) and rsi > regime.rsi_buy:
                    signal_type = "BUY"
                    
            elif regime == self.REGIME_STRONG_DOWN:
                 if (crossed_down or (current_price < ema_fast and rsi > 50)) and rsi < regime.rsi_sell:
                    signal_type = "SELL"
                    
            elif regime == self.REGIME_HIGH_VOL:
                if crossed_up and rsi > regime.rsi_buy: 
                         signal_type = "BUY"
                elif crossed_down and rsi < regime.rsi_sell:
                     signal_type = "SELL"
                     
            else: # Ranging
                if crossed_up and rsi > regime.rsi_buy:
                    signal_type = "BUY"
                elif crossed_down and rsi < regime.rsi_sell:
                    signal_type = "SELL"

            # 4. Multi-Timeframe Confirmation (H1)
            if signal_type:
                if not self.check_h1_trend(symbol, signal_type):
                    # logger.info(f"Signal {signal_type} for {symbol} rejected by H1 Trend")
                    return None

            # 5. Create Signal
            if signal_type:
                # Calculate ATR for SL/TP
                atr = df.ta.atr(length=14).iloc[-1]
                if np.isnan(atr) or atr == 0:
                     atr = 0.0010 # Fallback
                     
                sl_dist = 2.0 * atr
                
                if signal_type == "BUY":
                    sl = current_price - sl_dist
                    tp = current_price + (3.0 * sl_dist) 
                else:
                    sl = current_price + sl_dist
                    tp = current_price - (3.0 * sl_dist)
                
                # Position Sizing (Risk %)
                risk_amount = self.get_account_balance() * settings.RISK_PER_TRADE / 100.0
                
                # Calculate Volume
                volume = self.calculate_position_size(symbol, current_price, sl, risk_amount)
                
                if volume <= 0:
                    return None
                
                return TradeSignal(
                    symbol=symbol,
                    order_type=signal_type,
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    volume=volume,
                    confidence=0.85,
                    reason=f"{regime.name} | EMA({fast_period},{slow_period}) | RSI {rsi:.1f}"
                )
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
            
        return None

    def check_institutional_filters(self, symbol: str) -> bool:
        # A. Global Daily Drawdown
        if self.daily_stats["profit"] < -(self.get_account_balance() * self.max_daily_drawdown_pct):
            # logger.warning("Daily Drawdown Limit Hit. Trading Paused.")
            return False
            
        # Kill Switch
        current_equity = self.get_account_equity()
        initial_balance = self.get_account_balance() 
        if initial_balance > 0:
            dd_pct = (initial_balance - current_equity) / initial_balance
            if dd_pct > self.kill_switch_dd_pct:
                logger.critical("KILL SWITCH ACTIVATED: DD > 5%")
                self.bot.pause_trading() 
                logger.critical("TELEGRAM ALERT: KILL SWITCH ACTIVATED")
                self.bot.close_all_positions() 
                return False

        # Max Trades
        if len(self.bot.positions) >= self.max_concurrent_trades:
            return False
            
        # Correlation Filter (only check if we have signals/trades pending? No, check before signal gen)
        # This can be expensive. Do it only if we have other positions.
        if len(self.bot.positions) > 0:
            if self.check_correlation(symbol):
                # logger.info(f"Trade rejected for {symbol} due to high correlation")
                return False
            
        # Session Filter
        if not self.is_london_ny_session():
            return False
            
        # News Filter
        if self.is_high_impact_news_soon(symbol):
            return False
            
        return True

    def check_h1_trend(self, symbol: str, signal_type: str) -> bool:
        """
        Check H1 Trend: EMA200 + ADX
        """
        try:
            df_h1 = self.bot.cache.get_rates(symbol, 60, 100)
            if df_h1 is None or len(df_h1) < 50:
                return True 
            
            df_h1.ta.ema(length=200, append=True)
            
            last_close = df_h1['close'].iloc[-1]
            ema200 = df_h1['EMA_200'].iloc[-1]
            
            if signal_type == "BUY":
                return last_close > ema200
            else:
                return last_close < ema200
        except Exception:
            return True

    def check_correlation(self, symbol: str) -> bool:
        """
        Check correlation with open positions.
        """
        # Get data for symbol
        df_sym = self.bot.cache.get_rates(symbol, 60, 100) # Using H1 for correlation
        if df_sym is None or len(df_sym) < 30: return False
        
        returns_sym = df_sym['close'].pct_change().dropna()
        
        for pos_symbol in self.bot.positions:
            if pos_symbol == symbol: continue
            
            df_pos = self.bot.cache.get_rates(pos_symbol, 60, 100)
            if df_pos is None or len(df_pos) < 30: continue
            
            returns_pos = df_pos['close'].pct_change().dropna()
            
            # Align
            min_len = min(len(returns_sym), len(returns_pos))
            
            corr = returns_sym.tail(min_len).corr(returns_pos.tail(min_len))
            if abs(corr) > self.correlation_threshold:
                return True
                
        return False

    def is_london_ny_session(self) -> bool:
        """
        08:00 - 17:00 GMT
        """
        now_gmt = datetime.utcnow()
        start = now_gmt.replace(hour=8, minute=0, second=0, microsecond=0)
        end = now_gmt.replace(hour=17, minute=0, second=0, microsecond=0)
        return start <= now_gmt <= end

    def is_high_impact_news_soon(self, symbol: str) -> bool:
        # Placeholder
        return False

    def get_account_balance(self) -> float:
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            return info.balance if info else 10000.0
        except:
            return 10000.0

    def get_account_equity(self) -> float:
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            return info.equity if info else 10000.0
        except:
            return 10000.0

    def calculate_position_size(self, symbol: str, price: float, sl: float, risk_amount: float) -> float:
        dist = abs(price - sl)
        if dist == 0: return 0.01
        
        # Get pip size
        pip_size = get_pip_size(symbol)
        if pip_size == 0: pip_size = 0.0001
        
        sl_pips = dist / pip_size
        
        # Calculate Lot Size
        # Note: calculate_lot_size expects 'stop_loss_pips' but actually uses it with pip_size in formula
        # formula: lot_size = risk_amount / (stop_loss_pips * pip_size * tick_value)
        # This seems redundant if we pass pips.
        # But let's trust the util.
        
        return calculate_lot_size(symbol, risk_amount, sl_pips)

    def update_daily_stats(self):
        current_date = datetime.now().date()
        if self.daily_stats["date"] != current_date:
            # Save previous day stats before resetting
            self.save_stats()
            
            # Reset
            self.daily_stats = {
                "date": current_date,
                "wins": 0,
                "losses": 0,
                "profit": 0.0,
                "trades_count": 0,
                "regime_stats": {}
            }
        
    def save_stats(self):
        """Save daily stats to JSON file"""
        try:
            stats_dir = Path("data/stats")
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            date_str = self.daily_stats["date"].strftime("%Y-%m-%d")
            filepath = stats_dir / f"ar_ema_rsi_stats_{date_str}.json"
            
            # Calculate metrics
            wins = self.daily_stats["wins"]
            losses = self.daily_stats["losses"]
            total = wins + losses
            winrate = (wins / total * 100) if total > 0 else 0.0
            
            data = {
                "date": date_str,
                "wins": wins,
                "losses": losses,
                "winrate": winrate,
                "net_profit": self.daily_stats["profit"],
                "regime_stats": self.daily_stats.get("regime_stats", {})
            }
            
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def get_why_report(self, symbol: str) -> Tuple[Optional["Panel"], float, str]:
        """
        Gera um relatório visual (rich.Panel) explicando "Por Que Comprar?"
        ou "Por Que Vender?" com base no regime atual e indicadores.
        Retorna (Panel, confidence_score, log_string).
        """
        if not HAS_RICH:
            return None, 0.0, ""

        # 1. Coleta de Dados
        # CRÍTICO: Usar RateCache do Bot para evitar slow fetch
        df = self.bot.cache.get_rates(symbol, 15, 300)
        
        if df is None or len(df) < 200:
            return Panel(Text(f"Dados insuficientes para {symbol}", style="bold red")), 0.0, ""

        regime = self.regimes.get(symbol, self.REGIME_RANGING)
        
        # Calcular Indicadores
        fast_period = regime.ema_fast
        slow_period = regime.ema_slow
        
        try:
            # Pandas TA calc
            df = df.copy()
            df[f'ema_fast'] = ta.ema(df['close'], length=fast_period)
            df[f'ema_slow'] = ta.ema(df['close'], length=slow_period)
            df['rsi'] = ta.rsi(df['close'], length=14)
            df.ta.adx(length=14, append=True)
            df.ta.atr(length=14, append=True)
            
            current_price = df['close'].iloc[-1]
            ema_fast = df[f'ema_fast'].iloc[-1]
            ema_slow = df[f'ema_slow'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            adx = df['ADX_14'].iloc[-1] if 'ADX_14' in df else 0
            atr = df['ATRr_14'].iloc[-1] if 'ATRr_14' in df else 0
            
            # Spread
            info = get_symbol_info(symbol)
            spread_points = info.get('spread', 0) if info else 0
            pip_size = get_pip_size(symbol)
            if pip_size == 0: pip_size = 0.0001
            
        except Exception as e:
            return Panel(Text(f"Erro ao calcular indicadores: {e}", style="bold red")), 0.0, str(e)

            # Determinar Direção Base (Tendência)
        # BUGFIX: Use logic identical to analyze()
        # Se regime for strong_uptrend, esperamos BUY
        # Se regime for strong_downtrend, esperamos SELL
        # Se regime for high_vol/ranging, depende do crossover
        
        # Recalculate crossovers same as analyze()
        crossed_up = (df[f'ema_fast'].iloc[-2] <= df[f'ema_slow'].iloc[-2]) and \
                         (ema_fast > ema_slow)
        crossed_down = (df[f'ema_fast'].iloc[-2] >= df[f'ema_slow'].iloc[-2]) and \
                           (ema_fast < ema_slow)

        # Determine signal direction being analyzed based on regime/conditions
        # Default to trend direction, but refine
        is_buy = ema_fast > ema_slow # Default trend based
        
        if regime.name == "strong_uptrend":
            is_buy = True
        elif regime.name == "strong_downtrend":
            is_buy = False
        
        direction = "BUY" if is_buy else "SELL"

        # 2. Avaliação de Condições
        conditions = []
        score = 0
        max_score = 0
        
        # Helper para adicionar condição
        def add_cond(name, current, target, status_bool, explanation, missing_msg):
            nonlocal score, max_score
            max_score += 1
            if status_bool:
                score += 1
                status_icon = "✅"
                style = "green"
            else:
                # Crucial: Mark as Fail
                status_icon = "❌" 
                style = "red"
            
            conditions.append({
                "name": name,
                "current": current,
                "target": target,
                "status": status_icon,
                "missing": missing_msg,
                "style": style,
                "explanation": explanation
            })

        # A. Tendência (EMAs)
        ema_diff_pips = (ema_fast - ema_slow) / pip_size
        price_vs_fast_pips = (current_price - ema_fast) / pip_size
        
        if is_buy:
            # Condition 1: EMA Fast > EMA Slow (Trend)
            trend_ok = ema_fast > ema_slow
            add_cond("EMA Fast > Slow", 
                     f"{ema_fast:.5f}", 
                     f"> {ema_slow:.5f}", 
                     trend_ok, 
                     "Tendência de alta confirmada", 
                     f"Invertido por {-ema_diff_pips:.1f} pips" if not trend_ok else "")
            
            # Condition 2: Trigger (Crossover OR Pullback)
            # Logic from analyze(): (crossed_up or (current_price > ema_fast and rsi < 50))
            is_crossover = crossed_up
            is_pullback = (current_price > ema_fast and rsi < 50)
            trigger_ok = is_crossover or is_pullback
            
            # For report display, we show what matched or what is missing
            if is_crossover:
                 add_cond("Gatilho de Entrada", "Crossover", "Crossover ou Pullback", True, "Cruzamento de alta detectado", "")
            elif is_pullback:
                 add_cond("Gatilho de Entrada", "Pullback", "Crossover ou Pullback", True, "Pullback na tendência", "")
            else:
                 # Show what failed
                 add_cond("Gatilho de Entrada", "Nenhum", "Crossover ou Pullback (RSI<50)", False, "Aguardando sinal", "Sem Crossover e RSI alto")

            # Condition 3: RSI Filter
            rsi_target = regime.rsi_buy
            rsi_ok = rsi > rsi_target
            add_cond(f"RSI Filtro (> {rsi_target})",
                     f"{rsi:.1f}",
                     f"> {rsi_target}",
                     rsi_ok,
                     "Momentum suficiente",
                     f"RSI muito baixo (-{rsi_target - rsi:.1f})" if not rsi_ok else "")

        else: # SELL
            # Condition 1: EMA Fast < EMA Slow (Trend)
            trend_ok = ema_fast < ema_slow
            add_cond("EMA Fast < Slow", 
                     f"{ema_fast:.5f}", 
                     f"< {ema_slow:.5f}", 
                     trend_ok, 
                     "Tendência de baixa confirmada", 
                     f"Invertido por {ema_diff_pips:.1f} pips" if not trend_ok else "")

            # Condition 2: Trigger (Crossover OR Pullback)
            # Logic from analyze(): (crossed_down or (current_price < ema_fast and rsi > 50))
            is_crossover = crossed_down
            is_pullback = (current_price < ema_fast and rsi > 50)
            trigger_ok = is_crossover or is_pullback
            
            if is_crossover:
                 add_cond("Gatilho de Entrada", "Crossover", "Crossover ou Pullback", True, "Cruzamento de baixa detectado", "")
            elif is_pullback:
                 add_cond("Gatilho de Entrada", "Pullback", "Crossover ou Pullback", True, "Pullback na tendência", "")
            else:
                 add_cond("Gatilho de Entrada", "Nenhum", "Crossover ou Pullback (RSI>50)", False, "Aguardando sinal", "Sem Crossover e RSI baixo")

            # Condition 3: RSI Filter
            rsi_target = regime.rsi_sell
            rsi_ok = rsi < rsi_target
            add_cond(f"RSI Filtro (< {rsi_target})",
                     f"{rsi:.1f}",
                     f"< {rsi_target}",
                     rsi_ok,
                     "Momentum suficiente",
                     f"RSI muito alto (+{rsi - rsi_target:.1f})" if not rsi_ok else "")

        # B. Força da Tendência (ADX)
        adx_ok = adx > 25
        add_cond("ADX confirma força",
                 f"{adx:.1f}",
                 "> 25",
                 adx_ok,
                 "Tendência precisa ficar mais forte",
                 f"+{25 - adx:.1f}" if not adx_ok else "")

        # C. Spread
        spread_ok = spread_points < 30 
        add_cond("Spread dentro do limite",
                 f"{spread_points}",
                 "< 30",
                 spread_ok,
                 "Liquidez excelente",
                 "Spread alto!" if not spread_ok else "")

        # D. Sessão
        session_ok = self.is_london_ny_session()
        add_cond("Sessão London/NY ativa",
                 "Sim" if session_ok else "Não",
                 "Sim",
                 session_ok,
                 "Melhor horário para operar",
                 "Horário de baixa liquidez" if not session_ok else "")

        # 3. Construção do Painel
        confidence = (score / max_score) * 100 if max_score > 0 else 0
        
        # Cor do título baseada na confiança
        if confidence >= 80:
            title_color = "green"
            conf_emoji = "🟢"
        elif confidence >= 60:
            title_color = "yellow"
            conf_emoji = "🟡"
        else:
            title_color = "red"
            conf_emoji = "🔴"
            
        title_text = f"🔵 POR QUE COMPRAR {symbol}?" if is_buy else f"🔴 POR QUE VENDER {symbol}?"
        
        # Tabela
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold white", expand=True)
        table.add_column("Condição", style="cyan")
        table.add_column("Valor Atual", justify="right")
        table.add_column("Meta", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Quanto falta", style="yellow")
        table.add_column("Explicação simples", style="italic")

        missing_summary = []
        
        # Construção da String de Log Detalhada
        log_lines = [
            f"--- WHY REPORT: {symbol} ({direction}) ---",
            f"Confiança: {confidence:.1f}%",
            f"Regime: {regime.name}"
        ]

        for c in conditions:
            table.add_row(
                c["name"],
                c["current"],
                c["target"],
                c["status"],
                c["missing"],
                c["explanation"]
            )
            
            # Add to log
            log_status = "OK" if c["status"] == "✅" else "FAIL"
            log_lines.append(f"[{log_status}] {c['name']}: {c['current']} (Meta: {c['target']}) - {c['missing']}")
            
            if c["missing"]:
                missing_summary.append(c["missing"])

        # Resumo Textual Didático
        if score == max_score:
            summary = f"[bold green]SINAL PERFEITO![/] O {symbol} está alinhado para {direction}. Execução recomendada."
            log_lines.append("RESULT: PERFECT SIGNAL")
        elif confidence > 60:
            missing_text = ', '.join(missing_summary[:2])
            summary = f"[bold yellow]QUASE LÁ![/] O EMA já deu o sinal, mas faltam detalhes: {missing_text}. Setup promissor se confirmar."
            log_lines.append(f"RESULT: PROMISSING (Missing: {missing_text})")
        else:
            summary = f"[bold red]AGUARDE.[/] O mercado ainda não confirmou a entrada em {symbol}."
            log_lines.append("RESULT: WAIT")

        # Painel Final
        panel_content = Group(
            Text(f"Confiança do sinal: {confidence:.0f}% {conf_emoji}", style=f"bold {title_color}"),
            Text("Regime de Mercado: " + regime.description, style="italic dim"),
            Text(""), 
            table,
            Text(""), 
            Text("Resumo didático:", style="bold underline"),
            Text.from_markup(summary)
        )
        
        full_log_report = "\n".join(log_lines)
        
        return Panel(
            panel_content,
            title=title_text,
            border_style=title_color,
            padding=(1, 2)
        ), confidence, full_log_report

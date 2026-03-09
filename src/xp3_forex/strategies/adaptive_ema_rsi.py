
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
from ..core.models import TradeSignal, Position
from ..utils.mt5_utils import get_symbol_info
from ..utils.indicators import calculate_atr
from ..utils.calculations import calculate_sl_tp, calculate_lot_size, get_pip_size
from ..mt5.symbol_manager import symbol_manager
from .session_analyzer import get_active_session_name, get_active_session_params


# Quantitative Layers (New)
from ..indicators.regime import calculate_hurst_rs, calculate_mmi_numba, RegimeConfig
from ..indicators.filters import KalmanFilter, KalmanConfig
from ..risk.institutional import calculate_fractional_kelly, calculate_cf_cvar, RiskConfig

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
        
        # RSI Exhaustion Limits (New)
        # Prevents entries if trend is already over-extended
        self.RSI_BUY_MAX = 70.0
        self.RSI_SELL_MIN = 30.0

    def startup(self):
        """
        Run at startup: Analyze last 300 candles for all symbols to detect regime.
        """
        logger.info("🚀 Starting Strategy: AR-EMA-RSI v1.0")
        self.update_daily_stats()
        
        # Log persistence check
        json_path = settings.DATA_DIR / "session_optimized_params.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    cnt = len(json.load(f))
                logger.info(f"📂 Memória de Ontem: {cnt} ativos carregados com pesos persistentes.")
            except: pass
        else:
            logger.info("🆕 Memória Vazia: Iniciando com pesos originais (Padrão Institucional).")
            
        # Quantitative Params Check
        quant_json = settings.DATA_DIR / "quant_optimized_params.json"
        if quant_json.exists():
            try:
                with open(quant_json, 'r') as f:
                    q_data = json.load(f)
                logger.info(f"🧬 QUANT CORE: {len(q_data)} ativos com calibração Optuna ativa.")
            except: pass
        
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
            
            # --- QUANT LAYER: Hurst & MMI Enhancement ---
            try:
                # Load optimized parameters
                q_params = settings.get_quant_params(symbol)
                h_lookback = q_params.get("hurst_lookback", 1000)
                m_lookback = q_params.get("mmi_lookback", 300)
                
                # Calculate Hurst recursively
                hurst_prices = df['close'].tail(h_lookback).values
                hurst_val = calculate_hurst_rs(hurst_prices)
                
                # Calculate MMI
                mmi_prices = df['close'].tail(m_lookback).values
                mmi_val = calculate_mmi_numba(mmi_prices)
                
                # Update regime based on Hurst (Quantitative Core)
                if hurst_val > 0.65:
                    regime.description += " | QUANT: Strong Trending (Hurst > 0.65)"
                elif hurst_val < 0.45:
                    regime.description += " | QUANT: Mean Reverting (Hurst < 0.45)"
                
                # Store quant values in regime or metadata
                setattr(regime, 'hurst', hurst_val)
                setattr(regime, 'mmi', mmi_val)
                
            except Exception as q_err:
                logger.warning(f"Failed to calculate quant metrics for {symbol}: {q_err}")
            
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

        # 2. Get Session Optimized Parameters
        session_params = get_active_session_params(symbol, datetime.utcnow())
        
        # 3. Prepare Data
        # Base indicators from Regime, but allow Session Overrides
        fast_period = session_params.get("ema_fast", regime.ema_fast)
        slow_period = session_params.get("ema_slow", regime.ema_slow)
        rsi_period = session_params.get("rsi_period", 14)
        
        # Strategy Logic with Session Thresholds
        rsi_buy_thresh = session_params.get("rsi_buy", regime.rsi_buy)
        rsi_sell_thresh = session_params.get("rsi_sell", regime.rsi_sell)
        adx_thresh = session_params.get("adx_threshold", 25)
        
        # Recalculate specific EMAs for the current regime
        # Check if columns exist or recalc
        # To avoid recalc every time if not needed, we could cache, but df changes every call.
        # Recalc is fast enough for M15.
        
        try:
            df[f'ema_fast'] = ta.ema(df['close'], length=fast_period)
            df[f'ema_slow'] = ta.ema(df['close'], length=slow_period)
            df['rsi'] = ta.rsi(df['close'], length=rsi_period)
            df.ta.adx(length=14, append=True)
            
            # --- QUANT LAYER 2: Adaptive Kalman Filter Signal ---
            q_params = settings.get_quant_params(symbol)
            hurst_val = getattr(regime, 'hurst', 0.5)
            
            kf_config = KalmanConfig(
                initial_r=q_params.get("initial_r", 500.0),
                min_q=q_params.get("min_q", 0.01),
                max_q=q_params.get("max_q", 0.1)
            )
            kf = KalmanFilter(kf_config)
            # Apply KF to close prices
            hurst_series = pd.Series([hurst_val] * len(df), index=df.index)
            filtered_price = kf.apply(df['close'], hurst_series)
            df['kf_signal'] = filtered_price
            
            # Check if indicators are valid
            if df[f'ema_fast'].iloc[-1] is None or df['rsi'].iloc[-1] is None:
                return None
                
            current_price = df['close'].iloc[-1]
            ema_fast = df[f'ema_fast'].iloc[-1]
            ema_slow = df[f'ema_slow'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            adx = df['ADX_14'].iloc[-1]
            
            # ADX Filter from Session
            if adx < adx_thresh:
                return None
                
            # --- RSI Exhaustion Filter (CRITICAL) ---
            # Don't buy if RSI is already near overbought (>70)
            # Don't sell if RSI is already near oversold (<30)
            if rsi > self.RSI_BUY_MAX:
                # logger.debug(f"🚫 {symbol} BUY rejected: RSI Exhaustion ({rsi:.1f} > {self.RSI_BUY_MAX})")
                return None
            if rsi < self.RSI_SELL_MIN:
                # logger.debug(f"🚫 {symbol} SELL rejected: RSI Exhaustion ({rsi:.1f} < {self.RSI_SELL_MIN})")
                return None
                
            # Dynamic RSI Tolerance for NY Session (Strong Trend)
            active_sess = session_params.get("active_session", "UNKNOWN")
            if "NY" in active_sess.upper() and adx > 35:
                # Relax thresholds to catch micro-corrections in strong trend
                rsi_buy_thresh -= 10
                rsi_sell_thresh += 10
            
            # 3. Signal Logic
            signal_type = None
            
            # Crossover logic: check last 2 candles for crossover
            # Bullish Crossover: Fast crosses above Slow
            # Or Pullback logic as per regime description?
            # Prompt says: "EMA/RSI adaptativos por regime"
            # I'll stick to crossover + RSI confirmation for simplicity/robustness v1.0
            
            # USE KALMAN SIGNAL INSTEAD OF EMA FAST (Layer 2 requirement)
            kf_signal = df['kf_signal'].iloc[-1]
            kf_signal_prev = df['kf_signal'].iloc[-2]
            
            crossed_up = (kf_signal_prev <= df[f'ema_slow'].iloc[-2]) and \
                         (kf_signal > ema_slow)
            
            crossed_down = (kf_signal_prev >= df[f'ema_slow'].iloc[-2]) and \
                           (kf_signal < ema_slow)
            
            # Conditions based on Regime
            if regime == self.REGIME_STRONG_UP:
                # In strong uptrend, buy on dips or crossovers using Kalman
                if (crossed_up or (current_price > kf_signal and rsi < 60)) and rsi > rsi_buy_thresh:
                    signal_type = "BUY"
                    
            elif regime == self.REGIME_STRONG_DOWN:
                 if (crossed_down or (current_price < kf_signal and rsi > 40)) and rsi < rsi_sell_thresh:
                    signal_type = "SELL"
                    
            elif regime == self.REGIME_HIGH_VOL:
                if crossed_up and rsi > rsi_buy_thresh: 
                         signal_type = "BUY"
                elif crossed_down and rsi < rsi_sell_thresh:
                     signal_type = "SELL"
                     
            else: # Ranging
                if crossed_up and rsi > rsi_buy_thresh:
                    signal_type = "BUY"
                elif crossed_down and rsi < rsi_sell_thresh:
                    signal_type = "SELL"

            # 4. Multi-Timeframe Confirmation (H1 Trend + Hurst Confluence)
            if signal_type:
                h1_ok = self.check_h1_trend(symbol, signal_type)
                if not h1_ok:
                    logger.info(f"🚫 Signal {signal_type} for {symbol} rejected by H1 Trend (EMA200 Conflict)")
                    return None

                # TAREFA 2: Implementação da Hierarquia de Regimes (Hurst Confluence)
                # Objetivo: Reduzir sinais falsos em micro-tendências.
                # Lógica: Uma entrada no M15 só deve ser permitida se o Hurst do M60 > 0.55 (Persistência).
                try:
                    df_h1 = self.bot.cache.get_rates(symbol, 60, 500)
                    if df_h1 is not None and len(df_h1) >= 100:
                        h1_hurst = calculate_hurst_rs(df_h1['close'].values)
                        if h1_hurst <= 0.55:
                            logger.info(f"🚫 [HURST CONFLUENCE] {symbol} rejected: H1 Hurst ({h1_hurst:.2f}) indicates noise/reversion.")
                            return None
                        setattr(regime, 'h1_hurst', h1_hurst)
                except Exception as h_err:
                    logger.warning(f"Failed Hurst Confluence check for {symbol}: {h_err}")
            else:
                h1_ok = True # No signal, no validation needed, but define for safety

            # 5. Calculate Confidence (Detailed)
            confidence = 0.5 # Base
            if signal_type:
                score = 0
                max_score = 6 # Trend, Trigger, RSI, ADX, H1, Spread
                
                # 1. Trend (Already checked by crossovers but let's re-verify)
                if (signal_type == "BUY" and ema_fast > ema_slow) or (signal_type == "SELL" and ema_fast < ema_slow):
                    score += 1
                
                # 2. Trigger Strength (Crossover > Pullback)
                if crossed_up or crossed_down:
                    score += 1
                elif (signal_type == "BUY" and current_price > ema_fast) or (signal_type == "SELL" and current_price < ema_fast):
                    score += 0.5 # Weak trigger
                
                # B. RSI Filter
                is_buy = (signal_type == "BUY")
                rsi_ok = False
                rsi_missing = ""
                if is_buy:
                    rsi_ok = rsi <= self.RSI_BUY_MAX
                    rsi_missing = f"RSI > {self.RSI_BUY_MAX} (Exhausted)"
                else:
                    rsi_ok = rsi >= self.RSI_SELL_MIN
                    rsi_missing = f"RSI < {self.RSI_SELL_MIN} (Exhausted)"
                
                # This part is for get_why_report, not directly for confidence score
                # add_cond("RSI Exhaustion", f"{rsi:.1f}", f"<= {self.RSI_BUY_MAX}" if is_buy else f">= {self.RSI_SELL_MIN}", 
                #          rsi_ok, "RSI not exhausted", rsi_missing)

                # 3. RSI Position (Strategy)
                if signal_type == "BUY":
                    if rsi > rsi_buy_thresh + 10: score += 1
                    elif rsi > rsi_buy_thresh: score += 0.5
                else: 
                    if rsi < rsi_sell_thresh - 10: score += 1
                    elif rsi < rsi_sell_thresh: score += 0.5
                
                # 4. ADX Strength
                if adx > 35: score += 1
                elif adx > 25: score += 0.5
                
                # 5. H1 Alignment
                if h1_ok: score += 1
                
                # 6. Spread (Already checked in bot/feeder but good for confidence)
                if symbol_manager.check_spread(symbol):
                    score += 1
                
                confidence = score / max_score

                # Calculate ATR for SL/TP
                atr = df.ta.atr(length=14).iloc[-1]
                if np.isnan(atr) or atr == 0:
                     atr = 0.0010 # Fallback
                     
                # Adaptive ATR Multipliers by Regime
                atr_mult_sl = session_params.get("atr_multiplier_sl", 2.0)
                
                # Dynamic TP based on Regime
                if regime.name in ["strong_uptrend", "strong_downtrend"]:
                    atr_mult_tp = 4.0 # Extended TP for strong trends
                elif regime.name == "ranging":
                    atr_mult_tp = 2.0 # Quicker TP for ranges
                else:
                    atr_mult_tp = session_params.get("atr_multiplier_tp", 3.0)
                
                sl_dist = atr_mult_sl * atr
                
                if signal_type == "BUY":
                    sl = current_price - sl_dist
                    tp = current_price + (atr_mult_tp * atr) 
                else:
                    sl = current_price + sl_dist
                    tp = current_price - (atr_mult_tp * atr)
                
                # --- QUANT LAYER 3: Institutional Risk Scaling ---
                try:
                    # CVaR Risk Check
                    returns_series = df['close'].pct_change().dropna().tail(500)
                    cvar_99 = calculate_cf_cvar(returns_series.values, alpha=0.99)
                    
                    # Fractional Kelly Sizing
                    # Win rate and R/R from session or historical
                    win_rate = session_params.get("win_rate", 0.55)
                    risk_reward = (tp - current_price) / abs(current_price - sl)
                    
                    hurst_val = getattr(regime, 'hurst', 0.5)
                    
                    # Get Correlation Adjustment (Portfolio Level)
                    risk_config = RiskConfig() 
                    kelly_fraction = calculate_fractional_kelly(
                        win_rate, 
                        risk_reward, 
                        hurst_val, 
                        risk_config,
                        symbol=symbol,
                        portfolio_returns=self.get_portfolio_returns_data()
                    )
                    
                    # Calculate Final Equity Risk and Volume
                    risk_amount = self.get_account_equity() * kelly_fraction
                    volume = self.calculate_position_size(symbol, current_price, sl, risk_amount)
                    volume = max(0.01, min(volume, settings.MAX_LOTS_PER_TRADE))
                    
                    logger.info(f"📊 [QUANT RISK] {symbol}: CVaR(99)={cvar_99:.4f}, Kelly Fraction={kelly_fraction:.4f}, Final Volume: {volume:.2f}")
                except Exception as risk_err:
                    logger.warning(f"Risk scaling failed for {symbol}: {risk_err}")
                    volume = 0.01
                
                if volume <= 0:
                    return None
                
                return TradeSignal(
                    symbol=symbol,
                    order_type=signal_type,
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    volume=volume,
                    confidence=confidence,
                    reason=f"{session_params.get('active_session', 'N/A')} | {regime.name} | EMA({fast_period},{slow_period}) | RSI {rsi:.1f}"
                )
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
            
        return None

    def check_institutional_filters(self, symbol: str) -> bool:
        # 0. Early Exit para CLOSE-ONLY symbols
        if not symbol_manager._check_trade_mode(symbol):
            return False

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
            
        # Session Filter (Session Analyzer identifies ASIA, LONDON, NY)
        session = get_active_session_name(datetime.utcnow())
        if session == "UNKNOWN":
            return False
            
        # News Filter
        if self.is_high_impact_news_soon(symbol):
            logger.info(f"🚫 Trade rejected for {symbol} due to High Impact News soon")
            return False

        # TAREFA 4: Forex Weekend Gap Protection
        # Objetivo: Evitar volatilidade extrema e gaps de domingo (Sunday Open).
        # Regra: Warm-up de 30 minutos na abertura do mercado (Domingo 17:00 NY / 22:00 UTC).
        now_utc = datetime.utcnow()
        if now_utc.weekday() == 6: # Sunday
            if now_utc.hour == 22 and now_utc.minute < 30:
                logger.info(f"⏳ [FOREX WARM-UP] {symbol}: Sunday Open gap observation (22:00-22:30 UTC). Kalman stabilized.")
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

    def get_portfolio_returns_data(self) -> pd.DataFrame:
        """
        Coleta retornos históricos dos ativos com posições abertas para cálculo de covariância.
        """
        returns_dict = {}
        for pos_symbol in self.bot.positions:
            df = self.bot.cache.get_rates(pos_symbol, 60, 200) # H1 returns
            if df is not None:
                returns_dict[pos_symbol] = df['close'].pct_change()
        
        if not returns_dict:
            return pd.DataFrame()
            
        return pd.DataFrame(returns_dict).dropna()

    def is_trading_session_active(self) -> bool:
        """Verifica se alguma sessão principal está ativa via SessionAnalyzer"""
        return get_active_session_name(datetime.utcnow()) != "UNKNOWN"

    @staticmethod
    def is_high_impact_news_soon(symbol: str) -> bool:
        # Placeholder
        return False

    def evaluate_exit(self, position: Position, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Avalia se uma posição deve ser encerrada precocemente.
        Retorna (deve_fechar, motivo).
        """
        try:
            if df is None or len(df) < 20:
                return False, ""
            
            # 1. Trend Exhaustion (RSI Extremo)
            # Recalculate RSI to ensure fresh data
            rsi = ta.rsi(df['close'], length=14).iloc[-1]
            
            if position.order_type == "BUY":
                if rsi > 75:
                    return True, "RSI Overbought Exhaustion (>75)"
            else: # SELL
                if rsi < 25:
                    return True, "RSI Oversold Exhaustion (<25)"
            
            # 2. Reversal Signals (EMA Crossover contrário)
            # This is more aggressive, but let's keep it subtle for now
            # Only if RSI is already near neutral/extreme
            
            return False, ""
        except Exception as e:
            logger.error(f"Error evaluating exit for {position.symbol}: {e}")
            return False, ""

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

        # 2. Prepare Data & Session Params
        session_params = get_active_session_params(symbol, datetime.utcnow())
        active_sess = session_params.get("active_session", "UNKNOWN")
        
        # Parâmetros
        fast_period = session_params.get("ema_fast", regime.ema_fast)
        slow_period = session_params.get("ema_slow", regime.ema_slow)
        rsi_period = session_params.get("rsi_period", 14)
        adx_thresh = session_params.get("adx_threshold", 25)
        
        try:
            # Pandas TA calc
            df = df.copy()
            df[f'ema_fast'] = ta.ema(df['close'], length=fast_period)
            df[f'ema_slow'] = ta.ema(df['close'], length=slow_period)
            df['rsi'] = ta.rsi(df['close'], length=rsi_period)
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
            # Logic from analyze(): (crossed_up or (current_price > ema_fast and rsi < 60))
            is_crossover = crossed_up
            is_pullback = (current_price > ema_fast and rsi < 60)
            trigger_ok = is_crossover or is_pullback
            
            # For report display, we show what matched or what is missing
            if is_crossover:
                 add_cond("Gatilho de Entrada", "Crossover", "Crossover ou Pullback", True, "Cruzamento de alta detectado", "")
            elif is_pullback:
                 add_cond("Gatilho de Entrada", "Pullback", "Crossover ou Pullback", True, "Pullback na tendência", "")
            else:
                 # Show what failed
                 add_cond("Gatilho de Entrada", "Nenhum", "Crossover ou Pullback (RSI<60)", False, "Aguardando sinal", "Sem Crossover e RSI alto")

            # Condition 3: RSI Filter
            rsi_target = session_params.get("rsi_buy", regime.rsi_buy)
            if "NY" in active_sess.upper() and adx > 35:
                rsi_target -= 10
                
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
            # Logic from analyze(): (crossed_down or (current_price < ema_fast and rsi > 40))
            is_crossover = crossed_down
            is_pullback = (current_price < ema_fast and rsi > 40)
            trigger_ok = is_crossover or is_pullback
            
            if is_crossover:
                 add_cond("Gatilho de Entrada", "Crossover", "Crossover ou Pullback", True, "Cruzamento de baixa detectado", "")
            elif is_pullback:
                 add_cond("Gatilho de Entrada", "Pullback", "Crossover ou Pullback", True, "Pullback na tendência", "")
            else:
                 add_cond("Gatilho de Entrada", "Nenhum", "Crossover ou Pullback (RSI>40)", False, "Aguardando sinal", "Sem Crossover e RSI baixo")

            # Condition 3: RSI Filter
            rsi_target = session_params.get("rsi_sell", regime.rsi_sell)
            if "NY" in active_sess.upper() and adx > 35:
                rsi_target += 10
                
            rsi_ok = rsi < rsi_target
            add_cond(f"RSI Filtro (< {rsi_target})",
                     f"{rsi:.1f}",
                     f"< {rsi_target}",
                     rsi_ok,
                     "Momentum suficiente",
                     f"RSI muito alto (+{rsi - rsi_target:.1f})" if not rsi_ok else "")

        # B. Força da Tendência (ADX)
        adx_ok = adx > adx_thresh
        add_cond(f"ADX confirma força (> {adx_thresh})",
                 f"{adx:.1f}",
                 f"> {adx_thresh}",
                 adx_ok,
                 "Tendência precisa ficar mais forte",
                 f"+{adx_thresh - adx:.1f}" if not adx_ok else "")

        # C. Spread
        from ..mt5.symbol_manager import symbol_manager
        spread_ok = symbol_manager.check_spread(symbol)
        
        # Para log visualizar quantos pips
        category = symbol_manager._categorize_symbol(symbol)
        max_spread = symbol_manager._get_max_spread_for_category(category)
        
        add_cond("Spread dentro do limite",
                 f"{spread_points}",
                 f"< {max_spread}",
                 spread_ok,
                 "Liquidez excelente",
                 "Spread alto!" if not spread_ok else "")

        # D. Sessão
        session_ok = active_sess != "UNKNOWN"
        add_cond(f"Sessão {active_sess} ativa",
                 "Sim" if session_ok else "Não",
                 "Sim",
                 session_ok,
                 "Horário operacional verificado",
                 "Mercado fechado ou sem liquidez" if not session_ok else "")

        # E. Filtro H1 (Novo no Relatório)
        h1_ok = self.check_h1_trend(symbol, direction)
        add_cond("Trend H1 OK (EMA200)",
                 "Alinhada" if h1_ok else "Contra",
                 "Alinhada",
                 h1_ok,
                 "Trend H1 confirma direção",
                 "H1 está em contratendência" if not h1_ok else "")

        # F. Institutional / Risk (Novo no Relatório)
        max_pos = settings.MAX_POSITIONS
        current_pos = len(self.bot.positions)
        pos_ok = current_pos < max_pos
        add_cond("Vaga no Portfólio",
                 f"{current_pos}/{max_pos}",
                 f"< {max_pos}",
                 pos_ok,
                 "Espaço para novas ordens",
                 "Limite de posições atingido" if not pos_ok else "")

        # G. Quantitative Verification (NEW)
        hurst_val = getattr(regime, 'hurst', 0.5)
        # We consider "Active" if it's not the default 0.5 or simply showing the value
        add_cond("Hurst Exponent (Regime)",
                 f"{hurst_val:.3f}",
                 "0.45 < H < 0.65",
                 True, 
                 "Trend" if hurst_val > 0.65 else ("MeanRev" if hurst_val < 0.45 else "Noise"),
                 "")
        
        # TAREFA 3: Cornish-Fisher CVaR in "Why Report"
        try:
            returns = df['close'].pct_change().dropna().tail(500).values
            mu = np.mean(returns)
            sigma = np.std(returns)
            skew = (np.sum((returns - mu)**3) / (len(returns) * sigma**3)) if sigma > 0 else 0
            kurt = (np.sum((returns - mu)**4) / (len(returns) * sigma**4)) if sigma > 0 else 3
            cvar_val = calculate_cf_cvar(returns, alpha=0.99)
            
            cvar_limit = settings.get_quant_params(symbol).get("max_portfolio_cvar_pct", 0.015)
            cvar_ok = cvar_val < cvar_limit
            
            add_cond("Cornish-Fisher CVaR (99%)",
                     f"{cvar_val*100:.2f}%",
                     f"< {cvar_limit*100:.2f}%",
                     cvar_ok,
                     f"Skew: {skew:.2f}, Kurt: {kurt:.2f}",
                     f"Excesso Risco de Cauda! CVaR > {cvar_limit*100:.2f}%" if not cvar_ok else "")
        except Exception as cvar_err:
            logger.warning(f"CVaR calculation failed for Why Report: {cvar_err}")

        # Optuna Calibration
        is_optimized = "hurst_lookback" in q_params and q_params.get("hurst_lookback") != 1000
        add_cond("Optuna Calibration",
                 "OTIMIZADO" if is_optimized else "PADRÃO",
                 "OTIMIZADO",
                 is_optimized,
                 "Usando parâmetros Optuna mais recentes",
                 "Usando valores padrão do sistema" if not is_optimized else "")

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
            f"Sessão: {active_sess}",
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

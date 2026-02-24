import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

import MetaTrader5 as mt5

import config_forex as config
import utils_forex as utils

DB_PATH = os.path.join("data", "risk_engine.db")


def _db_connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS blocks ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "symbol TEXT, strategy TEXT, reason TEXT, "
        "start_ts INTEGER, end_ts INTEGER, "
        "loss_amount REAL, loss_pct REAL, consecutive_losses INTEGER)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS adjustments ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "position_id INTEGER, symbol TEXT, action TEXT, "
        "before TEXT, after TEXT, timestamp INTEGER, params TEXT)"
    )
    conn.commit()
    return conn


@dataclass
class BlockRule:
    loss_pct_threshold: float
    loss_abs_threshold: float
    consecutive_losses_threshold: int
    duration_minutes: int


class BlockManager:
    def __init__(self):
        self.rule = BlockRule(
            loss_pct_threshold=float(getattr(config, "BLOCK_LOSS_PCT_THRESHOLD", 0.05)),
            loss_abs_threshold=float(getattr(config, "BLOCK_LOSS_ABS_THRESHOLD", 100.0)),
            consecutive_losses_threshold=int(getattr(config, "BLOCK_CONSECUTIVE_LOSSES", 2)),
            duration_minutes=int(getattr(config, "BLOCK_DURATION_MINUTES", 60)),
        )
        self.loss_streaks: Dict[str, int] = {}

    def is_blocked(self, symbol: str, strategy: Optional[str] = None) -> Tuple[bool, str]:
        if not getattr(config, "ENABLE_RISK_BLOCKS", True):
            return False, ""
        try:
            conn = _db_connect()
            cur = conn.cursor()
            now_ts = int(time.time())
            cur.execute(
                "SELECT reason, end_ts FROM blocks WHERE symbol = ? AND end_ts > ? ORDER BY end_ts DESC LIMIT 1",
                (symbol, now_ts),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                reason, end_ts = row
                until = datetime.fromtimestamp(end_ts).strftime("%H:%M")
                return True, f"Bloqueado até {until} | {reason}"
            return False, ""
        except Exception:
            return False, ""

    def block(self, symbol: str, strategy: Optional[str], reason: str, loss_amount: float = 0.0, loss_pct: float = 0.0, consecutive_losses: int = 0):
        try:
            start_ts = int(time.time())
            end_ts = start_ts + self.rule.duration_minutes * 60
            conn = _db_connect()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO blocks (symbol, strategy, reason, start_ts, end_ts, loss_amount, loss_pct, consecutive_losses) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, strategy or "", reason, start_ts, end_ts, float(loss_amount), float(loss_pct), int(consecutive_losses)),
            )
            conn.commit()
            conn.close()
            # Integra com pausa em memória do bot (quando disponível)
            try:
                until_dt = datetime.fromtimestamp(end_ts)
                from bot_forex import PAUSED_SYMBOLS  # lazy import
                PAUSED_SYMBOLS[symbol] = {"until": until_dt, "reason": f"RiskBlock: {reason}"}
            except Exception:
                pass
            utils.logger.info(f"⛔ RiskBlock: {symbol} | {reason} | duração={self.rule.duration_minutes}min")
        except Exception as e:
            utils.logger.error(f"❌ Falha ao aplicar bloqueio para {symbol}: {e}")

    def on_trade_close(self, symbol: str, open_price: float, close_price: float, volume: float, profit: float, side: str):
        try:
            # calcula perda percentual relativa ao preço de entrada
            if open_price <= 0:
                loss_pct = 0.0
            else:
                if side == "BUY":
                    move = (close_price - open_price) / open_price
                else:
                    move = (open_price - close_price) / open_price
                # trata como perda se resultado foi negativo
                loss_pct = move if profit < 0 else 0.0
            loss_pct_abs = abs(loss_pct)
            loss_abs = abs(float(profit or 0.0))
            if profit < 0:
                self.loss_streaks[symbol] = self.loss_streaks.get(symbol, 0) + 1
            else:
                self.loss_streaks[symbol] = 0
            streak = self.loss_streaks.get(symbol, 0)
            should_block = False
            reason = ""
            if loss_pct_abs >= self.rule.loss_pct_threshold:
                should_block = True
                reason = f"Perda {loss_pct_abs:.1%} ≥ {self.rule.loss_pct_threshold:.0%}"
            elif loss_abs >= self.rule.loss_abs_threshold:
                should_block = True
                reason = f"Perda ${loss_abs:.2f} ≥ ${self.rule.loss_abs_threshold:.2f}"
            elif streak >= self.rule.consecutive_losses_threshold:
                should_block = True
                reason = f"{streak} perdas consecutivas (limite {self.rule.consecutive_losses_threshold})"
            if should_block:
                self.block(symbol, None, reason, loss_amount=profit, loss_pct=loss_pct_abs, consecutive_losses=streak)
        except Exception as e:
            utils.logger.error(f"❌ on_trade_close erro: {e}")


class ProfitOptimizer:
    def __init__(self):
        self.enable = bool(getattr(config, "PROFIT_OPTIMIZER_ENABLE", True))
        self.activation_pips = float(getattr(config, "TRAILING_STOP_ACTIVATION_PIPS", 20))
        self.distance_pips = float(getattr(config, "TRAILING_STOP_DISTANCE_PIPS", 15))
        self.step_pips = float(getattr(config, "TRAILING_STOP_STEP_PIPS", 5))
        self.tp_levels_pips: List[float] = list(getattr(config, "TP_LEVELS_PIPS", [40, 80, 120]))
        self.tp_partials: List[float] = list(getattr(config, "TP_PARTIAL_RATIOS", [0.33, 0.33, 0.34]))
        self.processed_levels: Dict[int, List[int]] = {}

    def _record_adjustment(self, pos_id: int, symbol: str, action: str, before: str, after: str, params: dict):
        try:
            conn = _db_connect()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO adjustments (position_id, symbol, action, before, after, timestamp, params) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pos_id, symbol, action, before, after, int(time.time()), str(params)),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _pips_from(self, pos, price: float) -> float:
        pip = utils.get_pip_size(pos.symbol)
        if pip <= 0:
            return 0.0
        if pos.type == mt5.POSITION_TYPE_BUY:
            return (price - pos.price_open) / pip
        else:
            return (pos.price_open - price) / pip

    def _apply_trailing(self, pos) -> None:
        try:
            with utils.mt5_lock:
                tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                return
            ref_price = tick.bid if pos.type == mt5.POSITION_TYPE_SELL else tick.ask
            gained_pips = self._pips_from(pos, ref_price)
            if gained_pips < self.activation_pips:
                return
            pip = utils.get_pip_size(pos.symbol)
            if pip <= 0:
                return
            distance = self.distance_pips * pip
            if pos.type == mt5.POSITION_TYPE_BUY:
                new_sl = ref_price - distance
            else:
                new_sl = ref_price + distance
            before = f"sl={pos.sl:.5f}"
            if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                self._record_adjustment(pos.ticket, pos.symbol, "TRAILING_SL", before, f"sl={new_sl:.5f}", {"activation_pips": self.activation_pips, "distance_pips": self.distance_pips})
        except Exception:
            pass

    def _apply_partial_tp(self, pos) -> None:
        try:
            with utils.mt5_lock:
                tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                return
            ref_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            gained_pips = self._pips_from(pos, ref_price)
            if gained_pips <= 0:
                return
            reached = [i for i, lvl in enumerate(self.tp_levels_pips) if gained_pips >= lvl]
            if not reached:
                return
            already = self.processed_levels.get(pos.ticket, [])
            for idx in reached:
                if idx in already:
                    continue
                part = self.tp_partials[idx] if idx < len(self.tp_partials) else 0.0
                vol = float(pos.volume) * float(part)
                info = utils.get_symbol_info(pos.symbol)
                step = info.volume_step if info else 0.01
                vol = max(getattr(config, "MIN_VOLUME", 0.01), vol)
                # ajusta para step
                try:
                    units = round(vol / step)
                    vol = units * step
                except Exception:
                    pass
                if vol <= 0:
                    continue
                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": vol,
                    "type": order_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": getattr(config, 'DEVIATION', 20),
                    "magic": getattr(config, 'MAGIC_NUMBER', 123456),
                    "comment": f"Partial TP L{idx+1}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                with utils.mt5_lock:
                    res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    before = f"vol={pos.volume:.2f}"
                    self._record_adjustment(pos.ticket, pos.symbol, "PARTIAL_TP", before, f"closed={vol:.2f}", {"level": idx+1, "gained_pips": gained_pips})
                    self.processed_levels.setdefault(pos.ticket, []).append(idx)
                    # move SL para o nível anterior (lock-in)
                    try:
                        prev_lvl = self.tp_levels_pips[idx]
                        pip = utils.get_pip_size(pos.symbol)
                        lock_dist = prev_lvl * pip * 0.5
                        if pos.type == mt5.POSITION_TYPE_BUY:
                            new_sl = pos.price_open + lock_dist
                        else:
                            new_sl = pos.price_open - lock_dist
                        utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp)
                    except Exception:
                        pass
        except Exception:
            pass

    def scan_and_optimize(self):
        if not self.enable:
            return
        try:
            with utils.mt5_lock:
                positions = mt5.positions_get()
            if not positions:
                return
            for pos in positions:
                if int(getattr(pos, "magic", 0)) != int(getattr(config, "MAGIC_NUMBER", 123456)):
                    continue
                self._apply_trailing(pos)
                self._apply_partial_tp(pos)
        except Exception:
            pass


# Singleton-like instances
block_manager = BlockManager()
profit_optimizer = ProfitOptimizer()
class AdaptiveTPManager:
    def __init__(self):
        self.enable = bool(getattr(config, "DYNAMIC_TP_SL_ENABLE", True))
        self.tp_step = float(getattr(config, "TP_ADJUST_STEP", 0.2))
        self.sl_step = float(getattr(config, "SL_ADJUST_STEP", 0.2))
        self.min_tp = float(getattr(config, "MIN_TP_ATR_MULT", 2.0))
        self.max_tp = float(getattr(config, "MAX_TP_ATR_MULT", 6.0))
        self.min_sl = float(getattr(config, "MIN_SL_ATR_MULT", 1.0))
        self.max_sl = float(getattr(config, "MAX_SL_ATR_MULT", 3.0))
        self.win_up = float(getattr(config, "WIN_RATE_THRESHOLD_UP", 0.55))
        self.pf_up = float(getattr(config, "PROFIT_FACTOR_THRESHOLD_UP", 1.3))
        self.win_down = float(getattr(config, "WIN_RATE_THRESHOLD_DOWN", 0.45))
        self.pf_down = float(getattr(config, "PROFIT_FACTOR_THRESHOLD_DOWN", 1.0))
        self.window = int(getattr(config, "PERFORMANCE_WINDOW_TRADES", 20))
        self._init_tables()
    def _init_tables(self):
        conn = _db_connect()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS adaptive_params (symbol TEXT, strategy TEXT, sl_mult REAL, tp_mult REAL, updated_ts INTEGER, PRIMARY KEY(symbol, strategy))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tp_sl_adjustments (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, strategy TEXT, old_sl REAL, old_tp REAL, new_sl REAL, new_tp REAL, reason TEXT, profit REAL, win_rate REAL, profit_factor REAL, timestamp INTEGER)"
        )
        conn.commit()
        conn.close()
    def _get_conn(self):
        return _db_connect()
    def _get_trades_conn(self):
        try:
            path = utils._get_db_path()
        except Exception:
            path = os.path.join("data", "trades.db")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return sqlite3.connect(path)
    def get_current_params(self, symbol: str, strategy: str, base_sl: float, base_tp: float) -> Tuple[float, float]:
        if not self.enable:
            return base_sl, base_tp
        now_ts = int(time.time())
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT sl_mult, tp_mult FROM adaptive_params WHERE symbol = ? AND strategy = ?", (symbol, strategy or ""))
        row = cur.fetchone()
        if row:
            slm, tpm = float(row[0]), float(row[1])
        else:
            slm = float(base_sl)
            tpm = float(base_tp)
            cur.execute("INSERT OR REPLACE INTO adaptive_params (symbol, strategy, sl_mult, tp_mult, updated_ts) VALUES (?, ?, ?, ?, ?)", (symbol, strategy or "", slm, tpm, now_ts))
            conn.commit()
        conn.close()
        slm = max(self.min_sl, min(self.max_sl, slm))
        tpm = max(self.min_tp, min(self.max_tp, tpm))
        return slm, tpm
    def _compute_perf(self, symbol: str, strategy: str) -> Tuple[float, float, float, int]:
        try:
            tconn = self._get_trades_conn()
            tcur = tconn.cursor()
            q = "SELECT profit, comment FROM trades WHERE symbol = ? AND magic = ? ORDER BY close_time DESC LIMIT ?"
            tcur.execute(q, (symbol, int(getattr(config, "MAGIC_NUMBER", 123456)), int(self.window)))
            rows = tcur.fetchall()
            tconn.close()
            filtered = []
            if strategy:
                s_key = f"XP3_{strategy}"
                for p, c in rows:
                    if isinstance(c, str) and s_key in c:
                        filtered.append(float(p or 0.0))
            else:
                filtered = [float(r[0] or 0.0) for r in rows]
            if not filtered:
                return 0.0, 0.0, 0.0, 0
            wins = [x for x in filtered if x > 0]
            losses = [x for x in filtered if x < 0]
            total = len(filtered)
            win_rate = (len(wins) / total) if total > 0 else 0.0
            pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
            avg_profit = (sum(filtered) / total) if total > 0 else 0.0
            return avg_profit, win_rate, pf, total
        except Exception:
            return 0.0, 0.0, 0.0, 0
    def _decide(self, avg_profit: float, win_rate: float, pf: float) -> int:
        if win_rate >= self.win_up and pf >= self.pf_up and avg_profit > 0:
            return 1
        if win_rate <= self.win_down or pf < self.pf_down:
            return -1
        return 0
    def adjust(self, symbol: str, strategy: str, base_sl: float, base_tp: float) -> Tuple[float, float, dict]:
        slm, tpm = self.get_current_params(symbol, strategy, base_sl, base_tp)
        avg_profit, win_rate, pf, total = self._compute_perf(symbol, strategy)
        decision = self._decide(avg_profit, win_rate, pf)
        reason = {"avg_profit": avg_profit, "win_rate": win_rate, "profit_factor": pf, "samples": total, "decision": decision}
        old_sl, old_tp = slm, tpm
        if decision > 0:
            tpm = min(self.max_tp, tpm + self.tp_step)
            slm = max(self.min_sl, slm - self.sl_step)
        elif decision < 0:
            tpm = max(self.min_tp, tpm - self.tp_step)
            slm = min(self.max_sl, slm + self.sl_step)
        tpm = max(self.min_tp, min(self.max_tp, tpm))
        slm = max(self.min_sl, min(self.max_sl, slm))
        now_ts = int(time.time())
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO adaptive_params (symbol, strategy, sl_mult, tp_mult, updated_ts) VALUES (?, ?, ?, ?, ?)", (symbol, strategy or "", slm, tpm, now_ts))
        cur.execute(
            "INSERT INTO tp_sl_adjustments (symbol, strategy, old_sl, old_tp, new_sl, new_tp, reason, profit, win_rate, profit_factor, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, strategy or "", float(old_sl), float(old_tp), float(slm), float(tpm), str(reason), float(avg_profit), float(win_rate), float(pf), now_ts),
        )
        conn.commit()
        conn.close()
        return slm, tpm, reason
    def on_trade_close(self, symbol: str, strategy: Optional[str], profit: float):
        try:
            base_sl = float(getattr(config, "DEFAULT_STOP_LOSS_ATR_MULTIPLIER", 2.0))
            base_tp = float(getattr(config, "DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER", 3.0))
            self.adjust(symbol, strategy or "", base_sl, base_tp)
        except Exception:
            pass
    def backtest(self, window: Optional[int] = None) -> Dict[str, float]:
        w = int(window or getattr(config, "ADAPTIVE_BACKTEST_DEFAULT_WINDOW", 100))
        try:
            tconn = self._get_trades_conn()
            tcur = tconn.cursor()
            tcur.execute("SELECT symbol, profit, comment FROM trades WHERE magic = ? ORDER BY close_time ASC LIMIT ?", (int(getattr(config, "MAGIC_NUMBER", 123456)), w))
            rows = tcur.fetchall()
            tconn.close()
            base_sl = float(getattr(config, "DEFAULT_STOP_LOSS_ATR_MULTIPLIER", 2.0))
            base_tp = float(getattr(config, "DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER", 3.0))
            pnl_static = 0.0
            pnl_dynamic = 0.0
            for sym, p, c in rows:
                pnl_static += float(p or 0.0)
                strat = ""
                if isinstance(c, str) and "XP3_" in c:
                    try:
                        strat = c.split("XP3_")[1].split()[0]
                    except Exception:
                        strat = ""
                slm, tpm, _ = self.adjust(str(sym or ""), strat, base_sl, base_tp)
                scale = (tpm / base_tp)
                pnl_dynamic += float(p or 0.0) * float(scale)
            return {"pnl_static": pnl_static, "pnl_dynamic_scaled": pnl_dynamic}
        except Exception:
            return {"pnl_static": 0.0, "pnl_dynamic_scaled": 0.0}
adaptive_manager = AdaptiveTPManager()

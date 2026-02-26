from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any
import math
try:
    import config_forex as config
except Exception:
    class _Cfg:
        ENABLE_ADAPTIVE_ENGINE = True
        ADAPTIVE_ENGINE_MAX_STRATEGY_CHANGES = 3
        ADAPTIVE_ENGINE_MIN_CONFIDENCE = 0.65
        ADAPTIVE_ENGINE_PANIC_THRESHOLD = 0.85
    config = _Cfg()

class SensorLayer:
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = market_data.get("symbol")
        price = (market_data.get("price_data") or {}).get("current_price", 0.0)
        vol = market_data.get("volatility") or {}
        vol_regime = vol.get("regime", "NORMAL")
        atr_pct = float(vol.get("atr_percent", 0.0))
        volume = market_data.get("volume") or {}
        vol_ratio = float(volume.get("volume_ratio", 1.0))
        session = market_data.get("session", "NORMAL")
        return {
            "symbol": symbol,
            "price": price,
            "vol_regime": vol_regime,
            "atr_percent": atr_pct,
            "volume_ratio": vol_ratio,
            "session": session
        }

class BrainLayer:
    def decide(self, sensor_out: Dict[str, Any]) -> Dict[str, Any]:
        vol_regime = sensor_out.get("vol_regime", "NORMAL")
        vol_ratio = float(sensor_out.get("volume_ratio", 1.0))
        base_conf = 0.7
        if vol_regime == "HIGH":
            ema_short = 10
            ema_long = 40
            rsi_low = 35
            rsi_high = 65
            base_conf -= 0.05
        elif vol_regime == "LOW":
            ema_short = 25
            ema_long = 60
            rsi_low = 28
            rsi_high = 72
            base_conf += 0.05
        else:
            ema_short = 20
            ema_long = 50
            rsi_low = 30
            rsi_high = 70
        if vol_ratio > 1.3:
            ema_short = max(8, ema_short - 2)
        if vol_ratio < 0.8:
            ema_long = min(80, ema_long + 5)
        confidence = max(0.0, min(1.0, base_conf))
        return {
            "parameter_adjustments": {
                "ema_short": ema_short,
                "ema_long": ema_long,
                "rsi_low": rsi_low,
                "rsi_high": rsi_high
            },
            "confidence": confidence
        }

class MechanicLayer:
    def apply(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        return {"applied": True, "params": adjustments}

class EvolutionLayer:
    def __init__(self):
        self.memory = deque(maxlen=getattr(config, "ADAPTIVE_ENGINE_MEMORY_SIZE", 1000))
        self.learning_rate = float(getattr(config, "ADAPTIVE_ENGINE_LEARNING_RATE", 0.1))
    def update(self, feedback: Dict[str, Any]) -> None:
        self.memory.append(feedback)

class PanicMode:
    def __init__(self):
        self.active = False
        self.reason = ""
    def evaluate(self, sensor_out: Dict[str, Any], brain_out: Dict[str, Any]) -> Dict[str, Any]:
        atr_pct = float(sensor_out.get("atr_percent", 0.0))
        vol_regime = sensor_out.get("vol_regime", "NORMAL")
        confidence = float(brain_out.get("confidence", 0.0))
        risk = 0.0
        if vol_regime == "HIGH":
            risk += 0.4
        risk += min(0.5, atr_pct / 10.0)
        risk -= max(0.0, (confidence - 0.5))
        threshold = float(getattr(config, "ADAPTIVE_ENGINE_PANIC_THRESHOLD", 0.85))
        self.active = risk >= threshold
        self.reason = "high_volatility" if self.active else ""
        return {"panic_mode_active": self.active, "panic_reason": self.reason}

class AdaptiveEngine:
    def __init__(self):
        self.sensor = SensorLayer()
        self.brain = BrainLayer()
        self.mechanic = MechanicLayer()
        self.evolution = EvolutionLayer()
        self.strategy_changes: Dict[str, deque] = {}
        self.max_changes = int(getattr(config, "ADAPTIVE_ENGINE_MAX_STRATEGY_CHANGES", 3))
        self.conf_min = float(getattr(config, "ADAPTIVE_ENGINE_MIN_CONFIDENCE", 0.65))
        self.panic = PanicMode()
    def _allow_strategy_change(self, symbol: str) -> bool:
        now = datetime.utcnow()
        dq = self.strategy_changes.get(symbol)
        if dq is None:
            dq = deque()
            self.strategy_changes[symbol] = dq
        while dq and (now - dq[0]) > timedelta(hours=1):
            dq.popleft()
        if len(dq) >= self.max_changes:
            return False
        dq.append(now)
        return True
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        sensor_out = self.sensor.analyze(market_data)
        brain_out = self.brain.decide(sensor_out)
        panic_out = self.panic.evaluate(sensor_out, brain_out)
        if brain_out.get("confidence", 0.0) < self.conf_min:
            adjustments = {}
        else:
            adjustments = brain_out.get("parameter_adjustments") or {}
        if not self._allow_strategy_change(sensor_out.get("symbol") or ""):
            adjustments = {}
        mech_out = self.mechanic.apply(adjustments)
        self.evolution.update({
            "symbol": sensor_out.get("symbol"),
            "session": sensor_out.get("session"),
            "confidence": brain_out.get("confidence"),
            "applied": mech_out.get("applied")
        })
        return {
            "panic_mode_active": panic_out.get("panic_mode_active", False),
            "panic_reason": panic_out.get("panic_reason", ""),
            "parameter_adjustments": adjustments
        }

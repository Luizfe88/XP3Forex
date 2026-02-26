import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config_forex as config

try:
    import optimizer_optuna_forex as optimizer
except Exception:
    optimizer = None

try:
    import otimizador_semanal_forex as weekly_optimizer
except Exception:
    weekly_optimizer = None

try:
    import utils_forex as utils
except Exception:
    utils = None


@dataclass
class AssetScore:
    symbol: str
    asset_class: str
    eligible: bool
    reasons: List[str]
    data_source: str
    presence_ratio: float
    avg_daily_liquidity_brl: float
    vol_annualized: float
    avg_spread_pips: float
    avg_abs_corr: float
    perf_score_3m: float
    perf_score_30d: float
    rank_score: float
    details: Dict[str, Any]

def _escape_html(text: str) -> str:
    try:
        import html
        return html.escape(text or "")
    except Exception:
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _now_utc() -> datetime:
    try:
        import pytz
        return datetime.now(pytz.utc)
    except Exception:
        return datetime.utcnow()


def _safe_divide(num: float, den: float, default: float = 0.0) -> float:
    try:
        if den == 0:
            return default
        return float(num) / float(den)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _min_max_scale(values: Dict[str, float], invert: bool = False) -> Dict[str, float]:
    if not values:
        return {}
    vs = np.array(list(values.values()), dtype=np.float64)
    finite = vs[np.isfinite(vs)]
    if finite.size == 0:
        return {k: 0.0 for k in values.keys()}
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    out: Dict[str, float] = {}
    for k, v in values.items():
        if not math.isfinite(v):
            s = 0.0
        elif hi == lo:
            s = 1.0
        else:
            s = (float(v) - lo) / (hi - lo)
        if invert:
            s = 1.0 - s
        out[k] = float(np.clip(s, 0.0, 1.0))
    return out


def _vol_score(vol_annualized: float, min_v: float = 0.05, max_v: float = 0.20) -> float:
    if not math.isfinite(vol_annualized):
        return 0.0
    if vol_annualized < min_v or vol_annualized > max_v:
        return 0.0
    mid = (min_v + max_v) / 2.0
    half = (max_v - min_v) / 2.0
    return float(np.clip(1.0 - abs(vol_annualized - mid) / half, 0.0, 1.0))


def _load_universe() -> List[str]:
    if hasattr(config, "SYMBOL_MAP"):
        try:
            return list(config.SYMBOL_MAP)
        except Exception:
            pass
    if hasattr(config, "FOREX_PAIRS"):
        try:
            return list(config.FOREX_PAIRS.keys())
        except Exception:
            pass
    return []

def get_asset_class(symbol: str) -> str:
    overrides = getattr(config, "ASSET_CLASS_OVERRIDES", {})
    if isinstance(overrides, dict) and symbol in overrides:
        try:
            return str(overrides[symbol]).upper()
        except Exception:
            pass

    s = str(symbol).upper()

    if ".NAS" in s or ".NYSE" in s:
        return "EQUITIES"

    index_tokens = ["US30", "UK100", "NAS100", "USTEC", "SPX", "US500", "GER", "DE40", "DAX", "JP225", "HK50"]
    if any(tok in s for tok in index_tokens):
        return "INDICES"

    if s.startswith("XAU") or s.startswith("XAG"):
        return "METALS"

    crypto_tokens = ["BTC", "ETH", "SOL", "BNB", "DOGE", "ADA", "XRP", "LTC", "AVAX", "DOT", "NETH"]
    if any(tok in s for tok in crypto_tokens):
        return "CRYPTO"

    if len(s) == 6 and s.isalpha():
        return "FX"
    if "JPY" in s and len(s) >= 6:
        return "FX"

    return "OTHER"

def _get_class_rules(asset_class: str) -> Dict[str, Any]:
    rules = getattr(config, "ASSET_CLASS_RULES", {})
    if isinstance(rules, dict) and asset_class in rules and isinstance(rules[asset_class], dict):
        return dict(rules[asset_class])
    return dict(getattr(config, "ASSET_CLASS_RULES_DEFAULT", {}))


def _load_data(symbol: str) -> Optional[Dict[str, Any]]:
    loader = None
    if weekly_optimizer and hasattr(weekly_optimizer, "load_data_v7"):
        loader = weekly_optimizer.load_data_v7
    if loader is None:
        return None
    try:
        return loader(symbol)
    except Exception:
        return None


def _daily_ohlc(df_m15: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df_m15.index, pd.DatetimeIndex):
        df_m15 = df_m15.copy()
        df_m15.index = pd.to_datetime(df_m15.index)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
    }
    if "real_volume" in df_m15.columns:
        agg["real_volume"] = "sum"
    daily = df_m15.resample("1D").agg(agg).dropna()
    daily = daily[daily["tick_volume"] > 0]
    return daily


def _calc_presence_ratio(daily: pd.DataFrame, lookback_days: int = 30, expected_mode: str = "BUSINESS") -> float:
    if daily.empty:
        return 0.0
    end = daily.index.max()
    start = end - pd.Timedelta(days=lookback_days)
    window = daily.loc[(daily.index >= start) & (daily.index <= end)]
    if window.empty:
        return 0.0
    expected_mode = str(expected_mode or getattr(config, "PRESENCE_EXPECTED_DAYS_MODE", "BUSINESS")).upper()
    expected = 0
    try:
        if expected_mode == "CALENDAR":
            expected = (end.normalize() - start.normalize()).days + 1
        elif expected_mode == "BUSINESS":
            expected = len(pd.bdate_range(start.normalize(), end.normalize()))
        else:
            expected = len(pd.bdate_range(start.normalize(), end.normalize()))
    except Exception:
        expected = (end.normalize() - start.normalize()).days + 1
    present = int(window.shape[0])
    return float(np.clip(present / max(1, expected), 0.0, 1.0))


def _calc_liquidity_brl(daily: pd.DataFrame) -> float:
    if daily.empty:
        return 0.0
    fx_usdbrl = float(getattr(config, "USD_BRL", 5.0))
    use_tick_volume = bool(getattr(config, "LIQUIDITY_USE_TICK_VOLUME", True))
    if not use_tick_volume:
        return 0.0
    units_per_tick = float(getattr(config, "LIQUIDITY_UNITS_PER_TICK", 1000.0))
    units_per_real = float(getattr(config, "LIQUIDITY_UNITS_PER_REAL_VOLUME", units_per_tick))
    volume_col = "real_volume" if "real_volume" in daily.columns and daily["real_volume"].fillna(0).sum() > 0 else "tick_volume"
    units = units_per_real if volume_col == "real_volume" else units_per_tick
    notional_native = (
        daily["close"].astype(np.float64)
        * daily[volume_col].astype(np.float64)
        * units
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if notional_native.empty:
        return 0.0
    currency = str(getattr(config, "LIQUIDITY_CURRENCY", "USD")).upper()
    if currency == "BRL":
        return float(notional_native.mean())
    return float(notional_native.mean() * fx_usdbrl)


def _calc_vol_annualized(daily: pd.DataFrame, lookback_days: int = 30) -> float:
    if daily.empty:
        return float("nan")
    end = daily.index.max()
    start = end - pd.Timedelta(days=lookback_days)
    window = daily.loc[(daily.index >= start) & (daily.index <= end)]
    if window.shape[0] < 10:
        return float("nan")
    rets = window["close"].pct_change().dropna()
    if rets.empty:
        return float("nan")
    vol = float(np.std(rets.values, ddof=1) * math.sqrt(252))
    return vol


def _calc_cost_spread_pips(data_bundle: Dict[str, Any], symbol: str) -> float:
    spread = _as_float(data_bundle.get("spread"), default=float("nan"))
    if math.isfinite(spread):
        return spread
    if utils and hasattr(utils, "get_symbol_info"):
        try:
            info = utils.get_symbol_info(symbol)
            if info and getattr(info, "point", 0) > 0:
                pip = 0.01 if "JPY" in symbol.upper() else 0.0001
                return float(getattr(info, "spread", 0) * info.point / pip)
        except Exception:
            pass
    return float("nan")


def _backtest_score(data_bundle: Dict[str, Any], symbol: str, params: Dict[str, Any], lookback_days: int) -> Tuple[float, Dict[str, Any]]:
    if optimizer is None:
        return 0.0, {"error": "optimizer_unavailable"}
    df = data_bundle.get("df")
    if df is None or df.empty:
        return 0.0, {"error": "no_df"}
    cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
    df_slice = df[df.index >= cutoff]
    if df_slice.shape[0] < 1500:
        df_slice = df.iloc[max(0, len(df) - 8000):]
    bundle = dict(data_bundle)
    bundle["df"] = df_slice
    if "news_mask" in bundle and isinstance(bundle["news_mask"], np.ndarray):
        if bundle["news_mask"].shape[0] == df.shape[0]:
            bundle["news_mask"] = bundle["news_mask"][-len(df_slice):]
    if "news_mask" not in bundle:
        bundle["news_mask"] = np.zeros(len(df_slice), dtype=np.bool_)
    try:
        metrics, trade_res, bal = optimizer.run_backtest_with_params(bundle, params)
        if not metrics:
            return 0.0, {"error": "no_metrics"}
        return float(metrics.get("score_v7", 0.0)), {"metrics": metrics, "trades": int(len(trade_res))}
    except Exception as e:
        return 0.0, {"error": str(e)}


def _params_for_symbol(symbol: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if hasattr(config, "FOREX_PAIRS") and symbol in getattr(config, "FOREX_PAIRS", {}):
        try:
            params.update(getattr(config, "FOREX_PAIRS")[symbol])
        except Exception:
            pass
    required = {
        "ema_short": 20,
        "ema_long": 50,
        "rsi_low": 30,
        "rsi_high": 70,
        "adx_threshold": 15,
        "sl_atr": 2.0,
        "tp_atr": 3.0,
        "ml_threshold": 0.6,
    }
    for k, v in required.items():
        params.setdefault(k, v)
    return params


def _corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    return returns.corr().fillna(0.0)


def compute_rankings(date: Optional[str] = None) -> Tuple[List[AssetScore], Dict[str, Any]]:
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    corr_lookback_days = int(getattr(config, "CORR_LOOKBACK_DAYS", 30))

    universe = _load_universe()
    items: List[Dict[str, Any]] = []
    daily_returns: Dict[str, pd.Series] = {}

    for sym in universe:
        data = _load_data(sym)
        asset_class = get_asset_class(sym)
        class_rules = _get_class_rules(asset_class)
        if not data or "df" not in data:
            items.append({"symbol": sym, "asset_class": asset_class, "error": "no_data"})
            continue
        df = data["df"]
        daily = _daily_ohlc(df)
        presence_mode = str(class_rules.get("presence_mode", getattr(config, "PRESENCE_EXPECTED_DAYS_MODE", "BUSINESS")))
        presence = _calc_presence_ratio(daily, lookback_days=30, expected_mode=presence_mode)
        liq_brl = _calc_liquidity_brl(daily.tail(30))
        vol = _calc_vol_annualized(daily, lookback_days=30)
        spread_pips = _calc_cost_spread_pips(data, sym)

        end = daily.index.max()
        start = end - pd.Timedelta(days=corr_lookback_days)
        window = daily.loc[(daily.index >= start) & (daily.index <= end)]
        if window.shape[0] >= 10:
            daily_returns[sym] = window["close"].pct_change().dropna()

        items.append(
            {
                "symbol": sym,
                "asset_class": asset_class,
                "class_rules": class_rules,
                "data": data,
                "presence": presence,
                "liq_brl": liq_brl,
                "vol": vol,
                "spread_pips": spread_pips,
                "source": str(data.get("source", "UNKNOWN")),
            }
        )

    ret_df = pd.DataFrame(daily_returns).dropna(how="all")
    corr = _corr_matrix(ret_df.dropna())
    avg_abs_corr_map: Dict[str, float] = {}
    if not corr.empty:
        for sym in corr.columns:
            others = corr[sym].drop(labels=[sym], errors="ignore")
            avg_abs_corr_map[sym] = float(np.mean(np.abs(others.values))) if others.size else 0.0

    perf_3m: Dict[str, float] = {}
    perf_30d: Dict[str, float] = {}
    liq_map: Dict[str, float] = {}
    vol_map: Dict[str, float] = {}
    cost_map: Dict[str, float] = {}

    details_map: Dict[str, Dict[str, Any]] = {}

    for it in items:
        sym = it["symbol"]
        liq_map[sym] = _as_float(it.get("liq_brl"), default=0.0)
        vol_map[sym] = _as_float(it.get("vol"), default=float("nan"))
        cost_map[sym] = _as_float(it.get("spread_pips"), default=float("nan"))

        data = it.get("data")
        if data and optimizer is not None:
            params = _params_for_symbol(sym)
            s3, d3 = _backtest_score(data, sym, params, lookback_days=90)
            s30, d30 = _backtest_score(data, sym, params, lookback_days=30)
            perf_3m[sym] = s3
            perf_30d[sym] = s30
            details_map[sym] = {"bt_3m": d3, "bt_30d": d30}
        else:
            perf_3m[sym] = 0.0
            perf_30d[sym] = 0.0
            details_map[sym] = {"bt_3m": {"error": "no_optimizer"}, "bt_30d": {"error": "no_optimizer"}}

    perf_30d_scaled = _min_max_scale(perf_30d, invert=False)
    liq_scaled = _min_max_scale(liq_map, invert=False)
    cost_scaled = _min_max_scale(cost_map, invert=True)

    rank_weights = getattr(
        config,
        "RANK_WEIGHTS",
        {"performance": 0.30, "liquidity": 0.25, "volatility": 0.20, "correlation": 0.15, "cost": 0.10},
    )
    w_perf = float(rank_weights.get("performance", 0.30))
    w_liq = float(rank_weights.get("liquidity", 0.25))
    w_vol = float(rank_weights.get("volatility", 0.20))
    w_corr = float(rank_weights.get("correlation", 0.15))
    w_cost = float(rank_weights.get("cost", 0.10))

    scores: List[AssetScore] = []
    for it in items:
        sym = it["symbol"]
        asset_class = str(it.get("asset_class", "OTHER"))
        class_rules = dict(it.get("class_rules") or _get_class_rules(asset_class))
        reasons: List[str] = []
        eligible = True

        presence = _as_float(it.get("presence"), 0.0)
        liq_brl = _as_float(it.get("liq_brl"), 0.0)
        vol = _as_float(it.get("vol"), float("nan"))
        spread = _as_float(it.get("spread_pips"), float("nan"))
        avg_abs_corr = _as_float(avg_abs_corr_map.get(sym, 0.0), 0.0)

        min_presence = float(class_rules.get("min_presence_ratio", getattr(config, "MIN_PRESENCE_RATIO", 0.80)))
        min_liq_brl = float(class_rules.get("min_daily_liquidity_brl", getattr(config, "MIN_DAILY_LIQUIDITY_BRL", 1_000_000.0)))
        min_vol = float(class_rules.get("min_vol_annualized", getattr(config, "MIN_VOL_ANNUALIZED", 0.05)))
        max_vol = float(class_rules.get("max_vol_annualized", getattr(config, "MAX_VOL_ANNUALIZED", 0.20)))

        if presence < min_presence:
            eligible = False
            reasons.append(f"presença<{min_presence:.0%} ({presence:.0%})")
        if liq_brl < min_liq_brl:
            eligible = False
            reasons.append(f"liquidez<{min_liq_brl:,.0f} (R$ {liq_brl:,.0f})")
        if not math.isfinite(vol) or vol < min_vol or vol > max_vol:
            eligible = False
            reasons.append(f"vol fora {min_vol:.0%}-{max_vol:.0%} ({vol:.1%})" if math.isfinite(vol) else "vol inválida")

        corr_score = float(np.clip(1.0 - avg_abs_corr, 0.0, 1.0))
        vol_score = _vol_score(vol, min_v=min_vol, max_v=max_vol)

        rank_score = (
            w_perf * perf_30d_scaled.get(sym, 0.0)
            + w_liq * liq_scaled.get(sym, 0.0)
            + w_vol * vol_score
            + w_corr * corr_score
            + w_cost * cost_scaled.get(sym, 0.0)
        )

        scores.append(
            AssetScore(
                symbol=sym,
                asset_class=asset_class,
                eligible=eligible,
                reasons=reasons,
                data_source=str(it.get("source", "UNKNOWN")),
                presence_ratio=presence,
                avg_daily_liquidity_brl=liq_brl,
                vol_annualized=vol,
                avg_spread_pips=spread,
                avg_abs_corr=avg_abs_corr,
                perf_score_3m=float(perf_3m.get(sym, 0.0)),
                perf_score_30d=float(perf_30d.get(sym, 0.0)),
                rank_score=float(rank_score),
                details=details_map.get(sym, {}),
            )
        )

    meta = {
        "date": date,
        "universe_size": len(universe),
        "eligible_count": int(sum(1 for s in scores if s.eligible)),
        "criteria": {
            "rules_by_class": getattr(config, "ASSET_CLASS_RULES", {}),
        },
        "weights": {"performance": w_perf, "liquidity": w_liq, "volatility": w_vol, "correlation": w_corr, "cost": w_cost},
    }
    return scores, meta


def select_assets(scores: List[AssetScore], target_min: int = 15, target_max: int = 25) -> Tuple[List[str], Dict[str, Any]]:
    max_pair_corr = float(getattr(config, "MAX_PAIRWISE_CORR", 0.75))
    corr_relax_step = float(getattr(config, "CORR_RELAX_STEP", 0.05))
    corr_lookback_days = int(getattr(config, "CORR_LOOKBACK_DAYS", 30))
    class_targets = getattr(config, "ASSET_CLASS_TARGETS", {})

    eligible = [s for s in scores if s.eligible]
    eligible.sort(key=lambda x: x.rank_score, reverse=True)

    universe = [s.symbol for s in eligible]
    ret_map: Dict[str, pd.Series] = {}
    for sym in universe:
        data = _load_data(sym)
        if not data or "df" not in data:
            continue
        daily = _daily_ohlc(data["df"])
        end = daily.index.max()
        start = end - pd.Timedelta(days=corr_lookback_days)
        window = daily.loc[(daily.index >= start) & (daily.index <= end)]
        if window.shape[0] < 10:
            continue
        ret_map[sym] = window["close"].pct_change().dropna()

    ret_df = pd.DataFrame(ret_map).dropna(how="all")
    corr = _corr_matrix(ret_df.dropna())

    selected: List[str] = []
    selected_by_class: Dict[str, int] = {}
    relaxed = max_pair_corr

    def _class_min_max(asset_class: str) -> Tuple[int, int]:
        if isinstance(class_targets, dict) and asset_class in class_targets and isinstance(class_targets[asset_class], dict):
            c = class_targets[asset_class]
            return int(c.get("min", 0)), int(c.get("max", 10_000))
        return 0, 10_000

    def _can_add(sym: str, asset_class: str, relaxed_thr: float) -> bool:
        _, cmax = _class_min_max(asset_class)
        if selected_by_class.get(asset_class, 0) >= cmax:
            return False
        if not selected:
            return True
        if not corr.empty and sym in corr.columns:
            for other in selected:
                if other in corr.columns:
                    if abs(float(corr.loc[sym, other])) > relaxed_thr:
                        return False
        return True

    def _can_add_force(sym: str, relaxed_thr: float = 0.95) -> bool:
        if not selected:
            return True
        if not corr.empty and sym in corr.columns:
            for other in selected:
                if other in corr.columns:
                    if abs(float(corr.loc[sym, other])) > relaxed_thr:
                        return False
        return True

    def _add(sym: str, asset_class: str) -> None:
        selected.append(sym)
        selected_by_class[asset_class] = selected_by_class.get(asset_class, 0) + 1

    if isinstance(class_targets, dict) and class_targets:
        for asset_class, target in class_targets.items():
            cmin = int(target.get("min", 0)) if isinstance(target, dict) else 0
            if cmin <= 0:
                continue
            for s in eligible:
                if len(selected) >= target_max:
                    break
                if s.symbol in selected:
                    continue
                if s.asset_class != asset_class:
                    continue
                if _can_add(s.symbol, asset_class, relaxed):
                    _add(s.symbol, asset_class)
                if selected_by_class.get(asset_class, 0) >= cmin:
                    break

    while True:
        selected = selected[:]
        for s in eligible:
            if len(selected) >= target_max:
                break
            sym = s.symbol
            if sym in selected:
                continue
            if _can_add(sym, s.asset_class, relaxed):
                _add(sym, s.asset_class)
        if len(selected) >= target_min or relaxed >= 0.95:
            break
        relaxed = min(0.95, relaxed + corr_relax_step)

    weekday_min_available = None
    pct_days_meeting_req = None
    if not ret_df.empty and selected:
        try:
            sub = ret_df[selected]
            try:
                end = sub.index.max()
                sub = sub[sub.index >= (end - pd.Timedelta(days=int(getattr(config, "AVAILABILITY_LOOKBACK_DAYS", 10))))]
            except Exception:
                pass
            try:
                mode = str(getattr(config, "AVAILABILITY_DAY_MODE", "BUSINESS")).upper()
                if mode == "BUSINESS":
                    sub = sub[sub.index.dayofweek < 5]
            except Exception:
                pass
            counts_by_day = sub.notna().sum(axis=1)
            min_required = int(getattr(config, "MIN_ASSETS_PER_DAY", 5))
            nonzero = counts_by_day[counts_by_day > 0]
            weekday_min_available = int(nonzero.min()) if not nonzero.empty else None
            pct_days_meeting_req = float((counts_by_day >= min_required).mean()) if not counts_by_day.empty else None
            if weekday_min_available is not None and weekday_min_available < min_required:
                for s in eligible:
                    if len(selected) >= target_max:
                        break
                    if s.symbol in selected:
                        continue
                    if _can_add_force(s.symbol, 0.95):
                        _add(s.symbol, s.asset_class)
                    else:
                        continue
                    sub = ret_df[selected]
                    try:
                        end = sub.index.max()
                        sub = sub[sub.index >= (end - pd.Timedelta(days=int(getattr(config, "AVAILABILITY_LOOKBACK_DAYS", 10))))]
                    except Exception:
                        pass
                    try:
                        mode = str(getattr(config, "AVAILABILITY_DAY_MODE", "BUSINESS")).upper()
                        if mode == "BUSINESS":
                            sub = sub[sub.index.dayofweek < 5]
                    except Exception:
                        pass
                    counts_by_day = sub.notna().sum(axis=1)
                    nonzero = counts_by_day[counts_by_day > 0]
                    weekday_min_available = int(nonzero.min()) if not nonzero.empty else weekday_min_available
                    pct_days_meeting_req = float((counts_by_day >= min_required).mean()) if not counts_by_day.empty else pct_days_meeting_req
                    if weekday_min_available is not None and weekday_min_available >= min_required:
                        break
        except Exception:
            weekday_min_available = None
            pct_days_meeting_req = None

    selected = selected[:target_max]
    selected_by_class_final: Dict[str, int] = {}
    for s in selected:
        cls = get_asset_class(s)
        selected_by_class_final[cls] = selected_by_class_final.get(cls, 0) + 1

    info = {
        "target_min": target_min,
        "target_max": target_max,
        "max_pairwise_corr_initial": max_pair_corr,
        "max_pairwise_corr_used": relaxed,
        "selected_count": len(selected),
        "min_assets_per_day_required": int(getattr(config, "MIN_ASSETS_PER_DAY", 5)),
        "min_assets_per_day_observed": weekday_min_available,
        "pct_days_meeting_min_assets": pct_days_meeting_req,
        "selected_by_class": selected_by_class_final,
    }
    return selected, info


def apply_weekly_turnover(new_selected: List[str], prev_selected: List[str], max_turnover_pct: float = 0.30) -> List[str]:
    if not prev_selected:
        return new_selected
    max_replace = int(math.floor(len(prev_selected) * max_turnover_pct))
    max_replace = max(0, max_replace)
    keep_target = max(0, len(prev_selected) - max_replace)

    keep = [s for s in prev_selected if s in new_selected][:keep_target]
    fill = [s for s in new_selected if s not in keep]
    result = keep + fill
    return result[: max(len(prev_selected), len(new_selected))]


def write_outputs(date_str: str, scores: List[AssetScore], meta: Dict[str, Any], selected: List[str], selection_meta: Dict[str, Any]) -> Dict[str, str]:
    out_dir = Path(getattr(config, "DATA_DIR", Path("data")))
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = out_dir / f"asset_ranking_{date_str}.json"
    selected_path = out_dir / "selected_assets.json"
    alerts_path = out_dir / f"asset_alerts_{date_str}.json"

    ranking_payload = {
        "meta": meta,
        "scores": [
            {
                "symbol": s.symbol,
                "asset_class": s.asset_class,
                "eligible": s.eligible,
                "reasons": s.reasons,
                "data_source": s.data_source,
                "presence_ratio": s.presence_ratio,
                "avg_daily_liquidity_brl": s.avg_daily_liquidity_brl,
                "vol_annualized": s.vol_annualized,
                "avg_spread_pips": s.avg_spread_pips,
                "avg_abs_corr": s.avg_abs_corr,
                "perf_score_3m": s.perf_score_3m,
                "perf_score_30d": s.perf_score_30d,
                "rank_score": s.rank_score,
                "details": s.details,
            }
            for s in sorted(scores, key=lambda x: x.rank_score, reverse=True)
        ],
    }

    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(ranking_payload, f, ensure_ascii=False, indent=2)

    prev_selected: List[str] = []
    if selected_path.exists():
        try:
            prev_payload = json.loads(selected_path.read_text(encoding="utf-8"))
            prev_selected = list(prev_payload.get("symbols", []))
        except Exception:
            prev_selected = []

    max_turnover_pct = float(getattr(config, "WEEKLY_MAX_TURNOVER_PCT", 0.30))
    rebalance_on_weekday = int(getattr(config, "REBALANCE_WEEKDAY", 0))
    today_weekday = datetime.now().weekday()
    final_selected = selected
    if today_weekday == rebalance_on_weekday:
        final_selected = apply_weekly_turnover(selected, prev_selected, max_turnover_pct=max_turnover_pct)

    selected_payload = {
        "date": date_str,
        "symbols": final_selected,
        "selection": selection_meta,
        "turnover": {
            "prev_count": len(prev_selected),
            "new_count": len(selected),
            "final_count": len(final_selected),
            "max_turnover_pct": max_turnover_pct,
        },
    }
    selected_path.write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    alerts = []
    current_eligible = {s.symbol: s for s in scores}
    for sym in prev_selected:
        s = current_eligible.get(sym)
        if s and not s.eligible:
            alerts.append({"symbol": sym, "reason": "; ".join(s.reasons)})

    with open(alerts_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "alerts": alerts}, f, ensure_ascii=False, indent=2)

    if alerts and utils and hasattr(utils, "send_telegram_alert"):
        try:
            msg = "<b>⚠️ Ativos fora dos critérios</b>\n" + "\n".join(
                [f"- {_escape_html(a['symbol'])}: {_escape_html(a['reason'])}" for a in alerts[:15]]
            )
            utils.send_telegram_alert(msg, "WARNING")
        except Exception:
            pass

    return {"ranking_path": str(ranking_path), "selected_path": str(selected_path), "alerts_path": str(alerts_path)}


def run_daily_update(date_str: Optional[str] = None) -> Dict[str, Any]:
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    scores, meta = compute_rankings(date_str)
    target_min = int(getattr(config, "TARGET_ASSETS_MIN", 15))
    target_max = int(getattr(config, "TARGET_ASSETS_MAX", 25))
    selected, selection_meta = select_assets(scores, target_min=target_min, target_max=target_max)
    paths = write_outputs(date_str, scores, meta, selected, selection_meta)

    return {"date": date_str, "selected": selected, "meta": meta, "selection": selection_meta, "paths": paths}

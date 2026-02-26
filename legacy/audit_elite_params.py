import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config_forex as config
import asset_selection as asset_sel
import otimizador_semanal_forex as opt_v7


REQUIRED_FIELDS = {
    "ema_short": int,
    "ema_long": int,
    "rsi_low": int,
    "rsi_high": int,
    "adx_threshold": int,
    "sl_atr": (int, float),
    "tp_atr": (int, float),
    "ml_threshold": (int, float),
}

BENCHMARK_RANGES = {
    "ema_short": (8, 30),
    "ema_long": (40, 200),
    "rsi_low": (20, 50),
    "rsi_high": (50, 80),
    "adx_threshold": (5, 25),
    "sl_atr": (1.0, 3.5),
    "tp_atr": (1.5, 5.0),
    "ml_threshold": (0.5, 0.7),
}


@dataclass
class SymbolAudit:
    symbol: str
    asset_class: str
    params: Dict[str, Any]
    schema_ok: bool
    schema_issues: List[str]
    range_issues: List[str]
    data_ok: bool
    data_source: str
    candles: int
    last_ts: Optional[str]
    age_days: Optional[float]
    vol_30d_ann: Optional[float]
    vol_90d_ann: Optional[float]
    rr_atr: Optional[float]
    heuristics: List[str]


def _now() -> datetime:
    return datetime.now()


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _escape_md(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _load_json(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("elite_params json não é um dict")
    for k, v in data.items():
        if not isinstance(v, dict):
            raise ValueError(f"elite_params[{k}] não é um dict")
    return data


def _schema_validate(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    for k, typ in REQUIRED_FIELDS.items():
        if k not in params:
            issues.append(f"campo ausente: {k}")
            continue
        if not isinstance(params[k], typ):
            issues.append(f"tipo inválido: {k}={type(params[k]).__name__}")
    return (len(issues) == 0), issues


def _range_validate(params: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for k, (lo, hi) in BENCHMARK_RANGES.items():
        if k not in params:
            continue
        v = _as_float(params.get(k))
        if v is None or not math.isfinite(v):
            issues.append(f"{k} inválido: {params.get(k)}")
            continue
        if v < lo or v > hi:
            issues.append(f"{k} fora do range [{lo}, {hi}]: {v}")
    if "rsi_low" in params and "rsi_high" in params:
        lo = _as_float(params.get("rsi_low"))
        hi = _as_float(params.get("rsi_high"))
        if lo is not None and hi is not None and lo >= hi:
            issues.append(f"rsi_low>=rsi_high ({lo}>={hi})")
    if "ema_short" in params and "ema_long" in params:
        es = _as_float(params.get("ema_short"))
        el = _as_float(params.get("ema_long"))
        if es is not None and el is not None and es >= el:
            issues.append(f"ema_short>=ema_long ({es}>={el})")
    return issues


def _daily_df(df_m15: pd.DataFrame) -> pd.DataFrame:
    df = df_m15.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}).dropna()
    daily = daily[daily["tick_volume"] > 0]
    return daily


def _vol_annualized(daily: pd.DataFrame, lookback_days: int) -> Optional[float]:
    if daily.empty:
        return None
    end = daily.index.max()
    start = end - pd.Timedelta(days=lookback_days)
    w = daily.loc[(daily.index >= start) & (daily.index <= end)]
    if w.shape[0] < 10:
        return None
    r = w["close"].pct_change().dropna()
    if r.empty:
        return None
    return float(np.std(r.values, ddof=1) * math.sqrt(252))


def _heuristics(symbol: str, asset_class: str, params: Dict[str, Any], vol_30d: Optional[float]) -> List[str]:
    out: List[str] = []
    sl = _as_float(params.get("sl_atr"))
    tp = _as_float(params.get("tp_atr"))
    adx = _as_float(params.get("adx_threshold"))
    ml = _as_float(params.get("ml_threshold"))

    if sl and tp and sl > 0:
        rr = tp / sl
        if rr >= 4.0:
            out.append(f"RR_ATR alto ({rr:.2f})")
        if rr <= 0.9:
            out.append(f"RR_ATR baixo ({rr:.2f})")

    if asset_class == "CRYPTO" and sl is not None and sl <= 1.5:
        out.append("SL_ATR potencialmente curto para cripto")
    if asset_class in ("INDICES", "METALS") and sl is not None and sl <= 1.0:
        out.append("SL_ATR potencialmente curto para índice/metal")

    if adx is not None and adx >= 22:
        out.append("ADX threshold alto (tendência forte apenas)")
    if adx is not None and adx <= 6:
        out.append("ADX threshold muito baixo (pode operar lateral)")

    if ml is not None and ml >= 0.65:
        out.append("ML threshold alto (mais seletivo)")
    if ml is not None and ml <= 0.5:
        out.append("ML threshold baixo (menos seletivo)")

    if vol_30d is not None and math.isfinite(vol_30d):
        if asset_class == "FX" and vol_30d > 0.30:
            out.append(f"Volatilidade 30d alta p/ FX ({vol_30d:.0%})")
        if asset_class == "CRYPTO" and vol_30d < 0.20:
            out.append(f"Volatilidade 30d baixa p/ cripto ({vol_30d:.0%})")

    return out


def audit(elite_json_path: Path, ref_date: str = "2026-01-17") -> Tuple[List[SymbolAudit], Dict[str, Any]]:
    ref_dt = datetime.strptime(ref_date, "%Y-%m-%d")
    data = _load_json(elite_json_path)

    use_mt5 = os.getenv("XP3_AUDIT_USE_MT5", "").strip().lower() in ("1", "true", "yes", "y")
    max_rates = int(os.getenv("XP3_AUDIT_MAX_RATES", "12000").strip() or "12000")

    mt5 = None
    utils = None
    if use_mt5:
        try:
            import MetaTrader5 as _mt5
            import utils_forex as _utils
            mt5 = _mt5
            utils = _utils
            path = getattr(config, "MT5_TERMINAL_PATH", None)
            ok = mt5.initialize(path=path) if path else mt5.initialize()
            if not ok:
                use_mt5 = False
        except Exception:
            use_mt5 = False

    audits: List[SymbolAudit] = []
    for sym, params in data.items():
        asset_class = asset_sel.get_asset_class(sym)
        schema_ok, schema_issues = _schema_validate(params)
        range_issues = _range_validate(params)

        data_bundle = None
        duka_file = Path("dukascopy_data") / f"{sym}_M15.csv"
        if duka_file.exists():
            try:
                df = pd.read_csv(duka_file)
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
                df = df[["open", "high", "low", "close", "tick_volume"]].dropna()
                data_bundle = {"df": df, "source": "DUKASCOPY"}
            except Exception:
                data_bundle = None
        elif use_mt5 and mt5 is not None and utils is not None:
            try:
                resolved = utils.normalize_symbol(sym)
                if mt5.symbol_select(resolved, True):
                    rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M15, 0, max_rates)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df["time"] = pd.to_datetime(df["time"], unit="s")
                        df.set_index("time", inplace=True)
                        df = df[["open", "high", "low", "close", "tick_volume"]].dropna()
                        data_bundle = {"df": df, "source": "MT5"}
            except Exception:
                data_bundle = None

        data_ok = bool(data_bundle and isinstance(data_bundle.get("df"), pd.DataFrame) and not data_bundle["df"].empty)
        candles = int(len(data_bundle["df"])) if data_ok else 0
        src = str(data_bundle.get("source", "UNKNOWN")) if data_ok else "UNKNOWN"
        last_ts = None
        age_days = None
        vol_30d = None
        vol_90d = None

        if data_ok:
            df = data_bundle["df"]
            try:
                last_dt = df.index.max().to_pydatetime() if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index).max().to_pydatetime()
                last_ts = last_dt.isoformat(timespec="seconds")
                age_days = (ref_dt - last_dt).total_seconds() / 86400.0
            except Exception:
                last_ts = None
                age_days = None

            try:
                daily = _daily_df(df)
                vol_30d = _vol_annualized(daily, 30)
                vol_90d = _vol_annualized(daily, 90)
            except Exception:
                vol_30d = None
                vol_90d = None

        rr = None
        sl = _as_float(params.get("sl_atr"))
        tp = _as_float(params.get("tp_atr"))
        if sl and tp and sl > 0:
            rr = tp / sl

        heur = _heuristics(sym, asset_class, params, vol_30d)

        audits.append(
            SymbolAudit(
                symbol=sym,
                asset_class=asset_class,
                params=params,
                schema_ok=schema_ok,
                schema_issues=schema_issues,
                range_issues=range_issues,
                data_ok=data_ok,
                data_source=src,
                candles=candles,
                last_ts=last_ts,
                age_days=age_days,
                vol_30d_ann=vol_30d,
                vol_90d_ann=vol_90d,
                rr_atr=rr,
                heuristics=heur,
            )
        )

    if use_mt5 and mt5 is not None:
        try:
            mt5.shutdown()
        except Exception:
            pass

    meta = {
        "ref_date": ref_date,
        "file": str(elite_json_path),
        "symbols": len(audits),
        "schema_ok": int(sum(1 for a in audits if a.schema_ok)),
        "data_ok": int(sum(1 for a in audits if a.data_ok)),
    }
    return audits, meta


def _plot_charts(audits: List[SymbolAudit], out_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_paths: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "symbol": a.symbol,
                "class": a.asset_class,
                "ema_short": a.params.get("ema_short"),
                "ema_long": a.params.get("ema_long"),
                "rsi_low": a.params.get("rsi_low"),
                "rsi_high": a.params.get("rsi_high"),
                "adx_threshold": a.params.get("adx_threshold"),
                "sl_atr": a.params.get("sl_atr"),
                "tp_atr": a.params.get("tp_atr"),
                "ml_threshold": a.params.get("ml_threshold"),
                "rr_atr": a.rr_atr,
                "vol_30d": a.vol_30d_ann,
                "age_days": a.age_days,
            }
            for a in audits
        ]
    )

    def savefig(name: str):
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        out_paths.append(str(path))

    plt.figure(figsize=(10, 4))
    df_sorted = df.sort_values("ml_threshold", ascending=False)
    plt.bar(df_sorted["symbol"], df_sorted["ml_threshold"])
    plt.xticks(rotation=60, ha="right")
    plt.title("ML threshold por símbolo")
    plt.ylim(0.45, 0.75)
    savefig("ml_threshold_by_symbol.png")

    plt.figure(figsize=(10, 4))
    df_rr = df.sort_values("rr_atr", ascending=False)
    plt.bar(df_rr["symbol"], df_rr["rr_atr"])
    plt.xticks(rotation=60, ha="right")
    plt.title("RR (tp_atr / sl_atr) por símbolo")
    savefig("rr_atr_by_symbol.png")

    plt.figure(figsize=(8, 5))
    sub = df.dropna(subset=["vol_30d", "sl_atr"])
    for cls, grp in sub.groupby("class"):
        plt.scatter(grp["vol_30d"], grp["sl_atr"], label=cls)
    plt.xlabel("Vol anualizada 30d (estimada)")
    plt.ylabel("sl_atr")
    plt.title("Vol 30d vs SL_ATR")
    plt.legend()
    savefig("vol30_vs_slatr.png")

    plt.figure(figsize=(8, 5))
    sub = df.dropna(subset=["vol_30d", "ml_threshold"])
    for cls, grp in sub.groupby("class"):
        plt.scatter(grp["vol_30d"], grp["ml_threshold"], label=cls)
    plt.xlabel("Vol anualizada 30d (estimada)")
    plt.ylabel("ml_threshold")
    plt.title("Vol 30d vs ML threshold")
    plt.legend()
    savefig("vol30_vs_mlthreshold.png")

    plt.figure(figsize=(10, 4))
    sub = df.dropna(subset=["age_days"])
    plt.bar(sub["symbol"], sub["age_days"])
    plt.xticks(rotation=60, ha="right")
    plt.title("Defasagem (dias) do último candle vs data referência")
    savefig("data_age_days.png")

    return out_paths


def write_report(elite_json_path: Path, ref_date: str = "2026-01-17") -> Path:
    audits, meta = audit(elite_json_path, ref_date=ref_date)
    out_root = elite_json_path.parent / f"elite_params_audit_{ref_date.replace('-', '')}"
    charts_dir = out_root / "charts"
    out_root.mkdir(parents=True, exist_ok=True)

    chart_paths = _plot_charts(audits, charts_dir)
    chart_files = [Path(p).name for p in chart_paths]

    issues_schema = [a for a in audits if not a.schema_ok]
    issues_range = [a for a in audits if a.range_issues]
    issues_data = [a for a in audits if (a.age_days is not None and a.age_days > 2)]

    lines: List[str] = []
    lines.append(f"# Auditoria Elite Params ({ref_date})")
    lines.append("")
    lines.append("## Sumário Executivo")
    lines.append(f"- Arquivo: `{elite_json_path.as_posix()}`")
    lines.append(f"- Símbolos auditados: {meta['symbols']} | Schema OK: {meta['schema_ok']} | Data OK: {meta['data_ok']}")
    lines.append(f"- Problemas de schema: {len(issues_schema)} | Problemas de range: {len(issues_range)} | Dados defasados (>2d): {len(issues_data)}")
    lines.append("")
    lines.append("## Tabela Comparativa (parâmetros vs benchmarks + mercado recente)")
    lines.append("| Símbolo | Classe | Fonte | Candles | Último candle | Age(d) | Vol30a | Vol90a | EMA(s/l) | RSI(l/h) | ADX> | SL_ATR | TP_ATR | RR | ML_thr | Alertas |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for a in sorted(audits, key=lambda x: (x.asset_class, x.symbol)):
        p = a.params
        alerts = []
        alerts.extend(a.schema_issues)
        alerts.extend(a.range_issues)
        alerts.extend(a.heuristics)
        if a.age_days is not None and a.age_days > 2:
            alerts.append(f"dados defasados ({a.age_days:.1f}d)")
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(a.symbol),
                    _escape_md(a.asset_class),
                    _escape_md(a.data_source),
                    str(a.candles),
                    _escape_md(a.last_ts or ""),
                    f"{a.age_days:.1f}" if a.age_days is not None else "",
                    f"{a.vol_30d_ann:.0%}" if a.vol_30d_ann is not None else "",
                    f"{a.vol_90d_ann:.0%}" if a.vol_90d_ann is not None else "",
                    f"{p.get('ema_short')}/{p.get('ema_long')}",
                    f"{p.get('rsi_low')}/{p.get('rsi_high')}",
                    str(p.get("adx_threshold")),
                    str(p.get("sl_atr")),
                    str(p.get("tp_atr")),
                    f"{a.rr_atr:.2f}" if a.rr_atr is not None else "",
                    str(p.get("ml_threshold")),
                    _escape_md("; ".join(alerts[:6])),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Evidências e Discrepâncias")
    if issues_schema:
        lines.append("- Schema")
        for a in issues_schema:
            lines.append(f"  - {a.symbol}: {', '.join(a.schema_issues)}")
    if issues_range:
        lines.append("- Ranges")
        for a in issues_range:
            lines.append(f"  - {a.symbol}: {', '.join(a.range_issues)}")
    if issues_data:
        lines.append("- Temporal (dados defasados)")
        for a in issues_data:
            lines.append(f"  - {a.symbol}: last={a.last_ts} age_days={a.age_days:.2f}")
    if not (issues_schema or issues_range or issues_data):
        lines.append("- Nenhuma discrepância estrutural/range/temporal crítica detectada.")
    lines.append("")

    lines.append("## Gráficos")
    for f in chart_files:
        lines.append(f"- charts/{f}")
    lines.append("")

    lines.append("## Recomendações")
    lines.append("- Priorizar símbolos com dados recentes (age_days próximo de 0) e volume de candles suficiente.")
    lines.append("- Em cripto/índices/metais, revisar SL_ATR muito curto (<=1.0–1.5) quando a vol30 estiver alta.")
    lines.append("- Para RR_ATR muito alto (>=4), validar se a taxa de acerto histórica sustenta o alvo; senão, reduzir TP_ATR.")
    lines.append("- Se houver muitos símbolos sem dados via Dukascopy, preferir MT5 com aliases/sufixos e Market Watch filtrado.")

    report_path = out_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    ref = os.getenv("XP3_REF_DATE", "2026-01-17").strip() or "2026-01-17"
    base = Path("optimizer_output") / f"v7_{ref.replace('-', '')}"
    path = base / f"elite_params_{ref.replace('-', '')}.json"
    if not path.exists():
        raise FileNotFoundError(str(path))
    out = write_report(path, ref_date=ref)
    print(str(out))


if __name__ == "__main__":
    main()

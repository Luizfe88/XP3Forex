import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import MetaTrader5 as mt5

import config_forex as config
import utils_forex as utils


def _get_allowed_symbols() -> Tuple[str, List[str]]:
    sector = str(getattr(config, "MT5_SECTOR_FILTER", "ALL")).upper().strip() or "ALL"
    sector_map = getattr(config, "SECTOR_MAP", None)

    symbols: List[str] = []
    if isinstance(sector_map, dict) and sector in sector_map:
        raw = sector_map.get(sector)
        if isinstance(raw, (list, set, tuple)):
            symbols = [str(s).strip() for s in raw if str(s).strip()]
    else:
        raw = getattr(config, "SYMBOL_MAP", None)
        if isinstance(raw, (list, set, tuple)):
            symbols = [str(s).strip() for s in raw if str(s).strip()]

    return sector, sorted(set(symbols))


def apply_market_watch_filter() -> Dict[str, Any]:
    sector, allowed = _get_allowed_symbols()
    out: Dict[str, Any] = {"sector": sector, "allowed_count": len(allowed), "kept": [], "removed": [], "missing": []}

    if not mt5.initialize(path=getattr(config, "MT5_TERMINAL_PATH", None)):
        return {"error": f"Falha ao inicializar MT5: {mt5.last_error()}"}

    try:
        result = utils.sync_market_watch(allowed)
        out.update(result)

        data_path = None
        try:
            ti = mt5.terminal_info()
            data_path = getattr(ti, "data_path", None) if ti else None
        except Exception:
            data_path = None

        if data_path:
            files_dir = Path(str(data_path)) / "MQL5" / "Files"
            files_dir.mkdir(parents=True, exist_ok=True)
            list_file = files_dir / "xp3_sector_symbols.txt"
            list_file.write_text("\n".join(result.get("kept", [])), encoding="utf-8")
            out["export_file"] = str(list_file)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    return out


def main() -> None:
    logs_dir = Path(getattr(config, "LOGS_DIR", Path("logs")))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "mt5_sector_filter.jsonl"

    report = apply_market_watch_filter()
    report["ts"] = datetime.now().isoformat(timespec="seconds")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()


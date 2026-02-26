import base64
import io
import os
import re
import zipfile
from pathlib import Path


EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
    "analysis_logs",
    "optimizer_output",
    "optimizer_output_forex",
    "data",
    "dukascopy_data",
    "logs",
}

EXCLUDE_FILES = {
    "xp3_single.py",
    "build_xp3_single.py",
}


def _should_skip(path: Path) -> bool:
    parts = {p.name for p in path.parents}
    if parts.intersection(EXCLUDE_DIRS):
        return True
    return False


def _sanitize_source(rel_path: str, source: str) -> str:
    if rel_path.replace("\\", "/").endswith("config_forex.py"):
        source = re.sub(
            r'^TELEGRAM_BOT_TOKEN\s*=.*$',
            'TELEGRAM_BOT_TOKEN = os.getenv("XP3_TELEGRAM_BOT_TOKEN", "").strip()',
            source,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r'^TELEGRAM_CHAT_ID\s*=.*$',
            'TELEGRAM_CHAT_ID = os.getenv("XP3_TELEGRAM_CHAT_ID", "").strip()',
            source,
            flags=re.MULTILINE,
        )
    return source


def build(output_file: Path) -> None:
    root = Path(__file__).resolve().parent
    py_files = []
    for p in root.rglob("*.py"):
        if _should_skip(p):
            continue
        if p.name in EXCLUDE_FILES:
            continue
        py_files.append(p)

    if not py_files:
        raise RuntimeError("Nenhum arquivo .py encontrado para empacotar.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(py_files):
            rel = str(p.relative_to(root)).replace("\\", "/")
            src = p.read_text(encoding="utf-8", errors="ignore")
            src = _sanitize_source(rel, src)
            zf.writestr(rel, src)

    payload_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    template = f'''#!/usr/bin/env python3
# xp3_single.py - XP3 PRO FOREX (single-file executable bundle)
"""
XP3 PRO FOREX - Executável Único

Dependências externas esperadas (instalar via pip conforme requirements.txt do projeto):
- MetaTrader5
- pandas
- numpy
- scipy
- scikit-learn
- numba
- optuna
- requests
- pytz
- psutil
- streamlit (opcional, se usar dashboard)

Uso (exemplos):
- Rodar bot:              python xp3_single.py bot
- Rodar rank diário:      python xp3_single.py rank
- Rodar otimizador v7:    python xp3_single.py optimizer

Variáveis de ambiente (segurança):
- XP3_TELEGRAM_BOT_TOKEN
- XP3_TELEGRAM_CHAT_ID

Seções importantes:
1) Bootstrap + extração do bundle
2) Runner (executa scripts como se fossem arquivos originais)
3) CLI principal com tratamento de erro
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import runpy
import sys
import tempfile
import zipfile
from pathlib import Path


PAYLOAD_B64 = """{payload_b64}"""


# ===========================
# 1) BOOTSTRAP / EXTRAÇÃO
# ===========================
def extract_bundle(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(PAYLOAD_B64.encode("ascii"))
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        zf.extractall(str(target_dir))


# ===========================
# 2) EXECUÇÃO DE SCRIPTS
# ===========================
def run_script(extract_dir: Path, script_name: str, script_args: list[str]) -> int:
    script_path = extract_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script não encontrado no bundle: {{script_name}}")

    old_argv = sys.argv[:]
    sys.argv = [str(script_path)] + list(script_args)
    try:
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    finally:
        sys.argv = old_argv


# ===========================
# 3) CLI PRINCIPAL
# ===========================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xp3_single.py")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["bot", "rank", "optimizer", "optimizer_v01", "dukascopy", "log_analyzer"],
        help="Qual ferramenta executar dentro do bundle",
    )
    parser.add_argument("args", nargs="*", help="Argumentos repassados ao comando")
    parser.add_argument("--extract-dir", default="", help="Diretório para extração (opcional)")

    ns = parser.parse_args(argv)
    if not ns.command:
        parser.print_help()
        print("\\nDica: rode com um comando, por exemplo:\\n  py xp3_single.py bot\\n  py xp3_single.py rank\\n  py xp3_single.py optimizer")
        return 0

    if ns.extract_dir:
        extract_dir = Path(ns.extract_dir).expanduser().resolve()
        extract_bundle(extract_dir)
    else:
        work = Path(tempfile.gettempdir()) / "xp3_single_extracted"
        extract_dir = work
        if not extract_dir.exists() or not any(extract_dir.glob("*.py")):
            extract_bundle(extract_dir)

    sys.path.insert(0, str(extract_dir))

    mapping = {{
        "bot": "bot_forex.py",
        "rank": "update_asset_rankings.py",
        "optimizer": "otimizador_semanal_forex.py",
        "optimizer_v01": "otimizador_semanal_forex_v01.py",
        "dukascopy": "dukascopy_downloader.py",
        "log_analyzer": "log_analyzer.py",
    }}

    script = mapping[ns.command]
    return run_script(extract_dir, script, ns.args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ Erro fatal no xp3_single: {{e}}", file=sys.stderr)
        raise
'''

    output_file.write_text(template, encoding="utf-8")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "xp3_single.py"
    build(out)
    print(f"✅ Gerado: {out}")

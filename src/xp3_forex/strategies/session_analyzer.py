"""
🔍 SESSION ANALYZER - XP3 PRO FOREX
Identifica a sessão ativa e fornece parâmetros otimizados.
"""

import json
import logging
from datetime import datetime, time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from xp3_forex.core.settings import settings

logger = logging.getLogger("XP3.SessionAnalyzer")

def is_time_between(begin_time: time, end_time: time, check_time: time) -> bool:
    """Verifica se check_time está entre begin e end (suporta virada de dia)"""
    if begin_time < end_time:
        return begin_time <= check_time <= end_time
    else: # Transpõe meia-noite (Asia)
        return check_time >= begin_time or check_time <= end_time

def get_active_session_name(current_time_utc: datetime) -> str:
    """
    Identifica a sessão ativa agora. 
    Prioridade: NY > LONDON > ASIA (Overlap NY/London prioriza NY).
    """
    t = current_time_utc.time()
    
    # Horários das Sessões (Extraídos do Settings)
    asia_start = datetime.strptime(settings.SESSION_ASIA.start_time_utc, "%H:%M").time()
    asia_end = datetime.strptime(settings.SESSION_ASIA.end_time_utc, "%H:%M").time()
    
    london_start = datetime.strptime(settings.SESSION_LONDON.start_time_utc, "%H:%M").time()
    london_end = datetime.strptime(settings.SESSION_LONDON.end_time_utc, "%H:%M").time()
    
    ny_start = datetime.strptime(settings.SESSION_NY.start_time_utc, "%H:%M").time()
    ny_end = datetime.strptime(settings.SESSION_NY.end_time_utc, "%H:%M").time()

    # Checagem de Prioridade (Overlap NY/London prioriza NY)
    if is_time_between(ny_start, ny_end, t):
        return "NY"
    elif is_time_between(london_start, london_end, t):
        return "LONDON"
    elif is_time_between(asia_start, asia_end, t):
        return "ASIA"
    
    return "UNKNOWN"

def get_active_session_params(symbol: str, current_time_utc: datetime) -> dict:
    """
    Lê o JSON de configuração e retorna o bloco de parâmetros exato.
    Se não houver para o símbolo, usa o bloco DEFAULT.
    """
    session_name = get_active_session_name(current_time_utc)
    
    # Path para o JSON
    json_path = settings.DATA_DIR / "session_optimized_params.json"
    
    try:
        if not json_path.exists():
            logger.warning(f"Arquivo de parâmetros de sessão não encontrado em {json_path}. Usando padrões.")
            return {}

        with open(json_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)
            
        # 1. Tenta Símbolo Exato
        symbol_params = all_params.get(symbol, all_params.get("DEFAULT", {}))
        
        # 2. Pega Parâmetros da Sessão
        session_params = symbol_params.get(session_name, {})
        
        if not session_params:
            logger.debug(f"Nenhum parâmetro específico para {symbol} na sessão {session_name}.")
            # Se não tem na sessão do símbolo, tenta na sessão do DEFAULT
            session_params = all_params.get("DEFAULT", {}).get(session_name, {})

        # Adiciona flag de sessão para rastreabilidade
        session_params["active_session"] = session_name
        return session_params

    except Exception as e:
        logger.error(f"Erro ao ler parâmetros de sessão: {e}")
        return {"active_session": session_name}

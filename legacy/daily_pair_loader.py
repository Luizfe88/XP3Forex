#!/usr/bin/env python3
"""
📅 CARREGADOR DE PARES DIÁRIOS - XP3 PRO v5.0
==============================================
Módulo para carregar os pares de moedas selecionados diariamente
pelo sistema de análise de mercado.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

# ===========================
# CONFIGURAÇÕES
# ===========================

DEFAULT_PAIRS_FILE = 'daily_selected_pairs.json'
SIMPLE_PAIRS_FILE = 'simple_pairs_list.json'
MAX_ANALYSIS_AGE_HOURS = 24  # Análise válida por 24 horas

# Pares padrão caso não haja análise recente
DEFAULT_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'
]

# ===========================
# FUNÇÕES DE CARREGAMENTO
# ===========================

def load_daily_pairs(filename: str = DEFAULT_PAIRS_FILE) -> List[str]:
    """
    Carrega os pares selecionados para o dia atual.
    
    Args:
        filename: Nome do arquivo JSON com a análise
        
    Returns:
        Lista de pares selecionados
    """
    try:
        if not os.path.exists(filename):
            logger.warning(f"⚠️ Arquivo {filename} não encontrado. Usando pares padrão.")
            return DEFAULT_PAIRS
        
        with open(filename, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Valida se a análise é recente
        if not is_analysis_recent(analysis_data):
            logger.warning("⚠️ Análise desatada. Usando pares padrão.")
            return DEFAULT_PAIRS
        
        # Extrai lista de pares
        selected_pairs = []
        for pair_data in analysis_data.get('selected_pairs', []):
            pair_name = pair_data.get('pair')
            if pair_name:
                selected_pairs.append(pair_name)
        
        if not selected_pairs:
            logger.warning("⚠️ Nenhum par encontrado na análise. Usando padrões.")
            return DEFAULT_PAIRS
        
        logger.info(f"✅ Carregados {len(selected_pairs)} pares do arquivo {filename}")
        return selected_pairs
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Erro ao decodificar JSON: {e}")
        return DEFAULT_PAIRS
    except Exception as e:
        logger.error(f"❌ Erro ao carregar pares: {e}")
        return DEFAULT_PAIRS

def load_simple_pairs_list(filename: str = SIMPLE_PAIRS_FILE) -> List[str]:
    """
    Carrega a lista simplificada de pares (apenas nomes).
    
    Args:
        filename: Nome do arquivo JSON simplificado
        
    Returns:
        Lista de pares
    """
    try:
        if not os.path.exists(filename):
            logger.warning(f"⚠️ Arquivo {filename} não encontrado.")
            return load_daily_pairs()  # Tenta o arquivo completo
        
        with open(filename, 'r', encoding='utf-8') as f:
            pairs_list = json.load(f)
        
        if isinstance(pairs_list, list) and pairs_list:
            logger.info(f"✅ Carregados {len(pairs_list)} pares do arquivo simplificado")
            return pairs_list
        else:
            logger.warning("⚠️ Formato inválido no arquivo simplificado")
            return load_daily_pairs()  # Tenta o arquivo completo
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar lista simplificada: {e}")
        return load_daily_pairs()  # Fallback para arquivo completo

def is_analysis_recent(analysis_data: Dict[str, Any], max_age_hours: int = MAX_ANALYSIS_AGE_HOURS) -> bool:
    """
    Verifica se a análise é recente o suficiente.
    
    Args:
        analysis_data: Dados da análise
        max_age_hours: Idade máxima em horas
        
    Returns:
        True se a análise é recente
    """
    try:
        analysis_date_str = analysis_data.get('analysis_date')
        if not analysis_date_str:
            return False
        
        analysis_date = datetime.fromisoformat(analysis_date_str)
        current_date = datetime.now()
        
        # Calcula diferença em horas
        age_hours = (current_date - analysis_date).total_seconds() / 3600
        
        is_recent = age_hours <= max_age_hours
        
        if not is_recent:
            logger.info(f"📅 Análise tem {age_hours:.1f} horas (máximo: {max_age_hours}h)")
        
        return is_recent
        
    except Exception as e:
        logger.error(f"❌ Erro ao verificar idade da análise: {e}")
        return False

def get_analysis_metadata(filename: str = DEFAULT_PAIRS_FILE) -> Dict[str, Any]:
    """
    Carrega metadados da análise (data, sentimento, etc).
    
    Args:
        filename: Nome do arquivo JSON
        
    Returns:
        Dicionário com metadados
    """
    try:
        if not os.path.exists(filename):
            return {}
        
        with open(filename, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        return {
            'analysis_date': analysis_data.get('analysis_date'),
            'market_session': analysis_data.get('market_session'),
            'market_sentiment': analysis_data.get('market_sentiment', {}).get('sentiment'),
            'total_pairs_analyzed': analysis_data.get('analysis_metadata', {}).get('total_pairs_analyzed'),
            'pairs_avoided_due_news': analysis_data.get('analysis_metadata', {}).get('pairs_avoided_due_news')
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar metadados: {e}")
        return {}

def validate_pairs_list(pairs: List[str]) -> List[str]:
    """
    Valida e limpa a lista de pares.
    
    Args:
        pairs: Lista de pares
        
    Returns:
        Lista validada
    """
    if not pairs:
        return DEFAULT_PAIRS
    
    # Remove duplicatas e converte para maiúsculas
    validated_pairs = list(set(pair.upper().strip() for pair in pairs if pair and len(pair) >= 6))
    
    # Valida formato básico (XXXYYY)
    valid_pairs = []
    for pair in validated_pairs:
        if len(pair) == 6 and pair.isalpha():
            valid_pairs.append(pair)
        else:
            logger.warning(f"⚠️ Par inválido ignorado: {pair}")
    
    if not valid_pairs:
        logger.warning("⚠️ Nenhum par válido encontrado. Usando padrões.")
        return DEFAULT_PAIRS
    
    return valid_pairs

# ===========================
# FUNÇÕES DE INTEGRAÇÃO
# ===========================

def get_daily_pairs_for_bot() -> List[str]:
    """
    Função principal para obter pares do dia para o bot.
    
    Returns:
        Lista de pares validados e prontos para uso
    """
    logger.info("📅 Carregando pares diários...")
    
    # Tenta carregar lista simplificada primeiro
    pairs = load_simple_pairs_list()
    
    # Se não conseguiu, tenta o arquivo completo
    if pairs == DEFAULT_PAIRS:
        pairs = load_daily_pairs()
    
    # Valida a lista
    validated_pairs = validate_pairs_list(pairs)
    
    # Log de resumo
    logger.info(f"🎯 Pares selecionados para hoje: {validated_pairs}")
    
    # Carrega metadados para log adicional
    metadata = get_analysis_metadata()
    if metadata:
        logger.info(f"📊 Data da análise: {metadata.get('analysis_date', 'N/A')}")
        logger.info(f"💡 Sentimento de mercado: {metadata.get('market_sentiment', 'N/A')}")
        logger.info(f"🌍 Sessão: {metadata.get('market_session', 'N/A')}")
    
    return validated_pairs

def should_refresh_analysis(max_age_hours: int = MAX_ANALYSIS_AGE_HOURS) -> bool:
    """
    Verifica se a análise precisa ser atualizada.
    
    Args:
        max_age_hours: Idade máxima em horas
        
    Returns:
        True se precisa atualizar
    """
    try:
        if not os.path.exists(DEFAULT_PAIRS_FILE):
            logger.info("📊 Arquivo de análise não encontrado. Necessário atualizar.")
            return True
        
        with open(DEFAULT_PAIRS_FILE, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        if not is_analysis_recent(analysis_data, max_age_hours):
            logger.info(f"📊 Análise desatada (> {max_age_hours}h). Necessário atualizar.")
            return True
        
        logger.info("✅ Análise está atualizada.")
        return False
        
    except Exception as e:
        logger.error(f"❌ Erro ao verificar necessidade de atualização: {e}")
        return True

# ===========================
# FUNÇÃO DE TESTE
# ===========================

def test_daily_loader():
    """Função de teste do carregador"""
    logger.info("🧪 Testando carregador de pares diários...")
    
    # Testa carregamento
    pairs = get_daily_pairs_for_bot()
    logger.info(f"✅ Teste concluído. Pares carregados: {pairs}")
    
    # Testa validação
    test_pairs = ['eurusd', 'GBPUSD', 'invalid', 'USDJPY', 'EURUSD']
    validated = validate_pairs_list(test_pairs)
    logger.info(f"✅ Validação testada: {validated}")
    
    # Testa metadados
    metadata = get_analysis_metadata()
    logger.info(f"✅ Metadados: {metadata}")
    
    return pairs

# ===========================
# EXECUÇÃO
# ===========================

if __name__ == "__main__":
    pairs = test_daily_loader()
    print(f"\n🎯 Pares do dia: {pairs}")
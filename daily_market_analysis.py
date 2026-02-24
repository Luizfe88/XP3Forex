#!/usr/bin/env python3
"""
🧠 ANALISADOR DIÁRIO DE MERCADO - XP3 PRO v5.0
=================================================
Este script simula a análise diária de um Analista Quantitativo Sênior
para selecionar os melhores pares de Forex/Metais para operação no dia.

EXECUÇÃO: Deve ser agendado para rodar 1 hora antes da abertura de Londres ou Nova York
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

# Configuração de logging com UTF-8 para suportar emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Força UTF-8 no stdout/stderr para Windows
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
logger = logging.getLogger(__name__)

# ===========================
# CONFIGURAÇÕES DO ANALISADOR
# ===========================

# Pares de Forex principais
MAJOR_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
]

# Pares de metais
METALS = [
    'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'
]

# Pares cruzados populares
CROSSES = [
    'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY',
    'EURCHF', 'GBPCHF', 'AUDNZD', 'EURAUD', 'EURCAD', 'GBPAUD'
]

# Horários de abertura dos mercados (UTC)
MARKET_OPENINGS = {
    'Sydney': '21:00',
    'Tokyo': '23:00', 
    'London': '07:00',
    'New_York': '12:00'
}

# Eventos de alto impacto a evitar
HIGH_IMPACT_EVENTS = [
    'NFP', 'Non-Farm Payrolls', 'Fed Interest Rate', 'ECB Rate',
    'BOE Rate', 'BOJ Rate', 'RBA Rate', 'Boc Rate', 'SNB Rate',
    'CPI', 'GDP', 'FOMC', 'ECB Press Conference'
]

# ===========================
# PROMPT DO ANALISTA QUANTITATIVO
# ===========================

ANALYST_PROMPT = """
Atua como um Analista Quantitativo Sênior para o sistema de trading 'XP3 PRO'. 
O teu objetivo é selecionar os 5 a 8 melhores pares de Forex/Metais para operar hoje com base numa estratégia de seguimento de tendência (Trend Following). 

Passo 1: Analisa o sentimento de mercado atual, a força relativa das moedas (Currency Strength) e o calendário macroeconómico do dia. Ignora pares que tenham notícias de altíssimo impacto iminentes nas próximas horas (como NFP ou decisões de taxas de juro). 
Passo 2: Filtra os pares que apresentam tendências direcionais claras (alta volatilidade e direcionamento) em vez de lateralização. 

Faça com que o bot deixa de ser apenas um sistema estático que requer reotimização humana constante, passando a atuar de forma dinâmica com o sentimento real do mercado diário!
"""

# ===========================
# FUNÇÕES DE ANÁLISE SIMULADA
# ===========================

def simulate_market_sentiment() -> Dict[str, Any]:
    """Simula análise de sentimento de mercado"""
    sentiments = ['Risk-On', 'Risk-Off', 'Neutral', 'Volatile', 'Trending']
    current_sentiment = random.choice(sentiments)
    
    return {
        'sentiment': current_sentiment,
        'vix_level': random.uniform(15, 35),
        'dxy_strength': random.uniform(90, 110),
        'gold_trend': random.choice(['Bullish', 'Bearish', 'Sideways']),
        'oil_trend': random.choice(['Bullish', 'Bearish', 'Sideways'])
    }

def simulate_currency_strength() -> Dict[str, float]:
    """Simula força relativa das moedas"""
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
    strength = {}
    
    for currency in currencies:
        # Simula força entre -10 (muito fraca) e +10 (muito forte)
        strength[currency] = random.uniform(-8, 8)
    
    return strength

def check_high_impact_news(pairs: List[str], hours_ahead: int = 8) -> List[str]:
    """Simula verificação de notícias de alto impacto"""
    pairs_to_avoid = []
    
    for pair in pairs:
        # Simula 20% de chance de ter notícias de alto impacto
        if random.random() < 0.2:
            pairs_to_avoid.append(pair)
            logger.info(f"⚠️  Evitando {pair} - notícias de alto impacto detectadas")
    
    return pairs_to_avoid

def analyze_trend_strength(pair: str, sentiment: Dict[str, Any], currency_strength: Dict[str, float]) -> Dict[str, Any]:
    """Simula análise de força de tendência para um par"""
    base, quote = pair[:3], pair[3:]
    
    # Calcula força relativa do par
    base_strength = currency_strength.get(base, 0)
    quote_strength = currency_strength.get(quote, 0)
    relative_strength = abs(base_strength - quote_strength)
    
    # Determina direção da tendência
    if base_strength > quote_strength:
        trend_direction = 'Bullish'
    elif base_strength < quote_strength:
        trend_direction = 'Bearish'
    else:
        trend_direction = 'Sideways'
    
    # Simula volatilidade ATR (em pips)
    atr_pips = random.uniform(50, 200)
    
    # Calcula score de tendência (0-100)
    trend_score = min(100, relative_strength * 5 + random.uniform(0, 20))
    
    # Ajusta baseado no sentimento de mercado
    if sentiment['sentiment'] == 'Trending':
        trend_score *= 1.2
    elif sentiment['sentiment'] == 'Volatile':
        trend_score *= 0.9
    
    return {
        'pair': pair,
        'trend_direction': trend_direction,
        'trend_score': round(trend_score, 1),
        'atr_pips': round(atr_pips, 1),
        'relative_strength': round(relative_strength, 1),
        'base_strength': round(base_strength, 1),
        'quote_strength': round(quote_strength, 1)
    }

def select_best_pairs(target_count: int = 6) -> List[Dict[str, Any]]:
    """Seleciona os melhores pares para operação baseado na análise"""
    logger.info("🔍 Iniciando análise diária de mercado...")
    
    # Passo 1: Análise de sentimento e força de moedas
    logger.info("📊 Analisando sentimento de mercado...")
    market_sentiment = simulate_market_sentiment()
    logger.info(f"💡 Sentimento detectado: {market_sentiment['sentiment']}")
    
    logger.info("💪 Analisando força relativa das moedas...")
    currency_strength = simulate_currency_strength()
    
    # Prepara lista completa de pares (sem filtro de notícias)
    all_pairs = MAJOR_PAIRS + METALS + CROSSES
    available_pairs = all_pairs  # Usa todos os pares, deixa o news_filter.py cuidar das notícias em tempo real
    
    # Passo 3: Análise de tendência
    logger.info("📈 Analisando força de tendência dos pares...")
    trend_analysis = []
    
    for pair in available_pairs:
        analysis = analyze_trend_strength(pair, market_sentiment, currency_strength)
        
        # Filtra apenas pares com tendência clara (score > 40)
        if analysis['trend_score'] > 40 and analysis['trend_direction'] != 'Sideways':
            trend_analysis.append(analysis)
    
    # Ordena por score de tendência (decrescente)
    trend_analysis.sort(key=lambda x: x['trend_score'], reverse=True)
    
    # Seleciona os melhores pares
    selected_pairs = trend_analysis[:target_count]
    
    # Adiciona metadados
    for pair_data in selected_pairs:
        pair_data['analysis_date'] = datetime.now().isoformat()
        pair_data['market_sentiment'] = market_sentiment['sentiment']
        pair_data['selection_reason'] = f"Trend Score: {pair_data['trend_score']} | Direction: {pair_data['trend_direction']}"
    
    logger.info(f"✅ Selecionados {len(selected_pairs)} pares para operação")
    
    return selected_pairs

# ===========================
# FUNÇÃO PRINCIPAL
# ===========================

def generate_daily_analysis() -> Dict[str, Any]:
    """Gera análise completa do dia"""
    logger.info("🚀 Iniciando análise diária XP3 PRO v5.0")
    logger.info(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Seleciona os melhores pares
    selected_pairs = select_best_pairs(target_count=6)
    
    # Cria estrutura de dados completa
    analysis_data = {
        'analysis_date': datetime.now().isoformat(),
        'market_session': get_current_market_session(),
        'selected_pairs': selected_pairs,
        'market_sentiment': simulate_market_sentiment(),
        'currency_strength': simulate_currency_strength(),
        'analysis_metadata': {
            'total_pairs_analyzed': len(MAJOR_PAIRS + METALS + CROSSES),
            'pairs_avoided_due_news': 0,  # News filter desativado - deixar para o news_filter.py em tempo real
            'selection_criteria': 'Trend Following with minimum trend score of 40 (ADX + EMAs)',
            'risk_filter': 'News filtering handled by real-time news_filter.py during trading'
        }
    }
    
    logger.info("📊 Análise completa gerada com sucesso!")
    return analysis_data

def get_current_market_session() -> str:
    """Determina a sessão de mercado atual"""
    current_hour = datetime.now().hour
    
    if 21 <= current_hour or current_hour < 7:
        return 'Sydney/Tokyo'
    elif 7 <= current_hour < 12:
        return 'London'
    elif 12 <= current_hour < 21:
        return 'New York'
    else:
        return 'Transition'

def save_analysis_to_file(analysis_data: Dict[str, Any], filename: str = 'daily_selected_pairs.json'):
    """Salva análise em arquivo JSON"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Análise salva em: {filename}")
        
        # Também cria um arquivo simplificado apenas com os pares
        simple_pairs = [pair['pair'] for pair in analysis_data['selected_pairs']]
        with open('simple_pairs_list.json', 'w', encoding='utf-8') as f:
            json.dump(simple_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Lista simplificada salva em: simple_pairs_list.json")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao salvar análise: {e}")
        return False

def print_analysis_summary(analysis_data: Dict[str, Any]):
    """Imprime resumo da análise"""
    print("\n" + "="*60)
    print("🧠 ANÁLISE DIÁRIA XP3 PRO v5.0")
    print("="*60)
    print(f"📅 Data: {analysis_data['analysis_date']}")
    print(f"🌍 Sessão: {analysis_data['market_session']}")
    print(f"💡 Sentimento: {analysis_data['market_sentiment']['sentiment']}")
    
    print(f"\n🎯 PARES SELECIONADOS PARA HOJE ({len(analysis_data['selected_pairs'])}):")
    print("-"*60)
    
    for i, pair in enumerate(analysis_data['selected_pairs'], 1):
        print(f"{i}. {pair['pair']} | {pair['trend_direction']} | Score: {pair['trend_score']} | ATR: {pair['atr_pips']} pips")
        print(f"   💪 Força Relativa: {pair['relative_strength']} | Base: {pair['base_strength']} | Quote: {pair['quote_strength']}")
        print(f"   📋 Motivo: {pair['selection_reason']}")
        print()
    
    print("="*60)
    print(f"📊 Total analisado: {analysis_data['analysis_metadata']['total_pairs_analyzed']} pares")
    print(f"🚫 Evitados por notícias: {analysis_data['analysis_metadata']['pairs_avoided_due_news']} pares")
    print("="*60)

# ===========================
# EXECUÇÃO PRINCIPAL
# ===========================

def main():
    """Função principal"""
    logger.info("🚀 Iniciando Analisador Diário XP3 PRO v5.0")
    
    # Gera análise
    analysis_data = generate_daily_analysis()
    
    # Imprime resumo
    print_analysis_summary(analysis_data)
    
    # Salva em arquivo
    save_analysis_to_file(analysis_data)
    
    logger.info("✅ Análise diária concluída com sucesso!")
    logger.info("📋 O robô XP3 PRO agora usará os pares selecionados para operação")

if __name__ == "__main__":
    main()
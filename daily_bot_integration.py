# daily_bot_integration.py - Integração de Pares Diários XP3 PRO v5.0
"""
📅 INTEGRAÇÃO DE ANÁLISE DIÁRIA DE MERCADO
============================================
Este módulo integra o sistema de análise diária ao bot_forex.py
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ===========================
# FUNÇÕES DE INTEGRAÇÃO
# ===========================

def integrate_daily_pairs_to_bot():
    """
    Integra o carregamento de pares diários ao bot_forex.py
    Esta função deve ser chamada no início do main() do bot
    """
    try:
        # Importa o carregador de pares diários
        from daily_pair_loader import get_daily_pairs_for_bot, should_refresh_analysis
        
        logger.info("📅 Integrando sistema de análise diária...")
        
        # Verifica se precisa atualizar a análise
        if should_refresh_analysis():
            logger.warning("⚠️ Análise desatada. Execute: python daily_market_analysis.py")
        
        # Obtém pares do dia
        daily_pairs = get_daily_pairs_for_bot()
        
        if daily_pairs and len(daily_pairs) > 0:
            logger.info(f"✅ Usando {len(daily_pairs)} pares da análise diária: {daily_pairs}")
            return daily_pairs
        else:
            logger.warning("⚠️ Nenhum par da análise diária. Usando configuração padrão.")
            return None
            
    except ImportError as e:
        logger.warning(f"⚠️ Módulo de análise diária não disponível: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Erro na integração de pares diários: {e}")
        return None

def create_bot_integration_patch():
    """
    Cria um patch para adicionar ao bot_forex.py
    """
    patch_content = '''
# === INTEGRAÇÃO ANÁLISE DIÁRIA XP3 PRO v5.0 ===
# Adicione este código no início da função main() do bot_forex.py
# Após as importações e antes de carregar os símbolos padrão

try:
    # Tenta carregar pares da análise diária
    from daily_pair_loader import get_daily_pairs_for_bot, should_refresh_analysis
    
    if should_refresh_analysis():
        logger.warning("⚠️ Análise diária desatada. Execute: python daily_market_analysis.py")
    
    daily_pairs = get_daily_pairs_for_bot()
    if daily_pairs and len(daily_pairs) > 0:
        logger.info(f"✅ Usando {len(daily_pairs)} pares da análise diária")
        
        # Atualiza os símbolos que serão usados
        if hasattr(config, 'ALL_AVAILABLE_SYMBOLS'):
            config.ALL_AVAILABLE_SYMBOLS = daily_pairs
        elif hasattr(config, 'SYMBOL_MAP'):
            # Se usar SYMBOL_MAP, filtra apenas os pares selecionados
            selected_set = set(daily_pairs)
            config.SYMBOL_MAP = [s for s in config.SYMBOL_MAP if s in selected_set]
        
        # Força uso dos pares diários
        allowed_symbols = daily_pairs
        logger.info(f"📊 Pares do dia: {daily_pairs}")
    else:
        logger.warning("⚠️ Análise diária não disponível. Usando configuração padrão.")
        
except ImportError as e:
    logger.warning(f"⚠️ Sistema de análise diária não disponível: {e}")
except Exception as e:
    logger.error(f"❌ Erro no sistema de análise diária: {e}")

# === FIM INTEGRAÇÃO ANÁLISE DIÁRIA ===
'''
    return patch_content

def create_config_additions():
    """
    Cria configurações adicionais para o config_forex.py
    """
    config_content = '''
# === CONFIGURAÇÕES ANÁLISE DIÁRIA XP3 PRO v5.0 ===
# Adicione estas configurações ao seu config_forex.py

# Ativa/desativa uso de análise diária
ENABLE_DAILY_MARKET_ANALYSIS = True  # True para ativar, False para desativar

# Arquivos de análise diária
DAILY_ANALYSIS_FILE = 'daily_selected_pairs.json'
DAILY_ANALYSIS_SIMPLE_FILE = 'simple_pairs_list.json'

# Tempo máximo de validade da análise (em horas)
DAILY_ANALYSIS_MAX_AGE_HOURS = 24  # Análise válida por 24 horas

# Mínimo de pares necessários da análise
DAILY_ANALYSIS_MIN_PAIRS = 3  # Mínimo de pares para operar

# Pares padrão caso análise falhe
DAILY_ANALYSIS_FALLBACK_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'XAUUSD'
]

# Horários recomendados para executar análise (UTC)
# 1 hora antes da abertura de Londres ou Nova York
DAILY_ANALYSIS_SCHEDULE = {
    'london': '06:00',   # 1h antes da abertura de Londres (07:00 UTC)
    'new_york': '11:00', # 1h antes da abertura de NY (12:00 UTC)
    'tokyo': '22:00',    # 1h antes da abertura de Tóquio (23:00 UTC)
}

# Debug da análise diária
DAILY_ANALYSIS_DEBUG = False  # True para logs detalhados
# === FIM CONFIGURAÇÕES ANÁLISE DIÁRIA ===
'''
    return config_content

def create_usage_instructions():
    """
    Cria instruções de uso da análise diária
    """
    instructions = '''
# 🧠 GUIA DE USO DA ANÁLISE DIÁRIA XP3 PRO v5.0

## 📋 PASSO A PASSO

### 1. CONFIGURAÇÃO INICIAL
- Adicione as configurações ao config_forex.py
- Certifique-se de que daily_market_analysis.py e daily_pair_loader.py estão no diretório

### 2. EXECUÇÃO MANUAL (TESTE)
```bash
# Execute a análise diária manualmente
python daily_market_analysis.py

# Teste o carregamento
python daily_pair_loader.py
```

### 3. INTEGRAÇÃO NO BOT
- Adicione o código de integração no início da função main() do bot_forex.py
- O bot automaticamente usará os pares da análise diária

### 4. AGENDAMENTO AUTOMÁTICO (RECOMENDADO)
```bash
# Linux/Mac (cron)
# Execute 1 hora antes da abertura de Londres (06:00 UTC)
0 6 * * 1-5 cd /caminho/do/seu/bot && python daily_market_analysis.py

# Windows (Task Scheduler)
# Crie uma tarefa para executar daily_market_analysis.py diariamente às 06:00 UTC
```

## 🎯 COMO FUNCIONA

### Análise Diária (`daily_market_analysis.py`)
- Simula análise de Analista Quantitativo Sênior
- Seleciona 5-8 melhores pares para Trend Following
- Evita pares com notícias de alto impacto
- Gera arquivos JSON com a seleção

### Carregamento (`daily_pair_loader.py`)
- Carrega pares do arquivo JSON
- Valida idade da análise (máximo 24h)
- Fornece fallback para pares padrão
- Integra-se perfeitamente ao bot

### Integração no Bot
- Carrega pares automaticamente ao iniciar
- Usa análise diária quando disponível
- Fallback para configuração padrão quando necessário

## ⚠️ BOAS PRÁTICAS

### Manutenção
- Execute análise diariamente antes do mercado abrir
- Monitore logs do bot para verificar uso correto
- Teste mensalmente o sistema de fallback

### Segurança
- Sempre tenha pares padrão configurados
- Configure mínimo de pares para operar
- Monitore idade da análise

### Performance
- Análise é rápida (< 1 segundo)
- Não impacta performance do bot
- Cache inteligente de dados

## 🔧 SOLUÇÃO DE PROBLEMAS

### Bot não usa pares diários
- Verifique se ENABLE_DAILY_MARKET_ANALYSIS = True
- Confirme que arquivos JSON foram criados
- Verifique logs de erro no carregamento

### Análise desatada
- Execute manualmente: python daily_market_analysis.py
- Verifique agendamento do sistema
- Confirme fuso horário correto

### Poucos pares selecionados
- Verifique critérios de seleção
- Ajuste filtros de notícias
- Monitore sentimento de mercado

---
🚀 Seu bot agora é dinâmico e se adapta ao mercado diariamente!
'''
    return instructions

# ===========================
# FUNÇÃO PRINCIPAL
# ===========================

def main():
    """Função principal de demonstração"""
    print("🔧 INTEGRADOR ANÁLISE DIÁRIA XP3 PRO v5.0")
    print("="*60)
    
    # Testa integração
    daily_pairs = integrate_daily_pairs_to_bot()
    
    if daily_pairs:
        print(f"✅ Pares carregados: {daily_pairs}")
    else:
        print("⚠️ Usando configuração padrão")
    
    # Cria arquivos de apoio
    patch = create_bot_integration_patch()
    with open('daily_analysis_bot_patch.py', 'w', encoding='utf-8') as f:
        f.write(patch)
    
    config = create_config_additions()
    with open('daily_analysis_config_additions.py', 'w', encoding='utf-8') as f:
        f.write(config)
    
    instructions = create_usage_instructions()
    with open('DAILY_ANALYSIS_USAGE.md', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("\n" + "="*60)
    print("✅ ARQUIVOS DE INTEGRAÇÃO CRIADOS:")
    print("  📄 daily_analysis_bot_patch.py")
    print("  ⚙️  daily_analysis_config_additions.py")
    print("  📖 DAILY_ANALYSIS_USAGE.md")
    print("\n🎯 Pronto para integrar análise diária ao seu bot!")

if __name__ == "__main__":
    main()
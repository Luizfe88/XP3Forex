
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

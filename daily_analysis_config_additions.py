
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

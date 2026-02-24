# dashboard_forex.py - XP3 PRO FOREX INSTITUTIONAL DASHBOARD
"""
🏛️ XP3 PRO FOREX INSTITUTIONAL DASHBOARD
✅ Design Profissional (Ported from Dashboard01)
✅ Métricas avançadas de performance e risco Forex
✅ Análise psicológica e disciplina
✅ Estatísticas por ativo, horário e estratégia
✅ Visualizações institucionais (Equity Curve, Drawdown, Heatmaps)
✅ Integração de dados de mercado em tempo real via MT5
"""

import os
# 🔇 SILENCIA LOGS DO TENSORFLOW
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import MetaTrader5 as mt5
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import sys
import logging
import io
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
from log_analyzer import LogAnalyzer

# ✅ FORÇA UTF-8 NO WINDOWS (CRÍTICO PARA EMOJIS)
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ===========================
# CONFIGURAÇÃO DE LOGGING
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard_forex_institutional.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ===========================
# IMPORTAÇÕES DO BOT
# ===========================
try:
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    import config_forex as config
    import utils_forex as utils
    try:
        import bot_forex as bot
    except Exception:
        bot = None
    # Inicializa ML Optimizer local para o dashboard se habilitado
    try:
        if getattr(config, 'ENABLE_ML_OPTIMIZER', False):
            from ml_optimizer import EnsembleOptimizer
            if not hasattr(utils, "ml_optimizer_instance") or utils.ml_optimizer_instance is None:
                utils.ml_optimizer_instance = EnsembleOptimizer()
    except Exception as e:
        logger.warning(f"⚠️ ML Optimizer indisponível no dashboard: {e}")
    # import news_filter # Opcional
    
    # Tenta importar bot state se possível, mas dashboard roda separado geralmente
    BOT_CONNECTED = False
    
except ImportError as e:
    logger.error(f"❌ Erro crítico: {e}")
    st.error(f"❌ Erro ao importar módulos: {e}")
    st.stop()

# ===========================
# CONFIGURAÇÃO DA PÁGINA - TEMA INSTITUCIONAL
# ===========================
st.set_page_config(
    page_title="XP3 Forex Institutional Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS INSTITUCIONAL AVANÇADO (Copiado do Dashboard01)
st.markdown("""
<style>
    /* Tema Principal - Clean & Professional */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers Hierárquicos */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        border-bottom: 3px solid #0f3460;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f3460;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #16213e;
        padding-left: 1rem;
    }
    
    /* Cards Profissionais */
    .metric-card {
        background: #000000;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #0f3460;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #000000;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .kpi-value {
        color: #000000;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        color: #000000;
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Cores Semânticas */
    .profit-positive {
        color: #00b894;
        font-weight: 700;
    }
    
    .profit-negative {
        color: #d63031;
        font-weight: 700;
    }
    
    .profit-neutral {
        color: #636e72;
        font-weight: 600;
    }
    
    /* Tabelas Profissionais */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f8f9fa;
    }
    
    /* Sidebar Melhorado */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        color: white;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Botões Estilizados */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Melhorias nos Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    .diag-card {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        overflow: hidden;
        margin-bottom: 16px;
        border: 1px solid #e6e6e6;
    }
    .diag-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .diag-header.ok {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    }
    .diag-header.warn {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        color: #1a1a2e;
    }
    .diag-header.error {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }
    .diag-status {
        font-size: 0.95rem;
    }
    .diag-body {
        padding: 12px 14px;
        background: #fafafa;
    }
    .progress {
        height: 8px;
        background: #e9ecef;
        border-radius: 8px;
        overflow: hidden;
        margin: 6px 0 12px 0;
    }
    .progress-fill {
        height: 8px;
        background: #2ecc71;
        border-radius: 8px;
        transition: width 0.4s ease;
    }
    .diag-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }
    .diag-check {
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #eee;
        background: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        font-size: 0.9rem;
    }
    .diag-check.badge-ok {
        border-left: 4px solid #2ecc71;
        background: #e8f8f0;
    }
    .diag-check.badge-error {
        border-left: 4px solid #e74c3c;
        background: #fdecea;
    }
    .diag-rule {
        font-weight: 700;
        color: #0f3460;
        margin-bottom: 4px;
    }
    .diag-meta {
        color: #555;
        font-size: 0.85rem;
    }
    .diag-obs {
        color: #333;
        font-size: 0.85rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# FUNÇÕES AUXILIARES AVANÇADAS
# ===========================
@st.cache_data(ttl=5)
def load_account_info():
    """Carrega informações da conta MT5"""
    try:
        if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
            return None
        
        acc = mt5.account_info()
        if acc:
            return {
                "balance": acc.balance,
                "equity": acc.equity,
                "margin": acc.margin,
                "free_margin": acc.margin_free,
                "margin_level": (acc.equity / acc.margin * 100) if acc.margin > 0 else 0,
                "profit": acc.profit,
                "login": acc.login,
                "server": acc.server,
                "leverage": acc.leverage,
                "currency": acc.currency
            }
    except Exception as e:
        logger.error(f"Erro MT5: {e}")
    return None

@st.cache_data(ttl=5)
def load_positions():
    """Carrega posições abertas"""
    try:
        if not mt5.initialize(path=config.MT5_TERMINAL_PATH): # Garante init
             return pd.DataFrame()

        positions = mt5.positions_get() or []
        if not positions:
            return pd.DataFrame()
        
        data = []
        for pos in positions:
            side = "COMPRA" if pos.type == mt5.POSITION_TYPE_BUY else "VENDA"
            
            # Cálculo de tempo
            time_open = datetime.fromtimestamp(pos.time)
            duration = datetime.now() - time_open
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            
            # P&L percentual
            if side == "COMPRA":
                pnl_pct = ((pos.price_current - pos.price_open) / pos.price_open) * 100
            else:
                pnl_pct = ((pos.price_open - pos.price_current) / pos.price_open) * 100
            
            data.append({
                "Ticket": pos.ticket,
                "Símbolo": pos.symbol,
                "Lado": side,
                "Volume": pos.volume,
                "Preço Entrada": pos.price_open,
                "Preço Atual": pos.price_current,
                "Stop Loss": pos.sl if pos.sl > 0 else 0,
                "Take Profit": pos.tp if pos.tp > 0 else 0,
                "P&L R$": pos.profit,
                "P&L ($)": f"${pos.profit:,.2f}", # ✅ Formatação Monetária Solicitada
                "P&L %": pnl_pct,
                "Tempo Aberto": time_str,
                "Timestamp": pos.time
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Erro ao carregar posições: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_historical_trades(days=30):
    """Carrega histórico de trades dos últimos N dias (Lê do MT5 ou Logs)"""
    # Tentar ler do MT5 History Deals para precisão
    try:
        if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
           return pd.DataFrame()
        
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now() + timedelta(days=1) # Até amanhã para garantir
        
        deals = mt5.history_deals_get(from_date, to_date)
        
        if deals:
            data = []
            for deal in deals:
                if deal.entry == mt5.DEAL_ENTRY_OUT: # Apenas saídas
                     symbol = deal.symbol
                     if not symbol: continue # Ignora depósitos/retiradas que não têm símbolo as vezes
                     
                     pnl = deal.profit
                     # Ignora transações de swap/comissão isoladas se desejar, mas normal é somar tudo
                     # Basic filter
                     
                     data.append({
                        'Data': datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d'),
                        'Timestamp': datetime.fromtimestamp(deal.time),
                        'Tipo': 'SAÍDA',
                        'Símbolo': symbol,
                        'Lado': 'VENDA' if deal.type == mt5.DEAL_TYPE_SELL else 'COMPRA', # Simplificado
                        'Volume': deal.volume,
                        'Preço': deal.price,
                        'P&L': pnl,
                        'Hora': datetime.fromtimestamp(deal.time).strftime('%H:%M:%S')
                     })
            return pd.DataFrame(data)

    except Exception as e:
        logger.error(f"Erro ao carregar histórico MT5: {e}")
        
    return pd.DataFrame()

def calculate_advanced_metrics(trades_df):
    """Calcula métricas avançadas de performance"""
    if trades_df.empty or 'P&L' not in trades_df.columns:
        return {}
    
    # Filtrar apenas trades fechados (SAÍDA)
    # No nosso load_historical_trades já filtramos ENTRY_OUT, então assumimos que tudo é trade fechado
    
    returns = trades_df['P&L'].values
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    total_trades = len(returns)
    winning_trades = len(wins)
    losing_trades = len(losses)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0
    
    # Expectativa matemática
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
    
    # Sharpe Ratio (simplificado)
    if len(returns) > 1:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    else:
        sharpe = 0
    
    # Maximum Drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    # Sortino Ratio
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 1 else 0
    sortino = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    # Calmar Ratio (Anualizado / MaxDD)
    if len(trades_df) > 1:
         total_seconds = (trades_df['Timestamp'].max() - trades_df['Timestamp'].min()).total_seconds()
         total_days = total_seconds / 86400
         total_days = max(total_days, 1)
         annualized_return = (returns.sum() / total_days) * 252
    else:
         annualized_return = 0
         
    calmar = (annualized_return / max_dd) if max_dd > 0 else 0
    
    # Maior sequência de ganhos/perdas
    win_streak = 0
    loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for ret in returns:
        if ret > 0:
            current_win_streak += 1
            current_loss_streak = 0
            win_streak = max(win_streak, current_win_streak)
        elif ret < 0:
            current_loss_streak += 1
            current_win_streak = 0
            loss_streak = max(loss_streak, current_loss_streak)
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'win_streak': win_streak,
        'loss_streak': loss_streak,
        'total_pnl': returns.sum(),
        'avg_trade': returns.mean(),
        'best_trade': returns.max() if len(returns) > 0 else 0,
        'worst_trade': returns.min() if len(returns) > 0 else 0
    }

@st.cache_data(ttl=5)
def load_real_time_market_data():
    """Carrega dados de mercado em tempo real via MT5 para Pares Forex"""
    try:
        if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
            return pd.DataFrame()
        
        # Use Forex Pairs from Config
        symbols = list(config.FOREX_PAIRS.keys()) # Todos os pares configurados
        
        data = []
        for sym in symbols:
            # Tenta encontrar o símbolo correto (normalização)
            real_sym = sym
            if not mt5.symbol_select(sym, True):
                # Tenta busca fuzzy rápida se não achar direto
                all_syms = mt5.symbols_get()
                for s in all_syms:
                    if sym in s.name:
                        real_sym = s.name
                        break
            
            tick = mt5.symbol_info_tick(real_sym)
            info = mt5.symbol_info(real_sym)
            
            status_str = "OK" if (tick and info) else "N/A"
            if tick and info:
                time_brasilia = datetime.fromtimestamp(tick.time) + timedelta(hours=3)
                digits = info.digits
                fmt_price = f"{{:.{digits}f}}"
                data.append({
                    'Símbolo': real_sym,
                    'Bid': fmt_price.format(tick.bid),
                    'Ask': fmt_price.format(tick.ask),
                    'Spread': info.spread,
                    'Último': fmt_price.format(tick.last),
                    'Volume': int(tick.volume),
                    'Horário (BSB)': time_brasilia.strftime('%H:%M:%S'),
                    'Status': status_str
                })
            else:
                data.append({
                    'Símbolo': real_sym,
                    'Bid': '-',
                    'Ask': '-',
                    'Spread': 0,
                    'Último': '-',
                    'Volume': 0,
                    'Horário (BSB)': '-',
                    'Status': status_str
                })
        
        # Ordenar por Símbolo para facilitar leitura
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('Símbolo')
            
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados reais: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_elite_config_dashboard():
    try:
        if bot and hasattr(bot, "load_elite_config"):
            cfg = bot.load_elite_config()
            return cfg or {}
    except Exception:
        pass
    try:
        import re, ast
        output_dir = Path(getattr(config, 'OPTIMIZER_OUTPUT', 'optimizer_output'))
        files = list(output_dir.glob("elite_settings_*.txt"))
        if not files:
            return {}
        latest = max(files, key=lambda f: f.stat().st_mtime)
        content = latest.read_text(encoding="utf-8")
        match = re.search(r"ELITE_CONFIG = (\{.*?\})", content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(1)) or {}
    except Exception:
        return {}
    return {}

def diagnosticar_ativo_forex(symbol):
    """
    Realiza um diagnóstico rápido do ativo forex para explicar por que o bot pode não estar entrando.
    Retorna uma lista de dicionários com os checks.
    """
    checks = []
    
    # 1. Verificação de Conexão e Símbolo
    if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
        checks.append({
            'Regra': "Conexão MT5", 'Valor': "Falha", 'Meta': "Conectado", 
            'Ok': False, 'Obs': "Terminal MT5 não conectado"
        })
        return checks
    
    # Normaliza o símbolo
    real_symbol = utils.normalize_symbol(symbol)
    found = mt5.symbol_select(real_symbol, True)
    
    if not found:
        checks.append({
            'Regra': "Market Watch", 'Valor': "N/A", 'Meta': "Presente", 
            'Ok': False, 'Obs': f"Ativo '{symbol}' não encontrado"
        })
        return checks
    
    checks.append({
        'Regra': "Market Watch", 'Valor': "OK", 'Meta': "Presente", 
        'Ok': True, 'Obs': f"Disponível ({real_symbol})"
    })

    # 2. Dados de Preço (M15)
    df = None
    erro_msg = None
    try:
        df = utils.safe_copy_rates(real_symbol, mt5.TIMEFRAME_M15, 100)
        if df is None:
            # Verificar motivo específico
            err = mt5.last_error()
            if err[0] != 1:
                erro_msg = f"Erro MT5: {err[1]}"
            else:
                erro_msg = "Sem histórico disponível"
    except Exception as e:
        erro_msg = f"Exceção: {str(e)[:30]}"
        logger.error(f"Erro ao obter dados para {real_symbol}: {e}")

    if df is None or len(df) < 50:
        obs_msg = erro_msg if erro_msg else "Falha ao baixar dados"
        checks.append({
            'Regra': "Dados M15", 'Valor': "Vazio", 'Meta': "100 candles", 
            'Ok': False, 'Obs': obs_msg
        })
        # NÃO retorna aqui - continua com checks básicos
    else:
        checks.append({
            'Regra': "Dados M15", 'Valor': f"{len(df)} candles", 'Meta': "100 candles", 
            'Ok': True, 'Obs': "Dados disponíveis"
        })

    # Se não tem dados, tenta pelo menos mostrar spread
    if df is None or len(df) < 50:
        # Mostra apenas spread e info básica
        info = mt5.symbol_info(real_symbol)
        if info:
            spread_pips = info.spread
            max_spread = getattr(config, 'MAX_SPREAD_FOREX', 25)
            checks.append({
                'Regra': "Spread (pips)", 
                'Valor': f"{spread_pips:.1f}", 
                'Meta': f"≤{max_spread}", 
                'Ok': spread_pips <= max_spread, 
                'Obs': "Alto" if spread_pips > max_spread else "Ok"
            })
        
        checks.append({
            'Regra': "Indicadores", 'Valor': "N/A", 'Meta': "Calculados", 
            'Ok': False, 'Obs': "Sem dados suficientes"
        })
        return checks

    # 3. Continua com cálculos se tem dados
    try:
        elite = load_elite_config_dashboard()
        base_params = config.FOREX_PAIRS.get(real_symbol, {})
        elite_params = elite.get(real_symbol, {})
        params = {**base_params, **elite_params}
        
        # Defaults
        ema_s_period = params.get('ema_short', getattr(config, 'EMA_SHORT_PERIOD', 20))
        ema_l_period = params.get('ema_long', getattr(config, 'EMA_LONG_PERIOD', 50))
        rsi_period = 14
        rsi_low = params.get('rsi_low', getattr(config, 'RSI_LOW_LIMIT', 30))
        rsi_high = params.get('rsi_high', getattr(config, 'RSI_HIGH_LIMIT', 70))
        adx_threshold = params.get('adx_threshold', getattr(config, 'ADX_THRESHOLD', 25))
        max_spread = getattr(config, 'MAX_SPREAD_FOREX', 25)

        # Obter indicadores
        ind = utils.get_indicators_forex(
            real_symbol,
            ema_short=ema_s_period,
            ema_long=ema_l_period,
            rsi_period=rsi_period,
            rsi_low=rsi_low,
            rsi_high=rsi_high
        )
        
        if ind is None or ind.get('error'):
            checks.append({
                'Regra': "Cálculo Indicadores", 'Valor': "Erro", 'Meta': "Sucesso", 
                'Ok': False, 'Obs': ind.get('message', 'Erro desconhecido')[:30] if ind else 'Retorno nulo'
            })
            return checks
        
        # 3. Tendência (EMA)
        ema_trend = ind.get('ema_trend', 'N/A')
        ema_short_val = ind.get('ema_short', 0)
        ema_long_val = ind.get('ema_long', 0)
        
        if ema_long_val > 0:
            dist_ema = abs(ema_short_val - ema_long_val) / ema_long_val * 100
        else:
            dist_ema = 0
        
        trend_emoji = "📈" if ema_trend == "UP" else "📉" if ema_trend == "DOWN" else "➡️"
        checks.append({
            'Regra': f"Tendência EMA {ema_s_period}/{ema_l_period}", 
            'Valor': f"{trend_emoji} {ema_trend}", 
            'Meta': "Definida", 
            'Ok': ema_trend == "UP", 
            'Obs': f"Dist: {dist_ema:.2f}%"
        })

        # 4. Spread
        spread_pips = ind.get('spread_pips', 0)
        spread_ok = spread_pips <= max_spread
        checks.append({
            'Regra': "Spread (pips)", 
            'Valor': f"{spread_pips:.1f}", 
            'Meta': f"≤{max_spread}", 
            'Ok': spread_ok, 
            'Obs': "Alto" if not spread_ok else "Ok"
        })
        
        # 5. Volume
        vol_ratio = ind.get('volume_ratio', 0)
        min_vol = getattr(config, 'MIN_VOLUME_COEFFICIENT', 0.4)
        vol_ok = vol_ratio >= min_vol
        checks.append({
            'Regra': "Volume (Ratio)", 
            'Valor': f"{vol_ratio:.2f}x", 
            'Meta': f"≥{min_vol}x", 
            'Ok': vol_ok, 
            'Obs': "Baixo" if not vol_ok else "Ok"
        })
        
        # 6. RSI
        rsi_val = ind.get('rsi', 50)
        rsi_msg = "Neutro"
        if rsi_val < rsi_low:
            rsi_msg = "Sobrevendido (Compra)"
        elif rsi_val > rsi_high:
            rsi_msg = "Sobrecomprado (Venda)"
        
        checks.append({
            'Regra': "RSI (14)", 
            'Valor': f"{rsi_val:.1f}", 
            'Meta': f"{rsi_low}-{rsi_high}", 
            'Ok': True,
            'Obs': rsi_msg
        })
        rsi_req = 50
        checks.append({
            'Regra': "RSI > 50", 
            'Valor': f"{rsi_val:.1f}", 
            'Meta': f">{rsi_req}", 
            'Ok': rsi_val > rsi_req,
            'Obs': "Abaixo" if not (rsi_val > rsi_req) else "Ok"
        })
        
        # 7. ADX
        adx_val = ind.get('adx', 0)
        adx_ok = adx_val >= adx_threshold
        checks.append({
            'Regra': "ADX (Força)", 
            'Valor': f"{adx_val:.1f}", 
            'Meta': f"≥{adx_threshold}", 
            'Ok': adx_ok, 
            'Obs': "Lateral" if not adx_ok else "Tendência"
        })
        
        # 8. ML Confidence
        ml_conf = ind.get('ml_confidence')
        try:
            if getattr(config, 'ENABLE_ML_OPTIMIZER', False) and hasattr(utils, "ml_optimizer_instance") and utils.ml_optimizer_instance:
                df_ml = df
                signal_for_ml = "BUY" if ema_trend == "UP" else "SELL" if ema_trend == "DOWN" else "NONE"
                ml_score_raw, _ = utils.ml_optimizer_instance.get_prediction_score(real_symbol, ind, df_ml, signal=signal_for_ml)
                ml_conf = (ml_score_raw / 100.0) if ml_score_raw is not None else ml_conf
        except Exception:
            pass
        default_ml = float(getattr(config, 'ML_CONFIDENCE_THRESHOLD', 0.65) or 0.65)
        base_ml = None
        elite_ml = None
        try:
            base_ml = float(base_params.get('ml_threshold')) if base_params.get('ml_threshold') is not None else None
        except Exception:
            base_ml = None
        try:
            elite_ml = float(elite_params.get('ml_threshold')) if elite_params.get('ml_threshold') is not None else None
        except Exception:
            elite_ml = None
        min_ml = elite_ml if elite_ml is not None else max(default_ml, base_ml or 0.0)
        if ml_conf is None:
            checks.append({
                'Regra': "ML Confidence", 
                'Valor': "N/A", 
                'Meta': f"≥{min_ml*100:.0f}", 
                'Ok': False, 
                'Obs': "Indisponível"
            })
        else:
            ml_ok = ml_conf >= min_ml
            checks.append({
                'Regra': "ML Confidence", 
                'Valor': f"{ml_conf*100:.0f}", 
                'Meta': f"≥{min_ml*100:.0f}", 
                'Ok': ml_ok, 
                'Obs': "Baixo" if not ml_ok else "Ok"
            })
        
        close_price = ind.get('close', 0)
        open_price = ind.get('open', 0)
        if close_price and open_price:
            checks.append({
                'Regra': "Candle de Alta", 
                'Valor': f"C:{close_price} O:{open_price}", 
                'Meta': "C>O", 
                'Ok': close_price > open_price, 
                'Obs': "Baixa" if not (close_price > open_price) else "Ok"
            })
        
        # 9. EMA 200 Filter (se habilitado)
        if getattr(config, 'ENABLE_EMA_200_FILTER', False):
            try:
                ema_200_data = utils.get_ema_200(real_symbol)
                if not ema_200_data.get('error'):
                    is_above = ema_200_data.get('is_above_ema', False)
                    ema_200_trend = ema_200_data.get('trend_direction', 'N/A')
                    checks.append({
                        'Regra': "EMA 200 Macro", 
                        'Valor': f"{ema_200_trend}", 
                        'Meta': "Alinhado", 
                        'Ok': is_above, 
                        'Obs': "Acima" if is_above else "Abaixo"
                    })
            except Exception:
                pass
        
        # 10. Multi-Timeframe (se habilitado)
        if getattr(config, 'ENABLE_MULTI_TIMEFRAME', False):
            try:
                # ✅ v5.3: Agora retorna score penalty
                mtf_penalty, mtf_reason, mtf_trend = utils.get_multi_timeframe_trend(real_symbol, "BUY")
                checks.append({
                    'Regra': "Multi-TF (H4)", 
                    'Valor': mtf_trend if mtf_trend else "N/A", 
                    'Meta': "Alinhado", 
                    'Ok': mtf_penalty == 0, 
                    'Obs': mtf_reason[:20] if mtf_reason else "Ok"
                })
            except Exception:
                pass
        
    except Exception as e:
        checks.append({
            'Regra': "Cálculo Indicadores", 
            'Valor': "Erro", 
            'Meta': "Sucesso", 
            'Ok': False, 
            'Obs': str(e)[:30]
        })
    
    return checks

# ===========================
# HEADER PRINCIPAL
# ===========================
st.markdown('<div class="main-header">🏛️ XP3 FOREX INSTITUTIONAL DASHBOARD</div>', unsafe_allow_html=True)

# ===========================
# SIDEBAR - CONTROLES E FILTROS
# ===========================
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/0f3460/FFFFFF?text=XP3+FOREX")
    
    st.markdown("---")
    st.subheader("⚙️ Configurações")
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Atualização Automática", value=True)
    refresh_interval = st.slider("Intervalo (segundos)", 5, 120, 30)
    
    if st.button("🔄 Atualizar Agora", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Período de Análise")
    
    analysis_period = st.selectbox(
        "Selecione o período",
        ["Hoje", "Últimos 7 dias", "Últimos 30 dias", "Este mês", "Personalizado"]
    )
    
    if analysis_period == "Personalizado":
        date_from = st.date_input("Data inicial", datetime.now() - timedelta(days=30))
        date_to = st.date_input("Data final", datetime.now())
    
    st.markdown("---")
    
    # Bot Status Info (Simplificado)
    st.info(f"Bot Mode: {'Active'}") # Placeholder

# ===========================
# CARREGAMENTO DE DADOS
# ===========================
acc = load_account_info()
positions_df = load_positions()

# Definir período baseado na seleção
if analysis_period == "Hoje":
    days_to_load = 1
elif analysis_period == "Últimos 7 dias":
    days_to_load = 7
elif analysis_period == "Últimos 30 dias":
    days_to_load = 30
else:
    days_to_load = 30
    
historical_trades = load_historical_trades(days=days_to_load)

if not acc:
    st.error("❌ Erro ao conectar com MT5. Verifique a configuração e se o terminal está aberto.")
    st.stop()

# ===========================
# SEÇÃO 1: PAINEL DE CONTROLE - KPI
# ===========================
col_title, col_status = st.columns([0.65, 0.35])

with col_title:
    st.markdown('<div class="section-header">📊 Painel de Controle - Visão Geral</div>', unsafe_allow_html=True)

with col_status:
    # Market Status Placeholder
    status_text = "MARKET OPEN" # Simplificação
    st.markdown(
            f"""
            <div style="
                background-color: white; 
                padding: 10px; 
                border-radius: 8px; 
                border: 1px solid #ddd;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                text-align: right;
                font-family: 'Segoe UI', sans-serif;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #1a1a2e; font-size: 0.9rem;">Status:</span>
                    <span style="font-weight: bold; color: #00b894; background: #00b89420; padding: 2px 8px; border-radius: 4px;">{status_text}</span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

# KPI Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">💰 Patrimônio Líquido</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">${acc["equity"]:,.2f}</div>', unsafe_allow_html=True)
    pnl_day = acc['profit']
    delta_color = "🟢" if pnl_day >= 0 else "🔴"
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">{delta_color} ${pnl_day:+,.2f} hoje</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with kpi2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">📈 Retorno Diário</div>', unsafe_allow_html=True)
    pnl_pct = (acc['profit'] / acc['balance'] * 100) if acc['balance'] > 0 else 0
    st.markdown(f'<div class="kpi-value">{pnl_pct:+.2f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">Base: ${acc["balance"]:,.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with kpi3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">🎯 Posições Ativas</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{len(positions_df)}</div>', unsafe_allow_html=True)
    max_pos = getattr(config, 'MAX_SYMBOLS', 4)
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">Limite: {max_pos} posições</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with kpi4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">🛡️ Nível de Margem</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{acc["margin_level"]:.1f}%</div>', unsafe_allow_html=True)
    margin_status = "Seguro" if acc['margin_level'] > 200 else "Atenção"
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">{margin_status}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with kpi5:
    lucro_flutuante = positions_df['P&L R$'].sum() if not positions_df.empty else 0
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">💸 P&L Flutuante</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">${lucro_flutuante:+,.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">{len(positions_df)} operações abertas</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================
# SEÇÃO 2: MÉTRICAS AVANÇADAS DE PERFORMANCE
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📈 Métricas Avançadas de Performance</div>', unsafe_allow_html=True)

metrics = calculate_advanced_metrics(historical_trades)
if metrics:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🎯 Taxa Acerto", f"{metrics['win_rate']:.1f}%")
    
    with col2:
        st.metric("💹 Profit Factor", f"{metrics['profit_factor']:.2f}")
    
    with col3:
        st.metric("📊 Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    
    with col4:
        st.metric("🍷 Sortino", f"{metrics['sortino_ratio']:.2f}",
                 help="Retorno ajustado pelo risco descendente")
    
    with col5:
        st.metric("🏆 Calmar", f"{metrics['calmar_ratio']:.2f}",
                 help="Retorno Anual / Max Drawdown")
    
    with col6:
        st.metric("📉 Max DD", f"${metrics['max_drawdown']:.2f}")
    
    # Linha adicional de métricas
    st.markdown("### 📊 Estatísticas Detalhadas")
    
    col7, col8, col9, col10, col11, col12 = st.columns(6)
    
    with col7:
        st.metric("🟢 Ganho Médio", f"${metrics['avg_win']:.2f}")
    
    with col8:
        st.metric("🔴 Perda Média", f"${metrics['avg_loss']:.2f}")
    
    with col9:
        rr_ratio = metrics['avg_win'] / metrics['avg_loss'] if metrics['avg_loss'] > 0 else 0
        st.metric("⚖️ Risk/Reward", f"{rr_ratio:.2f}",
                 help="Relação ganho médio / perda média")
    
    with col10:
        st.metric("🏆 Melhor Trade", f"${metrics['best_trade']:.2f}")
    
    with col11:
        st.metric("💔 Pior Trade", f"${metrics['worst_trade']:.2f}")
    
    with col12:
        recovery_factor = abs(metrics['total_pnl'] / metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        st.metric("🔄 Recovery Factor", f"{recovery_factor:.2f}",
                 help="Capacidade de recuperação")
else:
    st.info("📊 Aguardando histórico de operações para calcular métricas...")

# ===========================
# SEÇÃO 3: EQUITY CURVE E ANÁLISE DE DRAWDOWN
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📈 Curva de Equity e Análise de Drawdown</div>', unsafe_allow_html=True)

if not historical_trades.empty and 'P&L' in historical_trades.columns:
    # Como já filtramos por 'SAÍDA', usamos direto
    closed_trades = historical_trades.copy()
    
    if not closed_trades.empty:
        closed_trades = closed_trades.sort_values('Data')
        closed_trades['P&L Acumulado'] = closed_trades['P&L'].cumsum()
        closed_trades['Máximo Acumulado'] = closed_trades['P&L Acumulado'].cummax()
        closed_trades['Drawdown'] = closed_trades['P&L Acumulado'] - closed_trades['Máximo Acumulado']
        
        # Criar subplot com 2 gráficos
        fig_equity = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Curva de Equity', 'Underwater Chart (Drawdown)'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Equity Curve
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(closed_trades))),
                y=closed_trades['P&L Acumulado'],
                mode='lines',
                name='P&L Acumulado',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ),
            row=1, col=1
        )
        
        # Underwater Chart
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(closed_trades))),
                y=closed_trades['Drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='#d63031', width=2),
                fill='tozeroy',
                fillcolor='rgba(214, 48, 49, 0.2)'
            ),
            row=2, col=1
        )
        
        fig_equity.update_xaxes(title_text="Número da Operação", row=2, col=1)
        fig_equity.update_yaxes(title_text="P&L ($)", row=1, col=1)
        fig_equity.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
        
        fig_equity.update_layout(
            height=700,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_equity, width='stretch')
    else:
        st.info("📊 Aguardando trades fechados para análise de equity...")
else:
    st.info("📊 Aguardando histórico de trades...")


# ===========================
# SEÇÃO 4: ANÁLISE DE REJEIÇÕES (NOVO)
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">🔍 Análise de Oportunidades Rejeitadas</div>', unsafe_allow_html=True)

@st.cache_data(ttl=30)
def load_rejections_data():
    """Lê o log de análises do dia e estrutura os dados (Brasília Time)"""
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path("analysis_logs")
    files = sorted(list(log_dir.glob(f"analysis_log_{today}_*.txt")) + list(log_dir.glob(f"analysis_log_{today}_*.txt.gz")))
    if not files:
        legacy = log_dir / f"analysis_log_{today}.txt"
        if legacy.exists():
            files = [legacy]
    if not files:
        return pd.DataFrame(), {}
    data = []
    reasons_count = {}
    try:
        contents = []
        for fp in files:
            if fp.suffix == ".gz":
                import gzip
                with gzip.open(fp, 'rt', encoding='utf-8') as f:
                    contents.append(f.read())
            else:
                with open(fp, 'r', encoding='utf-8') as f:
                    contents.append(f.read())
        content = "\n".join(contents)
            
        # Divide por blocos de entrada (separados por linhas de =)
        blocks = content.split('='*80)
        
        for block in blocks:
            if "|" not in block or "Motivo:" not in block:
                continue
                
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            
            entry = {}
            for line in lines:
                if "|" in line and ("🕐" in line or "1" in line): # Timestamp line
                    parts = line.split("|")
                    if len(parts) >= 3:
                        entry['Time'] = parts[0].replace('🕐', '').strip()
                        entry['Symbol'] = parts[1].strip()
                        entry['Status'] = parts[2].strip()
                
                if "Sinal:" in line:
                    parts = line.split("|")
                    entry['Signal'] = parts[0].split(":")[1].strip()
                    try:
                         entry['Score'] = parts[2].split(":")[1].strip()
                    except:
                         entry['Score'] = "N/A"

                if "Motivo:" in line:
                    reason = line.split("Motivo:")[1].strip()
                    entry['Reason'] = reason
                    if 'REJEITADA' in entry.get('Status', '') or 'AGUARDANDO' in entry.get('Status', ''):
                        reasons_count[reason] = reasons_count.get(reason, 0) + 1

                if "RSI:" in line:
                    entry['RSI'] = line.split("RSI:")[1].strip()
                
                if "Volume:" in line:
                    entry['Volume'] = line.split("Volume:")[1].strip()
            
            # Filtra apenas o que não foi executado ou monitorando
            if entry and 'Status' in entry and ('REJEITADA' in entry['Status'] or 'AGUARDANDO' in entry['Status'] or 'MONITORANDO' in entry['Status']):
                data.append(entry)
                
    except Exception as e:
        logger.error(f"Erro ao ler logs: {e}")
        
    return pd.DataFrame(data), reasons_count

rejections_df, reasons_stats = load_rejections_data()

if not rejections_df.empty:
    r_col1, r_col2 = st.columns([2, 1])
    
    with r_col1:
        st.markdown("### 📋 Diário de Análise (Últimos 50)")
        # Show recent first
        st.dataframe(
            rejections_df.tail(50)[['Time', 'Symbol', 'Signal', 'Score', 'Reason', 'RSI', 'Volume']],
            width='stretch',
            height=300,
            hide_index=True
        )
    
    with r_col2:
        st.markdown("### 📉 Principais Motivos de Rejeição")
        if reasons_stats:
            reasons_df = pd.DataFrame(
                list(reasons_stats.items()), 
                columns=['Motivo', 'Contagem']
            ).sort_values('Contagem', ascending=False)
            
            fig_reasons = px.pie(
                reasons_df, 
                values='Contagem', 
                names='Motivo',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_reasons.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                height=250,
                showlegend=False
            )
            st.plotly_chart(fig_reasons, width='stretch')
            
            # Top 3 reasons text
            st.markdown("#### Top Bloqueios:")
            for i, row in reasons_df.head(5).iterrows():
                st.markdown(f"- **{row['Contagem']}x**: {row['Motivo']}")
else:
    st.info("✅ Nenhuma rejeição registrada hoje (ou arquivo de log ainda vazio).")

# ===========================
# SEÇÃO 4.5: RAIO-X DE EXECUÇÃO (DIAGNÓSTICO)
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">🔍 Raio-X de Execução: Por que o bot não entrou?</div>', unsafe_allow_html=True)

lista_ativos = list(config.FOREX_PAIRS.keys()) if getattr(config, "SHOW_ALL_SYMBOLS_IN_DASHBOARD", True) else list(config.FOREX_PAIRS.keys())[:12]
selected_ativos = st.multiselect("Selecionar ativos", lista_ativos, default=lista_ativos)
filtro_status = st.selectbox("Filtrar por status", ["Todos", "Pronto", "Avisos", "Erros"])
mostrar_apenas_erros = st.checkbox("Mostrar apenas checks com erro", value=False)
cols_diag = st.columns(3)
col_counter = 0
for ativo in selected_ativos:
    with cols_diag[col_counter % 3]:
        res = diagnosticar_ativo_forex(ativo)
        total_checks = len(res)
        oks = sum(1 for r in res if r['Ok'])
        erros = total_checks - oks
        aprovado = erros == 0
        if aprovado:
            header_class = "ok"
            status_icon = "✅"
            status_text = "PRONTO"
        elif erros <= 2:
            header_class = "warn"
            status_icon = "⚠️"
            status_text = f"{erros} AVISO(S)"
        else:
            header_class = "error"
            status_icon = "❌"
            status_text = f"{erros} ERRO(S)"
        status_tag = "Pronto" if header_class == "ok" else "Avisos" if header_class == "warn" else "Erros"
        if filtro_status != "Todos" and filtro_status != status_tag:
            col_counter += 1
            continue
        percent_ok = int((oks / total_checks) * 100) if total_checks > 0 else 0
        checks_to_show = res if not mostrar_apenas_erros else [r for r in res if not r['Ok']]
        html_checks = ""
        for r in checks_to_show:
            ok_class = "badge-ok" if r.get('Ok') else "badge-error"
            html_checks += f"""
            <div class="diag-check {ok_class}">
                <div class="diag-rule">{r.get('Regra','')}</div>
                <div><strong>Valor:</strong> {r.get('Valor','')}</div>
                <div class="diag-meta"><strong>Meta:</strong> {r.get('Meta','')}</div>
                <div class="diag-obs">{r.get('Obs','')}</div>
            </div>
            """
        card_html = f"""
        <style>
            .diag-card {{
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.08);
                overflow: hidden;
                margin-bottom: 16px;
                border: 1px solid #e6e6e6;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
            }}
            .diag-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 14px;
                color: #ffffff;
                font-weight: 700;
                letter-spacing: 0.3px;
            }}
            .diag-header.ok {{ background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }}
            .diag-header.warn {{ background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%); color: #1a1a2e; }}
            .diag-header.error {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }}
            .diag-status {{ font-size: 0.95rem; }}
            .diag-body {{ padding: 12px 14px; background: #fafafa; }}
            .progress {{ height: 8px; background: #e9ecef; border-radius: 8px; overflow: hidden; margin: 6px 0 12px 0; }}
            .progress-fill {{ height: 8px; background: #2ecc71; border-radius: 8px; transition: width 0.4s ease; }}
            .diag-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
            .diag-check {{ padding: 10px; border-radius: 8px; border: 1px solid #eee; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.03); font-size: 0.9rem; }}
            .diag-check.badge-ok {{ border-left: 4px solid #2ecc71; background: #e8f8f0; }}
            .diag-check.badge-error {{ border-left: 4px solid #e74c3c; background: #fdecea; }}
            .diag-rule {{ font-weight: 700; color: #0f3460; margin-bottom: 4px; }}
            .diag-meta {{ color: #555; font-size: 0.85rem; }}
            .diag-obs {{ color: #333; font-size: 0.85rem; margin-top: 4px; }}
        </style>
        <div class="diag-card">
            <div class="diag-header {header_class}">
                <div>{ativo}</div>
                <div class="diag-status">{status_icon} {status_text}</div>
            </div>
            <div class="diag-body">
                <div class="progress"><div class="progress-fill" style="width:{percent_ok}%"></div></div>
                <div class="diag-grid">
                    {html_checks if html_checks else '<div class="diag-check badge-ok">Sem bloqueios</div>'}
                </div>
            </div>
        </div>
        """
        comp_height = max(180, min(520, 120 + len(checks_to_show) * 90))
        components.html(card_html, height=comp_height, scrolling=True)
    col_counter += 1

# ===========================
# SEÇÃO 5: POSIÇÕES ABERTAS
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📋 Posições Abertas - Gestão Ativa</div>', unsafe_allow_html=True)

if not positions_df.empty:
    positions_display = positions_df.copy()
    
    # Status da posição
    def get_position_status(row):
        if row['P&L %'] > 1.0:
            return '🟢 Lucro Forte'
        elif row['P&L %'] > 0:
            return '🟡 Lucro'
        elif row['P&L %'] > -1.0:
            return '🟠 Pequena Perda'
        else:
            return '🔴 Perda'
    
    positions_display['Status'] = positions_display.apply(get_position_status, axis=1)
    
    # ✅ Exibe P&L ($) formatado
    cols_to_show = ['Símbolo', 'Lado', 'Volume', 'Preço Entrada', 'Preço Atual',
                    'Stop Loss', 'Take Profit', 'P&L ($)', 'P&L %', 'Status']
    
    # Check if cols exist (safety)
    safe_cols = [c for c in cols_to_show if c in positions_display.columns]
    positions_display = positions_display[safe_cols]
    
    st.dataframe(positions_display, width='stretch', hide_index=True)
else:
    st.info("✅ Nenhuma posição aberta no momento. O bot está analisando o mercado.")

# ===========================
# SEÇÃO 5: MONITORAMENTO AO VIVO
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📈 Dados de Mercado (Top Forex Pairs)</div>', unsafe_allow_html=True)

real_time_df = load_real_time_market_data()

m_col1, m_col2 = st.columns([0.75, 0.25])

with m_col1:
    if not real_time_df.empty:
        st.dataframe(
            real_time_df, 
            width='stretch', 
            hide_index=True,
            column_config={
                "Símbolo": st.column_config.TextColumn("Símbolo", width="small"),
                "Bid": st.column_config.TextColumn("Bid", width="medium"),
                "Ask": st.column_config.TextColumn("Ask", width="medium"),
                "Spread": st.column_config.NumberColumn("Spread", format="%d pts"),
                "Último": st.column_config.TextColumn("Último", width="medium"),
                "Volume": st.column_config.NumberColumn("Volume"),
                "Horário (BSB)": st.column_config.TextColumn("Horário (BSB)", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
            }
        )
    else:
        st.warning("⚠️ Carregando dados de mercado... (MT5 pode estar desconectado)")

with m_col2:
    st.markdown("### 🌍 Sessão Atual")
    try:
        session_info = utils.get_current_trading_session()
        bsb_time = utils.get_brasilia_time().strftime('%H:%M')
        
        st.markdown(
            f"""
            <div style="
                background-color: {session_info.get('color', 'gray')}20;
                border-left: 5px solid {session_info.get('color', 'gray')};
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="font-size: 3rem;">{session_info.get('emoji', '🌍')}</div>
                <div style="font-weight: bold; font-size: 1.2rem; color: #1a1a2e;">{session_info.get('display', 'N/A')}</div>
                <div style="font-size: 0.9rem; margin-top: 10px;">⏰ Brasília: {bsb_time}</div>
                <div style="font-size: 0.8rem; color: #666;">(UTC-3)</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Market Status Open/Close
        m_status = utils.get_market_status()
        st.markdown(f"**Status:** {m_status.get('emoji')} {m_status.get('message')}")
        
    except Exception as e:
        st.error(f"Erro sessão: {e}")

@st.cache_data(ttl=60)
def load_weekly_optimizer_summary():
    try:
        fp = Path("analysis_logs") / "optimizer_weekly.json"
        if not fp.exists():
            return {}
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

st.markdown("---")
st.markdown('<div class="section-header">📊 Otimizador Semanal</div>', unsafe_allow_html=True)
opt_summary = load_weekly_optimizer_summary()
if opt_summary:
    cols_top = st.columns(3)
    with cols_top[0]:
        st.metric("Candidatos avaliados", int(opt_summary.get("total_candidates", 0)))
    with cols_top[1]:
        st.metric("Elite selecionada", int(opt_summary.get("elite_count", 0)))
    with cols_top[2]:
        st.caption(f"Gerado em {opt_summary.get('generated_at','')}")
    items = opt_summary.get("items", [])
    rows = []
    for it in items:
        m = it.get("error_metrics", {}) or {}
        ab = it.get("ab_test", {}) or {}
        a_m = (ab.get("A") or {}).get("metrics", {}) or {}
        b_m = (ab.get("B") or {}).get("metrics", {}) or {}
        wr_oos = float(it.get("win_rate", 0.0)) * 100.0
        dd_oos = float(it.get("drawdown", 0.0)) * 100.0
        trades_oos = int(it.get("total_trades", 0))
        a_wr = float(a_m.get("win_rate", 0.0)) * 100.0
        b_wr = float(b_m.get("win_rate", 0.0)) * 100.0
        a_pf = float(a_m.get("profit_factor", 0.0))
        b_pf = float(b_m.get("profit_factor", 0.0))
        rows.append({
            "Símbolo": it.get("symbol", ""),
            "Rank": float(it.get("rank", 0.0)),
            "WR OOS (%)": wr_oos,
            "PF OOS": float(it.get("profit_factor", 0.0)),
            "DD OOS (%)": dd_oos,
            "Trades OOS": trades_oos,
            "MAE": float(m.get("mae") or 0.0),
            "RMSE": float(m.get("rmse") or 0.0),
            "MAPE": float(m.get("mape") or 0.0),
            "Acc (%)": float(m.get("accuracy") or 0.0) * 100.0,
            "A WR (%)": a_wr,
            "B WR (%)": b_wr,
            "Δ WR (B−A)": b_wr - a_wr,
            "A PF": a_pf,
            "B PF": b_pf,
            "Δ PF (B−A)": b_pf - a_pf,
        })
    df_opt = pd.DataFrame(rows)
    if not df_opt.empty:
        df_opt = df_opt.sort_values("Rank", ascending=False)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Top WR OOS (%)", round(df_opt["WR OOS (%)"].max(), 2))
        with k2:
            st.metric("Top PF OOS", round(df_opt["PF OOS"].max(), 2))
        with k3:
            st.metric("Melhor Acc (%)", round(df_opt["Acc (%)"].max(), 2))
        with k4:
            st.metric("Média Δ PF (B−A)", round(df_opt["Δ PF (B−A)"].mean(), 2))
        st.dataframe(
            df_opt,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Símbolo": st.column_config.TextColumn("Símbolo", width="small"),
                "Rank": st.column_config.NumberColumn("Rank"),
                "WR OOS (%)": st.column_config.NumberColumn("WR OOS (%)", format="%.2f"),
                "PF OOS": st.column_config.NumberColumn("PF OOS", format="%.2f"),
                "DD OOS (%)": st.column_config.NumberColumn("DD OOS (%)", format="%.2f"),
                "Trades OOS": st.column_config.NumberColumn("Trades OOS"),
                "MAE": st.column_config.NumberColumn("MAE", format="%.4f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.4f"),
                "MAPE": st.column_config.NumberColumn("MAPE", format="%.4f"),
                "Acc (%)": st.column_config.NumberColumn("Acc (%)", format="%.2f"),
                "A WR (%)": st.column_config.NumberColumn("A WR (%)", format="%.2f"),
                "B WR (%)": st.column_config.NumberColumn("B WR (%)", format="%.2f"),
                "Δ WR (B−A)": st.column_config.NumberColumn("Δ WR (B−A)", format="%.2f"),
                "A PF": st.column_config.NumberColumn("A PF", format="%.2f"),
                "B PF": st.column_config.NumberColumn("B PF", format="%.2f"),
                "Δ PF (B−A)": st.column_config.NumberColumn("Δ PF (B−A)", format="%.2f"),
            }
        )
    else:
        st.info("Sem itens no resumo semanal.")
else:
    st.info("Nenhum resumo do otimizador encontrado.")

# ===========================
# SEÇÃO 6: LOGS DE ANÁLISE (FILTROS)
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📝 Logs de Análise (Audit Trail)</div>', unsafe_allow_html=True)
try:
    analyzer = LogAnalyzer(log_dir="analysis_logs")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Data inicial", datetime.now() - timedelta(days=7))
    with c2:
        end_date = st.date_input("Data final", datetime.now())
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    all_entries = []
    current = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    while current <= end_dt:
        date_str = current.strftime('%Y-%m-%d')
        files = sorted(list(Path("analysis_logs").glob(f"analysis_log_{date_str}_*.txt")) + list(Path("analysis_logs").glob(f"analysis_log_{date_str}_*.txt.gz")))
        if not files:
            legacy = Path("analysis_logs") / f"analysis_log_{date_str}.txt"
            if legacy.exists():
                files = [legacy]
        raw = []
        blocks_all = []
        for fp in files:
            parsed = analyzer.parse_log_file(fp)
            raw.extend(parsed)
            try:
                if fp.suffix == ".gz":
                    import gzip
                    with gzip.open(fp, 'rt', encoding='utf-8') as f:
                        text = f.read()
                else:
                    text = fp.read_text(encoding="utf-8")
                blocks = [b for b in text.split("\n" + "="*80 + "\n") if b.strip()]
                blocks_all.extend(blocks)
            except Exception:
                pass
        try:
            for entry in raw:
                key = f"🕐 {entry['time']} | {entry['symbol']} |"
                block = next((b for b in blocks_all if key in b), "")
                user = None
                context = None
                if "👤 Usuário:" in block:
                    try:
                        user = block.split("👤 Usuário:")[1].splitlines()[0].strip()
                    except Exception:
                        user = None
                if "🧭 Contexto:" in block:
                    try:
                        ctx_line = block.split("🧭 Contexto:")[1].splitlines()[0].strip()
                        context = ctx_line
                    except Exception:
                        context = None
                entry['user'] = user or ""
                entry['context'] = context or ""
        except Exception:
            pass
        all_entries.extend(raw)
        current += timedelta(days=1)
    if not all_entries:
        st.info("Nenhum log encontrado no período selecionado.")
    else:
        df = pd.DataFrame(all_entries)
        def classify_action(status: str):
            s = status.upper()
            if "EXECUTADA" in s:
                return "ACEITE"
            if "MONITORANDO" in s:
                return "MONITORANDO"
            return "BLOQUEIO"
        df['Ação'] = df['status'].apply(classify_action)
        symbols = sorted(df['symbol'].unique().tolist())
        actions = ["ACEITE", "BLOQUEIO", "MONITORANDO"]
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_symbols = st.multiselect("Símbolos", symbols, default=symbols[:10])
        with f2:
            sel_actions = st.multiselect("Tipo de ação", actions, default=actions)
        with f3:
            reason_filter = st.text_input("Filtro por motivo (contains)")
        mask = df['symbol'].isin(sel_symbols) & df['Ação'].isin(sel_actions)
        if reason_filter:
            mask = mask & df['reason'].str.contains(reason_filter, case=False, na=False)
        df_f = df[mask].copy()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total", len(df_f))
        with m2:
            st.metric("Aceites", int((df_f['Ação'] == "ACEITE").sum()))
        with m3:
            st.metric("Bloqueios", int((df_f['Ação'] == "BLOQUEIO").sum()))
        show_cols = ["time", "symbol", "Ação", "signal", "strategy", "score", "rsi", "adx", "spread", "volume", "ema_trend", "reason"]
        if "user" in df_f.columns:
            show_cols.append("user")
        if "context" in df_f.columns:
            show_cols.append("context")
        st.dataframe(df_f[show_cols], use_container_width=True)
except Exception as e:
    st.error(f"Falha ao carregar logs: {e}")

# ===========================
# FOOTER
# ===========================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption(f"🕐 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} (Brasília)")
with footer_col2:
    st.caption("🏛️ XP3 PRO FOREX Institutional Dashboard v6.0")
with footer_col3:
    st.caption(f"Bot Status: {'Online' if BOT_CONNECTED else 'Standalone'}")

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

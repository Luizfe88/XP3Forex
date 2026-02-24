# daily_analysis_logger.py - Sistema de Log Diário de Análises
"""
📝 Logger de análises de sinais em arquivos TXT diários
✅ Um arquivo por dia (analysis_log_YYYY-MM-DD.txt)
✅ Registra TODAS as análises (executadas e rejeitadas)
✅ Formato legível para auditoria manual
"""

import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

# ===========================
# REJECTION CATEGORIES v5.0
# ===========================
REJECTION_CATEGORIES = {
    "EMA200": "Rejeitado: Contra tendência Macro (EMA 200)",
    "DRAWDOWN": "Rejeitado: Circuit Breaker Ativo (Risco Diário)",
    "ROLLOVER": "Rejeitado: Período de Rollover Bancário",
    "ADX_LOW": "Rejeitado: Mercado Lateral (ADX < 20)",
    "RSI_NO_REVERSAL": "Rejeitado: RSI sem reversão confirmada",
    "SPREAD": "Rejeitado: Spread Alto",
    "NEWS": "Rejeitado: Notícia Alto Impacto",
    "COOLDOWN": "Rejeitado: Cooldown ativo",
    "LIMIT": "Rejeitado: Limite de ordens atingido",
    "VOLUME": "Rejeitado: Volume insuficiente",
    "VETO_TECH": "Rejeitado: Veto técnico",
}

class DailyAnalysisLogger:
    """
    Logger que cria um arquivo TXT novo para cada dia
    registrando todas as análises de sinais
    """
    
    def __init__(self, log_dir: str = "analysis_logs", rotation_hours: int = 3, compress_after_hours: int = 24, retention_days: int = 7, max_files: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.rotation_hours = max(1, int(rotation_hours))
        self.compress_after_hours = max(1, int(compress_after_hours))
        self.retention_days = max(1, int(retention_days))
        self.max_files = max(1, int(max_files))
        self.current_window_start = None
        self.current_file = None
        self.lock = Lock()
        
    def _get_log_filename(self) -> Path:
        now = datetime.now()
        window_hour = (now.hour // self.rotation_hours) * self.rotation_hours
        window_dt = now.replace(hour=window_hour, minute=0, second=0, microsecond=0)
        self.current_window_start = window_dt
        date_part = window_dt.strftime("%Y-%m-%d")
        time_part = window_dt.strftime("%H%M")
        return self.log_dir / f"analysis_log_{date_part}_{time_part}.txt"
    
    def _check_date_rollover(self):
        now = datetime.now()
        window_hour = (now.hour // self.rotation_hours) * self.rotation_hours
        window_dt = now.replace(hour=window_hour, minute=0, second=0, microsecond=0)
        if self.current_window_start != window_dt or self.current_file is None:
            self.current_file = self._get_log_filename()
            if not self.current_file.exists():
                with open(self.current_file, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"📊 XP3 PRO FOREX - LOG DE ANÁLISES\n")
                    f.write(f"📅 Janela: {window_dt.strftime('%d/%m/%Y %H:%M')}\n")
                    f.write("="*80 + "\n\n")
            try:
                self._compress_old_logs()
                self._enforce_retention_policy()
            except Exception:
                pass
    
    def log_analysis(self, 
                     symbol: str,
                     signal: str,
                     strategy: str,
                     score: float,
                     rejected: bool,
                     reason: str,
                     indicators: dict,
                     ml_score: float = 0.0,
                     is_baseline: bool = False,
                     user: Optional[str] = None,
                     context: Optional[dict] = None):
        """
        Registra uma análise no arquivo do dia
        
        Args:
            symbol: Par analisado (ex: EURUSD)
            signal: Sinal detectado (BUY, SELL, NONE)
            strategy: Estratégia usada (TREND, REVERSION, N/A)
            score: Score calculado (0-120)
            rejected: Se foi rejeitado ou executado
            reason: Motivo da rejeição ou execução
            indicators: Dict com RSI, ADX, spread, etc
            ml_score: Score de confiança do ML
            is_baseline: Se o score do ML é vindo do backtest
        """
        
        with self.lock:
            try:
                # Verifica se precisa criar novo arquivo
                self._check_date_rollover()
                
                # Formata timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Define status visual
                if not rejected:
                    status_emoji = "✅ EXECUTADA"
                    status_line = "="
                elif signal == "NONE":
                    status_emoji = "🔵 MONITORANDO"
                    status_line = "."
                elif "já aberta" in reason.lower():
                    status_emoji = "📌 POSIÇÃO JÁ ABERTA"
                    status_line = "-"
                elif "limite" in reason.lower():
                    status_emoji = "🚫 LIMITE ATINGIDO"
                    status_line = "-"
                elif "aguardando" in reason.lower() or "pullback" in reason.lower():
                    status_emoji = "⏳ AGUARDANDO SETUP"
                    status_line = "-"
                else:
                    status_emoji = "❌ REJEITADA"
                    status_line = "-"
                
                # Monta a entrada do log
                log_entry = []
                log_entry.append(status_line * 80)
                log_entry.append(f"🕐 {timestamp} | {symbol} | {status_emoji}")
                log_entry.append(status_line * 80)
                
                # Sinal e Estratégia
                signal_display = signal if signal else "NONE"
                strategy_display = strategy if strategy else "N/A"
                
                # Land Trading: ML Score com sufixo (B) se for baseline
                ml_display = f"{ml_score:.0f}"
                if is_baseline:
                    ml_display += " (Confiança Estatística)"
                
                log_entry.append(f"📊 Sinal: {signal_display} | Estratégia: {strategy_display} | Score: {score:.0f} | ML: {ml_display}")
                
                # Indicadores
                rsi = indicators.get("rsi", 0)
                adx = indicators.get("adx", 0)
                spread = indicators.get("spread_pips", 0)
                volume_ratio = indicators.get("volume_ratio", 0)
                ema_trend = indicators.get("ema_trend", "N/A")
                
                log_entry.append(f"📈 Indicadores:")
                log_entry.append(f"   • RSI: {rsi:.1f}")
                log_entry.append(f"   • ADX: {adx:.1f}")
                log_entry.append(f"   • Spread: {spread:.2f} pips")
                log_entry.append(f"   • Volume: {volume_ratio:.2f}x")
                log_entry.append(f"   • Tendência EMA: {ema_trend}")
                
                # Motivo
                log_entry.append(f"💬 Motivo: {reason}")
                if user:
                    log_entry.append(f"👤 Usuário: {user}")
                if isinstance(context, dict) and context:
                    try:
                        ctx_items = []
                        for k, v in context.items():
                            ctx_items.append(f"{k}={v}")
                        log_entry.append(f"🧭 Contexto: " + "; ".join(ctx_items))
                    except Exception:
                        pass
                log_entry.append("")  # Linha em branco para separação
                
                # Escreve no arquivo
                with open(self.current_file, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(log_entry))
                    f.flush() # ✅ Land Trading: Força escrita imediata no disco
                
            except Exception as e:
                # Não queremos que erro no log quebre o bot
                print(f"⚠️ Erro ao escrever log de análise: {e}")
    
    def log_summary(self, total_analyzed: int, executed: int, rejected: int):
        """
        Adiciona um resumo ao final do arquivo
        """
        with self.lock:
            try:
                self._check_date_rollover()
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                summary = [
                    "\n" + "="*80,
                    f"📊 RESUMO PARCIAL - {timestamp}",
                    "="*80,
                    f"Total Analisado: {total_analyzed}",
                    f"Ordens Executadas: {executed}",
                    f"Sinais Rejeitados: {rejected}",
                    f"Taxa de Execução: {(executed/total_analyzed*100) if total_analyzed > 0 else 0:.1f}%",
                    "="*80 + "\n",
                ]
                
                with open(self.current_file, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(summary))
                    
            except Exception as e:
                print(f"⚠️ Erro ao escrever resumo: {e}")

    def _compress_old_logs(self):
        try:
            import gzip
            import shutil
            cutoff_ts = (datetime.now().timestamp() - self.compress_after_hours * 3600)
            for fp in sorted(self.log_dir.glob("analysis_log_*.txt")):
                if fp.suffix == ".gz":
                    continue
                try:
                    if fp.stat().st_mtime < cutoff_ts:
                        gz_fp = fp.with_suffix(fp.suffix + ".gz")
                        with open(fp, "rb") as f_in, gzip.open(gz_fp, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        try:
                            fp.unlink()
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            pass

    def _enforce_retention_policy(self):
        try:
            import time
            now_ts = time.time()
            cutoff_days_ts = now_ts - (self.retention_days * 86400)
            files = sorted(list(self.log_dir.glob("analysis_log_*.txt")) + list(self.log_dir.glob("analysis_log_*.txt.gz")), key=lambda p: p.stat().st_mtime)
            for fp in files:
                try:
                    if fp.stat().st_mtime < cutoff_days_ts:
                        try:
                            fp.unlink()
                        except Exception:
                            pass
                except Exception:
                    continue
            files = sorted(list(self.log_dir.glob("analysis_log_*.txt")) + list(self.log_dir.glob("analysis_log_*.txt.gz")), key=lambda p: p.stat().st_mtime)
            if len(files) > self.max_files:
                excess = len(files) - self.max_files
                for fp in files[:excess]:
                    try:
                        fp.unlink()
                    except Exception:
                        continue
        except Exception:
            pass

# Instância global para usar em todo o bot
daily_logger = DailyAnalysisLogger()


# ===========================
# INTEGRAÇÃO COM O BOT
# ===========================

def log_signal_analysis_to_file(symbol: str, signal: str, strategy: str, score: float,
                                rejected: bool, reason: str, indicators: dict, user: Optional[str] = None, context: Optional[dict] = None):
    """
    Função wrapper que pode ser chamada no bot_forex.py
    mantendo compatibilidade com o sistema atual
    """
    daily_logger.log_analysis(
        symbol=symbol,
        signal=signal,
        strategy=strategy,
        score=score,
        rejected=rejected,
        reason=reason,
        indicators=indicators,
        user=user,
        context=context
    )


# ===========================
# EXEMPLO DE USO
# ===========================

if __name__ == "__main__":
    # Testes
    logger = DailyAnalysisLogger()
    
    # Exemplo 1: Ordem executada
    logger.log_analysis(
        symbol="EURUSD",
        signal="BUY",
        strategy="TREND",
        score=95,
        rejected=False,
        reason="✅ ORDEM EXECUTADA!",
        indicators={
            "rsi": 35,
            "adx": 42,
            "spread_pips": 1.5,
            "volume_ratio": 1.3,
            "ema_trend": "UP"
        }
    )
    
    # Exemplo 2: Aguardando pullback
    logger.log_analysis(
        symbol="GBPUSD",
        signal="NONE",
        strategy="TREND",
        score=65,
        rejected=True,
        reason="⏳ Aguardando pullback (RSI 58 > 40)",
        indicators={
            "rsi": 58,
            "adx": 35,
            "spread_pips": 2.1,
            "volume_ratio": 1.1,
            "ema_trend": "UP"
        }
    )
    
    # Exemplo 3: Correlação alta
    logger.log_analysis(
        symbol="USDCHF",
        signal="SELL",
        strategy="REVERSION",
        score=72,
        rejected=True,
        reason="🔗 Correlação alta",
        indicators={
            "rsi": 75,
            "adx": 18,
            "spread_pips": 2.8,
            "volume_ratio": 0.9,
            "ema_trend": "DOWN"
        }
    )
    
    # Resumo
    logger.log_summary(total_analyzed=50, executed=3, rejected=47)
    
    print("✅ Arquivo de log criado com sucesso!")
    print(f"📁 Localização: {logger.current_file}")

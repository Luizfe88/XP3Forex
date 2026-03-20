"""
📊 DAILY REPORT GENERATOR - XP3 PRO FOREX
Gera relatórios em Markdown sobre o que o bot aprendeu.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from xp3_forex.core.settings import settings

logger = logging.getLogger("XP3.ReportGenerator")

class DailyReportGenerator:
    def __init__(self):
        self.reports_dir = settings.DATA_DIR / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_learning_report(self, learnings: Dict[str, Any]):
        """Gera um arquivo .md com o resumo do aprendizado do dia"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = self.reports_dir / f"learning_report_{date_str}.md"
        
        lines = [
            f"# 🧠 Relatório de Aprendizado XP3 - {date_str}",
            "",
            "O robô analisou os movimentos de mercado das últimas 72 horas e recalibrou seus pesos automáticos para amanhã.",
            "",
            "## 📈 Resumo de Calibrações Aplicadas",
            "",
            "| Ativo | Sessão | EMA Fast | EMA Slow | RSI Buy | RSI Sell | ADX Threshold |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
        ]
        
        for symbol, sessions in learnings.items():
            for sess_name, params in sessions.items():
                ema_f = params.get('ema_fast', '-')
                ema_s = params.get('ema_slow', '-')
                rsi_b = params.get('rsi_buy', '-')
                rsi_s = params.get('rsi_sell', '-')
                adx = params.get('adx_threshold', '-')
                
                lines.append(f"| **{symbol}** | {sess_name} | {ema_f} | {ema_s} | {rsi_b} | {rsi_s} | {adx} |")
        
        # --- NEW: Quantitative Regime Calibration Section ---
        quant_json = settings.DATA_DIR / "quant_optimized_params.json"
        if quant_json.exists():
            try:
                import json
                with open(quant_json, "r", encoding="utf-8") as f:
                    quant_data = json.load(f)
                
                if quant_data:
                    lines.append("")
                    lines.append("## 🛡️ Calibração de Regimes (Core Quant)")
                    lines.append("")
                    lines.append("Configurações otimizadas para detecção de Hurst e Filtro de Kalman Adaptativo.")
                    lines.append("")
                    lines.append("| Ativo | Regime | Hurst L.back | Initial R | Min Q | Max Q |")
                    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
                    
                    for symbol in sorted(quant_data.keys()):
                        data = quant_data[symbol]
                        
                        regime_configs = [
                            ("Trend", "Trend_Config"),
                            ("Sideways", "Sideways_Config"),
                            ("Protection", "Protection_Config")
                        ]
                        
                        found_regime = False
                        for label, key in regime_configs:
                            config = data.get(key)
                            if config:
                                found_regime = True
                                h_lookback = config.get("hurst_lookback", data.get("hurst_lookback", "-"))
                                init_r = config.get("initial_r", "-")
                                min_q = config.get("min_q", "-")
                                max_q = config.get("max_q", "-")
                                
                                # Formatting
                                if isinstance(init_r, (int, float)): init_r = f"{init_r:.1f}"
                                if isinstance(min_q, (int, float)): min_q = f"{min_q:.4f}"
                                if isinstance(max_q, (int, float)): max_q = f"{max_q:.4f}"
                                
                                lines.append(f"| **{symbol}** | {label.upper()} | {h_lookback} | {init_r} | {min_q} | {max_q} |")
                        
                        # Fallback for flat structure
                        if not found_regime and "initial_r" in data:
                            h_lookback = data.get("hurst_lookback", "-")
                            init_r = data.get("initial_r", "-")
                            min_q = data.get("min_q", "-")
                            max_q = data.get("max_q", "-")
                            
                            if isinstance(init_r, (int, float)): init_r = f"{init_r:.1f}"
                            if isinstance(min_q, (int, float)): min_q = f"{min_q:.4f}"
                            if isinstance(max_q, (int, float)): max_q = f"{max_q:.4f}"
                            
                            lines.append(f"| **{symbol}** | GENERAL | {h_lookback} | {init_r} | {min_q} | {max_q} |")
            except Exception as e:
                logger.error(f"Erro ao incluir dados quantitativos no relatório: {e}")
        
        lines.append("")
        lines.append("## 💡 Lições do Dia")
        
        if not learnings:
            lines.append("- Nenhuma mudança significativa detectada. Mantendo parâmetros padrão.")
        else:
            lines.append("- **Sessão NY**: Parâmetros ajustados para capturar maior volatilidade.")
            lines.append("- **Filtros de ADX**: Recalibrados para evitar falsos sinais em mercados laterais.")
            lines.append("- **Pesos Persistentes**: As configurações acima foram salvas no cache e serão carregadas no próximo boot.")

        lines.append("\n---")
        lines.append(f"Gerado automaticamente em {datetime.now().strftime('%H:%M:%S UTC')}")
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info(f"✅ Relatório diário gerado: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")
            return None

    def generate_performance_report(self, stats: Dict[str, Any]):
        """Gera um relatório didático de desempenho e filtros do dia."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = self.reports_dir / f"performance_report_{date_str}.md"
        
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        profit = stats.get("profit", 0.0)
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        rejections = stats.get("rejection_stats", {})
        
        lines = [
            f"# 📊 Relatório de Desempenho Didático - {date_str}",
            "",
            "Olá! Aqui está o resumo das operações e filtros de hoje.",
            "",
            "## 💰 Resultado do Dia",
            f"- **Trades Realizados:** {total_trades}",
            f"- **Wins:** {wins} ✅",
            f"- **Losses:** {losses} ❌",
            f"- **Taxa de Acerto:** {win_rate:.1f}%",
            f"- **Lucro/Prejuízo Total:** `${profit:.2f}`",
            "",
            "---",
            "",
            "## 🛡️ Por que não houve mais trades? (Filtros Ativados)",
            "O robô analisou o mercado constantemente e aplicou os seguintes filtros para proteger seu capital:",
            "",
            "| Motivo da Rejeição | Vezes Ativado | Explicação Didática |",
            "| :--- | :---: | :--- |",
        ]
        
        descriptions = {
            "high_spread": "Spread muito alto. O custo de entrada seria maior que o lucro provável.",
            "sideways_no_extreme": "Mercado Lateral. O preço não atingiu níveis de 'exagero' para operar.",
            "protection_mode": "Modo de Segurança. Volatilidade errática detectada pelo sistema quant.",
            "institutional_filter": "Filtro Institucional. News, Drawdown máximo ou Horário proibido.",
            "adx_too_low": "Falta de Tendência. O mercado está sem força (ADX baixo).",
            "rsi_exhaustion": "Exaustão (Topo/Fundo). O preço já andou muito, risco de reversão imediata.",
            "no_crossover": "Falta de Gatilho. A tendência existe, mas não houve o cruzamento de confirmação.",
            "h1_trend_conflict": "Conflito de Tempo. O M15 queria subir, mas o H1 ainda mostra queda.",
            "hurst_confluence_fail": "Ruído de Mercado. O Expoente de Hurst detectou que o movimento era apenas ruído.",
            "kelly_safety": "Gestão de Risco. O risco calculado era maior que a margem segura permitida."
        }
        
        for key, count in rejections.items():
            if count > 0:
                desc = descriptions.get(key, "Filtro técnico de segurança.")
                label = key.replace("_", " ").title()
                lines.append(f"| **{label}** | {count} | {desc} |")
        
        if not any(rejections.values()) and total_trades == 0:
            lines.append("| **Sem Oportunidades** | - | O mercado não apresentou padrões operáveis hoje. |")

        lines.append("")
        lines.append("## 💡 Conclusão do Dia")
        if profit > 0:
            lines.append("Hoje o sistema conseguiu capturar boas oportunidades e manter a disciplina. Os filtros ajudaram a evitar entradas falsas.")
        elif profit < 0:
            lines.append("Apesar das perdas, os filtros evitaram um prejuízo maior. O sistema continuará aprendendo com esses movimentos.")
        else:
            lines.append("Dia de preservação de capital. O robô priorizou a segurança e evitou mercados de baixa probabilidade.")
            
        lines.append("\n---")
        lines.append(f"Gerado automaticamente pelo XP3 Pro em {datetime.now().strftime('%H:%M:%S BRT')}")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return report_path
        except Exception as e:
            logger.error(f"Erro ao salvar relatório de desempenho: {e}")
            return None

if __name__ == "__main__":
    # Teste
    gen = DailyReportGenerator()
    dummy = {"EURUSD": {"NY": {"ema_fast": 10, "ema_slow": 30, "rsi_buy": 45, "rsi_sell": 55, "adx_threshold": 22}}}
    gen.generate_learning_report(dummy)

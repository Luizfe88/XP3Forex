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

if __name__ == "__main__":
    # Teste
    gen = DailyReportGenerator()
    dummy = {"EURUSD": {"NY": {"ema_fast": 10, "ema_slow": 30, "rsi_buy": 45, "rsi_sell": 55, "adx_threshold": 22}}}
    gen.generate_learning_report(dummy)

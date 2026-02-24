#!/usr/bin/env python3
"""
🕐 AGENDADOR DE ANÁLISE DIÁRIA XP3 PRO v5.0
=============================================
Script auxiliar para agendar e executar análise diária automaticamente
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ===========================
# DIRETÓRIO DO SCRIPT E LOG
# ===========================
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "daily_scheduler.log"

# ===========================
# CONFIGURAÇÃO DE LOGGING
# ===========================
# Configuração de logging com UTF-8 para suportar emojis (log em arquivo do projeto)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Força UTF-8 no stdout/stderr para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ===========================
# CONFIGURAÇÕES
# ===========================

# Arquivos e diretórios
ANALYSIS_SCRIPT = SCRIPT_DIR / "daily_market_analysis.py"
LOCK_FILE = SCRIPT_DIR / "daily_analysis.lock"
SCHEDULE_FILE = SCRIPT_DIR / "daily_schedule.json"

# Horários de execução recomendados (UTC)
DEFAULT_SCHEDULE = {
    "london": "06:00",    # 1h antes da abertura de Londres
    "new_york": "11:00",  # 1h antes da abertura de Nova York
    "tokyo": "22:00",     # 1h antes da abertura de Tóquio
    "sydney": "20:00"     # 1h antes da abertura de Sydney
}

# Dias da semana para executar (1=Segunda, 7=Domingo)
DEFAULT_WEEKDAYS = [1, 2, 3, 4, 5]  # Segunda a Sexta

# Timeout máximo para execução (segundos)
MAX_EXECUTION_TIME = 300  # 5 minutos

# ===========================
# FUNÇÕES UTILITÁRIAS
# ===========================

def is_locked() -> bool:
    """Verifica se há uma execução em andamento"""
    if LOCK_FILE.exists():
        try:
            # Verifica se o lock é antigo (mais de 30 minutos)
            lock_time = datetime.fromisoformat(LOCK_FILE.read_text().strip())
            if datetime.now() - lock_time > timedelta(minutes=30):
                logger.warning("⚠️ Lock antigo encontrado, removendo...")
                LOCK_FILE.unlink()
                return False
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao verificar lock: {e}")
            return False
    return False

def create_lock():
    """Cria arquivo de lock"""
    try:
        LOCK_FILE.write_text(datetime.now().isoformat())
    except Exception as e:
        logger.error(f"❌ Erro ao criar lock: {e}")

def remove_lock():
    """Remove arquivo de lock"""
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception as e:
        logger.error(f"❌ Erro ao remover lock: {e}")

def load_schedule() -> Dict:
    """Carrega configuração de agendamento"""
    try:
        if SCHEDULE_FILE.exists():
            with open(SCHEDULE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Cria configuração padrão
            schedule = {
                "enabled": True,
                "weekdays": DEFAULT_WEEKDAYS,
                "times": DEFAULT_SCHEDULE,
                "timezone": "UTC",
                "last_execution": None,
                "next_execution": None
            }
            save_schedule(schedule)
            return schedule
    except Exception as e:
        logger.error(f"❌ Erro ao carregar agendamento: {e}")
        return {"enabled": True, "weekdays": DEFAULT_WEEKDAYS, "times": DEFAULT_SCHEDULE}

def save_schedule(schedule: Dict):
    """Salva configuração de agendamento"""
    try:
        with open(SCHEDULE_FILE, 'w', encoding='utf-8') as f:
            json.dump(schedule, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Erro ao salvar agendamento: {e}")

def should_execute_now(schedule: Dict) -> bool:
    """Verifica se deve executar agora baseado no agendamento"""
    try:
        if not schedule.get("enabled", True):
            return False
        
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_weekday = now.isoweekday()
        
        # Verifica se é dia útil
        weekdays = schedule.get("weekdays", DEFAULT_WEEKDAYS)
        if current_weekday not in weekdays:
            return False
        
        # Verifica horário
        times = schedule.get("times", DEFAULT_SCHEDULE)
        for session, time_str in times.items():
            if current_time == time_str:
                logger.info(f"🕐 Horário de execução: {session} ({time_str})")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Erro ao verificar horário de execução: {e}")
        return False

def execute_analysis() -> bool:
    """Executa o script de análise"""
    try:
        logger.info("🚀 Iniciando análise diária...")
        
        # Verifica se o script existe
        if not ANALYSIS_SCRIPT.exists():
            logger.error(f"❌ Script não encontrado: {ANALYSIS_SCRIPT}")
            return False
        
        # Executa o script com tratamento robusto de encoding
        try:
            result = subprocess.run(
                [sys.executable, str(ANALYSIS_SCRIPT)],
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=False,  # Não forçar texto para evitar problemas de decodificação
                timeout=MAX_EXECUTION_TIME
            )
            
            # Decodifica a saída com tratamento de erros
            try:
                stdout = result.stdout.decode('utf-8', errors='replace')
            except:
                stdout = result.stdout.decode('latin-1', errors='replace')
                
            try:
                stderr = result.stderr.decode('utf-8', errors='replace')
            except:
                stderr = result.stderr.decode('latin-1', errors='replace')
            
            # Log da saída
            if stdout:
                logger.info(f"📊 Saída da análise:\n{stdout}")
            
            if stderr:
                logger.error(f"⚠️ Erros da análise:\n{stderr}")
            
        except UnicodeDecodeError as e:
            logger.warning(f"⚠️ Problema de decodificação (mas a análise pode ter funcionado): {e}")
            # Mesmo com erro de decodificação, verifica se o script executou com sucesso
            stdout = "Análise executada (saída com problemas de encoding)"
            stderr = ""
        
        # Verifica resultado
        success = result.returncode == 0
        if success:
            logger.info("✅ Análise diária concluída com sucesso!")
        else:
            logger.error(f"❌ Análise falhou com código: {result.returncode}")
        
        return success
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Análise excedeu tempo limite de {MAX_EXECUTION_TIME} segundos")
        return False
    except Exception as e:
        logger.error(f"❌ Erro ao executar análise: {e}")
        return False

def update_schedule_next_execution(schedule: Dict):
    """Atualiza próxima execução"""
    try:
        now = datetime.now()
        schedule["last_execution"] = now.isoformat()
        
        # Calcula próxima execução
        next_day = now + timedelta(days=1)
        next_executions = []
        
        for session, time_str in schedule["times"].items():
            hour, minute = map(int, time_str.split(':'))
            next_exec = next_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Verifica se é dia útil
            while next_exec.isoweekday() not in schedule.get("weekdays", DEFAULT_WEEKDAYS):
                next_exec += timedelta(days=1)
            
            next_executions.append(next_exec)
        
        # Pega a próxima execução mais próxima
        if next_executions:
            next_execution = min(next_executions)
            schedule["next_execution"] = next_execution.isoformat()
            logger.info(f"📅 Próxima execução: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}")
        
        save_schedule(schedule)
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar próxima execução: {e}")

# ===========================
# FUNÇÕES DE CONFIGURAÇÃO
# ===========================

def interactive_setup():
    """Configuração interativa do agendamento"""
    print("🕐 CONFIGURAÇÃO DO AGENDADOR DE ANÁLISE DIÁRIA")
    print("="*60)
    
    schedule = load_schedule()
    
    print(f"\n📅 Configuração atual:")
    print(f"  Ativado: {'Sim' if schedule.get('enabled') else 'Não'}")
    print(f"  Dias úteis: {schedule.get('weekdays', [])}")
    print(f"  Horários: {schedule.get('times', {})}")
    
    print("\n📝 Opções de configuração:")
    print("1. Ativar/desativar agendamento")
    print("2. Configurar dias da semana")
    print("3. Configurar horários")
    print("4. Ver status atual")
    print("5. Executar análise agora")
    print("6. Sair")
    
    choice = input("\nEscolha uma opção (1-6): ").strip()
    
    if choice == "1":
        schedule["enabled"] = not schedule.get("enabled", True)
        save_schedule(schedule)
        print(f"✅ Agendamento {'ativado' if schedule['enabled'] else 'desativado'}")
        
    elif choice == "2":
        print("\nDias da semana (1=Segunda, 7=Domingo)")
        print("Exemplo: 1,2,3,4,5 (segunda a sexta)")
        days_input = input("Dias para executar: ").strip()
        try:
            days = [int(d.strip()) for d in days_input.split(",") if d.strip()]
            if all(1 <= d <= 7 for d in days):
                schedule["weekdays"] = days
                save_schedule(schedule)
                print("✅ Dias configurados com sucesso!")
            else:
                print("❌ Dias inválidos. Use números de 1 a 7.")
        except ValueError:
            print("❌ Formato inválido. Use números separados por vírgula.")
            
    elif choice == "3":
        print("\nHorários de execução (formato HH:MM, 24h)")
        print("Exemplo: 06:00, 11:00, 22:00")
        
        for session in DEFAULT_SCHEDULE.keys():
            current_time = schedule["times"].get(session, DEFAULT_SCHEDULE[session])
            new_time = input(f"Horário {session} ({current_time}): ").strip()
            if new_time and len(new_time) == 5 and ":" in new_time:
                try:
                    hour, minute = map(int, new_time.split(":"))
                    if 0 <= hour < 24 and 0 <= minute < 60:
                        schedule["times"][session] = new_time
                    else:
                        print(f"❌ Horário inválido para {session}")
                except ValueError:
                    print(f"❌ Formato inválido para {session}")
        
        save_schedule(schedule)
        print("✅ Horários configurados com sucesso!")
        
    elif choice == "4":
        show_status()
        
    elif choice == "5":
        print("\n🚀 Executando análise manualmente...")
        if execute_analysis():
            print("✅ Análise concluída!")
        else:
            print("❌ Análise falhou!")
        
    elif choice == "6":
        print("👋 Até logo!")
        return False
    
    return True

def show_status():
    """Mostra status do agendamento"""
    schedule = load_schedule()
    
    print("\n📊 STATUS DO AGENDADOR")
    print("="*40)
    print(f"Status: {'🟢 Ativo' if schedule.get('enabled') else '🔴 Inativo'}")
    print(f"Dias úteis: {schedule.get('weekdays', [])}")
    print(f"Horários: {schedule.get('times', {})}")
    
    last_exec = schedule.get("last_execution")
    if last_exec:
        last_time = datetime.fromisoformat(last_exec)
        print(f"Última execução: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Última execução: Nunca")
    
    next_exec = schedule.get("next_execution")
    if next_exec:
        next_time = datetime.fromisoformat(next_exec)
        print(f"Próxima execução: {next_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Próxima execução: Não agendada")
    
    # Verifica arquivos de análise
    if Path("daily_selected_pairs.json").exists():
        file_time = datetime.fromtimestamp(Path("daily_selected_pairs.json").stat().st_mtime)
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        print(f"Análise atual: {file_time.strftime('%Y-%m-%d %H:%M')} ({age_hours:.1f}h atrás)")
    else:
        print("Análise atual: Arquivo não encontrado")

# ===========================
# MODO MONITORAMENTO
# ===========================

def monitor_mode():
    print("🕐 MODO MONITORAMENTO")
    print("="*40)
    print("Pressione Ctrl+C para parar")
    print("Monitorando horários de execução...")
    try:
        while True:
            schedule = load_schedule()
            now = datetime.now()
            next_time = None
            try:
                times = schedule.get("times", DEFAULT_SCHEDULE)
                weekdays = schedule.get("weekdays", DEFAULT_WEEKDAYS)
                candidate_times = []
                for i in range(0, 7):
                    day = now + timedelta(days=i)
                    if day.isoweekday() not in weekdays:
                        continue
                    for t in times.values():
                        h, m = map(int, t.split(":"))
                        candidate_times.append(day.replace(hour=h, minute=m, second=0, microsecond=0))
                future_times = [t for t in candidate_times if t >= now]
                if future_times:
                    next_time = min(future_times)
            except Exception:
                next_time = None
            locked = is_locked()
            status_line = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] enabled={schedule.get('enabled', True)} lock={'ON' if locked else 'OFF'}"
            if next_time:
                delta = next_time - now
                mins = int(delta.total_seconds() // 60)
                secs = int(delta.total_seconds() % 60)
                status_line += f" next={next_time.strftime('%H:%M')} T-{mins:02d}:{secs:02d}"
            print(status_line, flush=True)
            logger.info(status_line)
            if should_execute_now(schedule):
                if not locked:
                    create_lock()
                    try:
                        if execute_analysis():
                            update_schedule_next_execution(schedule)
                    finally:
                        remove_lock()
                else:
                    logger.warning("⚠️ Execução em andamento detectada, aguardando...")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n👋 Monitoramento encerrado")
    except Exception as e:
        logger.error(f"❌ Erro no monitoramento: {e}")

# ===========================
# FUNÇÃO PRINCIPAL
# ===========================

def main():
    """Função principal"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "monitor":
            monitor_mode()
        elif command == "status":
            show_status()
        elif command == "setup":
            while interactive_setup():
                pass
        elif command == "run":
            if execute_analysis():
                print("✅ Análise concluída!")
            else:
                print("❌ Análise falhou!")
                sys.exit(1)
        else:
            print("Comando desconhecido. Use: monitor, status, setup, run")
            print("Ou execute sem argumentos para configuração interativa")
    else:
        # Modo interativo
        while True:
            print("\n🕐 AGENDADOR DE ANÁLISE DIÁRIA XP3 PRO v5.0")
            print("="*60)
            print("1. Configurar agendamento")
            print("2. Ver status")
            print("3. Executar análise agora")
            print("4. Iniciar monitoramento")
            print("5. Sair")
            
            choice = input("\nEscolha uma opção (1-5): ").strip()
            
            if choice == "1":
                interactive_setup()
            elif choice == "2":
                show_status()
            elif choice == "3":
                if execute_analysis():
                    print("✅ Análise concluída!")
                else:
                    print("❌ Análise falhou!")
            elif choice == "4":
                monitor_mode()
            elif choice == "5":
                print("👋 Até logo!")
                break
            else:
                print("❌ Opção inválida")

if __name__ == "__main__":
    main()

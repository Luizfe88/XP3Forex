# fix_pause.py - Corrige estado de pausa travado
"""
O bot está pausado mas deveria estar ativo.
Este script força o despause.

Execute: python fix_pause.py
"""

import json
from pathlib import Path
from datetime import datetime

print("\n" + "="*80)
print("🔧 CORRIGINDO ESTADO DE PAUSA TRAVADO")
print("="*80 + "\n")

# 1. Edita bot_state.json
state_file = Path("bot_state.json")

if state_file.exists():
    print("📄 Encontrado: bot_state.json")
    
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        print(f"   Status atual: {'PAUSADO' if state.get('paused', False) else 'ATIVO'}")
        
        if state.get('paused', False):
            print(f"   Motivo: {state.get('pause_reason', 'Desconhecido')}")
        
        # FORÇA DESPAUSE
        state['paused'] = False
        state['pause_reason'] = ''
        
        # Salva backup
        backup_file = state_file.with_suffix('.json.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        print(f"   ✅ Backup salvo: {backup_file}")
        
        # Salva estado corrigido
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        print("   ✅ Estado alterado para: ATIVO")
        print()
    
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        print()
else:
    print("ℹ️  bot_state.json não encontrado (bot usa estado em memória)")
    print()

# 2. Cria arquivo de comando para o bot detectar
command_file = Path("RESUME_TRADING.txt")

try:
    with open(command_file, 'w', encoding='utf-8') as f:
        f.write(f"RESUME_TRADING\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Reason: Market is open, manual resume\n")
    
    print(f"✅ Comando criado: {command_file}")
    print("   O bot vai detectar no próximo ciclo (5-10 segundos)")
    print()
except Exception as e:
    print(f"❌ Erro ao criar comando: {e}")
    print()

# 3. Instruções adicionais
print("="*80)
print("📋 PRÓXIMOS PASSOS:")
print("="*80)
print()
print("1. ⏳ Aguarde 10-15 segundos")
print("2. 👀 Observe o painel do bot")
print("3. 🟢 O status deve mudar para: Bot: 🟢 ATIVO")
print()
print("Se NÃO despausar:")
print()
print("OPÇÃO A - Reiniciar o bot:")
print("   1. Pressione Ctrl+C para parar")
print("   2. Execute: python bot_forex.py")
print()
print("OPÇÃO B - Desabilitar filtro temporariamente:")
print("   1. Abra config_forex.py")
print("   2. Mude: ENABLE_SCHEDULE_FILTER = False")
print("   3. Salve e reinicie o bot")
print()
print("="*80 + "\n")
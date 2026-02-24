# ========================================
# Salve como: check_calls.py
# Execute: python check_calls.py
# ========================================

import re
import os

print("=" * 80)
print("🔍 PROCURANDO TODAS AS CHAMADAS DE calculate_position_size_atr_forex")
print("=" * 80)

pattern = re.compile(
    r'calculate_position_size_atr_forex\s*\([^)]+\)',
    re.DOTALL
)

found = []

for filename in os.listdir('.'):
    if not filename.endswith('.py'):
        continue
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for match in pattern.finditer(content):
            # Encontra linha
            line_num = content[:match.start()].count('\n') + 1
            
            # Pega contexto (3 linhas antes e depois)
            start_line = max(0, line_num - 3)
            end_line = min(len(lines), line_num + 3)
            context = '\n'.join(lines[start_line:end_line])
            
            found.append({
                'file': filename,
                'line': line_num,
                'call': match.group(0),
                'context': context
            })
    
    except Exception as e:
        print(f"⚠️ Erro ao ler {filename}: {e}")

if found:
    print(f"\n✅ ENCONTRADAS {len(found)} CHAMADAS:\n")
    
    for i, item in enumerate(found, 1):
        print(f"{'='*80}")
        print(f"#{i} - {item['file']}:{item['line']}")
        print(f"{'='*80}")
        print(item['call'])
        print(f"\n📋 Contexto:")
        print(item['context'])
        print("\n")
else:
    print("\n⚠️ Nenhuma chamada encontrada!")

print("=" * 80)
print("✅ BUSCA CONCLUÍDA")
print("=" * 80)

# Análise automática
print("\n🔧 ANÁLISE:\n")

for item in found:
    call = item['call']
    
    # Conta vírgulas (número de argumentos)
    # Remove strings entre aspas primeiro
    call_clean = re.sub(r'["\'].*?["\']', '', call)
    args_count = call_clean.count(',') + 1
    
    # Verifica se tem argumentos nomeados
    has_named = '=' in call
    
    print(f"📁 {item['file']}:{item['line']}")
    print(f"   Argumentos: {args_count}")
    print(f"   Nomeados: {'Sim' if has_named else 'Não'}")
    
    if not has_named and args_count == 4:
        print(f"   ⚠️ POSSÍVEL ERRO: 4 argumentos sem nomes")
        print(f"   💡 Deveria ser: symbol, price, atr_pips, None, existing_positions")
    elif not has_named and args_count == 5:
        print(f"   ✅ OK: 5 argumentos posicionais")
    elif has_named:
        print(f"   ✅ OK: Argumentos nomeados")
    else:
        print(f"   ❓ VERIFICAR: {args_count} argumentos")
    
    print()
<<<<<<< HEAD
# ========================================
# Salve como: find_error.py
# Execute: python find_error.py
# ========================================

import os
import re

print("=" * 70)
print("🔍 PROCURANDO ATR_MULTIPLIER_SL EM TODOS OS ARQUIVOS .py")
print("=" * 70)

# Padrão que vai causar erro (multiplicação direta)
dangerous_pattern = re.compile(
    r'(\w+\s*\*\s*config\.ATR_MULTIPLIER_SL|config\.ATR_MULTIPLIER_SL\s*\*\s*\w+)'
)

# Padrão seguro (com .get())
safe_pattern = re.compile(r'ATR_MULTIPLIER_SL\.get\(')

found_issues = []
found_safe = []

# Procura em todos os .py
for filename in os.listdir('.'):
    if not filename.endswith('.py'):
        continue
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Ignora comentários
            if line.strip().startswith('#'):
                continue
            
            # Verifica padrão perigoso
            if dangerous_pattern.search(line):
                found_issues.append({
                    'file': filename,
                    'line': line_num,
                    'content': line.strip(),
                    'type': 'PERIGOSO'
                })
            
            # Verifica uso seguro
            elif 'ATR_MULTIPLIER_SL' in line:
                if safe_pattern.search(line):
                    found_safe.append({
                        'file': filename,
                        'line': line_num,
                        'content': line.strip(),
                        'type': 'SEGURO'
                    })
                else:
                    # Pode ser config ou outro uso
                    found_issues.append({
                        'file': filename,
                        'line': line_num,
                        'content': line.strip(),
                        'type': 'VERIFICAR'
                    })
    
    except Exception as e:
        print(f"⚠️  Erro ao ler {filename}: {e}")

print("\n")
print("=" * 70)

if found_issues:
    print(f"❌ ENCONTRADAS {len(found_issues)} LINHAS PROBLEMÁTICAS:\n")
    
    for issue in found_issues:
        print(f"📁 {issue['file']} (linha {issue['line']}):")
        print(f"   {issue['type']}: {issue['content']}")
        
        if issue['type'] == 'PERIGOSO':
            print(f"   🔧 CORREÇÃO:")
            print(f"      Substitua por:")
            print(f"      atr_mult = config.ATR_MULTIPLIER_SL.get(symbol, 2.0)")
            print(f"      result = value * atr_mult")
        
        print()
else:
    print("✅ NENHUMA MULTIPLICAÇÃO DIRETA ENCONTRADA!")

if found_safe:
    print(f"\n✅ {len(found_safe)} usos SEGUROS encontrados:\n")
    for safe in found_safe[:3]:  # Mostra só os 3 primeiros
        print(f"   {safe['file']}:{safe['line']}")

print("=" * 70)

# Agora procura TODAS as menções (mesmo em comentários)
print("\n📋 TODAS AS REFERÊNCIAS A ATR_MULTIPLIER_SL:\n")

for filename in os.listdir('.'):
    if not filename.endswith('.py'):
        continue
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            if 'ATR_MULTIPLIER_SL' in line:
                print(f"{filename}:{line_num}: {line.rstrip()}")
    
    except:
        pass

print("\n" + "=" * 70)
print("✅ BUSCA CONCLUÍDA!")
=======
# ========================================
# Salve como: find_error.py
# Execute: python find_error.py
# ========================================

import os
import re

print("=" * 70)
print("🔍 PROCURANDO ATR_MULTIPLIER_SL EM TODOS OS ARQUIVOS .py")
print("=" * 70)

# Padrão que vai causar erro (multiplicação direta)
dangerous_pattern = re.compile(
    r'(\w+\s*\*\s*config\.ATR_MULTIPLIER_SL|config\.ATR_MULTIPLIER_SL\s*\*\s*\w+)'
)

# Padrão seguro (com .get())
safe_pattern = re.compile(r'ATR_MULTIPLIER_SL\.get\(')

found_issues = []
found_safe = []

# Procura em todos os .py
for filename in os.listdir('.'):
    if not filename.endswith('.py'):
        continue
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Ignora comentários
            if line.strip().startswith('#'):
                continue
            
            # Verifica padrão perigoso
            if dangerous_pattern.search(line):
                found_issues.append({
                    'file': filename,
                    'line': line_num,
                    'content': line.strip(),
                    'type': 'PERIGOSO'
                })
            
            # Verifica uso seguro
            elif 'ATR_MULTIPLIER_SL' in line:
                if safe_pattern.search(line):
                    found_safe.append({
                        'file': filename,
                        'line': line_num,
                        'content': line.strip(),
                        'type': 'SEGURO'
                    })
                else:
                    # Pode ser config ou outro uso
                    found_issues.append({
                        'file': filename,
                        'line': line_num,
                        'content': line.strip(),
                        'type': 'VERIFICAR'
                    })
    
    except Exception as e:
        print(f"⚠️  Erro ao ler {filename}: {e}")

print("\n")
print("=" * 70)

if found_issues:
    print(f"❌ ENCONTRADAS {len(found_issues)} LINHAS PROBLEMÁTICAS:\n")
    
    for issue in found_issues:
        print(f"📁 {issue['file']} (linha {issue['line']}):")
        print(f"   {issue['type']}: {issue['content']}")
        
        if issue['type'] == 'PERIGOSO':
            print(f"   🔧 CORREÇÃO:")
            print(f"      Substitua por:")
            print(f"      atr_mult = config.ATR_MULTIPLIER_SL.get(symbol, 2.0)")
            print(f"      result = value * atr_mult")
        
        print()
else:
    print("✅ NENHUMA MULTIPLICAÇÃO DIRETA ENCONTRADA!")

if found_safe:
    print(f"\n✅ {len(found_safe)} usos SEGUROS encontrados:\n")
    for safe in found_safe[:3]:  # Mostra só os 3 primeiros
        print(f"   {safe['file']}:{safe['line']}")

print("=" * 70)

# Agora procura TODAS as menções (mesmo em comentários)
print("\n📋 TODAS AS REFERÊNCIAS A ATR_MULTIPLIER_SL:\n")

for filename in os.listdir('.'):
    if not filename.endswith('.py'):
        continue
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            if 'ATR_MULTIPLIER_SL' in line:
                print(f"{filename}:{line_num}: {line.rstrip()}")
    
    except:
        pass

print("\n" + "=" * 70)
print("✅ BUSCA CONCLUÍDA!")
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
print("=" * 70)
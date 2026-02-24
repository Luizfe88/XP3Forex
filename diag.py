<<<<<<< HEAD
# ========================================
# DIAGNÓSTICO COMPLETO - Execute ANTES do bot
# ========================================
# Salve como: diagnostic_fix.py
# Execute: python diagnostic_fix.py

import sys
import traceback

print("=" * 60)
print("🔍 DIAGNÓSTICO XP3 - ENCONTRANDO ERRO DE MULTIPLICAÇÃO")
print("=" * 60)

# 1. Testa imports
print("\n1️⃣ Testando imports...")
try:
    import config_forex as config
    print("   ✅ config_forex importado")
except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)

try:
    import utils_forex as utils
    print("   ✅ utils_forex importado")
except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)

# 2. Verifica estrutura do config
print("\n2️⃣ Verificando ATR_MULTIPLIER_SL no config...")
print(f"   Tipo: {type(config.ATR_MULTIPLIER_SL)}")
print(f"   Valor: {config.ATR_MULTIPLIER_SL}")

if isinstance(config.ATR_MULTIPLIER_SL, dict):
    print("   ⚠️  É um dicionário (esperado)")
    print(f"   Keys: {list(config.ATR_MULTIPLIER_SL.keys())}")
else:
    print(f"   ❌ NÃO é dicionário! Tipo: {type(config.ATR_MULTIPLIER_SL)}")

# 3. Simula o erro
print("\n3️⃣ Simulando cálculo que causa o erro...")

test_symbol = "EURUSD"
test_atr_pips = 15.0

print(f"   Symbol: {test_symbol}")
print(f"   ATR: {test_atr_pips} pips")

# Tenta o jeito ERRADO (como estava antes)
print("\n   ❌ FORMA ERRADA (deve dar erro):")
try:
    wrong_result = test_atr_pips * config.ATR_MULTIPLIER_SL
    print(f"      Resultado: {wrong_result}")
    print("      ⚠️  NÃO DEU ERRO! O problema está em outro lugar!")
except TypeError as e:
    print(f"      💥 ERRO CONFIRMADO: {e}")
    print("      ✅ Este é o bug que precisa ser corrigido!")

# Testa o jeito CERTO
print("\n   ✅ FORMA CORRETA:")
try:
    multiplier = config.ATR_MULTIPLIER_SL.get(test_symbol, 
                 config.ATR_MULTIPLIER_SL.get("DEFAULT", 2.0))
    correct_result = test_atr_pips * multiplier
    print(f"      Multiplier: {multiplier}")
    print(f"      Resultado: {correct_result} pips")
    print("      ✅ FUNCIONOU!")
except Exception as e:
    print(f"      ❌ Erro: {e}")

# 4. Verifica código-fonte de utils_forex
print("\n4️⃣ Verificando código-fonte de calculate_position_size_atr_forex...")

import inspect

try:
    func_source = inspect.getsource(utils.calculate_position_size_atr_forex)
    
    # Procura pela linha problemática
    if "* config.ATR_MULTIPLIER_SL" in func_source:
        print("   ❌ ENCONTRADO! Linha com erro ainda está no código:")
        for i, line in enumerate(func_source.split('\n'), 1):
            if "* config.ATR_MULTIPLIER_SL" in line:
                print(f"      Linha {i}: {line.strip()}")
        print("\n   🔧 CORREÇÃO NECESSÁRIA:")
        print("      Substitua por:")
        print("      atr_multiplier = config.ATR_MULTIPLIER_SL.get(symbol, ")
        print("                       config.ATR_MULTIPLIER_SL.get('DEFAULT', 2.0))")
        print("      sl_distance_pips = atr_pips * atr_multiplier")
    else:
        print("   ✅ Linha problemática NÃO encontrada na função")
        print("   ℹ️  O erro pode estar em outra função")
        
except Exception as e:
    print(f"   ⚠️  Não foi possível inspecionar: {e}")

# 5. Procura TODAS as multiplicações com ATR_MULTIPLIER_SL
print("\n5️⃣ Procurando TODAS as referências a ATR_MULTIPLIER_SL...")

import re

try:
    with open('utils_forex.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procura padrões problemáticos
    pattern = r'(\w+\s*\*\s*config\.ATR_MULTIPLIER_SL|config\.ATR_MULTIPLIER_SL\s*\*\s*\w+)'
    matches = re.finditer(pattern, content)
    
    found_issues = []
    for match in matches:
        # Encontra linha
        line_num = content[:match.start()].count('\n') + 1
        line_content = content.split('\n')[line_num - 1].strip()
        found_issues.append((line_num, line_content))
    
    if found_issues:
        print(f"   ❌ ENCONTRADAS {len(found_issues)} OCORRÊNCIAS:")
        for line_num, line_content in found_issues:
            print(f"      Linha {line_num}: {line_content}")
    else:
        print("   ✅ Nenhuma multiplicação direta encontrada")
        
except FileNotFoundError:
    print("   ⚠️  Arquivo utils_forex.py não encontrado no diretório atual")
except Exception as e:
    print(f"   ⚠️  Erro: {e}")

# 6. Testa a função real
print("\n6️⃣ Testando função real calculate_position_size_atr_forex...")

try:
    # Mock básico para não precisar do MT5
    class MockAccount:
        balance = 1000000  # $1M
    
    import unittest.mock as mock
    
    with mock.patch('MetaTrader5.account_info', return_value=MockAccount()):
        with mock.patch('utils_forex.get_pip_value', return_value=10.0):
            with mock.patch('utils_forex.get_symbol_info', return_value=None):
                result = utils.calculate_position_size_atr_forex(
                    symbol="EURUSD",
                    price=1.10000,
                    atr_pips=15.0,
                    existing_positions=[]
                )
                print(f"   ✅ Função executou! Resultado: {result:.4f} lotes")
except TypeError as e:
    print(f"   ❌ ERRO ENCONTRADO: {e}")
    print(f"\n   📋 Stack trace completo:")
    traceback.print_exc()
    print("\n   🔧 Este é o erro que está travando seu bot!")
except Exception as e:
    print(f"   ⚠️  Outro erro: {e}")

print("\n" + "=" * 60)
print("✅ DIAGNÓSTICO CONCLUÍDO")
=======
# ========================================
# DIAGNÓSTICO COMPLETO - Execute ANTES do bot
# ========================================
# Salve como: diagnostic_fix.py
# Execute: python diagnostic_fix.py

import sys
import traceback

print("=" * 60)
print("🔍 DIAGNÓSTICO XP3 - ENCONTRANDO ERRO DE MULTIPLICAÇÃO")
print("=" * 60)

# 1. Testa imports
print("\n1️⃣ Testando imports...")
try:
    import config_forex as config
    print("   ✅ config_forex importado")
except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)

try:
    import utils_forex as utils
    print("   ✅ utils_forex importado")
except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)

# 2. Verifica estrutura do config
print("\n2️⃣ Verificando ATR_MULTIPLIER_SL no config...")
print(f"   Tipo: {type(config.ATR_MULTIPLIER_SL)}")
print(f"   Valor: {config.ATR_MULTIPLIER_SL}")

if isinstance(config.ATR_MULTIPLIER_SL, dict):
    print("   ⚠️  É um dicionário (esperado)")
    print(f"   Keys: {list(config.ATR_MULTIPLIER_SL.keys())}")
else:
    print(f"   ❌ NÃO é dicionário! Tipo: {type(config.ATR_MULTIPLIER_SL)}")

# 3. Simula o erro
print("\n3️⃣ Simulando cálculo que causa o erro...")

test_symbol = "EURUSD"
test_atr_pips = 15.0

print(f"   Symbol: {test_symbol}")
print(f"   ATR: {test_atr_pips} pips")

# Tenta o jeito ERRADO (como estava antes)
print("\n   ❌ FORMA ERRADA (deve dar erro):")
try:
    wrong_result = test_atr_pips * config.ATR_MULTIPLIER_SL
    print(f"      Resultado: {wrong_result}")
    print("      ⚠️  NÃO DEU ERRO! O problema está em outro lugar!")
except TypeError as e:
    print(f"      💥 ERRO CONFIRMADO: {e}")
    print("      ✅ Este é o bug que precisa ser corrigido!")

# Testa o jeito CERTO
print("\n   ✅ FORMA CORRETA:")
try:
    multiplier = config.ATR_MULTIPLIER_SL.get(test_symbol, 
                 config.ATR_MULTIPLIER_SL.get("DEFAULT", 2.0))
    correct_result = test_atr_pips * multiplier
    print(f"      Multiplier: {multiplier}")
    print(f"      Resultado: {correct_result} pips")
    print("      ✅ FUNCIONOU!")
except Exception as e:
    print(f"      ❌ Erro: {e}")

# 4. Verifica código-fonte de utils_forex
print("\n4️⃣ Verificando código-fonte de calculate_position_size_atr_forex...")

import inspect

try:
    func_source = inspect.getsource(utils.calculate_position_size_atr_forex)
    
    # Procura pela linha problemática
    if "* config.ATR_MULTIPLIER_SL" in func_source:
        print("   ❌ ENCONTRADO! Linha com erro ainda está no código:")
        for i, line in enumerate(func_source.split('\n'), 1):
            if "* config.ATR_MULTIPLIER_SL" in line:
                print(f"      Linha {i}: {line.strip()}")
        print("\n   🔧 CORREÇÃO NECESSÁRIA:")
        print("      Substitua por:")
        print("      atr_multiplier = config.ATR_MULTIPLIER_SL.get(symbol, ")
        print("                       config.ATR_MULTIPLIER_SL.get('DEFAULT', 2.0))")
        print("      sl_distance_pips = atr_pips * atr_multiplier")
    else:
        print("   ✅ Linha problemática NÃO encontrada na função")
        print("   ℹ️  O erro pode estar em outra função")
        
except Exception as e:
    print(f"   ⚠️  Não foi possível inspecionar: {e}")

# 5. Procura TODAS as multiplicações com ATR_MULTIPLIER_SL
print("\n5️⃣ Procurando TODAS as referências a ATR_MULTIPLIER_SL...")

import re

try:
    with open('utils_forex.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procura padrões problemáticos
    pattern = r'(\w+\s*\*\s*config\.ATR_MULTIPLIER_SL|config\.ATR_MULTIPLIER_SL\s*\*\s*\w+)'
    matches = re.finditer(pattern, content)
    
    found_issues = []
    for match in matches:
        # Encontra linha
        line_num = content[:match.start()].count('\n') + 1
        line_content = content.split('\n')[line_num - 1].strip()
        found_issues.append((line_num, line_content))
    
    if found_issues:
        print(f"   ❌ ENCONTRADAS {len(found_issues)} OCORRÊNCIAS:")
        for line_num, line_content in found_issues:
            print(f"      Linha {line_num}: {line_content}")
    else:
        print("   ✅ Nenhuma multiplicação direta encontrada")
        
except FileNotFoundError:
    print("   ⚠️  Arquivo utils_forex.py não encontrado no diretório atual")
except Exception as e:
    print(f"   ⚠️  Erro: {e}")

# 6. Testa a função real
print("\n6️⃣ Testando função real calculate_position_size_atr_forex...")

try:
    # Mock básico para não precisar do MT5
    class MockAccount:
        balance = 1000000  # $1M
    
    import unittest.mock as mock
    
    with mock.patch('MetaTrader5.account_info', return_value=MockAccount()):
        with mock.patch('utils_forex.get_pip_value', return_value=10.0):
            with mock.patch('utils_forex.get_symbol_info', return_value=None):
                result = utils.calculate_position_size_atr_forex(
                    symbol="EURUSD",
                    price=1.10000,
                    atr_pips=15.0,
                    existing_positions=[]
                )
                print(f"   ✅ Função executou! Resultado: {result:.4f} lotes")
except TypeError as e:
    print(f"   ❌ ERRO ENCONTRADO: {e}")
    print(f"\n   📋 Stack trace completo:")
    traceback.print_exc()
    print("\n   🔧 Este é o erro que está travando seu bot!")
except Exception as e:
    print(f"   ⚠️  Outro erro: {e}")

print("\n" + "=" * 60)
print("✅ DIAGNÓSTICO CONCLUÍDO")
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
print("=" * 60)
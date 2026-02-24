# ========================================
# Salve como: fix_cache.py
# Execute ANTES de rodar o bot
# ========================================

import os
import shutil
import sys

print("=" * 60)
print("🧹 LIMPANDO CACHE PYTHON")
print("=" * 60)

# 1. Remove __pycache__
pycache_dirs = []
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        pycache_path = os.path.join(root, '__pycache__')
        pycache_dirs.append(pycache_path)

if pycache_dirs:
    print(f"\n📁 Encontrados {len(pycache_dirs)} diretórios __pycache__")
    for path in pycache_dirs:
        try:
            shutil.rmtree(path)
            print(f"   ✅ Removido: {path}")
        except Exception as e:
            print(f"   ⚠️  Erro ao remover {path}: {e}")
else:
    print("\n✅ Nenhum __pycache__ encontrado")

# 2. Remove arquivos .pyc
pyc_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.pyc'):
            pyc_path = os.path.join(root, file)
            pyc_files.append(pyc_path)

if pyc_files:
    print(f"\n📄 Encontrados {len(pyc_files)} arquivos .pyc")
    for path in pyc_files:
        try:
            os.remove(path)
            print(f"   ✅ Removido: {path}")
        except Exception as e:
            print(f"   ⚠️  Erro ao remover {path}: {e}")
else:
    print("\n✅ Nenhum arquivo .pyc encontrado")

# 3. Limpa cache de imports do Python
if 'utils_forex' in sys.modules:
    print("\n🔄 Removendo utils_forex do cache de módulos...")
    del sys.modules['utils_forex']
    print("   ✅ Removido!")

if 'config_forex' in sys.modules:
    print("🔄 Removendo config_forex do cache de módulos...")
    del sys.modules['config_forex']
    print("   ✅ Removido!")

# 4. Testa importação limpa
print("\n🧪 Testando importação limpa...")
try:
    import config_forex as config
    import utils_forex as utils
    
    print("   ✅ config_forex importado")
    print("   ✅ utils_forex importado")
    
    # Testa a função corrigida
    print("\n🎯 Testando calculate_position_size_atr_forex...")
    
    # Verifica se função existe
    if hasattr(utils, 'calculate_position_size_atr_forex'):
        print("   ✅ Função encontrada!")
        
        # Verifica código-fonte
        import inspect
        source = inspect.getsource(utils.calculate_position_size_atr_forex)
        
        if 'ATR_MULTIPLIER_SL.get(' in source:
            print("   ✅ Código corrigido detectado (.get() presente)")
        else:
            print("   ❌ ATENÇÃO: Código ainda pode estar usando multiplicação direta!")
            
        if '* config.ATR_MULTIPLIER_SL' in source and 'ATR_MULTIPLIER_SL.get(' not in source:
            print("   ❌ ERRO: Multiplicação direta ainda presente!")
        else:
            print("   ✅ Sem multiplicação direta detectada")
    else:
        print("   ❌ Função não encontrada!")
        
except Exception as e:
    print(f"   ❌ Erro ao importar: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ LIMPEZA CONCLUÍDA!")
print("=" * 60)
print("\n🚀 Agora rode o bot: python bot_forex.py")
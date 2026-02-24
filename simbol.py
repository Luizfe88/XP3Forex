<<<<<<< HEAD
# diagnostico_simbolos.py - Descobre nomes corretos dos símbolos
"""
🔍 DIAGNÓSTICO DE SÍMBOLOS MT5
Descobre os nomes exatos dos símbolos na sua corretora
"""

import MetaTrader5 as mt5
from pathlib import Path

# Inicializa MT5
if not mt5.initialize():
    print("❌ Falha ao inicializar MT5")
    print(f"Erro: {mt5.last_error()}")
    exit()

print("✅ MT5 inicializado com sucesso\n")

# Pega TODOS os símbolos disponíveis
all_symbols = mt5.symbols_get()

if not all_symbols:
    print("❌ Nenhum símbolo encontrado!")
    mt5.shutdown()
    exit()

print(f"📊 Total de símbolos disponíveis: {len(all_symbols)}\n")

# Filtra por categorias
forex_majors = []
forex_crosses = []
metals = []
crypto = []
indices = []
exotics = []

for symbol in all_symbols:
    name = symbol.name.upper()

    # Forex Majors
    if any(pair in name for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]):
        forex_majors.append(symbol.name)

    # Forex Crosses
    elif any(pair in name for pair in ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "EURAUD", "GBPAUD"]):
        forex_crosses.append(symbol.name)

    # Metais
    elif any(metal in name for metal in ["XAU", "XAG", "GOLD", "SILVER"]):
        metals.append(symbol.name)

    # Crypto
    elif any(crypto_name in name for crypto_name in ["BTC", "ETH", "BNB", "SOL", "ADA"]):
        crypto.append(symbol.name)

    # Índices
    elif any(index in name for index in ["US30", "NAS100", "SPX", "UK100", "GER", "DAX"]):
        indices.append(symbol.name)

    # Exóticos
    elif any(exotic in name for exotic in ["MXN", "ZAR", "TRY", "BRL"]):
        exotics.append(symbol.name)

# Exibe resultados
print("="*80)
print("📋 SÍMBOLOS ENCONTRADOS POR CATEGORIA")
print("="*80)

print(f"\n🟢 FOREX MAJORS ({len(forex_majors)}):")
for sym in sorted(forex_majors):
    print(f"  • {sym}")

print(f"\n🟡 FOREX CROSSES ({len(forex_crosses)}):")
for sym in sorted(forex_crosses):
    print(f"  • {sym}")

print(f"\n🥇 METAIS ({len(metals)}):")
for sym in sorted(metals):
    print(f"  • {sym}")

print(f"\n₿ CRYPTO ({len(crypto)}):")
for sym in sorted(crypto):
    print(f"  • {sym}")

print(f"\n📊 ÍNDICES ({len(indices)}):")
for sym in sorted(indices):
    print(f"  • {sym}")

print(f"\n🌍 EXÓTICOS ({len(exotics)}):")
for sym in sorted(exotics):
    print(f"  • {sym}")

# Gera SYMBOL_MAP corrigido
print("\n" + "="*80)
print("📝 SYMBOL_MAP CORRIGIDO (copie e cole no config_forex.py)")
print("="*80)

print("\nSYMBOL_MAP = {")

if forex_majors:
    print('    "FOREX_MAJORS": [')
    for sym in sorted(forex_majors):
        print(f'        "{sym}",')
    print("    ],")

if forex_crosses:
    print('    "FOREX_CROSSES": [')
    for sym in sorted(forex_crosses):
        print(f'        "{sym}",')
    print("    ],")

if metals:
    print('    "METALS": [')
    for sym in sorted(metals):
        print(f'        "{sym}",')
    print("    ],")

if crypto:
    print('    "CRYPTO": [')
    for sym in sorted(crypto):
        print(f'        "{sym}",')
    print("    ],")

if indices:
    print('    "INDICES": [')
    for sym in sorted(indices):
        print(f'        "{sym}",')
    print("    ],")

if exotics:
    print('    "EXOTICS": [')
    for sym in sorted(exotics):
        print(f'        "{sym}",')
    print("    ],")

print("}")

# Salva em arquivo
output_file = Path("symbol_map_corrigido.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write("SYMBOL_MAP = {\n")

    if forex_majors:
        f.write('    "FOREX_MAJORS": [\n')
        for sym in sorted(forex_majors):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if forex_crosses:
        f.write('    "FOREX_CROSSES": [\n')
        for sym in sorted(forex_crosses):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if metals:
        f.write('    "METALS": [\n')
        for sym in sorted(metals):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if crypto:
        f.write('    "CRYPTO": [\n')
        for sym in sorted(crypto):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if indices:
        f.write('    "INDICES": [\n')
        for sym in sorted(indices):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if exotics:
        f.write('    "EXOTICS": [\n')
        for sym in sorted(exotics):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    f.write("}\n")

print(f"\n✅ Arquivo salvo em: {output_file}")

mt5.shutdown()
print("\n👋 Diagnóstico concluído!")
=======
# diagnostico_simbolos.py - Descobre nomes corretos dos símbolos
"""
🔍 DIAGNÓSTICO DE SÍMBOLOS MT5
Descobre os nomes exatos dos símbolos na sua corretora
"""

import MetaTrader5 as mt5
from pathlib import Path

# Inicializa MT5
if not mt5.initialize():
    print("❌ Falha ao inicializar MT5")
    print(f"Erro: {mt5.last_error()}")
    exit()

print("✅ MT5 inicializado com sucesso\n")

# Pega TODOS os símbolos disponíveis
all_symbols = mt5.symbols_get()

if not all_symbols:
    print("❌ Nenhum símbolo encontrado!")
    mt5.shutdown()
    exit()

print(f"📊 Total de símbolos disponíveis: {len(all_symbols)}\n")

# Filtra por categorias
forex_majors = []
forex_crosses = []
metals = []
crypto = []
indices = []
exotics = []

for symbol in all_symbols:
    name = symbol.name.upper()

    # Forex Majors
    if any(pair in name for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]):
        forex_majors.append(symbol.name)

    # Forex Crosses
    elif any(pair in name for pair in ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "EURAUD", "GBPAUD"]):
        forex_crosses.append(symbol.name)

    # Metais
    elif any(metal in name for metal in ["XAU", "XAG", "GOLD", "SILVER"]):
        metals.append(symbol.name)

    # Crypto
    elif any(crypto_name in name for crypto_name in ["BTC", "ETH", "BNB", "SOL", "ADA"]):
        crypto.append(symbol.name)

    # Índices
    elif any(index in name for index in ["US30", "NAS100", "SPX", "UK100", "GER", "DAX"]):
        indices.append(symbol.name)

    # Exóticos
    elif any(exotic in name for exotic in ["MXN", "ZAR", "TRY", "BRL"]):
        exotics.append(symbol.name)

# Exibe resultados
print("="*80)
print("📋 SÍMBOLOS ENCONTRADOS POR CATEGORIA")
print("="*80)

print(f"\n🟢 FOREX MAJORS ({len(forex_majors)}):")
for sym in sorted(forex_majors):
    print(f"  • {sym}")

print(f"\n🟡 FOREX CROSSES ({len(forex_crosses)}):")
for sym in sorted(forex_crosses):
    print(f"  • {sym}")

print(f"\n🥇 METAIS ({len(metals)}):")
for sym in sorted(metals):
    print(f"  • {sym}")

print(f"\n₿ CRYPTO ({len(crypto)}):")
for sym in sorted(crypto):
    print(f"  • {sym}")

print(f"\n📊 ÍNDICES ({len(indices)}):")
for sym in sorted(indices):
    print(f"  • {sym}")

print(f"\n🌍 EXÓTICOS ({len(exotics)}):")
for sym in sorted(exotics):
    print(f"  • {sym}")

# Gera SYMBOL_MAP corrigido
print("\n" + "="*80)
print("📝 SYMBOL_MAP CORRIGIDO (copie e cole no config_forex.py)")
print("="*80)

print("\nSYMBOL_MAP = {")

if forex_majors:
    print('    "FOREX_MAJORS": [')
    for sym in sorted(forex_majors):
        print(f'        "{sym}",')
    print("    ],")

if forex_crosses:
    print('    "FOREX_CROSSES": [')
    for sym in sorted(forex_crosses):
        print(f'        "{sym}",')
    print("    ],")

if metals:
    print('    "METALS": [')
    for sym in sorted(metals):
        print(f'        "{sym}",')
    print("    ],")

if crypto:
    print('    "CRYPTO": [')
    for sym in sorted(crypto):
        print(f'        "{sym}",')
    print("    ],")

if indices:
    print('    "INDICES": [')
    for sym in sorted(indices):
        print(f'        "{sym}",')
    print("    ],")

if exotics:
    print('    "EXOTICS": [')
    for sym in sorted(exotics):
        print(f'        "{sym}",')
    print("    ],")

print("}")

# Salva em arquivo
output_file = Path("symbol_map_corrigido.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write("SYMBOL_MAP = {\n")

    if forex_majors:
        f.write('    "FOREX_MAJORS": [\n')
        for sym in sorted(forex_majors):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if forex_crosses:
        f.write('    "FOREX_CROSSES": [\n')
        for sym in sorted(forex_crosses):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if metals:
        f.write('    "METALS": [\n')
        for sym in sorted(metals):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if crypto:
        f.write('    "CRYPTO": [\n')
        for sym in sorted(crypto):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if indices:
        f.write('    "INDICES": [\n')
        for sym in sorted(indices):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    if exotics:
        f.write('    "EXOTICS": [\n')
        for sym in sorted(exotics):
            f.write(f'        "{sym}",\n')
        f.write("    ],\n")

    f.write("}\n")

print(f"\n✅ Arquivo salvo em: {output_file}")

mt5.shutdown()
print("\n👋 Diagnóstico concluído!")
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd

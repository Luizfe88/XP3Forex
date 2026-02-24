# Filtro de Market Watch por Sector Map (MT5)

## Objetivo
Manter no Market Watch apenas os ativos que estão no seu “sector map” (lista permitida), executando automaticamente no startup do MT5.

## Opção A (Recomendado): EA no MT5 (automático no startup)
1. Copie o arquivo [XP3_SectorFilter.mq5](file:///c:/Users/luizf/Documents/xp3forex/mql5/XP3_SectorFilter.mq5) para:
   - `...\MetaTrader 5\Terminal\<hash>\MQL5\Experts\XP3_SectorFilter.mq5`
2. No MT5: `Arquivo → Abrir Pasta de Dados → MQL5 → Experts` (cole o arquivo ali).
3. Compile no MetaEditor.
4. Gere/atualize a lista permitida (arquivo em `MQL5/Files/xp3_sector_symbols.txt`):
   - Rode: `py mt5_sector_filter.py` (com MT5 aberto).
5. No MT5, abra qualquer gráfico e anexe o EA “XP3_SectorFilter”.
6. Ative “Algo Trading”.
7. Salve o perfil/template para carregar no startup do terminal (assim o EA roda automaticamente sempre que o MT5 iniciar).

Logs: verifique a aba “Experts” no MT5 (mensagens `XP3_SectorFilter:`).

## Opção B: Script Python (sem EA)
Se você preferir sem EA, rode o script sempre que abrir o MT5:
- `py mt5_sector_filter.py`

Ele:
- obtém a lista permitida do config (`SECTOR_MAP`/`SYMBOL_MAP`)
- aplica o filtro no Market Watch via `SymbolSelect`
- exporta `xp3_sector_symbols.txt` para `MQL5/Files` (se conseguir localizar o data_path do MT5)
- grava log em `logs/mt5_sector_filter.jsonl`

## Configuração do “setor”
No [config_forex.py](file:///c:/Users/luizf/Documents/xp3forex/config_forex.py), defina:
- `MT5_SECTOR_FILTER = "ALL"` (default)
- ou `MT5_SECTOR_FILTER = "FX"`, `"INDICES"`, `"METALS"`, `"CRYPTO"`, etc. (se você usar `SECTOR_MAP`)

## Integração com o Otimizador (v7)
O [otimizador_semanal_forex.py](file:///c:/Users/luizf/Documents/xp3forex/otimizador_semanal_forex.py) agora pode:
- definir o universo de símbolos via `MT5_SECTOR_FILTER/SECTOR_MAP` (`OPTIMIZER_USE_SECTOR_MAP`)
- aplicar o filtro no Market Watch durante o startup (`OPTIMIZER_APPLY_MT5_MARKET_WATCH_FILTER`)
- registrar logs de tempo/memória na inicialização

Para testar só a inicialização (sem rodar Optuna), use:
- `set XP3_OPTIMIZER_STARTUP_ONLY=1` e rode o otimizador normalmente.

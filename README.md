
# 🚀 XP3 PRO FOREX BOT v5.0

**Bot de Trading Institucional para MetaTrader 5**

Este projeto foi reestruturado para seguir padrões profissionais de engenharia de software (src-layout), utilizando Pydantic para configurações, Logging estruturado e CLI robusta.

---

## 📂 Nova Estrutura de Pastas

```
XP3Forex/
├── src/
│   └── xp3_forex/
│       ├── __init__.py
│       ├── cli.py                # Entrypoint principal
│       ├── config/               # Configurações (Pydantic)
│       ├── core/                 # Lógica core do bot
│       ├── mt5/                  # Integração MT5 (SymbolManager)
│       ├── strategies/           # Estratégias de trading
│       ├── utils/                # Utilitários
│       └── main.py
├── legacy/                       # Arquivos antigos (v4 e anteriores)
├── tests/                        # Testes unitários
├── .env.example                  # Modelo de variáveis de ambiente
├── pyproject.toml                # Definição do pacote e dependências
└── README.md
```

---

## 🛠️ Instalação

1. **Pré-requisitos**:
   - Python 3.10+
   - MetaTrader 5 Terminal instalado e logado.

2. **Instalar o pacote em modo editável**:
   ```bash
   pip install -e .
   ```

3. **Configuração**:
   Copie o arquivo de exemplo e edite suas configurações:
   ```bash
   # Windows
   copy .env.example .env
   
   # Linux/Mac
   cp .env.example .env
   ```
   
   Edite o arquivo `.env` com suas credenciais do MT5 e preferências de risco.

---

## 🚀 Como Executar

O projeto agora possui um comando CLI unificado: `xp3-forex`.

### 1. Iniciar o Robô
```bash
# Modo Demo (Padrão)
xp3-forex run

# Modo Live (Cuidado!)
xp3-forex run --mode live

# Sobrescrever símbolos via CLI
xp3-forex run --symbols "EURUSD,GBPUSD"
```

### 2. Monitoramento
Para visualizar logs e status em tempo real:
```bash
xp3-forex monitor
```

### 3. Ajuda
```bash
xp3-forex --help
```

---

## ✨ Principais Mudanças (v5.0)

- **Entrypoint Unificado**: Adeus `bot.bat`, `run_bot.py`, etc. Tudo agora é via `xp3-forex`.
- **Configuração Robusta**: Uso de `pydantic-settings` e `.env`.
- **SymbolManager 2.0**: Detecção automática de sufixos (ex: `EURUSD` -> `EURUSD.a`), Circuit Breaker para falhas de conexão e Cache inteligente.
- **Estrutura Limpa**: Separação clara de responsabilidades em `src/xp3_forex`.
- **Legacy**: Código antigo movido para `legacy/` para referência.

---

## ⚠️ Breaking Changes para Desenvolvedores

- A classe `XP3Bot` agora espera configurações via `settings` global, não mais arquivo JSON.
- `SymbolManager` é um Singleton importado de `xp3_forex.mt5.symbol_manager`.
- Scripts na raiz (`dashboard.py`, etc.) foram movidos para `legacy/`.

---

## 📝 Desenvolvimento

Para rodar testes (futuro):
```bash
pytest
```

# XP3 PRO FOREX (Institutional V5.0.1)

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![MetaTrader5](https://img.shields.io/badge/MetaTrader5-5.0-green.svg)
![License](https://img.shields.io/badge/license-Proprietary-red.svg)

**Bot de Trading Institucional de Alta Performance para MetaTrader 5.**
Desenvolvido com Clean Architecture, Pydantic v2 e padrões de design robustos para operação 24/7.

---

## 🚀 Instalação

### Pré-requisitos
- **Windows 10/11** ou Windows Server 2019+
- **Python 3.11** ou superior
- **MetaTrader 5** (Terminal instalado e logado)

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/Luizfe88/XP3Forex.git
   cd XP3Forex
   ```

2. **Crie e ative o ambiente virtual:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instale as dependências (Modo Editável):**
   ```bash
   pip install -e .
   ```
   > O comando `-e .` instala o pacote `xp3-forex` globalmente no seu venv, permitindo usar o comando CLI de qualquer lugar.

4. **Configure o ambiente:**
   Copie o exemplo e edite com suas credenciais:
   ```bash
   cp .env.example .env
   notepad .env
   ```

---

## 🖥️ Como Usar (CLI Unificado)

O projeto agora conta com um CLI (Command Line Interface) centralizado: `xp3-forex`.

### 1. Rodar o Bot (Modo Live)
Inicia o bot principal com conexão ao MT5.
```bash
xp3-forex run --mode live --symbols EURUSD,GBPUSD,USDJPY
```
*   `--mode`: `live` (conta real/demo) ou `paper` (simulação interna).
*   `--symbols`: Lista de pares separados por vírgula (opcional, sobrescreve `.env`).

### 2. Rodar o Scheduler (Agendador Diário)
Executa a análise diária e seleção de pares.
```bash
xp3-forex schedule
```

### 3. Dashboard de Monitoramento
Abre o painel de visualização em tempo real (Streamlit/Rich).
```bash
xp3-forex dashboard
```

### 4. Executar Testes
```bash
xp3-forex test
```

### 5. Verificar Instalação
```bash
xp3-forex check
```

---

## 📂 Estrutura do Projeto (src-layout)

```
xp3forex/
├── data/               # Dados de runtime (JSONs, DBs - ignorados no git)
├── legacy/             # Arquivos antigos (referência)
├── logs/               # Logs de execução (rotacionados)
├── reports/            # Relatórios HTML/PNG gerados
├── src/
│   └── xp3_forex/      # Pacote Principal
│       ├── core/       # Bot, Settings, Config
│       ├── mt5/        # Gerenciamento de Conexão e Símbolos
│       ├── strategies/ # Lógica de Trading
│       ├── utils/      # Helpers, Indicadores
│       └── cli.py      # Ponto de entrada do console
├── tests/              # Testes Unitários e Integração
├── .env                # Configurações (NÃO COMITAR)
├── .gitignore          # Regras de ignorar arquivos
├── pyproject.toml      # Configuração de Build e Dependências
└── README.md           # Documentação
```

## ✨ Funcionalidades V5.0.1

- **Clean Architecture:** Separação clara entre Core, Infraestrutura (MT5) e Estratégia.
- **Configuração Centralizada:** Pydantic v2 valida tipos e carrega de `.env`.
- **Resiliência:**
    - Reconnect automático com Exponential Backoff.
    - Circuit Breaker por símbolo (pausa após falhas consecutivas).
    - Cache de cotações para reduzir latência e chamadas à API.
- **Logging Estruturado:** Logs rotacionados em `logs/`, separados por nível.
- **Gestão de Símbolos:** `SymbolManager` resolve sufixos (ex: `EURUSD` -> `EURUSD.a`) automaticamente.

## ⚠️ Notas de Migração (Legacy -> V5)

Se você usava a versão antiga:
1.  **NÃO** use mais `python src/run_bot.py`. Use `xp3-forex run`.
2.  Os arquivos de configuração antigos em `legacy/config/` foram substituídos pelo `.env`.
3.  Logs e dados agora ficam organizados em pastas dedicadas, não na raiz.

---
**XP3 PRO FOREX** - *Institutional Trading Intelligence*

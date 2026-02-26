# 🚀 XP3 PRO FOREX BOT v5.0 (Institutional Edition)

**Bot de Trading Institucional para MetaTrader 5**

Este projeto foi completamente reestruturado para seguir os mais altos padrões de engenharia de software (Clean Architecture, src-layout), utilizando Pydantic v2 para configurações robustas, Logging estruturado e uma CLI unificada.

---

## 📂 Estrutura Profissional (v5.0)

A estrutura de diretórios foi limpa e organizada para facilitar a manutenção e escalabilidade.

```
XP3Forex/
├── src/
│   └── xp3_forex/
│       ├── __init__.py
│       ├── __main__.py           # Entrypoint (__main__)
│       ├── cli.py                # Interface de Linha de Comando (CLI)
│       ├── core/                 # Núcleo do Sistema
│       │   ├── bot.py            # Lógica Principal do Bot
│       │   ├── settings.py       # Configurações Centralizadas (Pydantic)
│       │   └── health_monitor.py # Monitoramento de Saúde
│       ├── mt5/                  # Integração MT5 (SymbolManager)
│       ├── strategies/           # Estratégias de Trading
│       ├── risk/                 # Gestão de Risco e Validação
│       ├── analysis/             # Análise de Mercado (News Filter)
│       └── utils/                # Utilitários Gerais
├── data/                         # Dados de Mercado e Cache (GitIgnored)
├── logs/                         # Logs de Execução (GitIgnored)
├── tests/                        # Testes Unitários e de Integração
├── legacy/                       # Código Legado (Referência v4)
├── .env.example                  # Modelo de Variáveis de Ambiente
├── pyproject.toml                # Definição do Pacote e Dependências
└── README.md
```

---

## 🛠️ Guia de Instalação

1. **Pré-requisitos**:
   - Python 3.10 ou superior
   - MetaTrader 5 Terminal instalado e logado na conta (Demo ou Real).

2. **Instalar o pacote em modo de desenvolvimento**:
   Recomendamos o uso de um ambiente virtual (`venv`).
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Instalar dependências e o pacote xp3-forex
   pip install -e .
   ```

3. **Configuração Inicial**:
   O sistema utiliza variáveis de ambiente para configuração.
   ```bash
   # Inicializar configuração (cria arquivo .env)
   xp3-forex init
   ```
   
   Edite o arquivo `.env` gerado com suas credenciais do MT5 e preferências:
   ```ini
   MT5_LOGIN=123456
   MT5_PASSWORD=sua_senha
   MT5_SERVER=MetaQuotes-Demo
   SYMBOLS=EURUSD,GBPUSD,XAUUSD
   RISK_PER_TRADE=1.0
   ```

---

## 🚀 Como Executar

O projeto possui um comando CLI unificado: `xp3-forex`.

### 1. Iniciar o Robô de Trading
```bash
# Modo Demo (Padrão) - Seguro para testes
xp3-forex run

# Modo Live (Requer confirmação) - Operações em conta REAL
xp3-forex run --mode live

# Sobrescrever símbolos temporariamente via CLI
xp3-forex run --symbols "EURUSD,GBPUSD"
```

### 2. Monitoramento e Dashboard
Para visualizar logs, status de conexão e saúde do sistema em tempo real:
```bash
xp3-forex monitor
```

### 3. Comandos Úteis
```bash
# Ver versão
xp3-forex --version

# Ajuda geral
xp3-forex --help

# Ajuda de comando específico
xp3-forex run --help
```

---

## ✨ Principais Melhorias (Refatoração Completa)

- **Src-Layout**: Código fonte isolado em `src/xp3_forex`, prevenindo importações acidentais e poluição do namespace global.
- **Configuração Centralizada**: `core/settings.py` unifica todas as constantes e configurações, com suporte a validação de tipos via Pydantic.
- **Entrypoint Robusto**: `xp3-forex` é o único ponto de entrada, gerenciado via `pyproject.toml`.
- **Limpeza da Raiz**: Arquivos de script antigos, backups e logs foram movidos para `legacy/` ou `logs/`, mantendo a raiz do projeto limpa e profissional.
- **Tipagem Estática**: Uso extensivo de Type Hints para melhor suporte de IDE e prevenção de erros.

---

## ⚠️ Notas de Migração

Se você está vindo de uma versão anterior:
1. Todos os scripts antigos (`bot_forex.py`, `run_bot.py`, etc.) foram movidos para a pasta `legacy/`. **Não os utilize para rodar o bot.**
2. Utilize apenas o comando `xp3-forex`.
3. Certifique-se de configurar corretamente o arquivo `.env`.

---
**Desenvolvido por Luiz** | XP3 PRO FOREX v5.0.0

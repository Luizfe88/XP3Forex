# 🚀 XP3 PRO FOREX

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

**XP3 PRO FOREX** é um trading bot profissional para Forex que utiliza Machine Learning, análise técnica avançada e otimização de parâmetros para maximizar retornos e minimizar riscos.

## 📊 Características Principais

- 🤖 **Machine Learning**: Algoritmos de ML para previsão de tendências e otimização de entradas
- 📈 **Análise Técnica Avançada**: Múltiplos indicadores técnicos (ADX, RSI, EMA, ATR)
- 🎯 **Estratégias Multi-Timeframe**: Operações em múltiplos timeframes simultaneamente
- ⚠️ **Gestão de Risco Inteligente**: Sistema de risk management com stops dinâmicos
- 📱 **Telegram Integration**: Notificações em tempo real via Telegram
- 🔄 **Otimização Automática**: Otimização de parâmetros com Optuna
- 📊 **Dashboard Web**: Interface web para monitoramento em tempo real
- 🔒 **Segurança**: Criptografia de dados sensíveis e gestão segura de credenciais

## 🏗️ Arquitetura

```
xp3-forex/
├── src/
│   └── xp3_forex/
│       ├── core/           # Core functionality
│       ├── strategies/     # Trading strategies
│       ├── indicators/     # Technical indicators
│       ├── risk/          # Risk management
│       ├── data/          # Data handling
│       ├── utils/         # Utilities
│       └── ml/            # Machine learning
├── tests/                 # Test suite
├── docs/                  # Documentation
├── config/                # Configuration files
├── scripts/               # Utility scripts
└── logs/                  # Log files
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- MetaTrader 5
- Conta demo/profissional de Forex

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/Luizfe88/XP3Forex.git
cd XP3Forex

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Configure o bot
cp config/config_template.json config/config.json
# Edite config.json com suas credenciais
```

## ⚙️ Configuração

### 1. Configuração do MetaTrader 5

1. Instale o MetaTrader 5
2. Configure sua conta demo/profissional
3. Ative o AutoTrading
4. Configure os símbolos que deseja operar

### 2. Configuração do Telegram (Opcional)

1. Crie um bot no Telegram com [@BotFather](https://t.me/botfather)
2. Obtenha o token do bot
3. Configure o chat ID para receber notificações

### 3. Configuração do Arquivo de Configuração

Edite `config/config.json`:

```json
{
  "mt5": {
    "login": 12345678,
    "password": "your_password",
    "server": "YourBroker-Demo",
    "path": "C:/Program Files/MetaTrader 5/terminal64.exe"
  },
  "telegram": {
    "token": "your_bot_token",
    "chat_id": "your_chat_id"
  },
  "trading": {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
    "timeframes": [15, 60, 240],
    "risk_per_trade": 0.02,
    "max_positions": 5
  }
}
```

## 🎯 Uso

### Iniciar o Bot

```bash
# Ative o ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Inicie o bot
python src/xp3_forex/bot_forex.py
```

### Dashboard Web

```bash
# Inicie o dashboard
python scripts/dashboard.py
```

Acesse: http://localhost:8080

### Monitoramento

```bash
# Monitor em tempo real
python scripts/monitor.py
```

## 📊 Estratégias

### Estratégia XP3 v4.2

- **Tipo**: Trend Following + Mean Reversion
- **Timeframes**: M15, H1, H4
- **Indicadores**: ADX, RSI, EMA, ATR
- **ML**: Random Forest para previsão de tendência
- **Risk Management**: ATR-based stops, position sizing dinâmico

### Otimização

O bot utiliza Optuna para otimização automática de parâmetros:

```bash
# Execute otimização
python scripts/optimizer.py --symbol EURUSD --days 30
```

## 🧪 Testes

```bash
# Execute todos os testes
pytest tests/

# Teste com cobertura
pytest tests/ --cov=src/xp3_forex --cov-report=html
```

## 📈 Performance

### Métricas de Performance

- **Sharpe Ratio**: > 1.5
- **Maximum Drawdown**: < 15%
- **Win Rate**: 60-70%
- **Profit Factor**: > 1.5

### Backtesting

```bash
# Execute backtest
python scripts/backtest.py --symbol EURUSD --start 2024-01-01 --end 2024-12-31
```

## 🔧 Desenvolvimento

### Setup de Desenvolvimento

```bash
# Instale dependências de desenvolvimento
pip install -r requirements-dev.txt

# Configure pre-commit hooks
pre-commit install
```

### Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ⚠️ Disclaimer

**Aviso de Risco**: Trading Forex envolve risco substancial de perda e não é adequado para todos os investidores. O desempenho passado não é indicativo de resultados futuros. Use este software por sua conta e risco.

## 🆘 Suporte

- 📧 Email: luizfe88@example.com
- 💬 Telegram: @luizfe88
- 🐛 Issues: [GitHub Issues](https://github.com/Luizfe88/XP3Forex/issues)

## 🙏 Agradecimentos

- MetaTrader 5 Team
- Optuna Team
- Scikit-learn Team
- Toda a comunidade open-source

---

**⭐ Se este projeto te ajudou, considere dar uma estrela no GitHub!**
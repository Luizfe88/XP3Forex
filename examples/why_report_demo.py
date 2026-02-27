import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Mock Rich BEFORE importing strategy
try:
    import rich
    print("Rich real encontrado.")
except ImportError:
    print("Rich não encontrado. Injetando Mocks...")
    
    # Create Mock Classes
    class MockPanel:
        def __init__(self, renderable, title=None, **kwargs):
            self.renderable = renderable
            self.title = title
        def __str__(self):
            return f"\n--- PANEL: {self.title} ---\n{self.renderable}\n------------------------\n"
        def __repr__(self):
            return self.__str__()

    class MockTable:
        def __init__(self, **kwargs):
            self.rows = []
            self.columns = []
        def add_column(self, name, **kwargs):
            self.columns.append(name)
        def add_row(self, *args, **kwargs):
            self.rows.append(args)
        def __str__(self):
            s = " | ".join(self.columns) + "\n"
            for row in self.rows:
                s += " | ".join(str(x) for x in row) + "\n"
            return s

    class MockText:
        def __init__(self, text, style=None):
            self.text = text
            self.style = style
        def __str__(self):
            return f"[{self.style}]{self.text}[/]"
        @classmethod
        def from_markup(cls, text):
            return cls(text)

    class MockGroup:
        def __init__(self, *renderables):
            self.renderables = renderables
        def __str__(self):
            return "\n".join(str(r) for r in self.renderables)

    class MockConsole:
        def print(self, obj):
            print(str(obj))

    # Inject into sys.modules
    sys.modules['rich'] = MagicMock()
    sys.modules['rich.panel'] = MagicMock()
    sys.modules['rich.panel'].Panel = MockPanel
    sys.modules['rich.table'] = MagicMock()
    sys.modules['rich.table'].Table = MockTable
    sys.modules['rich.text'] = MagicMock()
    sys.modules['rich.text'].Text = MockText
    sys.modules['rich.console'] = MagicMock()
    sys.modules['rich.console'].Console = MockConsole
    sys.modules['rich.console'].Group = MockGroup
    sys.modules['rich.box'] = MagicMock()
    sys.modules['rich.style'] = MagicMock()

# Mock Pydantic & Settings BEFORE importing strategy
# This prevents ImportError due to missing pydantic in environment
class MockSettings:
    MAX_DAILY_LOSS_PERCENT = 5.0
    MAX_POSITIONS = 5
    KILL_SWITCH_DD_PCT = 0.05
    RISK_PER_TRADE = 1.0

# Mock Pydantic module structure so imports work
mock_pydantic = MagicMock()
sys.modules['pydantic'] = mock_pydantic
sys.modules['pydantic_settings'] = MagicMock()

# Mock the settings module itself
mock_settings_module = MagicMock()
mock_settings_module.settings = MockSettings()
mock_settings_module.ELITE_CONFIG = {}
sys.modules['xp3_forex.core.settings'] = mock_settings_module
sys.modules['xp3_forex.core.models'] = MagicMock()

# Mock pandas_ta
class MockPandasTA:
    def ema(self, series, length=None, append=False):
        return pd.Series(np.random.uniform(1.0800, 1.0900, len(series)), index=series.index)
    def rsi(self, series, length=None):
        return pd.Series(np.random.uniform(30, 70, len(series)), index=series.index)
    def adx(self, length=None, append=False):
        # returns dataframe usually, but strategy uses df['ADX_14'] after append=True
        pass
    def atr(self, length=None, append=False):
        pass

sys.modules['pandas_ta'] = MockPandasTA()

# Patch DataFrame to have .ta accessor (simple mock)
# But since strategy uses `import pandas_ta as ta` and `df.ta.ema(...)`, 
# pandas_ta usually monkeypatches DataFrame.
# We can just mock the methods called on df.ta if needed, or rely on the fact that 
# the strategy calls `df.ta.ema` which might fail if accessor not present.
# Strategy uses:
# df.ta.ema(length=200, append=True)
# df.ta.adx(...)
# df.ta.atr(...)
# And also:
# ta.ema(df['close'], length=fast_period)

# So we need both module level and accessor level.
# For accessor, we can monkeypatch pd.DataFrame.

class OTAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    def ema(self, length=None, append=False):
        if append:
            self._obj[f'EMA_{length}'] = np.random.uniform(1.0800, 1.0900, len(self._obj))
        return self._obj[f'EMA_{length}'] if append else pd.Series(np.random.uniform(1.08, 1.09, len(self._obj)))
    def adx(self, length=None, append=False):
        if append:
            self._obj[f'ADX_{length}'] = np.random.uniform(20, 40, len(self._obj))
    def atr(self, length=None, append=False):
        if append:
            self._obj[f'ATRr_{length}'] = np.random.uniform(0.0010, 0.0020, len(self._obj))

try:
    pd.api.extensions.register_dataframe_accessor("ta")(OTAccessor)
except Exception:
    pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock dependencies
class MockBot:
    def __init__(self):
        self.cache = MockCache()
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]

class MockCache:
    def get_rates(self, symbol, timeframe, count):
        # Generate fake data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=300, freq='15min')
        data = {
            'time': dates,
            'open': np.random.uniform(1.0800, 1.0900, 300),
            'high': np.random.uniform(1.0800, 1.0900, 300),
            'low': np.random.uniform(1.0800, 1.0900, 300),
            'close': np.random.uniform(1.0800, 1.0900, 300),
            'tick_volume': np.random.randint(100, 1000, 300),
            'spread': np.random.randint(0, 20, 300),
            'real_volume': np.random.randint(100, 1000, 300)
        }
        df = pd.DataFrame(data)
        
        # Create uptrend scenario for EURUSD
        if symbol == "EURUSD":
            # Strong Uptrend
            t = np.linspace(0, 10, 300)
            trend = t * 0.002
            df['close'] = 1.0800 + trend + np.random.normal(0, 0.0002, 300)
            
        return df

# Mock mt5_utils
import xp3_forex.utils.mt5_utils
from dataclasses import dataclass

def mock_get_symbol_info(symbol):
    return {
        'spread': 12,
        'point': 0.00001,
        'trade_tick_value': 1.0
    }

xp3_forex.utils.mt5_utils.get_symbol_info = mock_get_symbol_info

import xp3_forex.utils.calculations
def mock_get_pip_size(symbol):
    return 0.0001
xp3_forex.utils.calculations.get_pip_size = mock_get_pip_size

# Import Strategy
# Reload to ensure mocks are used
if 'xp3_forex.strategies.adaptive_ema_rsi' in sys.modules:
    del sys.modules['xp3_forex.strategies.adaptive_ema_rsi']

from xp3_forex.strategies.adaptive_ema_rsi import AdaptiveEmaRsiStrategy

def run_demo():
    print("\n--- INICIANDO DEMO VISUAL (MOCK) ---\n")
    
    bot = MockBot()
    strategy = AdaptiveEmaRsiStrategy(bot)
    
    # Force regime update
    print("Calculando regimes...")
    strategy.startup()
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    for symbol in symbols:
        print(f"\n>>> Analisando {symbol}...")
        try:
            panel, confidence = strategy.get_why_report(symbol)
            if panel:
                print(str(panel))
                print(f"Confidence Score: {confidence}")
            else:
                print(f"Sem relatório para {symbol} (Confiança baixa ou erro)")
        except Exception as e:
            print(f"Erro: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_demo()

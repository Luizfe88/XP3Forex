import types

import utils_forex as utils


class DummyAccountInfo:
    def __init__(self, equity: float, balance: float):
        self.equity = equity
        self.balance = balance


def _setup_mt5_mock(monkeypatch, equity=10000.0, balance=10000.0, pip_value=10.0):
    mt5_mock = types.SimpleNamespace()
    mt5_mock.account_info = lambda: DummyAccountInfo(equity, balance)
    mt5_mock.symbol_info = lambda symbol: types.SimpleNamespace(volume_step=0.01, visible=True, point=0.00001, trade_tick_value_profit=1.0)
    monkeypatch.setattr(utils, "mt5", mt5_mock)

    def fake_get_tick_value(symbol: str) -> float:
        return pip_value

    monkeypatch.setattr(utils, "get_tick_value", fake_get_tick_value)


def test_calculate_position_size_atr_forex_basic(monkeypatch):
    _setup_mt5_mock(monkeypatch, equity=10000.0, balance=10000.0, pip_value=10.0)

    monkeypatch.setattr(utils.config, "RISK_PER_TRADE_PCT", 0.01, raising=False)
    monkeypatch.setattr(utils.config, "MIN_VOLUME", 0.01, raising=False)
    monkeypatch.setattr(utils.config, "MAX_VOLUME", 1.0, raising=False)

    vol = utils.calculate_position_size_atr_forex(
        symbol="EURUSD",
        price=1.1000,
        atr_pips=20.0,
        sl_atr_mult=2.0,
        risk_multiplier=1.0,
    )

    assert vol == 0.25


def test_calculate_position_size_atr_forex_safety_cap(monkeypatch):
    _setup_mt5_mock(monkeypatch, equity=1000000.0, balance=1000000.0, pip_value=10.0)

    monkeypatch.setattr(utils.config, "RISK_PER_TRADE_PCT", 0.01, raising=False)
    monkeypatch.setattr(utils.config, "MIN_VOLUME", 0.01, raising=False)
    monkeypatch.setattr(utils.config, "MAX_VOLUME", 10.0, raising=False)

    vol = utils.calculate_position_size_atr_forex(
        symbol="EURUSD",
        price=1.1000,
        atr_pips=5.0,
        sl_atr_mult=1.0,
        risk_multiplier=1.0,
    )

    assert vol == 0.01


def test_calculate_position_size_atr_forex_account_info_none(monkeypatch):
    mt5_mock = types.SimpleNamespace()
    mt5_mock.account_info = lambda: None
    monkeypatch.setattr(utils, "mt5", mt5_mock)
    monkeypatch.setattr(utils.config, "DEFAULT_LOT", 0.02, raising=False)
    vol = utils.calculate_position_size_atr_forex(
        symbol="EURUSD",
        price=1.1000,
        atr_pips=20.0,
        sl_atr_mult=2.0,
        risk_multiplier=1.0,
    )
    assert vol == 0.02

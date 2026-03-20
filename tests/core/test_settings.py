import pytest
from xp3_forex.core.settings import settings

def test_risk_settings_for_small_account():
    # Verify virtual balance settings
    assert settings.USE_VIRTUAL_BALANCE is True
    assert settings.VIRTUAL_BALANCE == 100.0
    
    # Verify core risk limits
    assert settings.RISK_PER_TRADE == 1.0
    assert settings.MAX_POSITIONS == 2
    assert settings.MAX_LOTS_PER_TRADE == 0.01
    
    # Verify protection thresholds
    assert settings.MAX_LOSS_DOLLARS == -2.0
    assert settings.BREAK_EVEN_TRIGGER == 1.0
    assert settings.PROFIT_ACTIVATION_THRESHOLD == 3.0
    
    # Verify asset restrictions
    assert settings.ALLOWED_ASSET_CATEGORIES == "major"
    
def test_symbol_categories_list():
    categories = [c.strip().lower() for c in settings.ALLOWED_ASSET_CATEGORIES.split(",") if c.strip()]
    assert "major" in categories
    assert "metal" not in categories
    assert "index" not in categories

if __name__ == "__main__":
    # Basic print for manual check if pytest is not used
    print(f"USE_VIRTUAL_BALANCE: {settings.USE_VIRTUAL_BALANCE}")
    print(f"VIRTUAL_BALANCE: {settings.VIRTUAL_BALANCE}")
    print(f"ALLOWED_CATEGORIES: {settings.ALLOWED_ASSET_CATEGORIES}")

import pandas as pd
from typing import Optional
from ..utils.mt5_utils import get_rates

class RateCache:
    """
    Wrapper for rate retrieval with caching.
    Uses the underlying get_rates function which already implements caching.
    """
    
    def get_rates(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """
        Get rates for a symbol and timeframe.
        """
        return get_rates(symbol, timeframe, count)

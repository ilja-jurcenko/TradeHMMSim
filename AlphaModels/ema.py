"""
Exponential Moving Average (EMA) alpha model.
"""

import pandas as pd
from typing import Tuple
from .base import AlphaModel


class EMA(AlphaModel):
    """
    Exponential Moving Average crossover strategy.
    """
    
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate EMA indicators.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (short_ema, long_ema)
        """
        short_ema = close.ewm(span=self.short_window, adjust=False).mean()
        long_ema = close.ewm(span=self.long_window, adjust=False).mean()
        
        return short_ema, long_ema

"""
Simple Moving Average (SMA) alpha model.
"""

import pandas as pd
from typing import Tuple
from .base import AlphaModel


class SMA(AlphaModel):
    """
    Simple Moving Average crossover strategy.
    """
    
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SMA indicators.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (short_sma, long_sma)
        """
        short_sma = close.rolling(window=self.short_window).mean()
        long_sma = close.rolling(window=self.long_window).mean()
        
        return short_sma, long_sma

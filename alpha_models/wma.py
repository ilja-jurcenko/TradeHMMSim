"""
Weighted Moving Average (WMA) alpha model.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .base import AlphaModel


class WMA(AlphaModel):
    """
    Weighted Moving Average crossover strategy.
    """
    
    def _calculate_wma(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate Weighted Moving Average.
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        window : int
            Window size
            
        Returns:
        --------
        pd.Series
            WMA values
        """
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == window else np.nan,
            raw=True
        )
    
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate WMA indicators.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (short_wma, long_wma)
        """
        short_wma = self._calculate_wma(close, self.short_window)
        long_wma = self._calculate_wma(close, self.long_window)
        
        return short_wma, long_wma

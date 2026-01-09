"""
Hull Moving Average (HMA) alpha model.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .base import AlphaModel


class HMA(AlphaModel):
    """
    Hull Moving Average crossover strategy.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    
    def _calculate_wma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == window else np.nan,
            raw=True
        )
    
    def _calculate_hma(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate Hull Moving Average.
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        window : int
            Window size
            
        Returns:
        --------
        pd.Series
            HMA values
        """
        half_length = int(window / 2)
        sqrt_length = int(np.sqrt(window))
        
        wma_half = self._calculate_wma(series, half_length)
        wma_full = self._calculate_wma(series, window)
        raw_hma = 2 * wma_half - wma_full
        
        return self._calculate_wma(raw_hma, sqrt_length)
    
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate HMA indicators.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (short_hma, long_hma)
        """
        short_hma = self._calculate_hma(close, self.short_window)
        long_hma = self._calculate_hma(close, self.long_window)
        
        return short_hma, long_hma

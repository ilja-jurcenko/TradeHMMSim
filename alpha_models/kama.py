"""
Kaufman's Adaptive Moving Average (KAMA) alpha model.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .base import AlphaModel


class KAMA(AlphaModel):
    """
    Kaufman's Adaptive Moving Average crossover strategy.
    """
    
    def _calculate_kama(self, series: pd.Series, window: int, 
                       fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
        """
        Calculate Kaufman's Adaptive Moving Average.
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        window : int
            Window for efficiency ratio calculation
        fast_ema : int
            Fast EMA period
        slow_ema : int
            Slow EMA period
            
        Returns:
        --------
        pd.Series
            KAMA values
        """
        # Efficiency Ratio
        change = abs(series - series.shift(window))
        volatility = series.diff().abs().rolling(window).sum()
        er = change / volatility
        er = er.fillna(0)
        
        # Smoothing constant
        fast_sc = 2 / (fast_ema + 1)
        slow_sc = 2 / (slow_ema + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        kama_vals = np.zeros(len(series))
        kama_vals[:window] = np.nan
        
        # Handle edge case where window >= series length
        if window >= len(series):
            return pd.Series(kama_vals, index=series.index)
        
        # Initialize KAMA at window position
        kama_vals[window] = float(series.iloc[window])
        
        # Calculate KAMA values
        for i in range(window + 1, len(series)):
            sc_val = float(sc.iloc[i])
            price_val = float(series.iloc[i])
            kama_vals[i] = kama_vals[i-1] + sc_val * (price_val - kama_vals[i-1])
        
        return pd.Series(kama_vals, index=series.index)
    
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate KAMA indicators.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (short_kama, long_kama)
        """
        short_kama = self._calculate_kama(close, self.short_window)
        long_kama = self._calculate_kama(close, self.long_window)
        
        return short_kama, long_kama

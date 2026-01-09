"""
Zero-Lag Exponential Moving Average (ZLEMA) alpha model.
"""

import pandas as pd
from typing import Tuple
from .base import AlphaModel


class ZLEMA(AlphaModel):
    """
    Zero-Lag Exponential Moving Average crossover strategy.
    ZLEMA = EMA(data + (data - data[lag]))
    """
    
    def _calculate_zlema(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate Zero-Lag Exponential Moving Average.
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        window : int
            Window size
            
        Returns:
        --------
        pd.Series
            ZLEMA values
        """
        lag = int((window - 1) / 2)
        ema_data = series + (series - series.shift(lag))
        
        return ema_data.ewm(span=window, adjust=False).mean()
    
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate ZLEMA indicators.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (short_zlema, long_zlema)
        """
        short_zlema = self._calculate_zlema(close, self.short_window)
        long_zlema = self._calculate_zlema(close, self.long_window)
        
        return short_zlema, long_zlema

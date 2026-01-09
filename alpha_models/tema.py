"""
Triple Exponential Moving Average (TEMA) alpha model.
"""

import pandas as pd
from typing import Tuple
from .base import AlphaModel


class TEMA(AlphaModel):
    """
    Triple Exponential Moving Average crossover strategy.
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    """
    
    def _calculate_tema(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate Triple Exponential Moving Average.
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        window : int
            Window size
            
        Returns:
        --------
        pd.Series
            TEMA values
        """
        ema1 = series.ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        
        return 3 * ema1 - 3 * ema2 + ema3
    
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate TEMA indicators.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (short_tema, long_tema)
        """
        short_tema = self._calculate_tema(close, self.short_window)
        long_tema = self._calculate_tema(close, self.long_window)
        
        return short_tema, long_tema

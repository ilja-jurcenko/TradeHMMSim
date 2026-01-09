"""
Base class for alpha models (trading signals).
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class AlphaModel(ABC):
    """
    Abstract base class for alpha models.
    """
    
    def __init__(self, short_window: int, long_window: int):
        """
        Initialize alpha model.
        
        Parameters:
        -----------
        short_window : int
            Short period window
        long_window : int
            Long period window
        """
        self.short_window = short_window
        self.long_window = long_window
        self.name = self.__class__.__name__
        
    @abstractmethod
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate short and long indicators.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (short_indicator, long_indicator)
        """
        pass
    
    def generate_signals(self, close: pd.Series) -> pd.Series:
        """
        Generate trading signals (1 = long, 0 = flat).
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        pd.Series
            Trading signals (1 = bullish, 0 = bearish/neutral)
        """
        short_ind, long_ind = self.calculate_indicators(close)
        
        # Bullish when short > long
        signals = (short_ind > long_ind).astype(int)
        signals = signals.fillna(0)
        
        return signals
    
    def get_name(self) -> str:
        """Get model name."""
        return self.name
    
    def get_parameters(self) -> dict:
        """Get model parameters."""
        return {
            'short_window': self.short_window,
            'long_window': self.long_window,
            'model_type': self.name
        }

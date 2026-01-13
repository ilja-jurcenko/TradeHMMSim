"""
Bollinger Bands alpha model for mean-reversion strategy.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .base import AlphaModel


class BollingerBands(AlphaModel):
    """
    Bollinger Bands mean-reversion strategy.
    
    Generates buy signals when price crosses below lower band (oversold).
    Generates sell signals when price crosses above upper band (overbought).
    Exits when price returns to middle band.
    
    This is a contrarian strategy suitable for bear/sideways markets.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 2):
        """
        Initialize Bollinger Bands model.
        
        Parameters:
        -----------
        short_window : int
            Period for calculating moving average and std dev (default: 20)
        long_window : int
            Number of standard deviations for bands (default: 2)
        """
        super().__init__(short_window, long_window)
        self.period = short_window
        self.num_std = long_window
        
    def calculate_indicators(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (middle_band, lower_band)
            Note: We return middle and lower bands for compatibility with base class
        """
        # Calculate middle band (SMA)
        middle_band = close.rolling(window=self.period).mean()
        
        # Calculate standard deviation
        std_dev = close.rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * self.num_std)
        lower_band = middle_band - (std_dev * self.num_std)
        
        # Store for signal generation
        self._middle_band = middle_band
        self._upper_band = upper_band
        self._lower_band = lower_band
        
        # Return middle and lower for base class compatibility
        return middle_band, lower_band
    
    def generate_signals(self, close: pd.Series) -> pd.Series:
        """
        Generate mean-reversion trading signals.
        
        Strategy:
        - Buy when price crosses below lower band (oversold)
        - Hold until price returns to middle band
        - Sell/Flat when price crosses above upper band (overbought)
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        pd.Series
            Trading signals (1 = long, 0 = flat, -1 = short)
        """
        # Calculate bands
        self.calculate_indicators(close)
        
        # Initialize signals
        signals = pd.Series(0, index=close.index)
        
        # Track position state
        in_position = False
        
        for i in range(1, len(close)):
            # Skip if we don't have valid band values
            if pd.isna(self._lower_band.iloc[i]) or pd.isna(self._upper_band.iloc[i]):
                signals.iloc[i] = signals.iloc[i-1]
                continue
            
            prev_price = close.iloc[i-1]
            curr_price = close.iloc[i]
            
            # Buy signal: price crosses below lower band (oversold)
            if not in_position and prev_price >= self._lower_band.iloc[i-1] and curr_price < self._lower_band.iloc[i]:
                signals.iloc[i] = 1
                in_position = True
            # Exit signal: price returns to or crosses middle band
            elif in_position and curr_price >= self._middle_band.iloc[i]:
                signals.iloc[i] = 0
                in_position = False
            # Hold position
            elif in_position:
                signals.iloc[i] = 1
            # Sell signal (optional): price crosses above upper band
            elif not in_position and prev_price <= self._upper_band.iloc[i-1] and curr_price > self._upper_band.iloc[i]:
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = signals.iloc[i-1] if not in_position else 1
        
        return signals
    
    def get_bands(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Get all three Bollinger Bands.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            (upper_band, middle_band, lower_band)
        """
        self.calculate_indicators(close)
        return self._upper_band, self._middle_band, self._lower_band

"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Any


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    
    def __init__(self, name: str):
        """
        Initialize base strategy.
        
        Parameters:
        -----------
        name : str
            Strategy name
        """
        self.name = name
    
    @abstractmethod
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> pd.Series:
        """
        Generate position signals based on strategy logic.
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals aligned to common_idx
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Additional strategy-specific parameters
            
        Returns:
        --------
        pd.Series
            Position signals (can be binary 0/1 or fractional)
        """
        pass
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for the strategy.
        
        Parameters:
        -----------
        positions : pd.Series
            Generated positions
        close : pd.Series
            Close price series
        alpha_signals : pd.Series
            Alpha model signals
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Additional strategy-specific parameters
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of log entries
        """
        return []
    
    def get_name(self) -> str:
        """Get strategy name."""
        return self.name
    
    @staticmethod
    def _determine_action(prev_position: float, curr_position: float) -> str:
        """
        Determine trading action based on position change.
        
        Parameters:
        -----------
        prev_position : float
            Previous position
        curr_position : float
            Current position
            
        Returns:
        --------
        str
            Action taken
        """
        if prev_position == curr_position:
            return 'HOLD'
        elif prev_position < curr_position:
            if prev_position == 0:
                return 'BUY'
            else:
                return 'INCREASE_POSITION'
        else:
            if curr_position == 0:
                return 'SELL'
            else:
                return 'DECREASE_POSITION'

"""
Alpha-only strategy (no HMM filtering).
"""

import pandas as pd
from typing import Dict, List, Any
from .base import BaseStrategy


class AlphaOnlyStrategy(BaseStrategy):
    """
    Strategy using only alpha model signals (no HMM filtering).
    """
    
    def __init__(self):
        super().__init__('alpha_only')
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> pd.Series:
        """
        Generate positions from alpha signals only.
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
            
        Returns:
        --------
        pd.Series
            Binary position signals (0 or 1)
        """
        # Alpha signals already processed through FSM logic
        return alpha_signals.loc[common_idx]
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for alpha-only strategy.
        """
        log_data = []
        
        for i in range(len(positions)):
            idx = positions.index[i]
            prev_pos = 0 if i == 0 else positions.iloc[i-1]
            curr_pos = positions.iloc[i]
            action = self._determine_action(prev_pos, curr_pos)
            
            log_data.append({
                'Date': idx,
                'Price': close.loc[idx],
                'Alpha_Signal': alpha_signals.loc[idx],
                'Position': curr_pos,
                'Action': action
            })
        
        return log_data

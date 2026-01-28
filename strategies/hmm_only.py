"""
HMM-only strategy (ignores alpha signals).
"""

import pandas as pd
from typing import Dict, List, Any
from .base import BaseStrategy


class HMMOnlyStrategy(BaseStrategy):
    """
    Strategy using only HMM regime detection (ignores alpha signals).
    Uses binary positions based on combined bull+neutral probability.
    """
    
    def __init__(self):
        super().__init__('hmm_only')
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> pd.Series:
        """
        Generate positions from HMM regime probabilities only.
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals (ignored)
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Must contain: bull_prob_combined, bull_prob_threshold
            
        Returns:
        --------
        pd.Series
            Binary position signals (0 or 1)
        """
        bull_prob_combined = kwargs['bull_prob_combined']
        bull_prob_threshold = kwargs['bull_prob_threshold']
        
        # Binary positions based on combined bull+neutral probability
        positions = (bull_prob_combined > bull_prob_threshold).astype(int)
        
        return positions
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for HMM-only strategy.
        """
        bear_prob = kwargs['bear_prob']
        bull_prob = kwargs['bull_prob']
        bull_prob_combined = kwargs['bull_prob_combined']
        bull_prob_threshold = kwargs['bull_prob_threshold']
        
        log_data = []
        
        for i, idx in enumerate(common_idx):
            prev_pos = 0 if i == 0 else positions.iloc[i-1]
            curr_pos = positions.iloc[i]
            action = self._determine_action(prev_pos, curr_pos)
            
            log_data.append({
                'Date': idx,
                'Price': close.loc[idx],
                'Alpha_Signal': alpha_signals.iloc[i],
                'Bear_Prob': bear_prob.iloc[i],
                'Bull_Prob': bull_prob.iloc[i],
                'Bull_Combined_Prob': bull_prob_combined.iloc[i],
                'HMM_Signal': 1 if bull_prob_combined.iloc[i] > bull_prob_threshold else 0,
                'Position': curr_pos,
                'Action': action,
                'Strategy': 'HMM_Only'
            })
        
        return log_data

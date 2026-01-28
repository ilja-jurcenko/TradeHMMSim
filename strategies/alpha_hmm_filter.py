"""
Alpha + HMM Filter strategy (HMM filters out bear regime signals).
"""

import pandas as pd
from typing import Dict, List, Any
from .base import BaseStrategy


class AlphaHMMFilterStrategy(BaseStrategy):
    """
    Strategy combining alpha signals with HMM bear regime filter.
    HMM only filters out positions during bear regime.
    """
    
    def __init__(self):
        super().__init__('alpha_hmm_filter')
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> pd.Series:
        """
        Generate positions by filtering alpha signals through HMM bear filter.
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Must contain: bear_prob, bear_prob_threshold
            
        Returns:
        --------
        pd.Series
            Filtered position signals (0 or 1)
        """
        bear_prob = kwargs['bear_prob']
        bear_prob_threshold = kwargs['bear_prob_threshold']
        
        # Filter: allow positions only when bear probability is low
        bear_filter = (bear_prob < bear_prob_threshold).astype(int)
        positions = alpha_signals * bear_filter
        
        return positions
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for alpha+HMM filter strategy.
        """
        bear_prob = kwargs['bear_prob']
        bull_prob = kwargs['bull_prob']
        neutral_prob = kwargs['neutral_prob']
        bear_prob_threshold = kwargs['bear_prob_threshold']
        bear_filter = (bear_prob < bear_prob_threshold).astype(int)
        
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
                'Neutral_Prob': neutral_prob.iloc[i],
                'Bear_Filter': bear_filter.iloc[i],
                'HMM_Signal': 0 if bear_prob.iloc[i] >= bear_prob_threshold else 1,
                'Position': curr_pos,
                'Action': action
            })
        
        return log_data

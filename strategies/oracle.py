"""
Oracle strategy (HMM with perfect future knowledge).
"""

import pandas as pd
from typing import Dict, List, Any
from .base import BaseStrategy


class OracleStrategy(BaseStrategy):
    """
    Oracle strategy using HMM fitted on entire dataset (look-ahead bias).
    Uses fractional positions: Bull=1.0, Neutral=0.5, Bear=0.0
    """
    
    def __init__(self):
        super().__init__('oracle')
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> pd.Series:
        """
        Generate fractional positions based on regime detection.
        
        Bull regime: 1.0 (100% in instrument)
        Neutral regime: 0.5 (50% in instrument, 50% cash)
        Bear regime: 0.0 (100% cash)
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals (ignored)
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Must contain: bear_prob, bull_prob, neutral_prob
            
        Returns:
        --------
        pd.Series
            Fractional position signals (0.0, 0.5, or 1.0)
        """
        bear_prob = kwargs['bear_prob']
        bull_prob = kwargs['bull_prob']
        neutral_prob = kwargs['neutral_prob']
        
        # Default to neutral (50%)
        positions = pd.Series(0.5, index=common_idx)
        
        # Detect regimes: highest probability wins
        bull_regime_mask = (bull_prob > bear_prob) & (bull_prob > neutral_prob)
        bear_regime_mask = (bear_prob > bull_prob) & (bear_prob > neutral_prob)
        neutral_regime_mask = ~(bull_regime_mask | bear_regime_mask)
        
        # Apply fractional positions
        positions.loc[bull_regime_mask] = 1.0    # Bull: 100% in instrument
        positions.loc[neutral_regime_mask] = 0.5  # Neutral: 50% in instrument, 50% cash
        positions.loc[bear_regime_mask] = 0.0    # Bear: 0% in instrument (100% cash)
        
        return positions
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for Oracle strategy.
        """
        bear_prob = kwargs['bear_prob']
        bull_prob = kwargs['bull_prob']
        neutral_prob = kwargs['neutral_prob']
        bull_prob_combined = kwargs['bull_prob_combined']
        
        # Determine regime masks
        bull_regime_mask = (bull_prob > bear_prob) & (bull_prob > neutral_prob)
        bear_regime_mask = (bear_prob > bull_prob) & (bear_prob > neutral_prob)
        
        log_data = []
        
        for i, idx in enumerate(common_idx):
            prev_pos = 0.5 if i == 0 else positions.iloc[i-1]
            curr_pos = positions.iloc[i]
            
            # Determine regime
            if bull_regime_mask.iloc[i]:
                regime_label = 'Bull'
            elif bear_regime_mask.iloc[i]:
                regime_label = 'Bear'
            else:
                regime_label = 'Neutral'
            
            # Determine action
            if prev_pos < curr_pos:
                action = 'INCREASE_POSITION'
            elif prev_pos > curr_pos:
                action = 'DECREASE_POSITION'
            else:
                action = 'HOLD'
            
            log_data.append({
                'Date': idx,
                'Price': close.loc[idx],
                'Alpha_Signal': alpha_signals.iloc[i],
                'Bear_Prob': bear_prob.iloc[i],
                'Bull_Prob': bull_prob.iloc[i],
                'Neutral_Prob': neutral_prob.iloc[i],
                'Bull_Combined_Prob': bull_prob_combined.iloc[i],
                'Regime': regime_label,
                'Position': curr_pos,
                'Action': action,
                'Strategy': 'Oracle'
            })
        
        return log_data

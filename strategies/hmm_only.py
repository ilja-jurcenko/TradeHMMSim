"""
HMM-only strategy (ignores alpha signals).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .base import BaseStrategy


class HMMOnlyStrategy(BaseStrategy):
    """
    Strategy using only HMM regime switches (ignores alpha signals).
    Uses regime switches (turning points) as buy and sell signals.
    
    Regime determination uses hybrid approach:
    1. Detect dominant regime by highest probability (bear/neutral/bull)
    2. Detect regime switches (turning points)
    3. Validate switch: new regime probability must exceed threshold before signaling
       - Bear threshold: bear_prob >= bear_threshold (default 0.65)
       - Bull threshold: bull_combined_prob > bull_threshold (default 0.65)
    
    Buy signals: 
    - Switch FROM Bear TO [Bull, Neutral] AND threshold exceeded
    - Switch between Bull <-> Neutral while OUT of market AND threshold exceeded
    Sell signals: Switch FROM [Bull, Neutral] TO Bear AND threshold exceeded
    Hold: Switch between Bull <-> Neutral while IN market OR threshold not met
    """
    
    def __init__(self):
        super().__init__('hmm_only')
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> pd.Series:
        """
        Generate positions from HMM regime switches only.
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals (ignored)
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Must contain: bear_prob, bull_prob, neutral_prob, bear_prob_threshold, bull_prob_threshold
            
        Returns:
        --------
        pd.Series
            Binary position signals (0 or 1)
        """
        bear_prob = kwargs['bear_prob']
        bull_prob = kwargs['bull_prob']
        neutral_prob = kwargs.get('neutral_prob', pd.Series(0, index=common_idx))
        bear_threshold = kwargs.get('bear_prob_threshold', 0.65)
        bull_threshold = kwargs.get('bull_prob_threshold', 0.65)
        
        # Step 1: Determine dominant regime by highest probability
        probs_df = pd.DataFrame({
            'bear': bear_prob,
            'neutral': neutral_prob,
            'bull': bull_prob
        }, index=common_idx)
        regime = probs_df.idxmax(axis=1)
        
        # Step 2: Detect regime switches
        regime_switches = regime != regime.shift(1)
        
        # Calculate combined bull probability for threshold checks
        bull_combined_prob = bull_prob + neutral_prob
        
        # Generate positions based on regime switches
        positions = pd.Series(0, index=common_idx)
        
        current_position = 0
        for i, idx in enumerate(common_idx):
            if i == 0:
                # Initial position: always start invested
                current_position = 1
            elif regime_switches.iloc[i]:
                # Regime switch detected
                prev_regime = regime.iloc[i-1]
                curr_regime = regime.iloc[i]
                
                # Step 3: Check if new regime probability exceeds threshold
                # Switch FROM bear TO [bull, neutral]
                if prev_regime == 'bear' and curr_regime in ['bull', 'neutral']:
                    # Validate: bull_combined_prob must exceed threshold
                    if bull_combined_prob.iloc[i] > bull_threshold:
                        current_position = 1
                # Switch FROM [bull, neutral] TO bear
                elif prev_regime in ['bull', 'neutral'] and curr_regime == 'bear':
                    # Validate: bear_prob must exceed threshold
                    if bear_prob.iloc[i] >= bear_threshold:
                        current_position = 0
                # Switch between bull <-> neutral while OUT of market
                elif current_position == 0 and prev_regime in ['bull', 'neutral'] and curr_regime in ['bull', 'neutral']:
                    # Validate: bull_combined_prob must exceed threshold
                    if bull_combined_prob.iloc[i] > bull_threshold:
                        current_position = 1
                # Otherwise (bull <-> neutral while IN market or threshold not met) -> maintain position
            
            positions.iloc[i] = current_position
        
        return positions
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for HMM-only strategy with regime information.
        """
        bear_prob = kwargs['bear_prob']
        bull_prob = kwargs['bull_prob']
        neutral_prob = kwargs.get('neutral_prob', pd.Series(0, index=common_idx))
        bear_threshold = kwargs.get('bear_prob_threshold', 0.65)
        bull_threshold = kwargs.get('bull_prob_threshold', 0.65)
        
        # Determine dominant regime by highest probability
        probs_df = pd.DataFrame({
            'bear': bear_prob,
            'neutral': neutral_prob,
            'bull': bull_prob
        }, index=common_idx)
        regime = probs_df.idxmax(axis=1)
        regime_switches = regime != regime.shift(1)
        
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
                'Neutral_Prob': neutral_prob.iloc[i],
                'Bull_Prob': bull_prob.iloc[i],
                'Regime': regime.iloc[i],
                'Regime_Switch': regime_switches.iloc[i] if i > 0 else False,
                'Position': curr_pos,
                'Action': action,
                'Strategy': 'HMM_Only'
            })
        
        return log_data

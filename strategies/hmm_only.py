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
    
    def _calculate_regime_from_probs(self, bear_prob: pd.Series, neutral_prob: pd.Series, 
                                    bull_prob: pd.Series, common_idx: pd.Index) -> pd.Series:
        """
        Calculate regime by highest probability (legacy method).
        Preserved for compatibility and testing.
        
        Parameters:
        -----------
        bear_prob : pd.Series
            Bear regime probability
        neutral_prob : pd.Series
            Neutral regime probability
        bull_prob : pd.Series
            Bull regime probability
        common_idx : pd.Index
            Common index for alignment
            
        Returns:
        --------
        pd.Series
            Regime at each time point
        """
        probs_df = pd.DataFrame({
            'bear': bear_prob,
            'neutral': neutral_prob,
            'bull': bull_prob
        }, index=common_idx)
        return probs_df.idxmax(axis=1)
    
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
            Optional: regime (from HMM), switches (from HMM)
            
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
        
        # Get regime and switches from HMM filter if available
        regime = kwargs.get('regime')
        switches = kwargs.get('switches')
        
        # Fallback: calculate regime from probabilities if not provided
        if regime is None:
            regime = self._calculate_regime_from_probs(bear_prob, neutral_prob, bull_prob, common_idx)
        
        # Map regime integer values to strings if needed
        # Identify regime mapping from probabilities
        bear_regime_id = None
        bull_regime_id = None
        neutral_regime_id = None
        
        if regime.dtype in [np.int32, np.int64, int]:
            # Regime is numeric - need to map to bear/neutral/bull
            # Use volatility-based identification from kwargs if available
            regime_info = kwargs.get('regime_info')
            if regime_info:
                bear_regime_id = regime_info['bear_regime']
                bull_regime_id = regime_info['bull_regime']
                neutral_regime_id = regime_info['neutral_regime']
            else:
                # Fallback: assume 0=low vol (bull), 1=mid vol (neutral), 2=high vol (bear)
                # This is a guess - better to pass regime_info
                unique_regimes = sorted(regime.unique())
                if len(unique_regimes) == 2:
                    bull_regime_id = unique_regimes[0]
                    bear_regime_id = unique_regimes[1]
                    neutral_regime_id = None
                elif len(unique_regimes) == 3:
                    bull_regime_id = unique_regimes[0]
                    neutral_regime_id = unique_regimes[1]
                    bear_regime_id = unique_regimes[2]
            
            # Convert numeric regime to string labels
            regime_str = regime.copy()
            if bear_regime_id is not None:
                regime_str = regime_str.replace(bear_regime_id, 'bear')
            if bull_regime_id is not None:
                regime_str = regime_str.replace(bull_regime_id, 'bull')
            if neutral_regime_id is not None:
                regime_str = regime_str.replace(neutral_regime_id, 'neutral')
            regime = regime_str
        
        # Create switch indicator aligned with common_idx
        #if switches is not None and len(switches) > 0:
            # switches is a Series with dates where switches occurred
            # Create boolean series indicating if each date has a switch
        #    regime_switches = pd.Series(False, index=common_idx)
        #    regime_switches.loc[regime_switches.index.intersection(switches.index)] = True
        #else:
            # Fallback: calculate switches from regime changes
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
                # Regime switch detected at this position
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
        
        # Get regime and switches from HMM filter if available
        regime = kwargs.get('regime')
        switches = kwargs.get('switches')
        
        # Fallback: calculate regime from probabilities if not provided
        if regime is None:
            regime = self._calculate_regime_from_probs(bear_prob, neutral_prob, bull_prob, common_idx)
        
        # Map numeric regime to string if needed
        if regime.dtype in [np.int32, np.int64, int]:
            regime_info = kwargs.get('regime_info')
            if regime_info:
                bear_regime_id = regime_info['bear_regime']
                bull_regime_id = regime_info['bull_regime']
                neutral_regime_id = regime_info['neutral_regime']
                
                regime_str = regime.copy()
                if bear_regime_id is not None:
                    regime_str = regime_str.replace(bear_regime_id, 'bear')
                if bull_regime_id is not None:
                    regime_str = regime_str.replace(bull_regime_id, 'bull')
                if neutral_regime_id is not None:
                    regime_str = regime_str.replace(neutral_regime_id, 'neutral')
                regime = regime_str
        
        # Create switch indicator
        if switches is not None and len(switches) > 0:
            regime_switches = pd.Series(False, index=common_idx)
            regime_switches.loc[regime_switches.index.intersection(switches.index)] = True
        else:
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

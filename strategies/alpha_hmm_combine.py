"""
Alpha + HMM Combine strategy (4-state variance-trend logic).
"""

import pandas as pd
from typing import Dict, List, Any, Tuple
from .base import BaseStrategy


class AlphaHMMCombineStrategy(BaseStrategy):
    """
    4-State Variance-Trend Strategy.
    
    Combines HMM (market variance/regime) with Alpha (trend direction):
      - HMM detects variance: Low (bull/neutral) vs High (bear)
      - Alpha detects trend: Bullish vs Bearish
    
    Four States:
      State 1: Low variance + Bullish trend  → BUY
      State 2: Low variance + Bearish trend  → HOLD
      State 3: High variance + Bullish trend → HOLD
      State 4: High variance + Bearish trend → SELL
    
    Trading Logic:
      - Enter long only in State 1 (safe bull market)
      - Exit long only in State 4 (dangerous bear market)
      - All other states: maintain current position
    """
    
    def __init__(self):
        super().__init__('alpha_hmm_combine')
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate positions using 4-state logic.
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Must contain: bear_prob, bull_prob_combined, bear_prob_threshold, bull_prob_threshold
            
        Returns:
        --------
        Tuple[pd.Series, Dict[str, Any]]
            (positions, state_info) - positions and state information for plotting
        """
        bear_prob = kwargs['bear_prob']
        bull_prob_combined = kwargs['bull_prob_combined']
        bear_prob_threshold = kwargs['bear_prob_threshold']
        bull_prob_threshold = kwargs['bull_prob_threshold']
        
        # Detect variance regimes
        low_variance = (bull_prob_combined > bull_prob_threshold).astype(bool)
        high_variance = (bear_prob > bear_prob_threshold).astype(bool)
        
        # Detect trend direction
        bullish_trend = (alpha_signals > 0).astype(bool)
        bearish_trend = ~bullish_trend
        
        # Define the 4 states
        state_1 = low_variance & bullish_trend   # Low variance + Bullish → BUY signal
        state_2 = low_variance & bearish_trend   # Low variance + Bearish → HOLD
        state_3 = high_variance & bullish_trend  # High variance + Bullish → HOLD
        state_4 = high_variance & bearish_trend  # High variance + Bearish → SELL signal
        
        # Create state labels
        state_labels = pd.Series('Unknown', index=common_idx)
        state_labels[state_1] = 'State 1: Low Var + Bull'
        state_labels[state_2] = 'State 2: Low Var + Bear'
        state_labels[state_3] = 'State 3: High Var + Bull'
        state_labels[state_4] = 'State 4: High Var + Bear'
        
        # Initialize positions
        positions = pd.Series(0, index=common_idx, dtype=int)
        
        # State machine: track position and respond to state transitions
        current_position = 0  # Start flat
        prev_state = None
        
        for i in range(len(common_idx)):
            # Determine current state
            if state_1.iloc[i]:
                current_state_num = 1
            elif state_2.iloc[i]:
                current_state_num = 2
            elif state_3.iloc[i]:
                current_state_num = 3
            elif state_4.iloc[i]:
                current_state_num = 4
            else:
                current_state_num = None
            
            # Detect state transitions and update position
            if prev_state is not None and current_state_num is not None:
                # State switch to 4: SELL
                if prev_state in [1, 2, 3] and current_state_num in [4] and current_position == 1:
                    current_position = 0
                # State switch to 1: BUY
                elif prev_state in [2, 3, 4] and current_state_num in [1] and current_position == 0:
                    current_position = 1
            
            # Update prev_state for next iteration
            prev_state = current_state_num
            
            positions.iloc[i] = current_position
        
        # Store state information
        state_info = {
            'state_labels': state_labels,
            'state_1': state_1,
            'state_2': state_2,
            'state_3': state_3,
            'state_4': state_4
        }
        
        # Print diagnostics
        print(f"  State 1 (Low var + Bull): {state_1.sum()} periods")
        print(f"  State 2 (Low var + Bear): {state_2.sum()} periods")
        print(f"  State 3 (High var + Bull): {state_3.sum()} periods")
        print(f"  State 4 (High var + Bear): {state_4.sum()} periods")
        
        return positions, state_info
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for alpha+HMM combine strategy.
        """
        bear_prob = kwargs['bear_prob']
        bull_prob = kwargs['bull_prob']
        bull_prob_combined = kwargs['bull_prob_combined']
        state_info = kwargs['state_info']
        
        state_1 = state_info['state_1']
        state_2 = state_info['state_2']
        state_3 = state_info['state_3']
        state_4 = state_info['state_4']
        
        # Detect variance regimes for logging
        bear_prob_threshold = kwargs['bear_prob_threshold']
        bull_prob_threshold = kwargs['bull_prob_threshold']
        low_variance = (bull_prob_combined > bull_prob_threshold).astype(bool)
        high_variance = (bear_prob > bear_prob_threshold).astype(bool)
        
        log_data = []
        prev_position = 0
        
        for i, idx in enumerate(common_idx):
            curr_pos = positions.iloc[i]
            
            # Determine which state
            if state_1.iloc[i]:
                state_label = 'State_1_Low_Var_Bull'
            elif state_2.iloc[i]:
                state_label = 'State_2_Low_Var_Bear'
            elif state_3.iloc[i]:
                state_label = 'State_3_High_Var_Bull'
            elif state_4.iloc[i]:
                state_label = 'State_4_High_Var_Bear'
            else:
                state_label = 'Unknown'
            
            action = self._determine_action(prev_position, curr_pos)
            
            log_data.append({
                'Date': idx,
                'Price': close.loc[idx],
                'Alpha_Signal': alpha_signals.iloc[i],
                'Bear_Prob': bear_prob.iloc[i],
                'Bull_Prob': bull_prob.iloc[i],
                'Bull_Combined_Prob': bull_prob_combined.iloc[i],
                'Low_Variance': low_variance.iloc[i],
                'High_Variance': high_variance.iloc[i],
                'State': state_label,
                'Position': curr_pos,
                'Action': action
            })
            
            prev_position = curr_pos
        
        return log_data

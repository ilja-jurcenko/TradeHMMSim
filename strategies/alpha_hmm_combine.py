"""
Alpha + HMM Combine strategy (4-state variance-trend logic).
"""

import pandas as pd
from typing import Dict, List, Any, Tuple
from .base import BaseStrategy


class AlphaHMMCombineStrategy(BaseStrategy):
    """
    Alpha + HMM Momentum Strategy.
    
    Combines Alpha model signals with HMM regime momentum:
      - Alpha: Trend direction (fast_MA > slow_MA = bullish)
      - HMM: Regime probability momentum and levels
    
    Entry Logic (BUY):
      - fast_MA > slow_MA (bullish trend)
      - P(Bull)_t - P(Bull)_{t-1} > 0.01 (positive bull momentum)
      - P(Bull) > 0.5 (bull regime confirmed)
    
    Exit Logic (SELL):
      - fast_MA < slow_MA (bearish trend), OR
      - P(Bear)_t - P(Bear)_{t-1} > 0.01 (positive bear momentum)
    
    Note: The original 4-state variance-trend logic is preserved in
    generate_positions_4state() method.
    """
    
    def __init__(self):
        super().__init__('alpha_hmm_combine')
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate positions using momentum-based logic.
        
        Entry (BUY):
          - fast_MA > slow_MA (alpha_signals > 0)
          - P(Bull)_t - P(Bull)_{t-1} > 0.01 (positive bull momentum)
          - P(Bull) > 0.5 (bull regime confirmed)
        
        Exit (SELL):
          - fast_MA < slow_MA (alpha_signals < 0), OR
          - P(Bear)_t - P(Bear)_{t-1} > 0.01 (positive bear momentum)
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals (>0 = bullish, <0 = bearish)
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Must contain: bear_prob, bull_prob_combined
            
        Returns:
        --------
        Tuple[pd.Series, Dict[str, Any]]
            (positions, state_info) - positions and momentum information
        """
        bear_prob = kwargs['bear_prob']
        bull_prob_combined = kwargs['bull_prob_combined']
        
        # Calculate probability momentum (change from previous period)
        bull_prob_momentum = bull_prob_combined.diff()
        bear_prob_momentum = bear_prob.diff()
        
        # Entry conditions - simplified to avoid late entries
        # BUY when: alpha is bullish AND (bull regime confirmed OR strong bull momentum)
        alpha_bullish = (alpha_signals > 0)
        bull_regime_confirmed = (bull_prob_combined > 0.50)  # Already in bull regime
        bull_momentum_spike = (bull_prob_momentum > 0.05)    # Transitioning to bull
        
        # Entry when alpha bullish AND (already bull OR becoming bull)
        buy_signal = alpha_bullish & (bull_regime_confirmed | bull_momentum_spike)
        
        # Exit conditions
        alpha_bearish = (alpha_signals < 0)
        bear_momentum_spike = (bear_prob_momentum > 0.1)  # Lowered from 0.15 to 0.01 for more responsive exits
        
        exit_signal = alpha_bearish | bear_momentum_spike
        
        # Initialize positions
        positions = pd.Series(0, index=common_idx, dtype=int)
        current_position = 0  # Start flat
        
        # State machine logic
        for i in range(len(common_idx)):
            # Entry logic: BUY when flat and all conditions met
            if current_position == 0 and buy_signal.iloc[i]:
                current_position = 1
            # Exit logic: SELL when long and exit conditions met
            elif current_position == 1 and exit_signal.iloc[i]:
                current_position = 0
            
            positions.iloc[i] = current_position
        
        # Store momentum information for logging/plotting
        state_info = {
            'bull_prob_momentum': bull_prob_momentum,
            'bear_prob_momentum': bear_prob_momentum,
            'buy_signal': buy_signal,
            'exit_signal': exit_signal,
            'alpha_bullish': alpha_bullish,
            'alpha_bearish': alpha_bearish,
            'bull_momentum_spike': bull_momentum_spike,
            'bear_momentum_spike': bear_momentum_spike,
            'bull_regime_confirmed': bull_regime_confirmed
        }
        
        # Print diagnostics
        print(f"  Buy signals triggered: {buy_signal.sum()} periods")
        print(f"  Exit signals triggered: {exit_signal.sum()} periods")
        print(f"  Bull momentum spikes (>0.01): {bull_momentum_spike.sum()} periods")
        print(f"  Bear momentum spikes (>0.01): {bear_momentum_spike.sum()} periods")
        print(f"  Time in market: {(positions == 1).sum() / len(positions) * 100:.1f}%")
        
        return positions, state_info
    
    def generate_positions_4state(self,
                                  alpha_signals: pd.Series,
                                  close: pd.Series,
                                  common_idx: pd.Index,
                                  **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        LEGACY: Generate positions using original 4-state variance-trend logic.
        
        This is the original implementation preserved for reference.
        Four States:
          State 1: Low variance + Bullish trend  → BUY
          State 2: Low variance + Bearish trend  → HOLD
          State 3: High variance + Bullish trend → HOLD
          State 4: High variance + Bearish trend → SELL
        
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
        Generate logging data for alpha+HMM momentum strategy.
        """
        bear_prob = kwargs['bear_prob']
        bull_prob = kwargs['bull_prob']
        bull_prob_combined = kwargs['bull_prob_combined']
        state_info = kwargs['state_info']
        
        # Extract momentum information
        bull_prob_momentum = state_info['bull_prob_momentum']
        bear_prob_momentum = state_info['bear_prob_momentum']
        buy_signal = state_info['buy_signal']
        exit_signal = state_info['exit_signal']
        
        log_data = []
        prev_position = 0
        
        for i, idx in enumerate(common_idx):
            curr_pos = positions.iloc[i]
            action = self._determine_action(prev_position, curr_pos)
            
            log_data.append({
                'Date': idx,
                'Price': close.loc[idx],
                'Alpha_Signal': alpha_signals.iloc[i],
                'Bear_Prob': bear_prob.iloc[i],
                'Bull_Prob': bull_prob.iloc[i],
                'Bull_Combined_Prob': bull_prob_combined.iloc[i],
                'Bull_Prob_Momentum': bull_prob_momentum.iloc[i],
                'Bear_Prob_Momentum': bear_prob_momentum.iloc[i],
                'Buy_Signal': buy_signal.iloc[i],
                'Exit_Signal': exit_signal.iloc[i],
                'Position': curr_pos,
                'Action': action
            })
            
            prev_position = curr_pos
        
        return log_data


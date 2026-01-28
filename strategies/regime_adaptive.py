"""
Regime-Adaptive Alpha strategy (switches between trend-following and mean-reversion).
"""

import pandas as pd
from typing import Dict, List, Any
from .base import BaseStrategy


class RegimeAdaptiveAlphaStrategy(BaseStrategy):
    """
    Regime-Adaptive Alpha: Switch strategies based on market regime.
    
    Strategy:
      - Bull/Neutral: Use trend-following alpha (standard alpha_model)
      - Bear: Use mean-reversion alpha (bear_alpha_model, typically Bollinger Bands)
    
    This adapts to market conditions: ride trends up, catch bounces down.
    """
    
    def __init__(self):
        super().__init__('regime_adaptive_alpha')
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> pd.Series:
        """
        Generate positions by switching between alpha models based on regime.
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Trend-following alpha model signals (for bull/neutral)
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
        **kwargs : dict
            Must contain: regime, bear_signals
            
        Returns:
        --------
        pd.Series
            Position signals (0 or 1)
        """
        regime = kwargs['regime']
        bear_signals = kwargs['bear_signals']
        
        # Initialize positions
        positions = pd.Series(0, index=common_idx)
        
        # Use trend-following in bull/neutral, mean-reversion in bear
        for i in range(len(common_idx)):
            idx = common_idx[i]
            current_regime = regime.loc[idx]
            
            if current_regime == 'bear':
                # Use bear market strategy (mean-reversion)
                positions.iloc[i] = bear_signals.iloc[i]
            else:
                # Use bull/neutral market strategy (trend-following)
                positions.iloc[i] = alpha_signals.iloc[i]
        
        # Print diagnostics
        trend_periods = (regime != 'bear').sum()
        mean_rev_periods = (regime == 'bear').sum()
        print(f"  Regime switches - Trend-following: {trend_periods} periods, "
              f"Mean-reversion: {mean_rev_periods} periods")
        
        return positions
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for regime-adaptive strategy.
        """
        regime = kwargs['regime']
        bear_signals = kwargs['bear_signals']
        
        log_data = []
        
        for i, idx in enumerate(common_idx):
            prev_pos = 0 if i == 0 else positions.iloc[i-1]
            curr_pos = positions.iloc[i]
            current_regime = regime.loc[idx]
            
            if current_regime == 'bear':
                active_model = 'Bear_Alpha'
            else:
                active_model = 'Bull_Alpha'
            
            action = self._determine_action(prev_pos, curr_pos)
            
            log_data.append({
                'Date': idx,
                'Price': close.loc[idx],
                'Alpha_Signal': alpha_signals.iloc[i],
                'Bear_Signal': bear_signals.iloc[i],
                'Regime': current_regime,
                'Active_Model': active_model,
                'Position': curr_pos,
                'Action': action
            })
        
        return log_data

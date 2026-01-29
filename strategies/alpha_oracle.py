"""
Alpha Oracle Strategy - ZigZag-based ideal timing labels.

This strategy identifies local minima and maxima based on a minimum price move threshold,
creating ideal BUY/SELL labels for timing evaluation. It does NOT use future information
beyond the pivot definition and is meant for benchmarking timing quality, not actual trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .base import BaseStrategy


class AlphaOracleStrategy(BaseStrategy):
    """
    Oracle strategy that generates ideal market timing labels using a ZigZag-like algorithm.
    
    This strategy identifies:
    - BUY signals at local minima (after price falls >= min_move% from prior high)
    - SELL signals at local maxima (after price rises >= min_move% from prior low)
    
    Signals alternate BUY → SELL → BUY, maintaining only one position at a time.
    
    This is NOT a tradable strategy - it's a timing benchmark to evaluate how well
    other strategies capture major market swings.
    """
    
    def __init__(self, min_move_pct: float = 0.05):
        """
        Initialize Alpha Oracle Strategy.
        
        Parameters:
        -----------
        min_move_pct : float
            Minimum percentage move (as decimal) to trigger a signal.
            Default 0.05 = 5% move required to identify a pivot.
        """
        super().__init__('alpha_oracle')
        self.min_move_pct = min_move_pct
    
    def generate_positions(self,
                          alpha_signals: pd.Series,
                          close: pd.Series,
                          common_idx: pd.Index,
                          **kwargs) -> pd.Series:
        """
        Generate ideal timing positions using ZigZag algorithm.
        
        This method ignores alpha_signals and generates positions purely from
        price action to identify local extrema.
        
        Parameters:
        -----------
        alpha_signals : pd.Series
            Alpha model signals (ignored for oracle)
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
            
        Returns:
        --------
        pd.Series
            Binary position signals (0 or 1) representing ideal timing
        """
        # Align close to common index
        close_aligned = close.loc[common_idx]
        
        # Generate zigzag signals
        signals = self._generate_zigzag_signals(close_aligned)
        
        # Convert signals to positions (1 between BUY and SELL)
        positions = self._signals_to_positions(signals)
        
        return positions
    
    def _generate_zigzag_signals(self, close: pd.Series) -> pd.Series:
        """
        Generate ZigZag BUY/SELL signals based on local extrema.
        
        Algorithm:
        - Start in 'searching_for_buy' state
        - Track last extreme price and its index
        - When searching for buy:
          - If price drops below extreme, update extreme
          - If price rises >= min_move% from extreme, signal BUY at extreme time
        - When searching for sell:
          - If price rises above extreme, update extreme
          - If price drops >= min_move% from extreme, signal SELL at extreme time
        
        Parameters:
        -----------
        close : pd.Series
            Close price series
            
        Returns:
        --------
        pd.Series
            Signal series: 1 = BUY, -1 = SELL, 0 = HOLD
        """
        if len(close) == 0:
            return pd.Series(0, index=close.index)
        
        signals = pd.Series(0, index=close.index, dtype=int)
        
        # State machine
        state = 'searching_for_buy'
        last_extreme_price = close.iloc[0]
        last_extreme_idx = 0
        
        for i in range(1, len(close)):
            current_price = close.iloc[i]
            
            if state == 'searching_for_buy':
                # Looking for a local minimum
                if current_price < last_extreme_price:
                    # New low found
                    last_extreme_price = current_price
                    last_extreme_idx = i
                elif current_price >= last_extreme_price * (1 + self.min_move_pct):
                    # Price rose enough from low - signal BUY at the low point
                    signals.iloc[last_extreme_idx] = 1
                    # Switch to searching for sell
                    state = 'searching_for_sell'
                    last_extreme_price = current_price
                    last_extreme_idx = i
            
            elif state == 'searching_for_sell':
                # Looking for a local maximum
                if current_price > last_extreme_price:
                    # New high found
                    last_extreme_price = current_price
                    last_extreme_idx = i
                elif current_price <= last_extreme_price * (1 - self.min_move_pct):
                    # Price dropped enough from high - signal SELL at the high point
                    signals.iloc[last_extreme_idx] = -1
                    # Switch to searching for buy
                    state = 'searching_for_buy'
                    last_extreme_price = current_price
                    last_extreme_idx = i
        
        return signals
    
    def _signals_to_positions(self, signals: pd.Series) -> pd.Series:
        """
        Convert BUY/SELL signals to position series.
        
        Position = 1 (long) from BUY signal until SELL signal.
        Position = 0 (cash) from SELL signal until BUY signal.
        
        Parameters:
        -----------
        signals : pd.Series
            Signal series: 1 = BUY, -1 = SELL, 0 = HOLD
            
        Returns:
        --------
        pd.Series
            Position series (0 or 1)
        """
        positions = pd.Series(0, index=signals.index, dtype=int)
        current_position = 0
        
        for i in range(len(signals)):
            signal = signals.iloc[i]
            
            if signal == 1:  # BUY signal
                current_position = 1
            elif signal == -1:  # SELL signal
                current_position = 0
            
            positions.iloc[i] = current_position
        
        return positions
    
    def generate_log_data(self,
                         positions: pd.Series,
                         close: pd.Series,
                         alpha_signals: pd.Series,
                         common_idx: pd.Index,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Generate logging data for alpha oracle strategy.
        
        Parameters:
        -----------
        positions : pd.Series
            Generated positions
        close : pd.Series
            Close price series
        alpha_signals : pd.Series
            Alpha model signals (not used by oracle)
        common_idx : pd.Index
            Common index for alignment
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of log entries with timing labels
        """
        log_data = []
        
        # Regenerate signals to identify pivot points
        close_aligned = close.loc[common_idx]
        signals = self._generate_zigzag_signals(close_aligned)
        
        for i in range(len(positions)):
            idx = positions.index[i]
            prev_pos = 0 if i == 0 else positions.iloc[i-1]
            curr_pos = positions.iloc[i]
            action = self._determine_action(prev_pos, curr_pos)
            signal = signals.loc[idx]
            
            # Calculate price change from last extreme
            price_change_pct = 0.0
            if i > 0:
                price_change_pct = (close.loc[idx] / close.iloc[i-1] - 1) * 100
            
            log_data.append({
                'Date': idx,
                'Price': close.loc[idx],
                'Signal': 'BUY' if signal == 1 else ('SELL' if signal == -1 else 'HOLD'),
                'Position': curr_pos,
                'Action': action,
                'Price_Change_Pct': price_change_pct,
                'Min_Move_Threshold': self.min_move_pct * 100
            })
        
        return log_data
    
    def get_pivot_info(self, close: pd.Series, common_idx: pd.Index) -> Dict[str, Any]:
        """
        Get detailed information about identified pivots.
        
        Parameters:
        -----------
        close : pd.Series
            Close price series
        common_idx : pd.Index
            Common index for alignment
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing pivot analysis:
            - num_pivots: Total number of pivots
            - num_buys: Number of BUY signals
            - num_sells: Number of SELL signals
            - avg_swing_pct: Average swing size
            - pivot_dates: List of pivot dates with types
        """
        close_aligned = close.loc[common_idx]
        signals = self._generate_zigzag_signals(close_aligned)
        
        buys = signals[signals == 1]
        sells = signals[signals == -1]
        
        # Calculate swing sizes
        swings = []
        buy_indices = buys.index
        sell_indices = sells.index
        
        all_pivots = []
        for date in buy_indices:
            all_pivots.append({'date': date, 'type': 'BUY', 'price': close.loc[date]})
        for date in sell_indices:
            all_pivots.append({'date': date, 'type': 'SELL', 'price': close.loc[date]})
        
        # Sort by date
        all_pivots.sort(key=lambda x: x['date'])
        
        # Calculate swing sizes
        for i in range(1, len(all_pivots)):
            prev_price = all_pivots[i-1]['price']
            curr_price = all_pivots[i]['price']
            swing_pct = abs((curr_price / prev_price - 1) * 100)
            swings.append(swing_pct)
        
        avg_swing = np.mean(swings) if swings else 0.0
        
        return {
            'num_pivots': len(all_pivots),
            'num_buys': len(buys),
            'num_sells': len(sells),
            'avg_swing_pct': avg_swing,
            'pivot_dates': all_pivots,
            'min_move_threshold_pct': self.min_move_pct * 100
        }

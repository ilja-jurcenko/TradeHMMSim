"""
Unit tests for Backtest Engine Finite State Machine behavior.
Tests that position transitions follow FSM rules.
"""

import unittest
import pandas as pd
import numpy as np
from backtest import BacktestEngine
from alpha_models.base import AlphaModel


class TestSignalAlphaModel(AlphaModel):
    """Test alpha model that generates predefined signal patterns."""
    
    def __init__(self, signal_pattern):
        """
        Initialize with custom signal pattern.
        
        Parameters:
        -----------
        signal_pattern : list
            List of signals (1=long, 0=flat, -1=short)
        """
        super().__init__(short_window=10, long_window=20)
        self.signal_pattern = signal_pattern
        
    def calculate_indicators(self, close: pd.Series):
        """Not needed for test model."""
        return close, close
        
    def generate_signals(self, close: pd.Series) -> pd.Series:
        """Generate test signals from pattern."""
        # Pad pattern with last value to match close length
        if len(self.signal_pattern) < len(close):
            pad_value = self.signal_pattern[-1] if self.signal_pattern else 0
            padded_pattern = self.signal_pattern + [pad_value] * (len(close) - len(self.signal_pattern))
        else:
            padded_pattern = self.signal_pattern[:len(close)]
        
        signals = pd.Series(padded_pattern, index=close.index)
        return signals
    
    def get_name(self) -> str:
        """Get model name."""
        return "TestSignalModel"


class TestBacktestEngineFSM(unittest.TestCase):
    """Test cases for FSM behavior in backtest engine."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        self.prices = pd.Series(
            np.linspace(100, 110, 50) + np.random.randn(50) * 0.5,
            index=dates,
            name='close'
        )
        
    def test_invalid_sell_from_flat(self):
        """Test that SELL signal is ignored when FLAT (no position to sell)."""
        # Pattern: Try to SELL when FLAT
        # Signal: 0 (flat) for several periods, should stay FLAT
        pattern = [0, 0, 0, 0, 0]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # All positions should be FLAT (0)
        self.assertTrue((positions == 0).all(), 
                       "Should stay FLAT when no position to sell")
        
    def test_invalid_buy_when_long(self):
        """Test that BUY signal is ignored when already LONG."""
        # Pattern: BUY once, then multiple BUY signals (should be ignored)
        pattern = [1, 1, 1, 1, 1]  # Buy signal repeated
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Should go LONG on first signal and stay LONG
        self.assertEqual(positions.iloc[0], 1, "Should enter LONG on first BUY")
        self.assertTrue((positions == 1).all(), 
                       "Should stay LONG, ignore redundant BUY signals")
        
    def test_invalid_short_when_short(self):
        """Test that SHORT signal is ignored when already SHORT."""
        # Pattern: SHORT once, then multiple SHORT signals (should be ignored)
        pattern = [-1, -1, -1, -1, -1]  # Short signal repeated
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Should go SHORT on first signal and stay SHORT
        self.assertEqual(positions.iloc[0], -1, "Should enter SHORT on first signal")
        self.assertTrue((positions == -1).all(), 
                       "Should stay SHORT, ignore redundant SHORT signals")
        
    def test_valid_flat_to_long(self):
        """Test valid transition: FLAT -> LONG."""
        # Pattern: FLAT, then BUY
        pattern = [0, 0, 1, 1, 1]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Should be FLAT then LONG
        self.assertEqual(positions.iloc[0], 0, "Should start FLAT")
        self.assertEqual(positions.iloc[1], 0, "Should stay FLAT")
        self.assertEqual(positions.iloc[2], 1, "Should go LONG on BUY signal")
        self.assertEqual(positions.iloc[3], 1, "Should stay LONG")
        
    def test_valid_flat_to_short(self):
        """Test valid transition: FLAT -> SHORT."""
        # Pattern: FLAT, then SHORT
        pattern = [0, 0, -1, -1, -1]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Should be FLAT then SHORT
        self.assertEqual(positions.iloc[0], 0, "Should start FLAT")
        self.assertEqual(positions.iloc[1], 0, "Should stay FLAT")
        self.assertEqual(positions.iloc[2], -1, "Should go SHORT on SHORT signal")
        self.assertEqual(positions.iloc[3], -1, "Should stay SHORT")
        
    def test_valid_long_to_flat(self):
        """Test valid transition: LONG -> FLAT (SELL)."""
        # Pattern: BUY, hold, then SELL
        pattern = [1, 1, 0, 0, 0]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Should go LONG then FLAT
        self.assertEqual(positions.iloc[0], 1, "Should enter LONG")
        self.assertEqual(positions.iloc[1], 1, "Should stay LONG")
        self.assertEqual(positions.iloc[2], 0, "Should go FLAT on SELL signal")
        self.assertEqual(positions.iloc[3], 0, "Should stay FLAT")
        
    def test_valid_short_to_flat(self):
        """Test valid transition: SHORT -> FLAT (COVER)."""
        # Pattern: SHORT, hold, then COVER
        pattern = [-1, -1, 0, 0, 0]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Should go SHORT then FLAT
        self.assertEqual(positions.iloc[0], -1, "Should enter SHORT")
        self.assertEqual(positions.iloc[1], -1, "Should stay SHORT")
        self.assertEqual(positions.iloc[2], 0, "Should go FLAT on COVER signal")
        self.assertEqual(positions.iloc[3], 0, "Should stay FLAT")
        
    def test_valid_long_to_short_direct(self):
        """Test valid transition: LONG -> SHORT (direct flip)."""
        # Pattern: BUY, then direct SHORT signal
        pattern = [1, 1, -1, -1, -1]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Should go LONG then SHORT
        self.assertEqual(positions.iloc[0], 1, "Should enter LONG")
        self.assertEqual(positions.iloc[1], 1, "Should stay LONG")
        self.assertEqual(positions.iloc[2], -1, "Should flip to SHORT")
        self.assertEqual(positions.iloc[3], -1, "Should stay SHORT")
        
    def test_valid_short_to_long_direct(self):
        """Test valid transition: SHORT -> LONG (direct flip)."""
        # Pattern: SHORT, then direct BUY signal
        pattern = [-1, -1, 1, 1, 1]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Should go SHORT then LONG
        self.assertEqual(positions.iloc[0], -1, "Should enter SHORT")
        self.assertEqual(positions.iloc[1], -1, "Should stay SHORT")
        self.assertEqual(positions.iloc[2], 1, "Should flip to LONG")
        self.assertEqual(positions.iloc[3], 1, "Should stay LONG")
        
    def test_complex_valid_sequence(self):
        """Test complex sequence of valid transitions."""
        # Pattern: FLAT -> LONG -> FLAT -> SHORT -> FLAT -> LONG
        pattern = [0, 1, 1, 0, -1, -1, 0, 1, 1, 0]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        expected = [0, 1, 1, 0, -1, -1, 0, 1, 1, 0]
        for i, exp in enumerate(expected):
            self.assertEqual(positions.iloc[i], exp,
                           f"Position at index {i} should be {exp}")
            
    def test_invalid_sequence_ignored(self):
        """Test that invalid signals in sequence are properly ignored."""
        # Pattern with invalid transitions that should be ignored
        # 0, 0 (valid: stay flat)
        # 1 (valid: flat->long)
        # 1 (invalid: already long, ignored)
        # 0 (valid: long->flat)
        # 0 (invalid: already flat, ignored)
        # -1 (valid: flat->short)
        # -1 (invalid: already short, ignored)
        # 0 (valid: short->flat)
        pattern = [0, 0, 1, 1, 0, 0, -1, -1, 0]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        expected = [0, 0, 1, 1, 0, 0, -1, -1, 0]
        for i, exp in enumerate(expected):
            self.assertEqual(positions.iloc[i], exp,
                           f"Position at index {i} should be {exp}")
            
    def test_trade_counting_with_fsm(self):
        """Test that trade counting works correctly with FSM."""
        # Pattern: Two complete round trips
        # FLAT -> LONG -> FLAT -> SHORT -> FLAT
        pattern = [0, 1, 1, 0, 0, -1, -1, 0, 0]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        num_trades = results['num_trades']
        
        # Should count 2 trades (1 long round trip + 1 short round trip)
        self.assertEqual(num_trades, 2, "Should count 2 complete trades")
        
    def test_fsm_with_alternating_signals(self):
        """Test FSM with rapidly alternating signals."""
        # Alternating pattern: should create frequent position changes
        pattern = [1, -1, 1, -1, 1, -1, 0, 1, 0, -1, 0]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Verify each transition is valid
        for i in range(1, min(len(positions), len(pattern))):
            prev_pos = positions.iloc[i-1]
            curr_pos = positions.iloc[i]
            
            # Check FSM rules
            if prev_pos == 0:  # FLAT
                self.assertIn(curr_pos, [0, 1, -1], "From FLAT: can go to any state")
            elif prev_pos == 1:  # LONG
                self.assertIn(curr_pos, [1, 0, -1], "From LONG: can stay, sell, or flip short")
            elif prev_pos == -1:  # SHORT
                self.assertIn(curr_pos, [-1, 0, 1], "From SHORT: can stay, cover, or flip long")
                
    def test_time_in_market_with_fsm(self):
        """Test time in market calculation with FSM-validated positions."""
        # Pattern: 40% long, 40% short, 20% flat
        pattern = [1] * 20 + [-1] * 20 + [0] * 10
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        time_in_market = results['time_in_market']
        
        # Should be 80% (40% long + 40% short)
        self.assertGreater(time_in_market, 0.75)
        self.assertLess(time_in_market, 0.85)
        
    def test_fsm_state_persistence(self):
        """Test that FSM correctly maintains state across signals."""
        # Pattern designed to test state persistence
        # Enter long, try to buy again (ignored), sell, try to sell again (ignored)
        pattern = [1, 1, 1, 0, 0, 0, -1, -1, 0, 0]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Expected: long at indices 0-2, flat at 3-5, short at 6-7, flat at 8-9
        self.assertEqual(positions.iloc[0], 1, "Should enter LONG")
        self.assertEqual(positions.iloc[1], 1, "Should stay LONG (ignore duplicate buy)")
        self.assertEqual(positions.iloc[2], 1, "Should stay LONG")
        self.assertEqual(positions.iloc[3], 0, "Should go FLAT")
        self.assertEqual(positions.iloc[4], 0, "Should stay FLAT (ignore duplicate sell)")
        self.assertEqual(positions.iloc[5], 0, "Should stay FLAT")
        self.assertEqual(positions.iloc[6], -1, "Should go SHORT")
        self.assertEqual(positions.iloc[7], -1, "Should stay SHORT (ignore duplicate short)")
        self.assertEqual(positions.iloc[8], 0, "Should go FLAT")
        self.assertEqual(positions.iloc[9], 0, "Should stay FLAT")
        
    def test_position_values_valid(self):
        """Test that all position values are valid FSM states."""
        # Random-ish pattern
        pattern = [0, 1, -1, 1, 0, -1, 1, -1, 0, 1]
        alpha_model = TestSignalAlphaModel(pattern)
        
        engine = BacktestEngine(self.prices, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # All positions must be -1, 0, or 1
        valid_states = {-1, 0, 1}
        for pos in positions.unique():
            self.assertIn(pos, valid_states,
                         f"Position {pos} is not a valid FSM state")
            
    def test_no_position_at_start(self):
        """Test that engine always starts in FLAT state."""
        patterns = [
            [1, 1, 1],  # Starts with BUY
            [-1, -1, -1],  # Starts with SHORT
            [0, 0, 0],  # Starts with FLAT
        ]
        
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                alpha_model = TestSignalAlphaModel(pattern)
                engine = BacktestEngine(self.prices, alpha_model)
                results = engine.run(strategy_mode='alpha_only')
                
                positions = results['positions']
                first_pos = positions.iloc[0]
                
                # First position depends on first signal
                if pattern[0] == 1:
                    self.assertEqual(first_pos, 1, "Should enter LONG on first BUY signal")
                elif pattern[0] == -1:
                    self.assertEqual(first_pos, -1, "Should enter SHORT on first SHORT signal")
                else:
                    self.assertEqual(first_pos, 0, "Should stay FLAT on first FLAT signal")


if __name__ == '__main__':
    unittest.main()

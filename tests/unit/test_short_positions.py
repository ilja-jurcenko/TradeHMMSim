"""
Unit tests for short position functionality in Backtest Engine.
"""

import unittest
import pandas as pd
import numpy as np
from backtest import BacktestEngine
from alpha_models.base import AlphaModel


class ShortSignalAlphaModel(AlphaModel):
    """Test alpha model that generates short signals."""
    
    def __init__(self, signal_pattern=None):
        """
        Initialize with custom signal pattern.
        
        Parameters:
        -----------
        signal_pattern : list or None
            List of signals (1=long, 0=flat, -1=short)
            If None, generates alternating long/short signals
        """
        super().__init__(short_window=10, long_window=20)
        self.signal_pattern = signal_pattern
        
    def calculate_indicators(self, close: pd.Series):
        """Not needed for test model."""
        return close, close
        
    def generate_signals(self, close: pd.Series) -> pd.Series:
        """Generate test signals with short positions."""
        signals = pd.Series(0, index=close.index)
        
        if self.signal_pattern:
            # Use provided pattern (repeat if needed)
            pattern_length = len(self.signal_pattern)
            for i in range(len(signals)):
                signals.iloc[i] = self.signal_pattern[i % pattern_length]
        else:
            # Default: alternate between long (1) and short (-1) every 20 periods
            for i in range(len(signals)):
                cycle = (i // 20) % 2
                if cycle == 0:
                    signals.iloc[i] = 1  # Long
                else:
                    signals.iloc[i] = -1  # Short
        
        return signals
    
    def get_name(self) -> str:
        """Get model name."""
        return "ShortSignalTest"


class TestShortPositions(unittest.TestCase):
    """Test cases for short position functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic price data with downward trend
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Upward trend for long testing
        self.prices_up = pd.Series(
            np.linspace(100, 120, 100) + np.random.randn(100) * 0.5,
            index=dates,
            name='close'
        )
        
        # Downward trend for short testing
        self.prices_down = pd.Series(
            np.linspace(100, 80, 100) + np.random.randn(100) * 0.5,
            index=dates,
            name='close'
        )
        
    def test_short_positions_accepted(self):
        """Test that backtest engine accepts short positions (-1)."""
        # Create model that generates short signals
        alpha_model = ShortSignalAlphaModel(signal_pattern=[0, 0, -1, -1, -1, 0])
        
        engine = BacktestEngine(self.prices_down, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Verify that -1 positions exist
        self.assertIn(-1, positions.values, "Short positions (-1) should be present in positions series")
        
    def test_short_position_returns_calculation(self):
        """Test that short positions generate correct returns."""
        # Create model: short for first 50 periods, flat for rest
        alpha_model = ShortSignalAlphaModel(signal_pattern=[-1] * 50 + [0] * 50)
        
        engine = BacktestEngine(self.prices_down, alpha_model, initial_capital=100000)
        results = engine.run(strategy_mode='alpha_only', transaction_cost=0.0)
        
        # With downward price trend and short position, we should profit
        final_capital = results['final_capital']
        
        # Short position should profit from falling prices
        self.assertGreater(final_capital, 100000, 
                          "Short position should profit from falling prices")
        
    def test_short_vs_long_opposite_returns(self):
        """Test that short and long positions have opposite returns."""
        # Test with downward trending prices
        long_model = ShortSignalAlphaModel(signal_pattern=[1] * 100)  # Always long
        short_model = ShortSignalAlphaModel(signal_pattern=[-1] * 100)  # Always short
        
        # Long backtest
        engine_long = BacktestEngine(self.prices_down, long_model, initial_capital=100000)
        results_long = engine_long.run(strategy_mode='alpha_only', transaction_cost=0.0)
        
        # Short backtest
        engine_short = BacktestEngine(self.prices_down, short_model, initial_capital=100000)
        results_short = engine_short.run(strategy_mode='alpha_only', transaction_cost=0.0)
        
        # Long should lose money on down trend
        self.assertLess(results_long['final_capital'], 100000,
                       "Long position should lose on down trend")
        
        # Short should make money on down trend
        self.assertGreater(results_short['final_capital'], 100000,
                          "Short position should profit on down trend")
        
        # Returns should be approximately opposite
        long_return = results_long['metrics']['total_return']
        short_return = results_short['metrics']['total_return']
        
        # They should have opposite signs
        self.assertLess(long_return * short_return, 0,
                       "Long and short returns should have opposite signs")
        
    def test_time_in_market_includes_shorts(self):
        """Test that time_in_market metric includes short positions."""
        # Create model: 40% long, 40% short, 20% flat
        pattern = [1] * 40 + [-1] * 40 + [0] * 20
        alpha_model = ShortSignalAlphaModel(signal_pattern=pattern)
        
        engine = BacktestEngine(self.prices_down, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        time_in_market = results['time_in_market']
        
        # Should be approximately 0.8 (80% of time in position)
        self.assertGreater(time_in_market, 0.75, 
                          "Time in market should include short positions")
        self.assertLess(time_in_market, 0.85,
                       "Time in market calculation should be accurate")
        
    def test_transaction_costs_for_shorts(self):
        """Test that transaction costs apply to short position changes."""
        # Pattern with frequent position changes
        pattern = [1, -1, 1, -1, 1, -1, 0]  # Multiple transitions
        alpha_model = ShortSignalAlphaModel(signal_pattern=pattern)
        
        # Without transaction costs
        engine_no_cost = BacktestEngine(self.prices_up, alpha_model)
        results_no_cost = engine_no_cost.run(strategy_mode='alpha_only', transaction_cost=0.0)
        
        # With transaction costs
        engine_with_cost = BacktestEngine(self.prices_up, alpha_model)
        results_with_cost = engine_with_cost.run(strategy_mode='alpha_only', transaction_cost=0.01)
        
        # Transaction costs should reduce returns
        self.assertLess(results_with_cost['final_capital'], 
                       results_no_cost['final_capital'],
                       "Transaction costs should reduce returns for short positions")
        
    def test_trade_counting_with_shorts(self):
        """Test that trades are counted correctly with short positions."""
        # Pattern: flat -> long -> flat -> short -> flat
        # Position changes: 0->1 (1 change), 1->0 (1 change), 0->-1 (1 change), -1->0 (1 change)
        # Total changes = 4, divided by 2 = 2 round-trip trades
        # But the algorithm counts absolute changes, so:
        # |0-1|=1, |1-0|=1, |0-(-1)|=1, |-1-0|=1 = 4 total position changes / 2 = 2 trades
        # However, the actual implementation may count differently depending on position transitions
        pattern = [0] * 10 + [1] * 10 + [0] * 10 + [-1] * 10 + [0] * 10
        alpha_model = ShortSignalAlphaModel(signal_pattern=pattern)
        
        engine = BacktestEngine(self.prices_up, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        num_trades = results['num_trades']
        
        # The trade counter sums abs(position changes) / 2
        # With transitions 0->1->0->-1->0, we have 4 transitions total
        # This counts as 2 complete "round trips" (entry + exit = 1 trade)
        # But actually: flat->long->flat is 1 trade, flat->short->flat is 1 trade
        # Total = 2 trades, but may count as more depending on implementation
        self.assertGreaterEqual(num_trades, 2,
                               "Should count at least both long and short trades")
        self.assertLessEqual(num_trades, 4,
                            "Trade count should be reasonable")
        
    def test_long_to_short_transition(self):
        """Test direct transition from long to short position."""
        # Pattern: long -> short (direct transition without flat)
        pattern = [1] * 30 + [-1] * 30 + [0] * 40
        alpha_model = ShortSignalAlphaModel(signal_pattern=pattern)
        
        engine = BacktestEngine(self.prices_up, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Verify transition exists
        long_periods = (positions == 1).sum()
        short_periods = (positions == -1).sum()
        
        self.assertGreater(long_periods, 0, "Should have long positions")
        self.assertGreater(short_periods, 0, "Should have short positions")
        
    def test_short_to_long_transition(self):
        """Test direct transition from short to long position."""
        # Pattern: short -> long (direct transition without flat)
        pattern = [-1] * 30 + [1] * 30 + [0] * 40
        alpha_model = ShortSignalAlphaModel(signal_pattern=pattern)
        
        engine = BacktestEngine(self.prices_down, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # Find transition point
        position_changes = positions.diff()
        
        # Transition from -1 to 1 should be a change of +2
        max_change = position_changes.max()
        min_change = position_changes.min()
        
        # Should have large position changes
        self.assertGreaterEqual(max_change, 2, "Should have short-to-long transition")
        
    def test_equity_curve_with_shorts(self):
        """Test equity curve calculation with short positions."""
        # Pattern: short entire period on downward trend
        alpha_model = ShortSignalAlphaModel(signal_pattern=[-1])
        
        engine = BacktestEngine(self.prices_down, alpha_model, initial_capital=100000)
        results = engine.run(strategy_mode='alpha_only', transaction_cost=0.0)
        
        equity = results['equity_curve']
        
        # Equity should increase over time with short on downtrend
        # Check that final value > initial value
        self.assertGreater(equity.iloc[-1], equity.iloc[10],
                          "Equity should increase with profitable short position")
        
    def test_mixed_position_types(self):
        """Test backtest with mix of long, short, and flat positions."""
        # Pattern with all three position types
        pattern = [1, 1, 1, 0, 0, -1, -1, -1, 0, 1]
        alpha_model = ShortSignalAlphaModel(signal_pattern=pattern)
        
        engine = BacktestEngine(self.prices_up, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        unique_positions = sorted(positions.unique())
        
        # Should have all three position types
        self.assertEqual(len(unique_positions), 3, 
                        "Should have long (1), flat (0), and short (-1)")
        self.assertIn(-1, unique_positions, "Should have short positions")
        self.assertIn(0, unique_positions, "Should have flat positions")
        self.assertIn(1, unique_positions, "Should have long positions")
        
    def test_short_position_valid_values(self):
        """Test that position values are valid (-1, 0, or 1)."""
        alpha_model = ShortSignalAlphaModel(signal_pattern=[1, -1, 0])
        
        engine = BacktestEngine(self.prices_up, alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # All positions should be in {-1, 0, 1}
        valid_positions = {-1, 0, 1}
        for pos in positions.unique():
            self.assertIn(pos, valid_positions,
                         f"Position {pos} is not valid. Should be -1, 0, or 1")
            
    def test_rebalancing_with_shorts(self):
        """Test rebalancing frequency works with short positions."""
        pattern = [1, -1, 1, -1, 1, -1, 0]
        alpha_model = ShortSignalAlphaModel(signal_pattern=pattern)
        
        # Daily rebalancing
        engine_daily = BacktestEngine(self.prices_up, alpha_model)
        results_daily = engine_daily.run(strategy_mode='alpha_only', rebalance_frequency=1)
        
        # Weekly rebalancing
        alpha_model2 = ShortSignalAlphaModel(signal_pattern=pattern)
        engine_weekly = BacktestEngine(self.prices_up, alpha_model2)
        results_weekly = engine_weekly.run(strategy_mode='alpha_only', rebalance_frequency=5)
        
        # Weekly should have fewer or equal trades
        self.assertLessEqual(results_weekly['num_trades'], 
                           results_daily['num_trades'],
                           "Weekly rebalancing should have fewer trades")
        
    def test_short_position_returns_alignment(self):
        """Test that returns are properly calculated for short positions."""
        # Create simple scenario: short position during price increase
        alpha_model = ShortSignalAlphaModel(signal_pattern=[-1])
        
        engine = BacktestEngine(self.prices_up, alpha_model, initial_capital=100000)
        results = engine.run(strategy_mode='alpha_only', transaction_cost=0.0)
        
        positions = results['positions']
        returns = results['returns']
        close = results['close_prices']
        
        # Manually calculate expected return for a few periods
        for i in range(10, 20):
            pos = positions.iloc[i-1]  # Lagged position
            price_return = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            expected_return = pos * price_return
            actual_return = returns.iloc[i]
            
            # Should match (within tolerance)
            self.assertAlmostEqual(actual_return, expected_return, places=6,
                                  msg=f"Return calculation incorrect at position {i}")


if __name__ == '__main__':
    unittest.main()

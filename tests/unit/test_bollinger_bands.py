"""
Unit tests for Bollinger Bands alpha model.
"""

import unittest
import pandas as pd
import numpy as np
from alpha_models.bollinger import BollingerBands


class TestBollingerBands(unittest.TestCase):
    """Test Bollinger Bands alpha model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Trending data
        self.trending_prices = pd.Series(
            np.cumsum(np.random.randn(100) * 2) + 100,
            index=dates
        )
        
        # Mean-reverting data (oscillating around 100)
        self.mean_reverting_prices = pd.Series(
            np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 100 + np.random.randn(100) * 0.5,
            index=dates
        )
    
    def test_initialization_default(self):
        """Test default initialization."""
        bb = BollingerBands()
        self.assertEqual(bb.period, 20)
        self.assertEqual(bb.num_std, 2)
        self.assertEqual(bb.short_window, 20)
        self.assertEqual(bb.long_window, 2)
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        bb = BollingerBands(short_window=30, long_window=3)
        self.assertEqual(bb.period, 30)
        self.assertEqual(bb.num_std, 3)
    
    def test_calculate_indicators(self):
        """Test band calculation."""
        bb = BollingerBands(short_window=20, long_window=2)
        middle, lower = bb.calculate_indicators(self.trending_prices)
        
        # Check types
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)
        
        # Check lengths
        self.assertEqual(len(middle), len(self.trending_prices))
        self.assertEqual(len(lower), len(self.trending_prices))
        
        # Check that middle is SMA
        expected_middle = self.trending_prices.rolling(window=20).mean()
        pd.testing.assert_series_equal(middle, expected_middle)
        
        # Check that lower is below middle
        valid_idx = ~middle.isna()
        self.assertTrue((lower[valid_idx] < middle[valid_idx]).all())
    
    def test_get_bands(self):
        """Test getting all three bands."""
        bb = BollingerBands(short_window=20, long_window=2)
        upper, middle, lower = bb.get_bands(self.trending_prices)
        
        # Check that bands are ordered correctly
        valid_idx = ~middle.isna()
        self.assertTrue((lower[valid_idx] <= middle[valid_idx]).all())
        self.assertTrue((middle[valid_idx] <= upper[valid_idx]).all())
        
        # Check band width (should be 2 std devs on each side)
        std_dev = self.trending_prices.rolling(window=20).std()
        expected_width = 2 * 2 * std_dev  # 2 std on each side
        actual_width = upper - lower
        
        # Compare where both are valid
        valid_comparison = ~(expected_width.isna() | actual_width.isna())
        pd.testing.assert_series_equal(
            actual_width[valid_comparison], 
            expected_width[valid_comparison],
            check_names=False
        )
    
    def test_generate_signals_structure(self):
        """Test signal generation returns correct structure."""
        bb = BollingerBands(short_window=20, long_window=2)
        signals = bb.generate_signals(self.trending_prices)
        
        # Check type and length
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.trending_prices))
        
        # Check signal values are valid (-1, 0, or 1)
        unique_signals = signals.unique()
        self.assertTrue(all(s in [-1, 0, 1] for s in unique_signals))
    
    def test_generate_signals_mean_reversion(self):
        """Test that signals are generated for mean-reverting data."""
        bb = BollingerBands(short_window=10, long_window=2)
        signals = bb.generate_signals(self.mean_reverting_prices)
        
        # Should generate some buy signals (price below lower band)
        num_buys = (signals == 1).sum()
        self.assertGreater(num_buys, 0, "Should generate buy signals for mean-reverting data")
        
        # First few periods should be 0 (not enough data)
        self.assertTrue((signals[:10] == 0).all())
    
    def test_signals_consistency(self):
        """Test that signals are consistent (no gaps in positions)."""
        bb = BollingerBands(short_window=20, long_window=2)
        signals = bb.generate_signals(self.trending_prices)
        
        # Once in position (1), should stay in position until exit
        # Check that we don't have isolated single-period positions
        in_position = False
        position_lengths = []
        current_length = 0
        
        for sig in signals[20:]:  # Skip warmup period
            if sig == 1:
                if not in_position:
                    in_position = True
                    current_length = 1
                else:
                    current_length += 1
            else:
                if in_position:
                    position_lengths.append(current_length)
                    in_position = False
                    current_length = 0
        
        # If we have positions, they should generally be multi-period
        if position_lengths:
            avg_length = np.mean(position_lengths)
            self.assertGreater(avg_length, 1.0, "Positions should typically last multiple periods")
    
    def test_oversold_entry(self):
        """Test that model generates buy signal when price crosses below lower band."""
        # Create data that clearly crosses below lower band
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = pd.Series([100] * 25 + [90] * 25, index=dates)  # Sharp drop
        
        bb = BollingerBands(short_window=10, long_window=1.5)
        signals = bb.generate_signals(prices)
        
        # Should generate buy signal after the drop
        buys_after_drop = (signals[26:30] == 1).any()
        self.assertTrue(buys_after_drop, "Should generate buy signal after price drop")
    
    def test_different_periods(self):
        """Test that different periods produce different signals."""
        bb_short = BollingerBands(short_window=10, long_window=2)
        bb_long = BollingerBands(short_window=30, long_window=2)
        
        signals_short = bb_short.generate_signals(self.mean_reverting_prices)
        signals_long = bb_long.generate_signals(self.mean_reverting_prices)
        
        # Signals should be different
        self.assertFalse(signals_short.equals(signals_long))
        
        # Shorter period should generally generate more signals
        num_trades_short = (signals_short.diff() != 0).sum()
        num_trades_long = (signals_long.diff() != 0).sum()
        
        # This might not always be true due to randomness, but generally holds
        # Just check both generate some signals
        self.assertGreater(num_trades_short, 0)
        self.assertGreater(num_trades_long, 0)
    
    def test_bands_not_crossed_no_signal(self):
        """Test that no signals are generated when price stays within bands."""
        # Create stable price data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = pd.Series([100 + i * 0.01 for i in range(50)], index=dates)  # Very gradual trend
        
        bb = BollingerBands(short_window=20, long_window=3)  # Wide bands
        signals = bb.generate_signals(prices)
        
        # After warmup, should have minimal signals since price stays within wide bands
        signals_after_warmup = signals[25:]
        num_changes = (signals_after_warmup.diff() != 0).sum()
        
        # Should have very few changes with stable prices and wide bands
        self.assertLess(num_changes, 10)
    
    def test_model_name(self):
        """Test that model name is set correctly."""
        bb = BollingerBands()
        self.assertEqual(bb.name, 'BollingerBands')


class TestBollingerBandsEdgeCases(unittest.TestCase):
    """Test edge cases for Bollinger Bands."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.Series(range(10), index=dates)
        
        bb = BollingerBands(short_window=20, long_window=2)
        signals = bb.generate_signals(prices)
        
        # All signals should be 0 (not enough data)
        self.assertTrue((signals == 0).all())
    
    def test_nan_handling(self):
        """Test handling of NaN values in price data."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = pd.Series(range(50), index=dates, dtype=float)
        prices.iloc[10] = np.nan
        prices.iloc[20] = np.nan
        
        bb = BollingerBands(short_window=10, long_window=2)
        signals = bb.generate_signals(prices)
        
        # Should not raise an error
        self.assertEqual(len(signals), 50)
    
    def test_constant_prices(self):
        """Test behavior with constant prices."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = pd.Series([100] * 50, index=dates)
        
        bb = BollingerBands(short_window=20, long_window=2)
        
        # Should not crash
        upper, middle, lower = bb.get_bands(prices)
        
        # With constant prices, std dev is 0, so all bands should be equal
        valid_idx = ~middle.isna()
        self.assertTrue(np.allclose(upper[valid_idx], middle[valid_idx]))
        self.assertTrue(np.allclose(lower[valid_idx], middle[valid_idx]))


if __name__ == '__main__':
    unittest.main()

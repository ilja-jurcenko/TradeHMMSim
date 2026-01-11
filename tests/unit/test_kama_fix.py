"""
Unit tests for KAMA model fix.
"""

import unittest
import pandas as pd
import numpy as np
from alpha_models.kama import KAMA


class TestKAMAFix(unittest.TestCase):
    """Test KAMA model edge cases and fixes."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 * (1 + np.random.randn(100).cumsum() * 0.01)
        self.close = pd.Series(prices, index=dates)
    
    def test_kama_basic_calculation(self):
        """Test KAMA calculates without errors."""
        model = KAMA(short_window=10, long_window=50)
        short_kama, long_kama = model.calculate_indicators(self.close)
        
        self.assertEqual(len(short_kama), len(self.close))
        self.assertEqual(len(long_kama), len(self.close))
        self.assertTrue(isinstance(short_kama, pd.Series))
        self.assertTrue(isinstance(long_kama, pd.Series))
    
    def test_kama_with_short_data(self):
        """Test KAMA handles data shorter than window size."""
        # Create very short data (40 points)
        short_close = self.close[:40]
        
        # This should not raise an error even though window=50 > len=40
        model = KAMA(short_window=10, long_window=50)
        short_kama, long_kama = model.calculate_indicators(short_close)
        
        self.assertEqual(len(long_kama), len(short_close))
        # Long KAMA should be all NaN since window > data length
        self.assertTrue(long_kama.isna().all())
    
    def test_kama_float_conversion(self):
        """Test KAMA handles float conversions properly."""
        model = KAMA(short_window=10, long_window=30)
        short_kama, long_kama = model.calculate_indicators(self.close)
        
        # Check that values are proper floats, not Series
        for val in long_kama[30:35]:
            if not pd.isna(val):
                self.assertTrue(isinstance(val, (float, np.floating)))
    
    def test_kama_signal_generation(self):
        """Test KAMA generates signals without errors."""
        model = KAMA(short_window=10, long_window=50)
        signals = model.generate_signals(self.close)
        
        self.assertEqual(len(signals), len(self.close))
        self.assertTrue(isinstance(signals, pd.Series))
        # Signals should be in {-1, 0, 1}
        self.assertTrue(signals.isin([-1, 0, 1]).all())
    
    def test_kama_with_dataframe_column(self):
        """Test KAMA works when passed a DataFrame column."""
        # Simulate yfinance returning DataFrame
        df = pd.DataFrame({'Close': self.close})
        close_df = df['Close']
        
        model = KAMA(short_window=10, long_window=50)
        short_kama, long_kama = model.calculate_indicators(close_df)
        
        self.assertEqual(len(short_kama), len(close_df))
        self.assertEqual(len(long_kama), len(close_df))


if __name__ == '__main__':
    unittest.main()

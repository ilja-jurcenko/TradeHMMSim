"""
Unit tests for AlphaModels.
"""

import unittest
import pandas as pd
import numpy as np
from AlphaModels import SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA


class TestAlphaModels(unittest.TestCase):
    """Test cases for alpha models."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        trend = np.linspace(100, 150, 200)
        noise = np.random.randn(200) * 2
        self.prices = pd.Series(trend + noise, index=dates, name='close')
        
    def test_sma_calculation(self):
        """Test SMA calculation."""
        model = SMA(short_window=10, long_window=30)
        short_ma, long_ma = model.calculate_indicators(self.prices)
        
        # Check that MAs are calculated
        self.assertIsInstance(short_ma, pd.Series)
        self.assertIsInstance(long_ma, pd.Series)
        
        # Check lengths match
        self.assertEqual(len(short_ma), len(self.prices))
        self.assertEqual(len(long_ma), len(self.prices))
        
        # Check that short MA has fewer NaN at start
        self.assertEqual(short_ma.isna().sum(), 9)
        self.assertEqual(long_ma.isna().sum(), 29)
        
        # Check values are reasonable (should be within price range)
        self.assertTrue(short_ma.dropna().min() >= self.prices.min() * 0.95)
        self.assertTrue(short_ma.dropna().max() <= self.prices.max() * 1.05)
        
    def test_sma_signals(self):
        """Test SMA signal generation."""
        model = SMA(short_window=10, long_window=30)
        signals = model.generate_signals(self.prices)
        
        # Check signals are binary
        unique_vals = signals.unique()
        self.assertTrue(all(val in [0, 1] for val in unique_vals))
        
        # Check we have both long and short signals (for trending data)
        self.assertGreater(signals.sum(), 0)
        
    def test_ema_calculation(self):
        """Test EMA calculation."""
        model = EMA(short_window=10, long_window=30)
        short_ema, long_ema = model.calculate_indicators(self.prices)
        
        self.assertIsInstance(short_ema, pd.Series)
        self.assertIsInstance(long_ema, pd.Series)
        self.assertEqual(len(short_ema), len(self.prices))
        
    def test_wma_calculation(self):
        """Test WMA calculation."""
        model = WMA(short_window=10, long_window=30)
        short_wma, long_wma = model.calculate_indicators(self.prices)
        
        self.assertIsInstance(short_wma, pd.Series)
        self.assertIsInstance(long_wma, pd.Series)
        self.assertEqual(len(short_wma), len(self.prices))
        
    def test_hma_calculation(self):
        """Test HMA calculation."""
        model = HMA(short_window=10, long_window=30)
        short_hma, long_hma = model.calculate_indicators(self.prices)
        
        self.assertIsInstance(short_hma, pd.Series)
        self.assertIsInstance(long_hma, pd.Series)
        self.assertEqual(len(short_hma), len(self.prices))
        
    def test_kama_calculation(self):
        """Test KAMA calculation."""
        model = KAMA(short_window=10, long_window=30)
        short_kama, long_kama = model.calculate_indicators(self.prices)
        
        self.assertIsInstance(short_kama, pd.Series)
        self.assertIsInstance(long_kama, pd.Series)
        self.assertEqual(len(short_kama), len(self.prices))
        
    def test_tema_calculation(self):
        """Test TEMA calculation."""
        model = TEMA(short_window=10, long_window=30)
        short_tema, long_tema = model.calculate_indicators(self.prices)
        
        self.assertIsInstance(short_tema, pd.Series)
        self.assertIsInstance(long_tema, pd.Series)
        self.assertEqual(len(short_tema), len(self.prices))
        
    def test_zlema_calculation(self):
        """Test ZLEMA calculation."""
        model = ZLEMA(short_window=10, long_window=30)
        short_zlema, long_zlema = model.calculate_indicators(self.prices)
        
        self.assertIsInstance(short_zlema, pd.Series)
        self.assertIsInstance(long_zlema, pd.Series)
        self.assertEqual(len(short_zlema), len(self.prices))
        
    def test_model_parameters(self):
        """Test model parameter storage."""
        model = SMA(short_window=10, long_window=30)
        params = model.get_parameters()
        
        self.assertEqual(params['short_window'], 10)
        self.assertEqual(params['long_window'], 30)
        self.assertEqual(params['model_type'], 'SMA')
        
    def test_signal_consistency(self):
        """Test that signals are consistent across calls."""
        model = SMA(short_window=10, long_window=30)
        signals1 = model.generate_signals(self.prices)
        signals2 = model.generate_signals(self.prices)
        
        pd.testing.assert_series_equal(signals1, signals2)
        
    def test_crossover_detection(self):
        """Test that crossovers generate signal changes."""
        # Create data with clear crossover
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices_up = pd.Series(range(100, 200), index=dates[:100])
        
        model = SMA(short_window=5, long_window=20)
        signals = model.generate_signals(prices_up)
        
        # For uptrend, should eventually get long signal
        self.assertGreater(signals.iloc[-20:].sum(), 0)
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Very short data
        short_prices = self.prices.iloc[:10]
        model = SMA(short_window=5, long_window=20)
        
        # Should not crash
        signals = model.generate_signals(short_prices)
        self.assertEqual(len(signals), len(short_prices))
        
    def test_all_models_generate_signals(self):
        """Test that all models can generate signals."""
        models = [
            SMA(10, 30),
            EMA(10, 30),
            WMA(10, 30),
            HMA(10, 30),
            KAMA(10, 30),
            TEMA(10, 30),
            ZLEMA(10, 30)
        ]
        
        for model in models:
            with self.subTest(model=model.get_name()):
                signals = model.generate_signals(self.prices)
                self.assertEqual(len(signals), len(self.prices))
                self.assertTrue(all(val in [0, 1] for val in signals.unique()))


if __name__ == '__main__':
    unittest.main()

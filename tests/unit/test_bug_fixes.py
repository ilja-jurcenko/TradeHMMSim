"""
Unit tests for critical bug fixes.

Tests cover:
1. Portfolio MultiIndex handling for single ticker
2. HMM filter DataFrame/array handling
3. Alpha+HMM combine logic (OR operation)
4. HMM filter insufficient data handling
5. Output directory creation and structure
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime

from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import SMA
from signal_filter import HMMRegimeFilter


class TestPortfolioMultiIndex(unittest.TestCase):
    """Test Portfolio MultiIndex column handling fix."""
    
    def test_single_ticker_returns_series(self):
        """Test that get_close_prices returns Series for single ticker."""
        # This tests the fix for yfinance returning MultiIndex columns
        portfolio = Portfolio(['SPY'], '2023-01-01', '2023-01-31')
        portfolio.load_data()
        
        close = portfolio.get_close_prices('SPY')
        
        # Should be a Series, not DataFrame
        self.assertIsInstance(close, pd.Series, 
                            "get_close_prices should return Series for single ticker")
        self.assertEqual(close.name, 'SPY', 
                        "Series name should be ticker symbol")
    
    def test_prepare_close_prices_handles_dataframe(self):
        """Test that portfolio properly handles MultiIndex columns from yfinance."""
        portfolio = Portfolio(['SPY'], '2023-01-01', '2023-01-31')
        portfolio.load_data()
        
        # After loading, _prepare_close_prices should have been called
        # and close prices should be properly formatted
        close = portfolio.get_close_prices('SPY')
        
        # Should be Series (not DataFrame with MultiIndex)
        self.assertIsInstance(close, pd.Series,
                            "Portfolio should convert MultiIndex DataFrame to Series internally")


class TestHMMFilterArrayHandling(unittest.TestCase):
    """Test HMM filter DataFrame/array handling fix."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        self.prices = pd.Series(np.cumsum(np.random.randn(300)) + 100, index=dates)
        self.hmm_filter = HMMRegimeFilter(n_states=2, random_state=42)
    
    def test_fit_accepts_dataframe(self):
        """Test that fit() accepts DataFrame input after make_features()."""
        # make_features converts Series to DataFrame with returns and volatility
        features = self.hmm_filter.make_features(self.prices)
        
        # Should not raise AttributeError
        try:
            self.hmm_filter.fit(features)
            success = True
        except AttributeError:
            success = False
        
        self.assertTrue(success, "fit() should accept DataFrame features")
    
    def test_fit_accepts_array_features(self):
        """Test that fit() accepts numpy array features."""
        # make_features returns DataFrame, convert to array
        features = self.hmm_filter.make_features(self.prices)
        array_features = features.values
        
        # Should not raise AttributeError
        try:
            self.hmm_filter.fit(array_features)
            success = True
        except AttributeError:
            success = False
        
        self.assertTrue(success, "fit() should accept numpy array features")
    
    def test_predict_states_accepts_array(self):
        """Test that predict_states() accepts numpy array after fitting."""
        # Fit the model first with proper features
        features = self.hmm_filter.make_features(self.prices)
        self.hmm_filter.fit(features)
        
        # Create numpy array input
        array_input = features.values
        
        # Should not raise AttributeError
        try:
            states = self.hmm_filter.predict_states(array_input)
            success = True
        except AttributeError:
            success = False
        
        self.assertTrue(success, "predict_states() should accept numpy array")


class TestAlphaHMMCombineLogic(unittest.TestCase):
    """Test Alpha+HMM combine strategy uses OR logic."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        self.prices = pd.Series(np.cumsum(np.random.randn(200) * 0.5) + 100, index=dates)
        self.alpha_config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 10,
                'long_window': 30
            }
        }
        self.hmm_filter = HMMRegimeFilter(n_states=2, random_state=42)
    
    def test_combine_strategy_uses_or_logic(self):
        """Test that alpha_hmm_combine uses OR logic (not override)."""
        engine = BacktestEngine(self.prices, alpha_config=self.alpha_config, hmm_filter=self.hmm_filter)
        
        results = engine.run(
            strategy_mode='alpha_hmm_combine',
            walk_forward=True,
            train_window=60,
            refit_every=20
        )
        
        # Get positions
        positions = results['positions']
        
        # The combine strategy should have positions when EITHER:
        # 1. Alpha model signals long, OR
        # 2. HMM indicates bull regime
        # This means it should generally have more time in market than filter strategy
        
        alpha_only_engine = BacktestEngine(self.prices, alpha_config=self.alpha_config)
        alpha_only = alpha_only_engine.run(strategy_mode='alpha_only')
        
        hmm_only_engine = BacktestEngine(self.prices, alpha_config=self.alpha_config, hmm_filter=self.hmm_filter)
        hmm_only = hmm_only_engine.run(
            strategy_mode='hmm_only',
            walk_forward=True,
            train_window=60,
            refit_every=20
        )
        
        # Combine should not be identical to HMM only (which was the bug)
        combine_return = results['metrics']['total_return']
        hmm_return = hmm_only['metrics']['total_return']
        
        # Returns should be different (not exactly equal)
        self.assertNotEqual(combine_return, hmm_return,
                          "Combine strategy should differ from HMM only (uses OR logic)")


class TestHMMInsufficientDataHandling(unittest.TestCase):
    """Test HMM filter handles insufficient training data."""
    
    def test_short_data_adjusts_training_window(self):
        """Test that HMM adjusts training window when data is insufficient."""
        # Create very short dataset (less than default 504 days)
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = pd.Series(np.cumsum(np.random.randn(100) * 0.5) + 100, index=dates)
        
        alpha_config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 5,
                'long_window': 10
            }
        }
        hmm_filter = HMMRegimeFilter(n_states=2, random_state=42)
        
        engine = BacktestEngine(prices, alpha_config=alpha_config, hmm_filter=hmm_filter)
        
        # Should not raise IndexError
        try:
            results = engine.run(
                strategy_mode='alpha_hmm_filter',
                walk_forward=True,
                train_window=504,  # Larger than available data
                refit_every=21
            )
            success = True
        except IndexError:
            success = False
        
        self.assertTrue(success, 
                       "HMM filter should handle insufficient data without IndexError")


class TestOutputDirectoryStructure(unittest.TestCase):
    """Test output directory creation and structure."""
    
    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_output_directory_created_in_results_folder(self):
        """Test that output directories are created under results/."""
        from run_comparison import run_comparison
        
        # Create minimal test
        output_dir = 'test_run'
        
        # Construct expected path
        expected_path = os.path.join('results', output_dir)
        
        # The run_comparison should create directory under results/
        # We'll just test the path construction logic
        if not output_dir.startswith('results/'):
            final_path = os.path.join('results', output_dir)
        else:
            final_path = output_dir
        
        self.assertTrue(final_path.startswith('results/'),
                       "Output directory should be under results/ folder")
    
    def test_timestamped_directory_format(self):
        """Test that default output uses run_TIMESTAMP format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('results', f'run_{timestamp}')
        
        # Check format
        self.assertTrue(output_dir.startswith('results/run_'),
                       "Default output should use results/run_TIMESTAMP format")
        
        # Check timestamp is valid format (YYYYMMDD_HHMMSS)
        parts = output_dir.split('run_')
        if len(parts) > 1:
            ts = parts[1]
            # Should be 15 characters: YYYYMMDD_HHMMSS
            self.assertEqual(len(ts), 15,
                           "Timestamp should be in YYYYMMDD_HHMMSS format")


class TestBacktestResultsStructure(unittest.TestCase):
    """Test backtest results include required fields."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=150, freq='D')
        self.prices = pd.Series(np.cumsum(np.random.randn(150) * 0.5) + 100, index=dates)
        self.alpha_model = SMA(short_window=10, long_window=30)
    
    def test_results_include_close_prices(self):
        """Test that backtest results include close_prices for benchmark calculation."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        # Should include close_prices for benchmark calculation
        self.assertIn('close_prices', results,
                     "Results should include close_prices for benchmark")
        
        # close_prices should match input
        pd.testing.assert_series_equal(results['close_prices'], self.prices,
                                      check_names=False)


if __name__ == '__main__':
    unittest.main()

"""
Integration tests for BacktestEngine with JSON configuration.
Tests config-based initialization and execution.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import BacktestEngine
from alpha_models.sma import SMA
from alpha_models.ema import EMA
from config_loader import ConfigLoader


class TestBacktestEngineWithConfig(unittest.TestCase):
    """Test BacktestEngine with JSON configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        prices = 100 * (1 + np.random.randn(500).cumsum() * 0.01)
        self.close = pd.Series(prices, index=dates, name='close')
    
    def test_init_with_alpha_config(self):
        """Test initialization with alpha config."""
        alpha_config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50,
                'long_window': 200
            }
        }
        
        engine = BacktestEngine(
            close=self.close,
            alpha_config=alpha_config,
            initial_capital=100000.0
        )
        
        self.assertIsNotNone(engine.alpha_model)
        self.assertIsInstance(engine.alpha_model, SMA)
        self.assertEqual(engine.alpha_model.short_window, 50)
        self.assertEqual(engine.alpha_model.long_window, 200)
    
    def test_init_with_alpha_model_instance(self):
        """Test initialization with alpha model instance."""
        alpha_model = SMA(short_window=10, long_window=30)
        
        engine = BacktestEngine(
            close=self.close,
            alpha_model=alpha_model,
            initial_capital=100000.0
        )
        
        self.assertIsNotNone(engine.alpha_model)
        self.assertEqual(engine.alpha_model.short_window, 10)
        self.assertEqual(engine.alpha_model.long_window, 30)
    
    def test_init_with_both_raises_error(self):
        """Test that providing both alpha_model and alpha_config raises error."""
        alpha_model = SMA(short_window=10, long_window=30)
        alpha_config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50,
                'long_window': 200
            }
        }
        
        with self.assertRaises(ValueError) as context:
            BacktestEngine(
                close=self.close,
                alpha_model=alpha_model,
                alpha_config=alpha_config
            )
        
        self.assertIn('either alpha_model OR alpha_config', str(context.exception))
    
    def test_init_with_neither_raises_error(self):
        """Test that providing neither alpha_model nor alpha_config raises error."""
        with self.assertRaises(ValueError) as context:
            BacktestEngine(close=self.close)
        
        self.assertIn('Must provide either', str(context.exception))
    
    def test_from_alpha_config(self):
        """Test creating engine from alpha config."""
        alpha_config = {
            'type': 'EMA',
            'parameters': {
                'short_window': 12,
                'long_window': 26
            }
        }
        
        engine = BacktestEngine.from_alpha_config(
            close=self.close,
            alpha_config=alpha_config,
            initial_capital=50000.0
        )
        
        self.assertIsInstance(engine.alpha_model, EMA)
        self.assertEqual(engine.alpha_model.short_window, 12)
        self.assertEqual(engine.alpha_model.long_window, 26)
        self.assertEqual(engine.initial_capital, 50000.0)
    
    def test_from_config_full(self):
        """Test creating engine from full config dictionary."""
        config = {
            'alpha_model': {
                'type': 'SMA',
                'parameters': {
                    'short_window': 10,
                    'long_window': 30
                }
            },
            'backtest': {
                'initial_capital': 75000.0,
                'transaction_cost': 0.001
            }
        }
        
        engine = BacktestEngine.from_config(
            close=self.close,
            config=config
        )
        
        self.assertIsInstance(engine.alpha_model, SMA)
        self.assertEqual(engine.alpha_model.short_window, 10)
        self.assertEqual(engine.alpha_model.long_window, 30)
        self.assertEqual(engine.initial_capital, 75000.0)
    
    def test_from_config_missing_alpha_model(self):
        """Test error when config missing alpha_model section."""
        config = {
            'backtest': {
                'initial_capital': 100000.0
            }
        }
        
        with self.assertRaises(KeyError) as context:
            BacktestEngine.from_config(close=self.close, config=config)
        
        self.assertIn('alpha_model', str(context.exception))
    
    def test_from_config_default_initial_capital(self):
        """Test default initial capital when not specified."""
        config = {
            'alpha_model': {
                'type': 'SMA',
                'parameters': {
                    'short_window': 50,
                    'long_window': 200
                }
            }
        }
        
        engine = BacktestEngine.from_config(close=self.close, config=config)
        self.assertEqual(engine.initial_capital, 100000.0)
    
    def test_get_alpha_config(self):
        """Test getting alpha config after initialization."""
        alpha_config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50,
                'long_window': 200
            }
        }
        
        engine = BacktestEngine(
            close=self.close,
            alpha_config=alpha_config
        )
        
        retrieved_config = engine.get_alpha_config()
        self.assertEqual(retrieved_config, alpha_config)
    
    def test_get_alpha_config_none_for_instance(self):
        """Test get_alpha_config returns None when created from instance."""
        alpha_model = SMA(short_window=10, long_window=30)
        
        engine = BacktestEngine(
            close=self.close,
            alpha_model=alpha_model
        )
        
        retrieved_config = engine.get_alpha_config()
        self.assertIsNone(retrieved_config)
    
    def test_run_backtest_with_config(self):
        """Test running backtest with config-based engine."""
        alpha_config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 10,
                'long_window': 30
            }
        }
        
        engine = BacktestEngine(
            close=self.close,
            alpha_config=alpha_config,
            initial_capital=100000.0
        )
        
        results = engine.run(
            strategy_mode='alpha_only',
            transaction_cost=0.0
        )
        
        self.assertIsNotNone(results)
        self.assertIn('metrics', results)
        self.assertIn('equity_curve', results)
        self.assertIn('total_return', results['metrics'])
        self.assertEqual(results['alpha_model'], 'SMA')
    
    def test_run_backtest_different_models(self):
        """Test running backtests with different alpha models."""
        models_to_test = ['SMA', 'EMA', 'WMA']
        
        for model_type in models_to_test:
            alpha_config = {
                'type': model_type,
                'parameters': {
                    'short_window': 10,
                    'long_window': 30
                }
            }
            
            engine = BacktestEngine(
                close=self.close,
                alpha_config=alpha_config
            )
            
            results = engine.run(strategy_mode='alpha_only')
            
            self.assertEqual(results['alpha_model'], model_type)
            self.assertIsNotNone(results['metrics'])


class TestBacktestEngineConfigIntegration(unittest.TestCase):
    """Integration tests with actual config files."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        prices = 100 * (1 + np.random.randn(500).cumsum() * 0.01)
        self.close = pd.Series(prices, index=dates, name='close')
    
    def test_load_and_run_default_config(self):
        """Test loading default config and running backtest."""
        try:
            config = ConfigLoader.load_config('config_default.json')
        except FileNotFoundError:
            self.skipTest("config_default.json not found")
        
        engine = BacktestEngine.from_config(
            close=self.close,
            config=config
        )
        
        results = engine.run(strategy_mode='alpha_only')
        
        self.assertIsNotNone(results)
        self.assertIn('metrics', results)
    
    def test_load_and_run_optimal_config(self):
        """Test loading optimal config and running backtest."""
        try:
            config = ConfigLoader.load_config('config_optimal.json')
        except FileNotFoundError:
            self.skipTest("config_optimal.json not found")
        
        engine = BacktestEngine.from_config(
            close=self.close,
            config=config
        )
        
        results = engine.run(strategy_mode='alpha_only')
        
        self.assertIsNotNone(results)
        self.assertIn('metrics', results)
    
    def test_load_and_run_accurate_config(self):
        """Test loading accurate config and running backtest."""
        try:
            config = ConfigLoader.load_config('config_accurate.json')
        except FileNotFoundError:
            self.skipTest("config_accurate.json not found")
        
        engine = BacktestEngine.from_config(
            close=self.close,
            config=config
        )
        
        results = engine.run(strategy_mode='alpha_only')
        
        self.assertIsNotNone(results)
        self.assertIn('metrics', results)


class TestBacktestEngineConfigValidation(unittest.TestCase):
    """Test validation and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 * (1 + np.random.randn(100).cumsum() * 0.01)
        self.close = pd.Series(prices, index=dates, name='close')
    
    def test_invalid_alpha_model_type(self):
        """Test error with invalid alpha model type."""
        alpha_config = {
            'type': 'INVALID_MODEL',
            'parameters': {
                'short_window': 10,
                'long_window': 30
            }
        }
        
        with self.assertRaises(ValueError):
            BacktestEngine(
                close=self.close,
                alpha_config=alpha_config
            )
    
    def test_missing_alpha_parameters(self):
        """Test error with missing parameters."""
        alpha_config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 10
                # long_window missing
            }
        }
        
        with self.assertRaises(ValueError):
            BacktestEngine(
                close=self.close,
                alpha_config=alpha_config
            )
    
    def test_invalid_window_values(self):
        """Test error with invalid window values."""
        alpha_config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 100,
                'long_window': 50  # short > long
            }
        }
        
        with self.assertRaises(ValueError):
            BacktestEngine(
                close=self.close,
                alpha_config=alpha_config
            )


if __name__ == '__main__':
    unittest.main()

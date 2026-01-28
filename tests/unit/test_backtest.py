"""
Unit tests for Backtest Engine.
"""

import unittest
import pandas as pd
import numpy as np
from backtest import BacktestEngine
from alpha_models import SMA, EMA
from signal_filter import HMMRegimeFilter


class TestBacktestEngine(unittest.TestCase):
    """Test cases for backtest engine."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic price data
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        trend = np.linspace(100, 150, 300)
        noise = np.random.randn(300) * 2
        self.prices = pd.Series(trend + noise, index=dates, name='close')
        
        # Create alpha model
        self.alpha_model = SMA(short_window=10, long_window=30)
        
    def test_backtest_initialization(self):
        """Test backtest engine initialization."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        
        self.assertIsNotNone(engine.close)
        self.assertIsNotNone(engine.alpha_model)
        self.assertEqual(engine.initial_capital, 100000.0)
        
    def test_alpha_only_backtest(self):
        """Test backtest with alpha model only."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        
        results = engine.run(strategy_mode='alpha_only')
        
        # Check results structure
        self.assertIn('metrics', results)
        self.assertIn('equity_curve', results)
        self.assertIn('positions', results)
        self.assertIn('returns', results)
        self.assertIn('num_trades', results)
        
        # Check that we have results
        self.assertGreater(len(results['equity_curve']), 0)
        self.assertGreaterEqual(results['num_trades'], 0)  # May be 0 if no crossovers
        
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        metrics = results['metrics']
        
        # Check all expected metrics exist
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'profit_factor', 'win_rate', 'calmar_ratio'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
    def test_rebalancing_frequency(self):
        """Test rebalancing frequency."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        
        # Run with different rebalancing frequencies
        results_daily = engine.run(strategy_mode='alpha_only', rebalance_frequency=1)
        
        engine2 = BacktestEngine(self.prices, self.alpha_model)
        results_weekly = engine2.run(strategy_mode='alpha_only', rebalance_frequency=5)
        
        # Weekly rebalancing should have fewer trades
        self.assertLessEqual(results_weekly['num_trades'], results_daily['num_trades'])
        
    def test_transaction_costs(self):
        """Test that transaction costs reduce returns."""
        engine_no_cost = BacktestEngine(self.prices, self.alpha_model)
        results_no_cost = engine_no_cost.run(strategy_mode='alpha_only', transaction_cost=0.0)
        
        engine_with_cost = BacktestEngine(self.prices, self.alpha_model)
        results_with_cost = engine_with_cost.run(strategy_mode='alpha_only', transaction_cost=0.001)
        
        # With transaction costs, returns should be lower
        self.assertLessEqual(
            results_with_cost['metrics']['total_return'],
            results_no_cost['metrics']['total_return']
        )
        
    def test_hmm_only_strategy(self):
        """Test HMM-only strategy."""
        hmm_filter = HMMRegimeFilter(n_states=3)
        engine = BacktestEngine(self.prices, self.alpha_model, hmm_filter=hmm_filter)
        
        results = engine.run(
            strategy_mode='hmm_only',
            walk_forward=False,
            train_window=100
        )
        
        # Check that HMM-specific results are present
        self.assertIn('regime_probs', results)
        self.assertIn('regime', results)
        self.assertIn('regime_info', results)
        
    def test_alpha_hmm_filter_strategy(self):
        """Test alpha + HMM filter strategy."""
        hmm_filter = HMMRegimeFilter(n_states=3)
        engine = BacktestEngine(self.prices, self.alpha_model, hmm_filter=hmm_filter)
        
        results = engine.run(
            strategy_mode='alpha_hmm_filter',
            walk_forward=False
        )
        
        # Should have results
        self.assertGreater(len(results['equity_curve']), 0)
        
    def test_alpha_hmm_combine_strategy(self):
        """Test alpha + HMM combine strategy."""
        hmm_filter = HMMRegimeFilter(n_states=3)
        engine = BacktestEngine(self.prices, self.alpha_model, hmm_filter=hmm_filter)
        
        results = engine.run(
            strategy_mode='alpha_hmm_combine',
            walk_forward=False
        )
        
        # Should have results
        self.assertGreater(len(results['equity_curve']), 0)
        
    def test_benchmark_comparison(self):
        """Test benchmark comparison."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        engine.run(strategy_mode='alpha_only')
        
        comparison = engine.compare_with_benchmark()
        
        # Check structure
        self.assertIn('strategy_metrics', comparison)
        self.assertIn('benchmark_metrics', comparison)
        self.assertIn('total_return_diff', comparison)
        self.assertIn('sharpe_diff', comparison)
        
    def test_equity_curve_calculation(self):
        """Test equity curve calculation."""
        engine = BacktestEngine(self.prices, self.alpha_model, initial_capital=10000)
        results = engine.run(strategy_mode='alpha_only')
        
        equity = results['equity_curve']
        
        # First value should be close to initial capital (after first return)
        self.assertGreater(equity.iloc[1], 9000)
        self.assertLess(equity.iloc[1], 11000)
        
        # Final capital should match
        self.assertEqual(results['final_capital'], float(equity.iloc[-1]))
        
    def test_time_in_market_calculation(self):
        """Test time in market calculation."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        time_in_market = results['time_in_market']
        
        # Should be between 0 and 1
        self.assertGreaterEqual(time_in_market, 0.0)
        self.assertLessEqual(time_in_market, 1.0)
        
    def test_different_alpha_models(self):
        """Test backtest with different alpha models."""
        models = [
            SMA(10, 30),
            EMA(10, 30)
        ]
        
        for model in models:
            with self.subTest(model=model.get_name()):
                engine = BacktestEngine(self.prices, model)
                results = engine.run(strategy_mode='alpha_only')
                
                # Should complete without error
                self.assertIn('metrics', results)
                self.assertGreater(len(results['equity_curve']), 0)
                
    def test_print_results(self):
        """Test that print_results doesn't crash."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        engine.run(strategy_mode='alpha_only')
        
        # Should not raise error
        try:
            engine.print_results(include_benchmark=True)
        except Exception as e:
            self.fail(f"print_results raised {e}")
            
    def test_error_handling_no_run(self):
        """Test error handling when accessing results before running."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        
        # Should raise error
        with self.assertRaises(ValueError):
            engine.print_results()
        
        with self.assertRaises(ValueError):
            engine.compare_with_benchmark()
            
    def test_positions_valid(self):
        """Test that positions are valid."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        
        # All positions should be 0 or 1
        self.assertTrue(all(pos in [0, 1] for pos in positions.unique()))
        
    def test_returns_alignment(self):
        """Test that returns are properly aligned with positions."""
        engine = BacktestEngine(self.prices, self.alpha_model)
        results = engine.run(strategy_mode='alpha_only')
        
        positions = results['positions']
        returns = results['returns']
        
        # Lengths should match
        self.assertEqual(len(positions), len(returns))
        
        # Indices should match
        self.assertTrue(positions.index.equals(returns.index))


if __name__ == '__main__':
    unittest.main()

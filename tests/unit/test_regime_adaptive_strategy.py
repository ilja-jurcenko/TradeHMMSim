"""
Unit tests for regime-adaptive alpha strategy.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from backtest import BacktestEngine
from alpha_models.sma import SMA
from alpha_models.bollinger import BollingerBands
from signal_filter.hmm_filter import HMMRegimeFilter


class TestRegimeAdaptiveStrategy(unittest.TestCase):
    """Test regime-adaptive alpha strategy in BacktestEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        self.close = pd.Series(
            np.cumsum(np.random.randn(200) * 2) + 100,
            index=dates
        )
        
        self.sma_model = SMA(short_window=10, long_window=30)
        self.bb_model = BollingerBands(short_window=20, long_window=2)
        self.hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
    
    def test_initialization_with_bear_alpha_model(self):
        """Test that BacktestEngine accepts bear_alpha_model parameter."""
        engine = BacktestEngine(
            self.close, 
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=self.bb_model
        )
        
        self.assertIsNotNone(engine.bear_alpha_model)
        self.assertEqual(engine.bear_alpha_model.name, 'BollingerBands')
    
    def test_regime_adaptive_requires_bear_model(self):
        """Test that regime_adaptive_alpha strategy requires bear_alpha_model."""
        engine = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter
            # No bear_alpha_model provided
        )
        
        with self.assertRaises(ValueError) as context:
            engine.run(
                strategy_mode='regime_adaptive_alpha',
                walk_forward=True,
                train_window=50,
                refit_every=10
            )
        
        self.assertIn('bear_alpha_model', str(context.exception))
    
    def test_regime_adaptive_runs_successfully(self):
        """Test that regime-adaptive strategy runs without errors."""
        engine = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=self.bb_model
        )
        
        results = engine.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10,
            transaction_cost=0.001
        )
        
        # Check that results contain expected keys
        self.assertIn('metrics', results)
        self.assertIn('equity_curve', results)
        self.assertIn('positions', results)
        self.assertIn('strategy_mode', results)
        
        # Check strategy mode is recorded
        self.assertEqual(results['strategy_mode'], 'regime_adaptive_alpha')
    
    def test_regime_adaptive_generates_positions(self):
        """Test that regime-adaptive strategy generates position signals."""
        engine = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=self.bb_model
        )
        
        results = engine.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10
        )
        
        positions = results['positions']
        
        # Should have positions
        self.assertGreater(len(positions), 0)
        
        # Should have some non-zero positions
        self.assertTrue((positions != 0).any())
    
    def test_regime_adaptive_switches_models(self):
        """Test that strategy switches between models based on regime."""
        engine = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=self.bb_model
        )
        
        results = engine.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10
        )
        
        # Check that regime information is stored
        self.assertIsNotNone(engine.regime)
        self.assertIsNotNone(engine.regime_probs)
        
        # Check that we have multiple regimes detected
        unique_regimes = engine.regime.unique()
        self.assertGreater(len(unique_regimes), 1, "Should detect multiple regimes")
    
    def test_regime_adaptive_performance_metrics(self):
        """Test that performance metrics are calculated correctly."""
        engine = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=self.bb_model
        )
        
        results = engine.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10
        )
        
        metrics = results['metrics']
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'profit_factor', 'win_rate'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsNotNone(metrics[metric])
    
    def test_regime_adaptive_with_transaction_costs(self):
        """Test that transaction costs are applied correctly."""
        engine = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=self.bb_model
        )
        
        # Run without transaction costs
        results_no_cost = engine.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10,
            transaction_cost=0.0
        )
        
        # Run with transaction costs
        engine2 = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=self.bb_model
        )
        
        results_with_cost = engine2.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10,
            transaction_cost=0.01  # 1% cost
        )
        
        # Returns with cost should be lower (assuming some trades occur)
        if results_no_cost['num_trades'] > 0:
            self.assertLess(
                results_with_cost['metrics']['total_return'],
                results_no_cost['metrics']['total_return']
            )
    
    def test_regime_adaptive_equity_curve(self):
        """Test that equity curve is properly generated."""
        engine = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=self.bb_model
        )
        
        results = engine.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10
        )
        
        equity_curve = results['equity_curve']
        
        # Check structure
        self.assertIsInstance(equity_curve, pd.Series)
        # Walk-forward testing uses a subset of data (after train_window)
        self.assertGreater(len(equity_curve), 0)
        self.assertLessEqual(len(equity_curve), len(self.close))
        
        # Check that equity starts at initial capital
        self.assertAlmostEqual(equity_curve.iloc[0], 100000.0, places=2)
        
        # Check that equity is always positive
        self.assertTrue((equity_curve > 0).all())
    
    def test_different_bear_models(self):
        """Test using different bear market models."""
        # Test with another SMA as bear model (different parameters)
        bear_sma = SMA(short_window=5, long_window=15)
        
        engine = BacktestEngine(
            self.close,
            alpha_model=self.sma_model,
            hmm_filter=self.hmm_filter,
            bear_alpha_model=bear_sma
        )
        
        results = engine.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10
        )
        
        # Should run successfully
        self.assertIn('metrics', results)
        self.assertGreater(results['num_trades'], 0)


class TestRegimeAdaptiveIntegration(unittest.TestCase):
    """Integration tests for regime-adaptive strategy."""
    
    def test_regime_adaptive_vs_alpha_only(self):
        """Compare regime-adaptive strategy against alpha-only."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        close = pd.Series(
            np.cumsum(np.random.randn(200) * 2) + 100,
            index=dates
        )
        
        sma_model = SMA(short_window=10, long_window=30)
        bb_model = BollingerBands(short_window=20, long_window=2)
        
        # Alpha only
        engine_alpha = BacktestEngine(close, alpha_model=sma_model)
        results_alpha = engine_alpha.run(strategy_mode='alpha_only')
        
        # Regime-adaptive
        hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
        engine_adaptive = BacktestEngine(
            close,
            alpha_model=sma_model,
            hmm_filter=hmm_filter,
            bear_alpha_model=bb_model
        )
        results_adaptive = engine_adaptive.run(
            strategy_mode='regime_adaptive_alpha',
            walk_forward=True,
            train_window=50,
            refit_every=10
        )
        
        # Both should complete successfully
        self.assertIn('metrics', results_alpha)
        self.assertIn('metrics', results_adaptive)
        
        # Both should have valid returns
        self.assertIsNotNone(results_alpha['metrics']['total_return'])
        self.assertIsNotNone(results_adaptive['metrics']['total_return'])
    
    def test_all_strategies_comparison(self):
        """Test that regime-adaptive can be run alongside other strategies."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=150, freq='D')
        close = pd.Series(
            np.cumsum(np.random.randn(150) * 2) + 100,
            index=dates
        )
        
        sma_model = SMA(short_window=10, long_window=30)
        bb_model = BollingerBands(short_window=20, long_window=2)
        hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
        
        strategies = [
            'alpha_only',
            'hmm_only',
            'alpha_hmm_filter',
            'alpha_hmm_combine',
            'regime_adaptive_alpha'
        ]
        
        results = {}
        for strategy in strategies:
            if strategy == 'alpha_only':
                engine = BacktestEngine(close, alpha_model=sma_model)
                results[strategy] = engine.run(strategy_mode=strategy)
            elif strategy == 'regime_adaptive_alpha':
                engine = BacktestEngine(
                    close,
                    alpha_model=sma_model,
                    hmm_filter=HMMRegimeFilter(n_states=3, random_state=42),
                    bear_alpha_model=bb_model
                )
                results[strategy] = engine.run(
                    strategy_mode=strategy,
                    walk_forward=True,
                    train_window=50,
                    refit_every=10
                )
            else:
                engine = BacktestEngine(
                    close,
                    alpha_model=sma_model,
                    hmm_filter=HMMRegimeFilter(n_states=3, random_state=42)
                )
                results[strategy] = engine.run(
                    strategy_mode=strategy,
                    walk_forward=True,
                    train_window=50,
                    refit_every=10
                )
        
        # All strategies should complete
        self.assertEqual(len(results), 5)
        
        # All should have metrics
        for strategy, result in results.items():
            self.assertIn('metrics', result, f"Strategy {strategy} missing metrics")
            self.assertIn('total_return', result['metrics'])


if __name__ == '__main__':
    unittest.main()

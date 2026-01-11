"""
Unit tests for the plotter module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from plotter import BacktestPlotter


class TestBacktestPlotter(unittest.TestCase):
    """Test cases for BacktestPlotter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        n = len(dates)
        
        # Sample close prices (with trend)
        self.close = pd.Series(
            100 + np.cumsum(np.random.randn(n) * 0.5),
            index=dates,
            name='Close'
        )
        
        # Sample results dictionary
        returns = np.random.randn(n) * 0.01
        positions = (np.random.rand(n) > 0.3).astype(int)
        
        self.results = {
            'equity_curve': pd.Series(100000 * (1 + returns).cumprod(), index=dates),
            'returns': pd.Series(returns, index=dates),
            'positions': pd.Series(positions, index=dates),
            'metrics': {
                'total_return': 0.15,
                'annualized_return': 0.12,
                'volatility': 0.18,
                'sharpe_ratio': 1.5,
                'sortino_ratio': 1.8,
                'max_drawdown': -0.10,
                'profit_factor': 1.8,
                'win_rate': 0.55,
                'calmar_ratio': 1.2
            },
            'num_trades': 25,
            'initial_capital': 100000,
            'strategy_mode': 'alpha_hmm_override',
            'alpha_model': 'SMA'
        }
        
        # Sample regime data
        self.probs = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], n),
            index=dates,
            columns=[0, 1, 2]
        )
        self.regime = pd.Series(
            np.random.choice([0, 1, 2], n),
            index=dates
        )
        self.switches = self.regime[self.regime.ne(self.regime.shift(1))].dropna()
    
    def test_plot_results_basic(self):
        """Test basic plot_results functionality."""
        try:
            # Close all existing figures
            plt.close('all')
            
            # Should not raise an exception
            BacktestPlotter.plot_results(self.results, self.close)
            
            # Verify figure was created
            self.assertEqual(len(plt.get_fignums()), 1)
            
            plt.close('all')
        except Exception as e:
            self.fail(f"plot_results raised exception: {e}")
    
    def test_plot_results_with_regime(self):
        """Test plot_results with regime probabilities."""
        try:
            plt.close('all')
            
            # Add regime data to results
            results_with_regime = self.results.copy()
            results_with_regime['regime_probs'] = self.probs
            results_with_regime['regime_info'] = {
                'bear_regime': 2,
                'bull_regime': 0,
                'neutral_regime': 1
            }
            
            BacktestPlotter.plot_results(results_with_regime, self.close)
            
            self.assertEqual(len(plt.get_fignums()), 1)
            
            plt.close('all')
        except Exception as e:
            self.fail(f"plot_results with regime raised exception: {e}")
    
    def test_plot_regime_analysis(self):
        """Test regime analysis plotting."""
        try:
            plt.close('all')
            
            BacktestPlotter.plot_regime_analysis(
                self.probs,
                self.regime,
                self.close,
                self.switches
            )
            
            self.assertEqual(len(plt.get_fignums()), 1)
            
            plt.close('all')
        except Exception as e:
            self.fail(f"plot_regime_analysis raised exception: {e}")
    
    def test_plot_comparison(self):
        """Test comparison plotting."""
        try:
            plt.close('all')
            
            # Create second results set
            results2 = self.results.copy()
            results2['equity_curve'] = self.results['equity_curve'] * 0.95
            
            BacktestPlotter.plot_comparison(
                [self.results, results2],
                ['Strategy 1', 'Strategy 2']
            )
            
            self.assertEqual(len(plt.get_fignums()), 1)
            
            plt.close('all')
        except Exception as e:
            self.fail(f"plot_comparison raised exception: {e}")
    
    def test_plot_metrics_comparison(self):
        """Test metrics comparison plotting."""
        try:
            plt.close('all')
            
            # Create second results set
            results2 = self.results.copy()
            results2['metrics'] = {
                'total_return': 0.12,
                'annualized_return': 0.10,
                'volatility': 0.16,
                'sharpe_ratio': 1.3,
                'sortino_ratio': 1.6,
                'max_drawdown': -0.08,
                'profit_factor': 1.6,
                'win_rate': 0.52,
                'calmar_ratio': 1.0
            }
            
            BacktestPlotter.plot_metrics_comparison(
                [self.results, results2],
                ['Strategy 1', 'Strategy 2']
            )
            
            self.assertEqual(len(plt.get_fignums()), 1)
            
            plt.close('all')
        except Exception as e:
            self.fail(f"plot_metrics_comparison raised exception: {e}")
    
    def test_plot_with_empty_positions(self):
        """Test plotting with no positions taken."""
        try:
            plt.close('all')
            
            # Create results with no positions
            results_empty = self.results.copy()
            results_empty['positions'] = pd.Series(
                np.zeros(len(self.close)),
                index=self.close.index
            )
            
            BacktestPlotter.plot_results(results_empty, self.close)
            
            self.assertEqual(len(plt.get_fignums()), 1)
            
            plt.close('all')
        except Exception as e:
            self.fail(f"plot_results with empty positions raised exception: {e}")
    
    def test_plot_with_custom_figsize(self):
        """Test plotting with custom figure size."""
        try:
            plt.close('all')
            
            BacktestPlotter.plot_results(self.results, self.close, figsize=(10, 8))
            
            fig = plt.gcf()
            self.assertEqual(fig.get_figwidth(), 10)
            self.assertEqual(fig.get_figheight(), 8)
            
            plt.close('all')
        except Exception as e:
            self.fail(f"plot_results with custom figsize raised exception: {e}")


if __name__ == '__main__':
    unittest.main()

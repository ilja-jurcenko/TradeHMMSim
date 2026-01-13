"""
Unit tests for Portfolio weight management and regime-based allocation.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from portfolio import Portfolio


class TestPortfolioWeights(unittest.TestCase):
    """Test portfolio weight management."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock loader
        self.mock_loader = Mock()
        
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_spy_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 400,
            'Open': np.random.randn(100).cumsum() + 400,
            'High': np.random.randn(100).cumsum() + 405,
            'Low': np.random.randn(100).cumsum() + 395,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        mock_agg_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 110,
            'Open': np.random.randn(100).cumsum() + 110,
            'High': np.random.randn(100).cumsum() + 111,
            'Low': np.random.randn(100).cumsum() + 109,
            'Volume': np.random.randint(500000, 5000000, 100)
        }, index=dates)
        
        def mock_load_ticker(ticker, start, end, progress=False):
            if ticker == 'SPY':
                return mock_spy_data.copy()
            elif ticker == 'AGG':
                return mock_agg_data.copy()
            return None
        
        self.mock_loader.load_ticker = mock_load_ticker
    
    def test_default_equal_weights(self):
        """Test that default weights are equal."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31', 
                            loader=self.mock_loader)
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['SPY'], 0.5)
        self.assertEqual(weights['AGG'], 0.5)
    
    def test_custom_initial_weights(self):
        """Test custom initial weights."""
        custom_weights = {'SPY': 0.6, 'AGG': 0.4}
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader, weights=custom_weights)
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['SPY'], 0.6)
        self.assertEqual(weights['AGG'], 0.4)
    
    def test_weights_validation_sum(self):
        """Test that weights must sum to 1.0."""
        with self.assertRaises(ValueError) as context:
            Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                     loader=self.mock_loader, weights={'SPY': 0.6, 'AGG': 0.5})
        
        self.assertIn('sum to 1.0', str(context.exception))
    
    def test_weights_validation_keys(self):
        """Test that weight keys must match tickers."""
        with self.assertRaises(ValueError) as context:
            Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                     loader=self.mock_loader, weights={'SPY': 1.0})
        
        self.assertIn('must match', str(context.exception))
    
    def test_set_weights(self):
        """Test setting new weights."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        new_weights = {'SPY': 0.7, 'AGG': 0.3}
        portfolio.set_weights(new_weights)
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['SPY'], 0.7)
        self.assertEqual(weights['AGG'], 0.3)
    
    def test_set_weights_validation(self):
        """Test weight validation in set_weights."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        # Test sum validation
        with self.assertRaises(ValueError):
            portfolio.set_weights({'SPY': 0.5, 'AGG': 0.6})
        
        # Test keys validation
        with self.assertRaises(ValueError):
            portfolio.set_weights({'SPY': 0.5, 'QQQ': 0.5})
    
    def test_get_weighted_returns(self):
        """Test calculating weighted portfolio returns."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader, weights={'SPY': 0.6, 'AGG': 0.4})
        portfolio.load_data(progress=False)
        
        weighted_returns = portfolio.get_weighted_returns()
        
        # Check that returns were calculated
        self.assertIsInstance(weighted_returns, pd.Series)
        self.assertGreater(len(weighted_returns), 0)
        
        # Check that returns use correct weights
        spy_returns = portfolio.get_returns('SPY')
        agg_returns = portfolio.get_returns('AGG')
        expected_returns = spy_returns * 0.6 + agg_returns * 0.4
        expected_returns.name = None
        
        pd.testing.assert_series_equal(weighted_returns, expected_returns)


class TestRegimeBasedRebalancing(unittest.TestCase):
    """Test regime-based portfolio rebalancing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock loader
        self.mock_loader = Mock()
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 400,
            'Open': np.random.randn(100).cumsum() + 400,
            'High': np.random.randn(100).cumsum() + 405,
            'Low': np.random.randn(100).cumsum() + 395,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        self.mock_loader.load_ticker = Mock(return_value=mock_data.copy())
    
    def test_bull_regime_allocation(self):
        """Test that bull regime allocates 100% to SPY."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        portfolio.rebalance_regime_based('bull')
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['SPY'], 1.0)
        self.assertEqual(weights['AGG'], 0.0)
    
    def test_neutral_regime_allocation(self):
        """Test that neutral regime allocates 100% to SPY."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        portfolio.rebalance_regime_based('neutral')
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['SPY'], 1.0)
        self.assertEqual(weights['AGG'], 0.0)
    
    def test_bear_regime_allocation(self):
        """Test that bear regime allocates 100% to AGG."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        portfolio.rebalance_regime_based('bear')
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['SPY'], 0.0)
        self.assertEqual(weights['AGG'], 1.0)
    
    def test_custom_tickers_allocation(self):
        """Test regime-based allocation with custom tickers."""
        portfolio = Portfolio(['QQQ', 'TLT'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        portfolio.rebalance_regime_based('bull', aggressive_ticker='QQQ', 
                                        defensive_ticker='TLT')
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['QQQ'], 1.0)
        self.assertEqual(weights['TLT'], 0.0)
        
        portfolio.rebalance_regime_based('bear', aggressive_ticker='QQQ',
                                        defensive_ticker='TLT')
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['QQQ'], 0.0)
        self.assertEqual(weights['TLT'], 1.0)
    
    def test_invalid_regime(self):
        """Test that invalid regime raises error."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        with self.assertRaises(ValueError) as context:
            portfolio.rebalance_regime_based('sideways')
        
        self.assertIn('Unknown regime', str(context.exception))
    
    def test_missing_tickers_error(self):
        """Test error when required tickers not in portfolio."""
        portfolio = Portfolio(['SPY'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        with self.assertRaises(ValueError) as context:
            portfolio.rebalance_regime_based('bull')
        
        self.assertIn('must be in portfolio', str(context.exception))
    
    def test_regime_switching_sequence(self):
        """Test switching between regimes."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        # Start in bull
        portfolio.rebalance_regime_based('bull')
        self.assertEqual(portfolio.get_weights()['SPY'], 1.0)
        
        # Switch to bear
        portfolio.rebalance_regime_based('bear')
        self.assertEqual(portfolio.get_weights()['AGG'], 1.0)
        
        # Switch to neutral (back to SPY)
        portfolio.rebalance_regime_based('neutral')
        self.assertEqual(portfolio.get_weights()['SPY'], 1.0)


class TestMultiAssetPortfolio(unittest.TestCase):
    """Test multi-asset portfolio functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = Mock()
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        mock_data = pd.DataFrame({
            'Close': np.random.randn(50).cumsum() + 100,
            'Open': np.random.randn(50).cumsum() + 100,
            'High': np.random.randn(50).cumsum() + 101,
            'Low': np.random.randn(50).cumsum() + 99,
            'Volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)
        
        self.mock_loader.load_ticker = Mock(return_value=mock_data.copy())
    
    def test_three_asset_portfolio(self):
        """Test portfolio with three assets."""
        portfolio = Portfolio(['SPY', 'AGG', 'GLD'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader,
                            weights={'SPY': 0.5, 'AGG': 0.3, 'GLD': 0.2})
        
        weights = portfolio.get_weights()
        self.assertEqual(weights['SPY'], 0.5)
        self.assertEqual(weights['AGG'], 0.3)
        self.assertEqual(weights['GLD'], 0.2)
    
    def test_weights_independence(self):
        """Test that get_weights returns a copy."""
        portfolio = Portfolio(['SPY', 'AGG'], '2023-01-01', '2023-12-31',
                            loader=self.mock_loader)
        
        weights1 = portfolio.get_weights()
        weights1['SPY'] = 0.9  # Modify the returned dict
        
        weights2 = portfolio.get_weights()
        self.assertEqual(weights2['SPY'], 0.5)  # Should still be 0.5


if __name__ == '__main__':
    unittest.main()

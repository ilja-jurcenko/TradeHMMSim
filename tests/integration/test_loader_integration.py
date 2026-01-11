"""
Integration test to verify loader refactoring works correctly with real data.
This test uses real yfinance data to ensure backward compatibility.
"""

import unittest
from portfolio import Portfolio
from loaders import YFinanceLoader, CachedYFinanceLoader


class TestLoaderIntegration(unittest.TestCase):
    """Integration tests for data loaders with real data."""
    
    def test_portfolio_basic_usage(self):
        """Test basic portfolio usage with default CachedYFinanceLoader."""
        # Create portfolio (should use CachedYFinanceLoader by default)
        portfolio = Portfolio(['SPY'], '2024-01-01', '2024-01-31')
        
        # Load data
        portfolio.load_data(progress=False)
        
        # Verify data was loaded
        self.assertGreater(len(portfolio.data), 0, "No data loaded")
        self.assertIn('SPY', portfolio.data, "SPY not in data")
        
        # Get close prices
        close = portfolio.get_close_prices('SPY')
        self.assertGreater(len(close), 0, "No close prices")
    
    def test_portfolio_custom_loader(self):
        """Test portfolio with explicitly provided loader."""
        # Create loader explicitly
        loader = YFinanceLoader()
        
        # Create portfolio with explicit loader
        portfolio = Portfolio(['SPY'], '2024-01-01', '2024-01-31', loader=loader)
        
        # Load data
        portfolio.load_data(progress=False)
        
        # Verify data was loaded
        self.assertGreater(len(portfolio.data), 0, "No data loaded")
        self.assertIn('SPY', portfolio.data, "SPY not in data")
        
        close = portfolio.get_close_prices('SPY')
        self.assertGreater(len(close), 0, "No close prices")
    
    def test_loader_directly(self):
        """Test YFinanceLoader directly."""
        loader = YFinanceLoader()
        
        # Load single ticker
        data = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
        
        self.assertIsNotNone(data, "Failed to load data")
        self.assertGreater(len(data), 0, "No data returned")
        self.assertIn('Close', data.columns, "Close column missing")
        
        # Load multiple tickers
        data_dict = loader.load_multiple_tickers(['SPY', 'QQQ'], '2024-01-01', '2024-01-31', progress=False)
        
        self.assertEqual(len(data_dict), 2, "Should have loaded 2 tickers")
        self.assertIn('SPY', data_dict, "SPY missing")
        self.assertIn('QQQ', data_dict, "QQQ missing")


if __name__ == '__main__':
    unittest.main()

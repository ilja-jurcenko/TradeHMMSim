"""
Unit tests for the loaders module.
Tests BaseDataLoader, YFinanceLoader, and portfolio integration.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import BaseDataLoader, YFinanceLoader, CachedYFinanceLoader
from portfolio import Portfolio


class MockDataLoader(BaseDataLoader):
    """Mock data loader for testing."""
    
    def __init__(self, data_dict=None):
        self.data_dict = data_dict or {}
        self.call_count = 0
        
    def load_ticker(self, ticker: str, start_date: str, end_date: str, 
                   progress: bool = False) -> pd.DataFrame:
        """Return mock data for testing."""
        self.call_count += 1
        return self.data_dict.get(ticker)
    
    def load_multiple_tickers(self, tickers, start_date: str, 
                             end_date: str, progress: bool = False):
        """Return mock data for multiple tickers."""
        result = {}
        for ticker in tickers:
            data = self.load_ticker(ticker, start_date, end_date, progress)
            if data is not None:
                result[ticker] = data
        return result


class TestBaseDataLoader(unittest.TestCase):
    """Test BaseDataLoader functionality."""
    
    def test_base_loader_is_abstract(self):
        """Test that BaseDataLoader cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseDataLoader()
    
    def test_validate_data_with_valid_ohlcv(self):
        """Test validation with complete OHLCV data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        
        loader = MockDataLoader()
        self.assertTrue(loader.validate_data(data))
    
    def test_validate_data_with_close_only(self):
        """Test validation with Close column only."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Close': np.random.rand(10) * 100
        }, index=dates)
        
        loader = MockDataLoader()
        self.assertTrue(loader.validate_data(data))
    
    def test_validate_data_with_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        data = pd.DataFrame()
        loader = MockDataLoader()
        self.assertFalse(loader.validate_data(data))
    
    def test_validate_data_with_none(self):
        """Test validation with None."""
        loader = MockDataLoader()
        self.assertFalse(loader.validate_data(None))
    
    def test_validate_data_without_required_columns(self):
        """Test validation without required columns."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'SomeOtherColumn': np.random.rand(10) * 100
        }, index=dates)
        
        loader = MockDataLoader()
        self.assertFalse(loader.validate_data(data))


class TestYFinanceLoader(unittest.TestCase):
    """Test YFinanceLoader functionality."""
    
    @patch('loaders.yfinance_loader.yf.download')
    def test_load_ticker_success(self, mock_download):
        """Test successful ticker data loading."""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        
        mock_download.return_value = mock_data
        
        loader = YFinanceLoader()
        result = loader.load_ticker('SPY', '2023-01-01', '2023-01-10')
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10)
        self.assertIn('Close', result.columns)
        mock_download.assert_called_once_with('SPY', start='2023-01-01', 
                                              end='2023-01-10', progress=False)
    
    @patch('loaders.yfinance_loader.yf.download')
    def test_load_ticker_empty_data(self, mock_download):
        """Test loading ticker with no data."""
        mock_download.return_value = pd.DataFrame()
        
        loader = YFinanceLoader()
        result = loader.load_ticker('INVALID', '2023-01-01', '2023-01-10')
        
        self.assertIsNone(result)
    
    @patch('loaders.yfinance_loader.yf.download')
    def test_load_ticker_exception(self, mock_download):
        """Test loading ticker with exception."""
        mock_download.side_effect = Exception("Network error")
        
        loader = YFinanceLoader()
        result = loader.load_ticker('SPY', '2023-01-01', '2023-01-10')
        
        self.assertIsNone(result)
    
    @patch('loaders.yfinance_loader.yf.download')
    def test_load_multiple_tickers(self, mock_download):
        """Test loading multiple tickers."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        def mock_download_side_effect(ticker, start, end, progress):
            return pd.DataFrame({
                'Open': np.random.rand(10) * 100,
                'High': np.random.rand(10) * 100,
                'Low': np.random.rand(10) * 100,
                'Close': np.random.rand(10) * 100,
                'Volume': np.random.randint(1000000, 10000000, 10)
            }, index=dates)
        
        mock_download.side_effect = mock_download_side_effect
        
        loader = YFinanceLoader()
        result = loader.load_multiple_tickers(['SPY', 'QQQ'], '2023-01-01', '2023-01-10')
        
        self.assertEqual(len(result), 2)
        self.assertIn('SPY', result)
        self.assertIn('QQQ', result)
        self.assertEqual(mock_download.call_count, 2)
    
    @patch('loaders.yfinance_loader.yf.download')
    def test_normalize_multiindex_columns(self, mock_download):
        """Test normalization of MultiIndex columns from yfinance."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        # Create MultiIndex columns (as yfinance sometimes returns)
        columns = pd.MultiIndex.from_product([['Close', 'Open', 'High', 'Low', 'Volume'], ['SPY']])
        mock_data = pd.DataFrame(
            np.random.rand(10, 5),
            index=dates,
            columns=columns
        )
        
        mock_download.return_value = mock_data
        
        loader = YFinanceLoader()
        result = loader.load_ticker('SPY', '2023-01-01', '2023-01-10')
        
        self.assertIsNotNone(result)
        # After normalization, should have simple column names
        self.assertFalse(isinstance(result.columns, pd.MultiIndex))
        self.assertIn('Close', result.columns)


class TestPortfolioIntegration(unittest.TestCase):
    """Test Portfolio integration with data loaders."""
    
    def test_portfolio_with_default_loader(self):
        """Test Portfolio uses CachedYFinanceLoader by default."""
        portfolio = Portfolio(['SPY'], '2023-01-01', '2023-12-31')
        self.assertIsInstance(portfolio.loader, CachedYFinanceLoader)
    
    def test_portfolio_with_custom_loader(self):
        """Test Portfolio with custom data loader."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        
        mock_loader = MockDataLoader({'SPY': mock_data})
        portfolio = Portfolio(['SPY'], '2023-01-01', '2023-01-10', loader=mock_loader)
        
        self.assertIsInstance(portfolio.loader, MockDataLoader)
        
        # Load data using mock loader
        portfolio.load_data()
        
        self.assertEqual(len(portfolio.data), 1)
        self.assertIn('SPY', portfolio.data)
        self.assertEqual(mock_loader.call_count, 1)
    
    def test_portfolio_load_multiple_tickers(self):
        """Test loading multiple tickers with mock loader."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        mock_data_spy = pd.DataFrame({
            'Close': np.random.rand(10) * 400
        }, index=dates)
        
        mock_data_qqq = pd.DataFrame({
            'Close': np.random.rand(10) * 300
        }, index=dates)
        
        mock_loader = MockDataLoader({'SPY': mock_data_spy, 'QQQ': mock_data_qqq})
        portfolio = Portfolio(['SPY', 'QQQ'], '2023-01-01', '2023-01-10', loader=mock_loader)
        
        portfolio.load_data()
        
        self.assertEqual(len(portfolio.data), 2)
        self.assertIn('SPY', portfolio.data)
        self.assertIn('QQQ', portfolio.data)
        self.assertIsNotNone(portfolio.close_prices)
        self.assertEqual(len(portfolio.close_prices.columns), 2)
    
    def test_portfolio_add_ticker_with_loader(self):
        """Test adding a ticker using custom loader."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        mock_data_spy = pd.DataFrame({
            'Close': np.random.rand(10) * 400
        }, index=dates)
        
        mock_data_qqq = pd.DataFrame({
            'Close': np.random.rand(10) * 300
        }, index=dates)
        
        mock_loader = MockDataLoader({'SPY': mock_data_spy, 'QQQ': mock_data_qqq})
        portfolio = Portfolio(['SPY'], '2023-01-01', '2023-01-10', loader=mock_loader)
        
        portfolio.load_data()
        self.assertEqual(len(portfolio.data), 1)
        
        # Add new ticker
        portfolio.add_ticker('QQQ')
        
        self.assertEqual(len(portfolio.data), 2)
        self.assertIn('QQQ', portfolio.data)
        self.assertEqual(len(portfolio.tickers), 2)
    
    def test_portfolio_with_failed_ticker_load(self):
        """Test Portfolio handles failed ticker loads gracefully."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data_spy = pd.DataFrame({
            'Close': np.random.rand(10) * 400
        }, index=dates)
        
        # Only SPY has data, INVALID returns None
        mock_loader = MockDataLoader({'SPY': mock_data_spy, 'INVALID': None})
        portfolio = Portfolio(['SPY', 'INVALID'], '2023-01-01', '2023-01-10', 
                            loader=mock_loader)
        
        portfolio.load_data()
        
        # Only SPY should be loaded
        self.assertEqual(len(portfolio.data), 1)
        self.assertIn('SPY', portfolio.data)
        self.assertNotIn('INVALID', portfolio.data)
    
    def test_portfolio_get_close_prices(self):
        """Test getting close prices after loading."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        close_prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        mock_data = pd.DataFrame({
            'Close': close_prices
        }, index=dates)
        
        mock_loader = MockDataLoader({'SPY': mock_data})
        portfolio = Portfolio(['SPY'], '2023-01-01', '2023-01-10', loader=mock_loader)
        portfolio.load_data()
        
        # Get close prices
        close = portfolio.get_close_prices('SPY')
        
        self.assertEqual(len(close), 10)
        self.assertEqual(close.iloc[0], 100)
        self.assertEqual(close.iloc[-1], 109)


if __name__ == '__main__':
    unittest.main()

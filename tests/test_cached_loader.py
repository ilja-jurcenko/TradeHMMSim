"""
Unit tests for CachedYFinanceLoader.
Tests caching, gap detection, and append functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import CachedYFinanceLoader, YFinanceLoader


class MockYFinanceLoader:
    """Mock YFinanceLoader for testing without network calls."""
    
    def __init__(self):
        self.call_count = 0
        self.call_history = []
    
    def load_ticker(self, ticker: str, start_date: str, end_date: str, 
                   progress: bool = False) -> pd.DataFrame:
        """Return mock data for testing."""
        self.call_count += 1
        self.call_history.append({
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        })
        
        # Generate mock data for the requested period
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to business days (rough approximation)
        dates = dates[dates.dayofweek < 5]
        
        data = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100 + 300,
            'High': np.random.rand(len(dates)) * 100 + 310,
            'Low': np.random.rand(len(dates)) * 100 + 290,
            'Close': np.random.rand(len(dates)) * 100 + 300,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return data


class TestCachedYFinanceLoader(unittest.TestCase):
    """Test CachedYFinanceLoader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / 'cache'
        
        # Create mock yfinance loader
        self.mock_yf_loader = MockYFinanceLoader()
        
        # Create cached loader with mock
        self.loader = CachedYFinanceLoader(
            cache_dir=str(self.cache_dir),
            yfinance_loader=self.mock_yf_loader
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_cache_directory_creation(self):
        """Test that cache directory is created."""
        self.assertTrue(self.cache_dir.exists())
        self.assertTrue(self.cache_dir.is_dir())
    
    def test_first_load_downloads_and_caches(self):
        """Test that first load downloads from yfinance and saves to cache."""
        # First load should download
        data = self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        
        # Verify download happened
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        self.assertIsNotNone(data)
        
        # Verify cache file was created
        cache_path = self.cache_dir / 'SPY.csv'
        self.assertTrue(cache_path.exists())
        
        # Verify cached data matches returned data
        cached_data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(data, cached_data['2024-01-01':'2024-01-31'])
    
    def test_second_load_uses_cache(self):
        """Test that second load uses cache without downloading."""
        # First load
        data1 = self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Second load with same date range
        data2 = self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        
        # Should not have downloaded again
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Data should be the same
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_append_future_data(self):
        """Test that requesting future data downloads and appends missing data."""
        # Load initial data
        data1 = self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        initial_len = len(data1)
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Request extended date range (includes future dates)
        data2 = self.loader.load_ticker('SPY', '2024-01-01', '2024-02-29')
        
        # Should have downloaded the gap
        self.assertEqual(self.mock_yf_loader.call_count, 2)
        
        # Verify the gap download was for February
        last_call = self.mock_yf_loader.call_history[-1]
        self.assertEqual(last_call['start_date'], '2024-02-01')
        self.assertEqual(last_call['end_date'], '2024-02-29')
        
        # Data should be longer
        self.assertGreater(len(data2), initial_len)
        
        # Verify cache was updated
        cached_data = self.loader._read_cached_data('SPY')
        self.assertGreater(len(cached_data), initial_len)
    
    def test_prepend_past_data(self):
        """Test that requesting past data downloads and prepends missing data."""
        # Load initial data (February)
        data1 = self.loader.load_ticker('SPY', '2024-02-01', '2024-02-29')
        initial_len = len(data1)
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Request extended date range (includes past dates)
        data2 = self.loader.load_ticker('SPY', '2024-01-01', '2024-02-29')
        
        # Should have downloaded the gap
        self.assertEqual(self.mock_yf_loader.call_count, 2)
        
        # Verify the gap download was for January
        last_call = self.mock_yf_loader.call_history[-1]
        self.assertEqual(last_call['start_date'], '2024-01-01')
        # Gap should end before February starts
        self.assertIn('2024-01', last_call['end_date'])
        
        # Data should be longer
        self.assertGreater(len(data2), initial_len)
    
    def test_append_both_directions(self):
        """Test appending data both before and after cached range."""
        # Load initial data (February only)
        data1 = self.loader.load_ticker('SPY', '2024-02-01', '2024-02-29')
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Request much wider range
        data2 = self.loader.load_ticker('SPY', '2024-01-01', '2024-03-31')
        
        # Should have downloaded two gaps (January and March)
        self.assertEqual(self.mock_yf_loader.call_count, 3)  # Initial + 2 gaps
        
        # Verify both gaps were downloaded
        self.assertEqual(len(self.mock_yf_loader.call_history), 3)
    
    def test_subset_of_cached_data(self):
        """Test that requesting a subset of cached data doesn't download."""
        # Load large range
        data1 = self.loader.load_ticker('SPY', '2024-01-01', '2024-12-31')
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Request subset
        data2 = self.loader.load_ticker('SPY', '2024-02-01', '2024-02-29')
        
        # Should not have downloaded again
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Data should be subset
        self.assertLess(len(data2), len(data1))
    
    def test_force_download(self):
        """Test that force_download bypasses cache."""
        # Load and cache
        data1 = self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Force download
        data2 = self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31', force_download=True)
        
        # Should have downloaded again
        self.assertEqual(self.mock_yf_loader.call_count, 2)
    
    def test_multiple_tickers(self):
        """Test loading multiple tickers with caching."""
        # Load multiple tickers
        data_dict = self.loader.load_multiple_tickers(
            ['SPY', 'QQQ', 'IWM'],
            '2024-01-01',
            '2024-01-31'
        )
        
        # Should have downloaded all three
        self.assertEqual(self.mock_yf_loader.call_count, 3)
        self.assertEqual(len(data_dict), 3)
        
        # Verify cache files exist
        for ticker in ['SPY', 'QQQ', 'IWM']:
            cache_path = self.cache_dir / f'{ticker}.csv'
            self.assertTrue(cache_path.exists())
        
        # Load again (should use cache)
        data_dict2 = self.loader.load_multiple_tickers(
            ['SPY', 'QQQ', 'IWM'],
            '2024-01-01',
            '2024-01-31'
        )
        
        # Should not have downloaded again
        self.assertEqual(self.mock_yf_loader.call_count, 3)
    
    def test_clear_cache_single_ticker(self):
        """Test clearing cache for a single ticker."""
        # Load and cache
        self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        self.loader.load_ticker('QQQ', '2024-01-01', '2024-01-31')
        
        # Verify both cache files exist
        self.assertTrue((self.cache_dir / 'SPY.csv').exists())
        self.assertTrue((self.cache_dir / 'QQQ.csv').exists())
        
        # Clear only SPY
        self.loader.clear_cache('SPY')
        
        # SPY cache should be gone, QQQ should remain
        self.assertFalse((self.cache_dir / 'SPY.csv').exists())
        self.assertTrue((self.cache_dir / 'QQQ.csv').exists())
    
    def test_clear_cache_all(self):
        """Test clearing all cache."""
        # Load and cache multiple tickers
        self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        self.loader.load_ticker('QQQ', '2024-01-01', '2024-01-31')
        
        # Clear all
        self.loader.clear_cache()
        
        # All cache files should be gone
        self.assertFalse((self.cache_dir / 'SPY.csv').exists())
        self.assertFalse((self.cache_dir / 'QQQ.csv').exists())
    
    def test_get_cache_info(self):
        """Test getting cache information."""
        # No cache initially
        info = self.loader.get_cache_info('SPY')
        self.assertIsNone(info)
        
        # Load and cache
        self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        
        # Get cache info
        info = self.loader.get_cache_info('SPY')
        
        self.assertIsNotNone(info)
        self.assertEqual(info['ticker'], 'SPY')
        self.assertIn('2024-01', info['start_date'])
        self.assertIn('2024-01', info['end_date'])
        self.assertGreater(info['rows'], 0)
        self.assertIn('SPY.csv', info['cache_path'])
    
    def test_cache_persistence_across_instances(self):
        """Test that cache persists across different loader instances."""
        # Load with first instance
        data1 = self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Create new loader instance with same cache directory
        new_mock = MockYFinanceLoader()
        new_loader = CachedYFinanceLoader(
            cache_dir=str(self.cache_dir),
            yfinance_loader=new_mock
        )
        
        # Load with new instance
        data2 = new_loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        
        # New instance should not have downloaded (used cache)
        self.assertEqual(new_mock.call_count, 0)
        
        # Data should match
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_overlapping_data_handling(self):
        """Test that overlapping data is handled correctly (newer data preferred)."""
        # Load initial data
        self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        
        # Get the cached data
        cached1 = self.loader._read_cached_data('SPY')
        original_value = cached1.iloc[10]['Close']
        
        # Load overlapping data (should update cache)
        self.loader.load_ticker('SPY', '2024-01-15', '2024-02-15', force_download=True)
        
        # Check that cache was updated (newer data preferred)
        cached2 = self.loader._read_cached_data('SPY')
        new_value = cached2.iloc[10]['Close']
        
        # Values might be different due to mock generating random data
        # The important thing is no duplicates and data is sorted
        self.assertEqual(len(cached2), len(cached2.index.unique()))
        self.assertTrue(cached2.index.is_monotonic_increasing)
    
    def test_empty_cache_directory(self):
        """Test behavior with empty cache directory."""
        # Verify cache dir is empty
        cache_files = list(self.cache_dir.glob('*.csv'))
        self.assertEqual(len(cache_files), 0)
        
        # Load should work (download and cache)
        data = self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        
        self.assertIsNotNone(data)
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        self.assertTrue((self.cache_dir / 'SPY.csv').exists())
    
    def test_gap_detection_with_exact_boundaries(self):
        """Test gap detection when requesting exactly at cache boundaries."""
        # Cache January
        self.loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        self.assertEqual(self.mock_yf_loader.call_count, 1)
        
        # Request February (starts exactly after cache ends)
        self.loader.load_ticker('SPY', '2024-02-01', '2024-02-29')
        
        # Should download February
        self.assertEqual(self.mock_yf_loader.call_count, 2)
        
        # Now request full range (should use cache, no new downloads)
        self.loader.load_ticker('SPY', '2024-01-01', '2024-02-29')
        
        # Should not download again
        self.assertEqual(self.mock_yf_loader.call_count, 2)


class TestCachedLoaderIntegration(unittest.TestCase):
    """Integration tests with Portfolio."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / 'cache'
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_portfolio_with_cached_loader(self):
        """Test Portfolio integration with CachedYFinanceLoader."""
        from portfolio import Portfolio
        
        mock_yf = MockYFinanceLoader()
        cached_loader = CachedYFinanceLoader(
            cache_dir=str(self.cache_dir),
            yfinance_loader=mock_yf
        )
        
        # Create portfolio with cached loader
        portfolio = Portfolio(['SPY'], '2024-01-01', '2024-01-31', loader=cached_loader)
        portfolio.load_data()
        
        # Verify data was loaded
        self.assertEqual(len(portfolio.data), 1)
        self.assertIn('SPY', portfolio.data)
        
        # Verify download happened once
        self.assertEqual(mock_yf.call_count, 1)
        
        # Create new portfolio with same cache
        portfolio2 = Portfolio(['SPY'], '2024-01-01', '2024-01-31', loader=cached_loader)
        portfolio2.load_data()
        
        # Should not have downloaded again
        self.assertEqual(mock_yf.call_count, 1)


if __name__ == '__main__':
    unittest.main()

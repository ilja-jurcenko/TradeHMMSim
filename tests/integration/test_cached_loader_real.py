"""
Integration test for CachedYFinanceLoader with real yfinance data.
Demonstrates caching, appending, and performance benefits.
"""

import unittest
import shutil
from pathlib import Path
import time

from loaders import CachedYFinanceLoader
from portfolio import Portfolio


class TestCachedLoaderReal(unittest.TestCase):
    """Integration tests for CachedYFinanceLoader with real data."""
    
    def setUp(self):
        """Set up test cache directory."""
        self.cache_dir = './test_cache_integration'
    
    def tearDown(self):
        """Clean up cache directory."""
        if Path(self.cache_dir).exists():
            shutil.rmtree(self.cache_dir)
    
    def test_basic_caching(self):
        """Test basic caching with real data."""
        loader = CachedYFinanceLoader(cache_dir=self.cache_dir)
        
        # First load - should download
        start = time.time()
        data1 = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
        time1 = time.time() - start
        
        # Second load - should use cache
        start = time.time()
        data2 = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
        time2 = time.time() - start
        
        # Both should return valid data
        self.assertGreater(len(data1), 0, "First load should return data")
        self.assertEqual(len(data1), len(data2), "Data should be identical")
        
        # Get cache info
        info = loader.get_cache_info('SPY')
        self.assertIsNotNone(info)
        self.assertGreater(info['rows'], 0)
    
    def test_append_missing_data(self):
        """Test appending missing data."""
        loader = CachedYFinanceLoader(cache_dir=self.cache_dir)
        
        # Load January data
        data1 = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
        rows1 = len(data1)
        
        # Extend to include February (should append)
        data2 = loader.load_ticker('SPY', '2024-01-01', '2024-02-29', progress=False)
        rows2 = len(data2)
        
        # Should have more data
        self.assertGreater(rows2, rows1, "Extended data should have more rows")
        
        # Load the extended range again (should be fully cached)
        start = time.time()
        data3 = loader.load_ticker('SPY', '2024-01-01', '2024-02-29', progress=False)
        time3 = time.time() - start
        
        # Should be fast (fully cached)
        self.assertLess(time3, 1.0, "Cached load should be fast")
        self.assertEqual(len(data3), rows2, "Data should be identical")
    
    def test_portfolio_integration(self):
        """Test CachedYFinanceLoader with Portfolio."""
        loader = CachedYFinanceLoader(cache_dir=self.cache_dir)
        
        # First portfolio load
        portfolio1 = Portfolio(['SPY', 'QQQ'], '2024-01-01', '2024-01-31', loader=loader)
        portfolio1.load_data(progress=False)
        
        # Second portfolio load (should use cache)
        portfolio2 = Portfolio(['SPY', 'QQQ'], '2024-01-01', '2024-01-31', loader=loader)
        portfolio2.load_data(progress=False)
        
        # Verify data was loaded
        self.assertGreater(len(portfolio1.data), 0, "First load should return data")
        self.assertGreater(len(portfolio2.data), 0, "Second load should return data")
        
        # Verify data has same shape
        spy_close1 = portfolio1.get_close_prices('SPY')
        spy_close2 = portfolio2.get_close_prices('SPY')
        self.assertEqual(len(spy_close1), len(spy_close2), "Data should have same length")
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        loader = CachedYFinanceLoader(cache_dir=self.cache_dir)
        
        # Load multiple tickers
        tickers = ['SPY', 'QQQ', 'IWM']
        for ticker in tickers:
            loader.load_ticker(ticker, '2024-01-01', '2024-01-31', progress=False)
            info = loader.get_cache_info(ticker)
            self.assertIsNotNone(info)
            self.assertGreater(info['rows'], 0)
        
        # Show cache directory contents
        cache_path = Path(self.cache_dir)
        cache_files = list(cache_path.glob('*.csv'))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        self.assertEqual(len(cache_files), len(tickers), "Should have one file per ticker")
        self.assertGreater(total_size, 0, "Cache files should have content")


if __name__ == '__main__':
    unittest.main()

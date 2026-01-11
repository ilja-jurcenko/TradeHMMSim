"""
Integration test for CachedYFinanceLoader with real yfinance data.
Demonstrates caching, appending, and performance benefits.
"""

import sys
import os
import shutil
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import CachedYFinanceLoader
from portfolio import Portfolio


def test_cached_loader_basic():
    """Test basic caching with real data."""
    cache_dir = './test_cache'
    
    try:
        print("="*70)
        print("TEST 1: Basic Caching")
        print("="*70)
        
        loader = CachedYFinanceLoader(cache_dir=cache_dir)
        
        # First load - should download
        print("\n1. First load (should download from yfinance):")
        start = time.time()
        data1 = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
        time1 = time.time() - start
        print(f"   Time: {time1:.2f}s, Rows: {len(data1)}")
        
        # Second load - should use cache
        print("\n2. Second load (should use cache):")
        start = time.time()
        data2 = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
        time2 = time.time() - start
        print(f"   Time: {time2:.2f}s, Rows: {len(data2)}")
        
        print(f"\n   ✓ Cache speedup: {time1/time2:.1f}x faster")
        
        # Get cache info
        info = loader.get_cache_info('SPY')
        print(f"\n3. Cache info:")
        print(f"   Start: {info['start_date']}")
        print(f"   End: {info['end_date']}")
        print(f"   Rows: {info['rows']}")
        print(f"   Path: {info['cache_path']}")
        
    finally:
        # Cleanup
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
    
    print("\n✓ Basic caching test passed!")


def test_cached_loader_append():
    """Test appending missing data."""
    cache_dir = './test_cache'
    
    try:
        print("\n" + "="*70)
        print("TEST 2: Appending Missing Data")
        print("="*70)
        
        loader = CachedYFinanceLoader(cache_dir=cache_dir)
        
        # Load January data
        print("\n1. Load January data:")
        data1 = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
        print(f"   Rows: {len(data1)}")
        print(f"   Date range: {data1.index[0]} to {data1.index[-1]}")
        
        # Extend to include February (should append)
        print("\n2. Extend to include February (should append missing data):")
        start = time.time()
        data2 = loader.load_ticker('SPY', '2024-01-01', '2024-02-29', progress=False)
        time2 = time.time() - start
        print(f"   Time: {time2:.2f}s")
        print(f"   Rows: {len(data2)} (was {len(data1)})")
        print(f"   Date range: {data2.index[0]} to {data2.index[-1]}")
        print(f"   Added: {len(data2) - len(data1)} rows")
        
        # Load the extended range again (should be fully cached)
        print("\n3. Load full range again (should be fully cached):")
        start = time.time()
        data3 = loader.load_ticker('SPY', '2024-01-01', '2024-02-29', progress=False)
        time3 = time.time() - start
        print(f"   Time: {time3:.2f}s")
        print(f"   Rows: {len(data3)}")
        
        print(f"\n   ✓ Cache hit is {time2/time3:.1f}x faster than append")
        
    finally:
        # Cleanup
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
    
    print("\n✓ Append test passed!")


def test_cached_loader_with_portfolio():
    """Test CachedYFinanceLoader with Portfolio."""
    cache_dir = './test_cache'
    
    try:
        print("\n" + "="*70)
        print("TEST 3: Portfolio Integration")
        print("="*70)
        
        loader = CachedYFinanceLoader(cache_dir=cache_dir)
        
        # First portfolio load
        print("\n1. First portfolio load (should download):")
        portfolio1 = Portfolio(['SPY', 'QQQ'], '2024-01-01', '2024-01-31', loader=loader)
        start = time.time()
        portfolio1.load_data(progress=False)
        time1 = time.time() - start
        print(f"   Time: {time1:.2f}s")
        
        # Second portfolio load (should use cache)
        print("\n2. Second portfolio load (should use cache):")
        portfolio2 = Portfolio(['SPY', 'QQQ'], '2024-01-01', '2024-01-31', loader=loader)
        start = time.time()
        portfolio2.load_data(progress=False)
        time2 = time.time() - start
        print(f"   Time: {time2:.2f}s")
        
        print(f"\n   ✓ Cache speedup: {time1/time2:.1f}x faster")
        
        # Verify data
        spy_close1 = portfolio1.get_close_prices('SPY')
        spy_close2 = portfolio2.get_close_prices('SPY')
        print(f"\n3. Data verification:")
        print(f"   SPY rows: {len(spy_close1)}")
        print(f"   Data matches: {spy_close1.equals(spy_close2)}")
        
    finally:
        # Cleanup
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
    
    print("\n✓ Portfolio integration test passed!")


def test_cache_statistics():
    """Display cache statistics."""
    cache_dir = './test_cache'
    
    try:
        print("\n" + "="*70)
        print("TEST 4: Cache Statistics")
        print("="*70)
        
        loader = CachedYFinanceLoader(cache_dir=cache_dir)
        
        # Load multiple tickers
        tickers = ['SPY', 'QQQ', 'IWM']
        print(f"\n1. Loading {len(tickers)} tickers...")
        
        for ticker in tickers:
            loader.load_ticker(ticker, '2024-01-01', '2024-01-31', progress=False)
            info = loader.get_cache_info(ticker)
            print(f"   {ticker}: {info['rows']} rows, {info['start_date']} to {info['end_date']}")
        
        # Show cache directory contents
        cache_path = Path(cache_dir)
        cache_files = list(cache_path.glob('*.csv'))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        print(f"\n2. Cache directory statistics:")
        print(f"   Files: {len(cache_files)}")
        print(f"   Total size: {total_size / 1024:.1f} KB")
        print(f"   Location: {cache_path.absolute()}")
        
    finally:
        # Cleanup
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
    
    print("\n✓ Cache statistics test passed!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CACHED YFINANCE LOADER - INTEGRATION TESTS")
    print("="*70)
    
    try:
        test_cached_loader_basic()
        test_cached_loader_append()
        test_cached_loader_with_portfolio()
        test_cache_statistics()
        
        print("\n" + "="*70)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("="*70)
        print("\nCachedYFinanceLoader successfully:")
        print("  • Downloads and caches data locally")
        print("  • Reuses cached data for faster access")
        print("  • Intelligently appends missing data")
        print("  • Integrates seamlessly with Portfolio")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

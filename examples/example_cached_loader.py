"""
Complete example demonstrating CachedYFinanceLoader usage.
Shows caching, appending, and performance benefits.
"""

from loaders import CachedYFinanceLoader
from portfolio import Portfolio
import time


def example_basic_caching():
    """Example 1: Basic caching."""
    print("="*70)
    print("EXAMPLE 1: Basic Caching")
    print("="*70)
    
    loader = CachedYFinanceLoader(cache_dir='./cache')
    
    print("\n1. First load (downloads from yfinance):")
    start = time.time()
    data = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
    elapsed = time.time() - start
    print(f"   Loaded {len(data)} rows in {elapsed:.2f}s")
    
    print("\n2. Second load (uses cache):")
    start = time.time()
    data = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
    elapsed = time.time() - start
    print(f"   Loaded {len(data)} rows in {elapsed:.2f}s (much faster!)")
    
    print("\n3. Cache info:")
    info = loader.get_cache_info('SPY')
    print(f"   Cached data: {info['start_date']} to {info['end_date']}")
    print(f"   Total rows: {info['rows']}")


def example_gap_filling():
    """Example 2: Intelligent gap filling."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Intelligent Gap Filling")
    print("="*70)
    
    loader = CachedYFinanceLoader(cache_dir='./cache')
    
    print("\n1. Load January data:")
    data1 = loader.load_ticker('QQQ', '2024-01-01', '2024-01-31', progress=False)
    print(f"   Loaded {len(data1)} rows")
    
    print("\n2. Extend to include February (only downloads February!):")
    data2 = loader.load_ticker('QQQ', '2024-01-01', '2024-02-29', progress=False)
    print(f"   Now have {len(data2)} rows (added {len(data2) - len(data1)} rows)")
    
    print("\n3. Request subset (instant, no download):")
    data3 = loader.load_ticker('QQQ', '2024-01-15', '2024-01-31', progress=False)
    print(f"   Subset has {len(data3)} rows (from cache)")


def example_with_portfolio():
    """Example 3: Using with Portfolio."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Portfolio Integration")
    print("="*70)
    
    loader = CachedYFinanceLoader(cache_dir='./cache')
    
    print("\n1. Create portfolio with cached loader:")
    portfolio = Portfolio(['SPY'], '2024-01-01', '2024-01-31', loader=loader)
    portfolio.load_data(progress=False)
    
    close = portfolio.get_close_prices('SPY')
    print(f"   Loaded {len(close)} days of SPY data")
    print(f"   First: ${close.iloc[0]:.2f}")
    print(f"   Last: ${close.iloc[-1]:.2f}")
    print(f"   Return: {((close.iloc[-1] / close.iloc[0]) - 1) * 100:.2f}%")


def example_cache_management():
    """Example 4: Cache management."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Cache Management")
    print("="*70)
    
    loader = CachedYFinanceLoader(cache_dir='./cache')
    
    print("\n1. Check cache for non-existent ticker:")
    info = loader.get_cache_info('AAPL')
    print(f"   Cache exists: {info is not None}")
    
    print("\n2. Load data:")
    loader.load_ticker('AAPL', '2024-01-01', '2024-01-31', progress=False)
    
    print("\n3. Check cache again:")
    info = loader.get_cache_info('AAPL')
    if info:
        print(f"   Cache exists: True")
        print(f"   Rows: {info['rows']}")
        print(f"   File: {info['cache_path']}")
    
    print("\n4. Clear cache:")
    loader.clear_cache('AAPL')
    info = loader.get_cache_info('AAPL')
    print(f"   Cache exists after clear: {info is not None}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("CACHED YFINANCE LOADER - COMPLETE EXAMPLES")
    print("="*70)
    
    try:
        example_basic_caching()
        example_gap_filling()
        example_with_portfolio()
        example_cache_management()
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nKey Takeaways:")
        print("  • CachedYFinanceLoader caches data locally for faster access")
        print("  • Intelligently detects and downloads only missing data")
        print("  • Works seamlessly with Portfolio")
        print("  • Perfect for development and backtesting")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

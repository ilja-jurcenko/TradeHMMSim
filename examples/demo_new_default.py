"""
Quick demonstration of the new default behavior.
Shows that Portfolio now uses CachedYFinanceLoader by default.
"""

from portfolio import Portfolio
from loaders import CachedYFinanceLoader, YFinanceLoader
import time
import shutil
from pathlib import Path


def demo_new_default():
    """Demonstrate new default caching behavior."""
    print("="*70)
    print("DEMONSTRATION: New Default Behavior")
    print("="*70)
    
    # Clean up any existing cache
    cache_path = Path('./data')
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print("Cleaned existing cache\n")
    
    print("1. Creating Portfolio (no explicit loader):")
    portfolio = Portfolio(['SPY'], '2024-01-15', '2024-01-20')
    print(f"   Loader type: {type(portfolio.loader).__name__}")
    print(f"   Cache directory: {portfolio.loader.cache_dir}")
    
    # Verify it's the cached loader
    assert isinstance(portfolio.loader, CachedYFinanceLoader), \
        "Default should be CachedYFinanceLoader"
    
    print("\n2. First load (will download and cache):")
    start = time.time()
    portfolio.load_data(progress=False)
    time1 = time.time() - start
    print(f"   Time: {time1:.2f}s")
    print(f"   Data cached to: {cache_path}/SPY.csv")
    print(f"   Cache exists: {(cache_path / 'SPY.csv').exists()}")
    
    print("\n3. Second load (will use cache):")
    portfolio2 = Portfolio(['SPY'], '2024-01-15', '2024-01-20')
    start = time.time()
    portfolio2.load_data(progress=False)
    time2 = time.time() - start
    print(f"   Time: {time2:.2f}s")
    print(f"   Speedup: {time1/time2:.1f}x faster!")
    
    print("\n4. To disable caching (if needed):")
    print("   loader = YFinanceLoader()  # No caching")
    print("   portfolio = Portfolio(['SPY'], '2024-01-15', '2024-01-20', loader=loader)")
    
    print("\n" + "="*70)
    print("✓ New default provides automatic caching for better performance!")
    print("="*70)
    
    # Cleanup
    if cache_path.exists():
        shutil.rmtree(cache_path)


if __name__ == '__main__':
    demo_new_default()

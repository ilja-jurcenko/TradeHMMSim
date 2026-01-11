"""
Quick integration test to verify loader refactoring works correctly.
This test uses real data to ensure backward compatibility.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio import Portfolio
from loaders import YFinanceLoader, CachedYFinanceLoader

def test_portfolio_basic_usage():
    """Test basic portfolio usage with default CachedYFinanceLoader."""
    print("Testing Portfolio with default CachedYFinanceLoader...")
    
    # Create portfolio (should use CachedYFinanceLoader by default)
    portfolio = Portfolio(['SPY'], '2024-01-01', '2024-01-31')
    
    # Load data
    portfolio.load_data(progress=False)
    
    # Verify data was loaded
    assert len(portfolio.data) > 0, "No data loaded"
    assert 'SPY' in portfolio.data, "SPY not in data"
    
    # Get close prices
    close = portfolio.get_close_prices('SPY')
    assert len(close) > 0, "No close prices"
    
    print(f"✓ Loaded {len(close)} days of SPY data")
    print("✓ Default loader works correctly")


def test_portfolio_custom_loader():
    """Test portfolio with explicitly provided loader."""
    print("\nTesting Portfolio with explicit YFinanceLoader...")
    
    # Create loader explicitly
    loader = YFinanceLoader()
    
    # Create portfolio with explicit loader
    portfolio = Portfolio(['SPY'], '2024-01-01', '2024-01-31', loader=loader)
    
    # Load data
    portfolio.load_data(progress=False)
    
    # Verify data was loaded
    assert len(portfolio.data) > 0, "No data loaded"
    assert 'SPY' in portfolio.data, "SPY not in data"
    
    close = portfolio.get_close_prices('SPY')
    assert len(close) > 0, "No close prices"
    
    print(f"✓ Loaded {len(close)} days of SPY data")
    print("✓ Custom loader works correctly")


def test_loader_directly():
    """Test YFinanceLoader directly."""
    print("\nTesting YFinanceLoader directly...")
    
    loader = YFinanceLoader()
    
    # Load single ticker
    data = loader.load_ticker('SPY', '2024-01-01', '2024-01-31', progress=False)
    
    assert data is not None, "Failed to load data"
    assert len(data) > 0, "No data returned"
    assert 'Close' in data.columns, "Close column missing"
    
    print(f"✓ Loaded {len(data)} days of SPY data")
    
    # Load multiple tickers
    data_dict = loader.load_multiple_tickers(['SPY', 'QQQ'], '2024-01-01', '2024-01-31', progress=False)
    
    assert len(data_dict) == 2, "Should have loaded 2 tickers"
    assert 'SPY' in data_dict, "SPY missing"
    assert 'QQQ' in data_dict, "QQQ missing"
    
    print(f"✓ Loaded data for {len(data_dict)} tickers")
    print("✓ Direct loader usage works correctly")


if __name__ == '__main__':
    print("="*60)
    print("LOADER REFACTORING INTEGRATION TEST")
    print("="*60)
    
    try:
        test_portfolio_basic_usage()
        test_portfolio_custom_loader()
        test_loader_directly()
        
        print("\n" + "="*60)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

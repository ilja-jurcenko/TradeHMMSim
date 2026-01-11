# Data Loaders Module

This module provides a flexible, pluggable architecture for loading market data from various sources.

## Overview

The Portfolio class has been refactored to separate data loading concerns from portfolio management. This allows you to easily switch between different data providers (Yahoo Finance, CSV files, databases, APIs, etc.) without changing your core logic.

### Available Loaders

1. **YFinanceLoader** - Direct downloads from Yahoo Finance
2. **CachedYFinanceLoader** - Yahoo Finance with intelligent local caching
3. **CSVLoader** - Load from local CSV files  
4. **Custom Loaders** - Easy to implement for your own data sources

## Architecture

```
loaders/
├── __init__.py              # Module exports
├── base_loader.py           # Abstract base class
├── yfinance_loader.py       # Yahoo Finance implementation
└── csv_loader.py            # CSV file implementation
```

### Key Components

1. **BaseDataLoader**: Abstract base class defining the loader interface
2. **YFinanceLoader**: Loads data from Yahoo Finance (default)
3. **CSVLoader**: Loads data from local CSV files
4. **Custom Loaders**: Easy to implement for your own data sources

## Usage

### Basic Usage with Default Loader

The Portfolio class uses `YFinanceLoader` by default:

```python
from portfolio import Portfolio

# Uses YFinanceLoader automatically
portfolio = Portfolio(['SPY', 'QQQ'], '2020-01-01', '2024-12-31')
portfolio.load_data()

close_prices = portfolio.get_close_prices('SPY')
```

### Using a Specific Loader

You can explicitly provide a loader:

```python
from portfolio import Portfolio
from loaders import YFinanceLoader

loader = YFinanceLoader()
portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31', loader=loader)
portfolio.load_data()
```

### Using CSV Loader

Load data from local CSV files:

```python
from portfolio import Portfolio
from loaders import CSVLoader

# CSV files should be in ./data/ directory
# Expected files: ./data/SPY.csv, ./data/QQQ.csv
loader = CSVLoader(data_dir='./data')
portfolio = Portfolio(['SPY', 'QQQ'], '2020-01-01', '2024-12-31', loader=loader)
portfolio.load_data()
```

### Using Cached YFinance Loader (Recommended for Development)

Best for development and backtesting - downloads once, caches locally:

```python
from portfolio import Portfolio
from loaders import CachedYFinanceLoader

# Creates ./cache/ directory with CSV files
loader = CachedYFinanceLoader(cache_dir='./cache')
portfolio = Portfolio(['SPY', 'QQQ'], '2020-01-01', '2024-12-31', loader=loader)
portfolio.load_data()  # First time: downloads and caches

# Second run: instant, uses cache
portfolio2 = Portfolio(['SPY', 'QQQ'], '2020-01-01', '2024-12-31', loader=loader)
portfolio2.load_data()  # Lightning fast! No network call
```

**Key Features:**
- Automatically caches downloaded data to CSV
- Detects missing data ranges and downloads only what's needed
- Intelligently appends new data to existing cache
- 2-10x faster for repeated access
- Perfect for iterative development

See [Cached Loader Guide](../docs/CACHED_LOADER_GUIDE.md) for detailed documentation.

### Using Loaders Directly

You can also use loaders independently of Portfolio:

```python
from loaders import YFinanceLoader

loader = YFinanceLoader()

# Load single ticker
data = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')
print(data.head())

# Load multiple tickers
data_dict = loader.load_multiple_tickers(
    ['SPY', 'QQQ', 'IWM'], 
    '2020-01-01', 
    '2024-12-31'
)

for ticker, data in data_dict.items():
    print(f"{ticker}: {len(data)} rows")
```

## Creating Custom Loaders

To create your own data loader, inherit from `BaseDataLoader` and implement the required methods:

```python
from loaders import BaseDataLoader
import pandas as pd
from typing import Optional, List, Dict

class MyCustomLoader(BaseDataLoader):
    """Load data from my custom source."""
    
    def load_ticker(self, ticker: str, start_date: str, end_date: str, 
                   progress: bool = False) -> Optional[pd.DataFrame]:
        """
        Load data for a single ticker.
        
        Returns:
        --------
        pd.DataFrame with DatetimeIndex and columns:
            - Open, High, Low, Close, Volume (at minimum Close is required)
        """
        # Your implementation here
        # Must return DataFrame with DatetimeIndex
        pass
    
    def load_multiple_tickers(self, tickers: List[str], start_date: str, 
                             end_date: str, progress: bool = False) -> Dict[str, pd.DataFrame]:
        """Load data for multiple tickers."""
        data_dict = {}
        for ticker in tickers:
            data = self.load_ticker(ticker, start_date, end_date, progress)
            if data is not None:
                data_dict[ticker] = data
        return data_dict

# Use your custom loader
loader = MyCustomLoader()
portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31', loader=loader)
```

## CSV File Format

For `CSVLoader`, your CSV files should have this format:

```csv
Date,Open,High,Low,Close,Volume
2020-01-02,324.87,325.62,323.10,325.12,128432000
2020-01-03,324.52,326.63,324.20,325.81,95654000
...
```

**Requirements:**
- Date column (can be index or named column)
- At minimum, a `Close` column is required
- Recommended: Open, High, Low, Close, Volume

## Testing

Run the loader tests:

```bash
python3 tests/test_loaders.py
```

Run integration tests with real data:

```bash
python3 tests/test_loader_integration.py
```

## Benefits of This Architecture

1. **Separation of Concerns**: Portfolio management is decoupled from data sources
2. **Flexibility**: Easy to switch between data providers
3. **Testability**: Mock loaders can be used for testing without network calls
4. **Extensibility**: Add new data sources by implementing BaseDataLoader
5. **Backward Compatibility**: Existing code works with default YFinanceLoader

## Migration Guide

If you have existing code using Portfolio, no changes are needed! The default behavior uses YFinanceLoader automatically.

### Before (still works):
```python
portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31')
portfolio.load_data()
```

### After (optional, more explicit):
```python
from loaders import YFinanceLoader

loader = YFinanceLoader()
portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31', loader=loader)
portfolio.load_data()
```

## Examples

See the `tests/` directory for comprehensive examples:
- `tests/test_loaders.py` - Unit tests with mock data
- `tests/test_loader_integration.py` - Integration tests with real data

## Future Enhancements

Potential additional loaders:
- **Database Loader**: Load from PostgreSQL, MySQL, etc.
- **API Loader**: Load from custom REST APIs
- **Parquet Loader**: Load from Apache Parquet files
- **Cache Loader**: Wrapper that caches data locally
- **Combined Loader**: Try multiple sources with fallback logic

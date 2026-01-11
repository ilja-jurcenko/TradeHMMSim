# Data Loaders Quick Reference

## Quick Start

### Default Usage (No Changes Needed)
```python
from portfolio import Portfolio

portfolio = Portfolio(['SPY', 'QQQ'], '2020-01-01', '2024-12-31')
portfolio.load_data()
```

### Using CSV Files
```python
from portfolio import Portfolio
from loaders import CSVLoader

loader = CSVLoader(data_dir='./data')
portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31', loader=loader)
portfolio.load_data()
```

### Direct Loader Usage
```python
from loaders import YFinanceLoader

loader = YFinanceLoader()
data = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')
print(data.head())
```

## Creating Custom Loaders

```python
from loaders import BaseDataLoader
import pandas as pd

class MyLoader(BaseDataLoader):
    def load_ticker(self, ticker, start_date, end_date, progress=False):
        # Your implementation
        # Must return DataFrame with DatetimeIndex and 'Close' column
        return pd.DataFrame(...)
    
    def load_multiple_tickers(self, tickers, start_date, end_date, progress=False):
        return {t: self.load_ticker(t, start_date, end_date) for t in tickers}
```

## Testing with Mock Data

```python
from loaders import BaseDataLoader
import pandas as pd
import numpy as np

class MockLoader(BaseDataLoader):
    def load_ticker(self, ticker, start_date, end_date, progress=False):
        dates = pd.date_range(start_date, end_date, freq='D')
        return pd.DataFrame({
            'Close': np.random.rand(len(dates)) * 100
        }, index=dates)
    
    def load_multiple_tickers(self, tickers, start_date, end_date, progress=False):
        return {t: self.load_ticker(t, start_date, end_date) for t in tickers}

# Use in tests
mock_loader = MockLoader()
portfolio = Portfolio(['SPY'], '2020-01-01', '2020-12-31', loader=mock_loader)
portfolio.load_data()  # Fast, no network calls!
```

## Available Loaders

| Loader | Purpose | Usage |
|--------|---------|-------|
| `YFinanceLoader` | Yahoo Finance (default) | `loader = YFinanceLoader()` |
| `CSVLoader` | Local CSV files | `loader = CSVLoader(data_dir='./data')` |
| `BaseDataLoader` | Custom loader base class | Inherit and implement |

## CSV File Format

Place CSV files in your data directory (e.g., `./data/SPY.csv`):

```csv
Date,Open,High,Low,Close,Volume
2020-01-02,324.87,325.62,323.10,325.12,128432000
2020-01-03,324.52,326.63,324.20,325.81,95654000
```

## Running Tests

```bash
# Unit tests (with mocks, fast)
python3 tests/test_loaders.py

# Integration tests (real data, requires internet)
python3 tests/test_loader_integration.py
```

## Common Patterns

### Fallback Logic
```python
class FallbackLoader(BaseDataLoader):
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
    
    def load_ticker(self, ticker, start_date, end_date, progress=False):
        data = self.primary.load_ticker(ticker, start_date, end_date, progress)
        if data is None:
            data = self.fallback.load_ticker(ticker, start_date, end_date, progress)
        return data
```

### Caching Wrapper
```python
class CachedLoader(BaseDataLoader):
    def __init__(self, inner_loader):
        self.inner = inner_loader
        self.cache = {}
    
    def load_ticker(self, ticker, start_date, end_date, progress=False):
        key = (ticker, start_date, end_date)
        if key not in self.cache:
            self.cache[key] = self.inner.load_ticker(ticker, start_date, end_date, progress)
        return self.cache[key]
```

## Migration Checklist

- [ ] No changes needed for existing code (uses YFinanceLoader by default)
- [ ] To use CSV: Create `loaders` directory with CSV files
- [ ] To create custom loader: Inherit from `BaseDataLoader`
- [ ] Run tests to verify: `python3 tests/test_loaders.py`

## Full Documentation

See `loaders/README.md` for complete documentation.

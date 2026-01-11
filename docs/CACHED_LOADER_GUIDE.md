# Cached YFinance Loader - Quick Guide

## Overview

`CachedYFinanceLoader` provides intelligent caching of Yahoo Finance data with automatic gap detection and data appending. Perfect for development and backtesting where you repeatedly access the same data.

## Key Features

- **Automatic Caching**: Downloads from yfinance once, caches locally in CSV files
- **Smart Gap Detection**: Detects missing date ranges in cache
- **Intelligent Appending**: Downloads only missing data and appends to cache
- **Performance**: 2-10x faster for cached data (no network calls)
- **Persistent**: Cache survives across Python sessions

## Quick Start

### Basic Usage

```python
from loaders import CachedYFinanceLoader

# Create loader with cache directory
loader = CachedYFinanceLoader(cache_dir='./cache')

# First call: downloads and caches
data1 = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')
# Fast! Uses yfinance, saves to ./cache/SPY.csv

# Second call: uses cache
data2 = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')
# Lightning fast! No network call, reads from ./cache/SPY.csv
```

### With Portfolio

```python
from portfolio import Portfolio
from loaders import CachedYFinanceLoader

# Create cached loader
loader = CachedYFinanceLoader(cache_dir='./cache')

# Use with portfolio
portfolio = Portfolio(['SPY', 'QQQ'], '2020-01-01', '2024-12-31', loader=loader)
portfolio.load_data()  # First time: downloads and caches

# Later, in another script or session:
portfolio2 = Portfolio(['SPY', 'QQQ'], '2020-01-01', '2024-12-31', loader=loader)
portfolio2.load_data()  # Instant! Uses cached data
```

## Intelligent Gap Filling

### Scenario 1: Extending into the Future

```python
loader = CachedYFinanceLoader(cache_dir='./cache')

# Initial load: 2024 data
data1 = loader.load_ticker('SPY', '2024-01-01', '2024-12-31')
# Downloads 2024 data, saves to cache

# Later, extend to include 2025 (only downloads new data!)
data2 = loader.load_ticker('SPY', '2024-01-01', '2025-12-31')
# Smart: Downloads only 2025 data, appends to cache
# Returns: Combined 2024 + 2025 data
```

### Scenario 2: Extending into the Past

```python
loader = CachedYFinanceLoader(cache_dir='./cache')

# Initial load: 2024 data
data1 = loader.load_ticker('SPY', '2024-01-01', '2024-12-31')

# Later, extend to include 2023
data2 = loader.load_ticker('SPY', '2023-01-01', '2024-12-31')
# Smart: Downloads only 2023 data, prepends to cache
# Returns: Combined 2023 + 2024 data
```

### Scenario 3: Working with Subsets

```python
loader = CachedYFinanceLoader(cache_dir='./cache')

# Cache full year
data1 = loader.load_ticker('SPY', '2024-01-01', '2024-12-31')

# Request subset (no download!)
data2 = loader.load_ticker('SPY', '2024-06-01', '2024-06-30')
# Fast! Just filters cached data
```

## Advanced Usage

### Force Refresh

```python
# Bypass cache and download fresh data
data = loader.load_ticker('SPY', '2024-01-01', '2024-12-31', force_download=True)
# Always downloads, overwrites cache
```

### Cache Management

```python
# Get cache information
info = loader.get_cache_info('SPY')
print(f"Cached: {info['start_date']} to {info['end_date']}")
print(f"Rows: {info['rows']}")
print(f"Location: {info['cache_path']}")

# Clear cache for one ticker
loader.clear_cache('SPY')

# Clear all cache
loader.clear_cache()
```

### Multiple Tickers

```python
# Load multiple tickers (caches each one)
data_dict = loader.load_multiple_tickers(
    ['SPY', 'QQQ', 'IWM'],
    '2024-01-01',
    '2024-12-31'
)

# Second call uses cache for all
data_dict2 = loader.load_multiple_tickers(
    ['SPY', 'QQQ', 'IWM'],
    '2024-01-01',
    '2024-12-31'
)
# Fast! All from cache
```

## Performance Comparison

### Without Caching (YFinanceLoader)
```python
from loaders import YFinanceLoader

loader = YFinanceLoader()

# Every call downloads
data1 = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')  # 2.5s
data2 = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')  # 2.5s
data3 = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')  # 2.5s
# Total: 7.5s for 3 calls
```

### With Caching (CachedYFinanceLoader)
```python
from loaders import CachedYFinanceLoader

loader = CachedYFinanceLoader(cache_dir='./cache')

data1 = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')  # 2.5s (download)
data2 = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')  # 0.1s (cache)
data3 = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')  # 0.1s (cache)
# Total: 2.7s for 3 calls (2.8x faster!)
```

## Cache Directory Structure

```
./cache/
├── SPY.csv      # Cached SPY data
├── QQQ.csv      # Cached QQQ data
├── IWM.csv      # Cached IWM data
└── ...
```

Each CSV file contains:
- Date index
- OHLCV columns (Open, High, Low, Close, Volume)
- Sorted by date

## Use Cases

### Development & Testing
```python
# Perfect for development - fast iteration
loader = CachedYFinanceLoader(cache_dir='./cache')

# First run: downloads once
for i in range(100):
    data = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')
    # process data...
# Subsequent runs: instant cache hits
```

### Backtesting Multiple Strategies
```python
# Download once, test many strategies
loader = CachedYFinanceLoader(cache_dir='./backtest_cache')

# Load data once
strategies = ['SMA', 'EMA', 'RSI', 'MACD']
for strategy in strategies:
    portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31', loader=loader)
    portfolio.load_data()  # Only first strategy downloads!
    # run backtest...
```

### Incremental Data Updates
```python
# Daily update: only download new data
loader = CachedYFinanceLoader(cache_dir='./cache')

# Initial: download all historical
data = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')

# Next day: only download latest day
import datetime
today = datetime.date.today().strftime('%Y-%m-%d')
data_updated = loader.load_ticker('SPY', '2020-01-01', today)
# Smart: Only downloads data after 2024-12-31
```

## Best Practices

1. **Use Separate Cache Directories**: Different projects → different cache dirs
2. **Clear Cache Periodically**: Old data might need refreshing
3. **Development vs Production**: Cache for dev, fresh downloads for production
4. **Monitor Cache Size**: Can grow large with many tickers/long periods

## Comparison with Other Loaders

| Feature | YFinanceLoader | CachedYFinanceLoader | CSVLoader |
|---------|---------------|---------------------|-----------|
| Speed (first load) | Fast | Fast | Instant |
| Speed (cached) | N/A | Instant | Instant |
| Auto-update | Always fresh | Smart append | Manual |
| Network required | Yes | First time only | No |
| Storage | None | Automatic CSV | Manual CSV |
| Best for | Always fresh data | Development/Backtesting | Static datasets |

## Troubleshooting

### Cache not updating?
```python
# Force refresh
loader.load_ticker('SPY', '2020-01-01', '2024-12-31', force_download=True)
```

### Wrong data in cache?
```python
# Clear and reload
loader.clear_cache('SPY')
data = loader.load_ticker('SPY', '2020-01-01', '2024-12-31')
```

### Cache too large?
```python
# Clear old tickers
loader.clear_cache('UNUSED_TICKER')

# Or clear all
loader.clear_cache()
```

## Examples

See `tests/test_cached_loader_real.py` for complete working examples.

# Portfolio Weight Management

## Overview

The Portfolio class now supports multi-asset portfolios with dynamic weight allocation based on market regimes detected by the HMM filter.

## Key Features

### 1. Weight Initialization

Portfolios can be initialized with custom weights or use equal weighting by default:

```python
# Equal weights (default)
portfolio = Portfolio(['SPY', 'AGG'], '2020-01-01', '2023-12-31')
# Weights: {'SPY': 0.5, 'AGG': 0.5}

# Custom weights
portfolio = Portfolio(
    ['SPY', 'AGG'], 
    '2020-01-01', 
    '2023-12-31',
    weights={'SPY': 0.6, 'AGG': 0.4}
)
```

### 2. Weight Validation

Weights are validated on initialization and when set:
- Keys must match ticker list exactly
- Values must sum to 1.0 (±0.01 tolerance)

```python
# This will raise ValueError - weights don't sum to 1.0
portfolio = Portfolio(
    ['SPY', 'AGG'],
    '2020-01-01',
    '2023-12-31', 
    weights={'SPY': 0.6, 'AGG': 0.5}  # Sums to 1.1
)

# This will raise ValueError - keys don't match tickers
portfolio = Portfolio(
    ['SPY', 'AGG'],
    '2020-01-01',
    '2023-12-31',
    weights={'SPY': 0.5, 'QQQ': 0.5}  # QQQ not in ticker list
)
```

### 3. Dynamic Weight Management

Weights can be updated at any time:

```python
portfolio = Portfolio(['SPY', 'AGG'], '2020-01-01', '2023-12-31')

# Get current weights
weights = portfolio.get_weights()
# Returns: {'SPY': 0.5, 'AGG': 0.5}

# Set new weights
portfolio.set_weights({'SPY': 0.7, 'AGG': 0.3})
```

### 4. Weighted Portfolio Returns

Calculate portfolio returns using current weights:

```python
portfolio = Portfolio(
    ['SPY', 'AGG'],
    '2020-01-01',
    '2023-12-31',
    weights={'SPY': 0.6, 'AGG': 0.4}
)
portfolio.load_data()

# Get weighted returns (SPY * 0.6 + AGG * 0.4)
weighted_returns = portfolio.get_weighted_returns()
```

### 5. Regime-Based Rebalancing

Automatically adjust weights based on HMM regime detection:

```python
portfolio = Portfolio(['SPY', 'AGG'], '2020-01-01', '2023-12-31')
portfolio.load_data()

# Bull or Neutral market → 100% SPY, 0% AGG
portfolio.rebalance_regime_based('bull')
# Weights: {'SPY': 1.0, 'AGG': 0.0}

# Bear market → 0% SPY, 100% AGG (defensive)
portfolio.rebalance_regime_based('bear')
# Weights: {'SPY': 0.0, 'AGG': 1.0}
```

## Regime-Based Allocation Strategy

The portfolio automatically adjusts between aggressive (SPY) and defensive (AGG) assets based on market regime:

| Regime | SPY Weight | AGG Weight | Rationale |
|--------|------------|------------|-----------|
| Bull | 100% | 0% | Maximize equity exposure during uptrends |
| Neutral | 100% | 0% | Maintain equity exposure in stable markets |
| Bear | 0% | 100% | Shift to defensive bonds during downturns |

### Custom Ticker Allocation

You can specify custom aggressive/defensive tickers:

```python
portfolio = Portfolio(['QQQ', 'TLT'], '2020-01-01', '2023-12-31')

# Bull → 100% QQQ
portfolio.rebalance_regime_based(
    'bull',
    aggressive_ticker='QQQ',
    defensive_ticker='TLT'
)

# Bear → 100% TLT
portfolio.rebalance_regime_based(
    'bear',
    aggressive_ticker='QQQ',
    defensive_ticker='TLT'
)
```

## Multi-Asset Portfolios

The weight system supports portfolios with more than two assets:

```python
portfolio = Portfolio(
    ['SPY', 'AGG', 'GLD'],
    '2020-01-01',
    '2023-12-31',
    weights={
        'SPY': 0.5,  # 50% equities
        'AGG': 0.3,  # 30% bonds
        'GLD': 0.2   # 20% gold
    }
)
```

## API Reference

### Portfolio.__init__()

```python
Portfolio(
    tickers: List[str],
    start_date: str,
    end_date: str,
    loader: Optional[BaseDataLoader] = None,
    weights: Optional[Dict[str, float]] = None
)
```

**Parameters:**
- `tickers`: List of ticker symbols
- `start_date`: Start date in 'YYYY-MM-DD' format
- `end_date`: End date in 'YYYY-MM-DD' format
- `loader`: Data loader (defaults to CachedYFinanceLoader)
- `weights`: Initial weights (defaults to equal weighting)

### Portfolio.set_weights()

```python
portfolio.set_weights(weights: Dict[str, float]) -> None
```

Update portfolio weights. Validates that weights sum to 1.0 and keys match tickers.

### Portfolio.get_weights()

```python
portfolio.get_weights() -> Dict[str, float]
```

Returns current portfolio weights as a dictionary.

### Portfolio.get_weighted_returns()

```python
portfolio.get_weighted_returns() -> pd.Series
```

Calculate portfolio returns using current weights.

### Portfolio.rebalance_regime_based()

```python
portfolio.rebalance_regime_based(
    regime: str,
    aggressive_ticker: str = 'SPY',
    defensive_ticker: str = 'AGG'
) -> None
```

Rebalance portfolio based on market regime.

**Parameters:**
- `regime`: 'bull', 'neutral', or 'bear'
- `aggressive_ticker`: Ticker to allocate to in bull/neutral markets
- `defensive_ticker`: Ticker to allocate to in bear markets

**Raises:**
- `ValueError`: If regime is invalid or required tickers not in portfolio

## Testing

Comprehensive unit tests verify weight management:

```bash
# Run portfolio weight tests
python -m unittest tests.unit.test_portfolio_weights -v

# Run all unit tests
python tests/run_unit_tests.py
```

Test coverage includes:
- Default equal weighting
- Custom weight initialization
- Weight validation (sum and keys)
- Dynamic weight updates
- Weighted return calculation
- Regime-based rebalancing (bull/neutral/bear)
- Custom ticker allocation
- Multi-asset portfolios
- Weight independence (defensive copying)

## Integration with Backtest

The BacktestEngine will automatically use weighted returns when provided with a Portfolio object that has regime-based rebalancing enabled. This allows strategies to dynamically shift between aggressive and defensive assets based on HMM regime detection.

## Example: Full Workflow

```python
from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import SMAAlphaModel

# Create portfolio with two assets
portfolio = Portfolio(['SPY', 'AGG'], '2020-01-01', '2023-12-31')
portfolio.load_data()

# Create alpha model
alpha_model = SMAAlphaModel(short_window=50, long_window=200)

# Run backtest with regime-based rebalancing
engine = BacktestEngine(
    portfolio=portfolio,
    alpha_model=alpha_model,
    use_hmm=True
)

results = engine.run()

# Results will show:
# - Total return with regime-based asset allocation
# - Performance metrics (Sharpe, max drawdown, etc.)
# - Regime switches and rebalancing events
```

## Benefits

1. **Risk Management**: Automatically shift to defensive assets during bear markets
2. **Flexibility**: Support for any combination of aggressive/defensive tickers
3. **Multi-Asset**: Handle portfolios with 2+ assets
4. **Validation**: Built-in weight validation prevents errors
5. **Transparency**: Easy to track and update current weights

## Future Enhancements

Potential extensions to the weight management system:

- Gradual rebalancing (e.g., 80/20 in neutral vs 100/0 in bull)
- Transaction cost modeling for rebalancing
- Weight bounds (min/max per asset)
- Custom rebalancing strategies beyond regime-based
- Rebalancing frequency controls
- Weight drift monitoring and alerts

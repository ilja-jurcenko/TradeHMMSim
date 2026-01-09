# Plotting Module Documentation

## Overview

The `plotter.py` module provides comprehensive visualization capabilities for backtesting results. It includes functions for plotting equity curves, drawdowns, regime probabilities, trading signals, and strategy comparisons.

## Features

### 1. Comprehensive Backtest Visualization
- **Equity Curves**: Compare strategy performance vs buy-and-hold benchmark
- **Price & Signals**: Visualize price action with entry/exit points and position highlights
- **Regime Probabilities**: Display HMM regime detection and probability evolution (when available)
- **Drawdown Analysis**: Compare drawdowns between strategy and benchmark

### 2. Strategy Comparison
- **Side-by-side Equity Curves**: Compare multiple strategies simultaneously
- **Drawdown Comparison**: Visual comparison of risk profiles
- **Metrics Bar Charts**: Compare key performance metrics across strategies

### 3. HMM Regime Analysis
- **Price Coloring by Regime**: Visualize market regimes on price chart
- **Regime Probabilities**: Track regime probability evolution
- **Regime Switches**: Highlight regime change points

## Usage Examples

### Example 1: Single Backtest Visualization

```python
from portfolio import Portfolio
from backtest import BacktestEngine
from AlphaModels import SMA
from SignalFilter import HMMRegimeFilter
from plotter import BacktestPlotter

# Setup and run backtest
portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

alpha_model = SMA(short_window=10, long_window=30)
hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)

engine = BacktestEngine(close, alpha_model, hmm_filter=hmm_filter)
results = engine.run(strategy_mode='alpha_hmm_override', walk_forward=True)

# Generate 4-panel visualization
BacktestPlotter.plot_results(results, close)
```

This creates a comprehensive 4-panel plot showing:
1. **Cumulative Returns**: Strategy vs Benchmark
2. **Price & Signals**: Price with entry/exit markers and position highlights
3. **Regime Probabilities**: HMM regime detection (if available)
4. **Drawdowns**: Strategy vs Benchmark drawdown comparison

### Example 2: Compare Multiple Strategies

```python
# Run multiple strategies
strategies = ['alpha_only', 'alpha_hmm_filter', 'alpha_hmm_override']
results_list = []
labels = []

for strategy in strategies:
    engine = BacktestEngine(close, alpha_model, hmm_filter=hmm_filter)
    result = engine.run(strategy_mode=strategy, walk_forward=True)
    results_list.append(result)
    labels.append(strategy.replace('_', ' ').title())

# Plot equity curves and drawdowns
BacktestPlotter.plot_comparison(results_list, labels)

# Plot metrics comparison (bar charts)
BacktestPlotter.plot_metrics_comparison(results_list, labels)
```

### Example 3: HMM Regime Analysis

```python
from SignalFilter import HMMRegimeFilter

# Train HMM
hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)

# Prepare features (returns + volatility)
returns = close.pct_change()
volatility = returns.rolling(20).std()
features = pd.DataFrame({'returns': returns, 'volatility': volatility}).dropna()

# Fit and get regime data
hmm_filter.fit(features.values)
probs = hmm_filter.filtered_state_probs(features.values)
probs_df = pd.DataFrame(probs, index=features.index, columns=range(3))
regime = pd.Series(probs.argmax(axis=1), index=features.index)
switches = regime[regime.ne(regime.shift(1))].dropna()

# Visualize regime analysis
BacktestPlotter.plot_regime_analysis(probs_df, regime, close, switches)
```

### Example 4: Compare Different Alpha Models

```python
from AlphaModels import SMA, EMA, KAMA

models = [
    ('SMA', SMA(10, 30)),
    ('EMA', EMA(10, 30)),
    ('KAMA', KAMA(10, 30))
]

results_list = []
labels = []

for name, model in models:
    engine = BacktestEngine(close, model)
    result = engine.run(strategy_mode='alpha_only')
    results_list.append(result)
    labels.append(name)

# Compare models
BacktestPlotter.plot_comparison(results_list, labels)
BacktestPlotter.plot_metrics_comparison(results_list, labels)
```

## API Reference

### BacktestPlotter.plot_results()

```python
BacktestPlotter.plot_results(results: dict, close: pd.Series, figsize: tuple = (14, 12)) -> None
```

**Parameters:**
- `results`: Backtest results dictionary from `BacktestEngine.run()`
- `close`: Close price series
- `figsize`: Figure size (width, height) in inches

**Generates:**
- 4-panel visualization with cumulative returns, price/signals, regime probs (if available), and drawdowns

### BacktestPlotter.plot_comparison()

```python
BacktestPlotter.plot_comparison(results_list: list, labels: list, figsize: tuple = (14, 8)) -> None
```

**Parameters:**
- `results_list`: List of backtest result dictionaries
- `labels`: List of strategy labels
- `figsize`: Figure size (width, height) in inches

**Generates:**
- 2-panel visualization comparing equity curves and drawdowns

### BacktestPlotter.plot_metrics_comparison()

```python
BacktestPlotter.plot_metrics_comparison(results_list: list, labels: list, figsize: tuple = (12, 8)) -> None
```

**Parameters:**
- `results_list`: List of backtest result dictionaries
- `labels`: List of strategy labels
- `figsize`: Figure size (width, height) in inches

**Generates:**
- 4-panel bar chart comparing Total Return, Sharpe Ratio, Max Drawdown, and Win Rate

### BacktestPlotter.plot_regime_analysis()

```python
BacktestPlotter.plot_regime_analysis(probs: pd.DataFrame, regime: pd.Series, 
                                     close: pd.Series, switches: pd.Series = None,
                                     figsize: tuple = (14, 10)) -> None
```

**Parameters:**
- `probs`: Regime probabilities DataFrame (time x states)
- `regime`: Active regime at each time point
- `close`: Close price series
- `switches`: Regime switch points (optional)
- `figsize`: Figure size (width, height) in inches

**Generates:**
- 3-panel visualization with price coloring by regime, probabilities, and regime evolution

## Running Examples

The framework includes a comprehensive examples script:

```bash
# Run all plotting examples
python example_plotting.py

# Run specific example
python example_plotting.py 1  # Single backtest visualization
python example_plotting.py 2  # Strategy comparison
python example_plotting.py 3  # Alpha model comparison
python example_plotting.py 4  # HMM regime analysis
```

## Integration with Main Scripts

### In example.py

```python
from plotter import BacktestPlotter

# After running backtest
results = engine.run(...)

# Add visualization
BacktestPlotter.plot_results(results, close)
```

### In run_comparison.py

```bash
# Run with plotting flag
python run_comparison.py --plot
```

## Customization

All plotting functions support custom figure sizes:

```python
# Large figures for presentations
BacktestPlotter.plot_results(results, close, figsize=(20, 16))

# Compact figures for reports
BacktestPlotter.plot_comparison(results_list, labels, figsize=(10, 6))
```

## Testing

The plotter module includes comprehensive unit tests:

```bash
# Run plotter tests
python -m unittest tests/test_plotter.py -v

# Run all tests (including plotter)
python -m unittest discover tests
```

All 7 plotter tests verify:
- Basic plotting functionality
- Plotting with regime data
- Comparison plots
- Metrics comparison
- Edge cases (empty positions, custom sizes)

## Technical Notes

### Non-Interactive Backend

For testing, the module uses matplotlib's `Agg` backend (non-interactive). This allows tests to run without requiring a display.

### Figure Management

The module properly manages matplotlib figures:
- Creates figures with appropriate sizes
- Uses subplots for multi-panel views
- Applies consistent styling (grids, colors, labels)
- Calls `plt.tight_layout()` for optimal spacing

### Data Alignment

All plotting functions handle:
- Misaligned indices between data series
- Missing data (NaN values)
- Empty position arrays
- Variable-length time series

## Performance

The plotting functions are optimized for:
- Large datasets (1000+ data points)
- Multiple simultaneous plots
- Fast rendering for interactive exploration

# TradeHMMSim - Backtesting Framework with HMM Regime Detection

A comprehensive backtesting framework for testing trading strategies with Hidden Markov Model (HMM) regime detection and filtering.

## Overview

This project provides a modular framework for:
- Testing various alpha models (moving average strategies)
- Applying HMM-based regime detection for signal filtering
- Running walk-forward backtests with rebalancing
- Comparing strategy performance with and without HMM filtering
- Comprehensive performance metrics and statistics

## Project Structure

```
TradeHMMSim/
├── AlphaModels/              # Trading signal generators
│   ├── __init__.py
│   ├── base.py              # Base alpha model class
│   ├── sma.py               # Simple Moving Average
│   ├── ema.py               # Exponential Moving Average
│   ├── wma.py               # Weighted Moving Average
│   ├── hma.py               # Hull Moving Average
│   ├── kama.py              # Kaufman Adaptive Moving Average
│   ├── tema.py              # Triple Exponential Moving Average
│   └── zlema.py             # Zero-Lag Exponential Moving Average
│
├── SignalFilter/             # HMM regime detection
│   ├── __init__.py
│   └── hmm_filter.py        # HMM regime filter implementation
│
├── tests/                    # Unit tests
│   ├── test_alpha_models/
│   │   └── test_alpha_models.py
│   ├── test_signal_filter/
│   │   └── test_hmm_filter.py
│   └── test_backtest.py
│
├── portfolio.py              # Portfolio management (data loading)
├── statistics.py             # Performance metrics calculation
├── backtest.py               # Backtest engine
├── plotter.py                # Visualization module
├── run_comparison.py         # Main comparison script
├── example.py                # Simple usage examples
├── example_plotting.py       # Plotting examples
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features

### Alpha Models
Seven different moving average strategies:
- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average (more weight on recent prices)
- **WMA**: Weighted Moving Average (linear weighting)
- **HMA**: Hull Moving Average (reduced lag)
- **KAMA**: Kaufman Adaptive Moving Average (adapts to volatility)
- **TEMA**: Triple Exponential Moving Average (smoothest)
- **ZLEMA**: Zero-Lag Exponential Moving Average (fastest response)

### HMM Regime Filter
- Automatic regime detection (bull/bear/neutral markets)
- Walk-forward training with periodic refitting
- Filtered probability calculation (no lookahead bias)
- Regime switch detection with hysteresis and confirmation

### Backtest Engine
- Multiple strategy modes:
  - **Alpha Only**: Pure alpha model signals
  - **HMM Only**: HMM regime-based signals
  - **Alpha + HMM Filter**: HMM filters out bear regimes
  - **Alpha + HMM Combine**: Combine alpha and HMM signals
- Configurable rebalancing frequency
- Transaction cost modeling
- Walk-forward testing
- Comprehensive performance metrics

### Performance Metrics
- Total Return
- Annualized Return
- Volatility (annualized)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Profit Factor
- Win Rate
- Calmar Ratio

## Installation

1. Clone the repository:
```bash
cd /Users/user/Dev/TradeHMMSim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Comparison

Compare all alpha models with and without HMM filtering:

```bash
python run_comparison.py
```

With custom parameters:
```bash
python run_comparison.py SPY 2018-01-01 2024-12-31
```

With visualization:
```bash
python run_comparison.py --plot
```

### Visualization Examples

Run comprehensive plotting examples:

```bash
# Run all examples
python example_plotting.py

# Run specific example
python example_plotting.py 1  # Single backtest visualization
python example_plotting.py 2  # Strategy comparison
python example_plotting.py 3  # Alpha model comparison
python example_plotting.py 4  # HMM regime analysis
```

### Custom Backtest Example

```python
from portfolio import Portfolio
from backtest import BacktestEngine
from AlphaModels import SMA
from SignalFilter import HMMRegimeFilter
from plotter import BacktestPlotter

# Load data
portfolio = Portfolio(['SPY'], '2018-01-01', '2024-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Create alpha model
alpha_model = SMA(short_window=10, long_window=30)

# Create HMM filter
hmm_filter = HMMRegimeFilter(n_states=3)

# Run backtest
engine = BacktestEngine(close, alpha_model, hmm_filter=hmm_filter)
results = engine.run(
    strategy_mode='alpha_hmm_combine',
    walk_forward=True,
    rebalance_frequency=5,  # Rebalance every 5 days
    transaction_cost=0.001  # 0.1% transaction cost
)

# Print results
engine.print_results(include_benchmark=True)

# Plot results
BacktestPlotter.plot_results(results, close)
```

### Visualization Functions

The `BacktestPlotter` class provides several plotting methods:

```python
# 1. Comprehensive backtest visualization (4 subplots)
BacktestPlotter.plot_results(results, close)

# 2. Compare multiple strategies
BacktestPlotter.plot_comparison(
    [results1, results2, results3],
    ['Strategy 1', 'Strategy 2', 'Strategy 3']
)

# 3. Compare performance metrics (bar charts)
BacktestPlotter.plot_metrics_comparison(
    [results1, results2],
    ['Strategy 1', 'Strategy 2']
)

# 4. HMM regime analysis visualization
BacktestPlotter.plot_regime_analysis(probs, regime, close, switches)
```

### Strategy Modes

1. **alpha_only**: Use only the alpha model signals
```python
results = engine.run(strategy_mode='alpha_only')
```

2. **hmm_only**: Use only HMM regime detection
```python
results = engine.run(strategy_mode='hmm_only', walk_forward=True)
```

3. **alpha_hmm_filter**: HMM filters out bear regimes
```python
results = engine.run(strategy_mode='alpha_hmm_filter', walk_forward=True)
```

4. **alpha_hmm_combine**: Combine alpha and HMM signals
```python
results = engine.run(strategy_mode='alpha_hmm_combine', walk_forward=True)
```

### Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run specific test suite:
```bash
python -m pytest tests/test_alpha_models/
python -m pytest tests/test_signal_filter/
python -m pytest tests/test_backtest.py
```

Run with unittest:
```bash
python -m unittest discover tests
```

## Walk-Forward Testing

The framework supports walk-forward testing to avoid lookahead bias:

```python
results = engine.run(
    strategy_mode='alpha_hmm_combine',
    walk_forward=True,
    train_window=504,      # 2 years of training data
    refit_every=21,        # Refit HMM every ~month
)
```

## Rebalancing Frequency

Control how often positions are adjusted:

```python
results = engine.run(
    rebalance_frequency=1,   # Daily (default)
    # rebalance_frequency=5,   # Weekly
    # rebalance_frequency=21,  # Monthly
)
```

## Performance Analysis

The comparison script provides:
- Individual strategy performance for all alpha models
- Average performance by strategy type
- HMM impact analysis (improvement from adding HMM)
- Benchmark comparison (Buy & Hold)
- Results saved to CSV for further analysis

## Research Questions

This framework is designed to test hypotheses such as:
1. **Do advanced moving averages outperform simple SMA?**
2. **Does HMM regime filtering improve strategy performance?**
3. **What's the optimal rebalancing frequency?**
4. **Which combination of alpha model + HMM works best?**
5. **How much does transaction cost impact different strategies?**

## Example Output

```
COMPARISON RESULTS
================================================================================
Top 10 Strategies by Total Return:
   Model              Strategy  Total Return (%)  Sharpe Ratio  Max Drawdown (%)
    HMA  Alpha + HMM Override             45.23          1.52            -15.34
   TEMA  Alpha + HMM Override             43.87          1.48            -16.12
    SMA  Alpha + HMM Override             42.15          1.45            -17.23
...

AVERAGE PERFORMANCE BY STRATEGY TYPE
================================================================================
                        Total Return (%)  Sharpe Ratio  Max Drawdown (%)
Strategy                                                                  
Alpha + HMM Filter                 38.45          1.35            -18.67
Alpha + HMM Override               41.23          1.42            -16.89
Alpha Only                         35.12          1.21            -21.34
HMM Only                           36.78          1.28            -19.45

HMM IMPACT ANALYSIS
================================================================================
SMA:
  Alpha Only Return: 35.45%
  HMM Only Return: 37.23%
  Alpha + Filter Return: 39.12%
  Alpha + Override Return: 42.15%
  HMM Filter Impact: 3.67%
  HMM Override Impact: 6.70%
```

## Architecture Decisions

### Modularity
- Each component (Portfolio, AlphaModels, SignalFilter, Statistics, Backtest) is independent
- Easy to add new alpha models or filters
- Testable in isolation

### Walk-Forward Testing
- Avoids lookahead bias by using only historical data for predictions
- Periodic refitting simulates real-world conditions
- Configurable training window and refit frequency

### Rebalancing
- Simulates realistic trading constraints
- Reduces transaction costs
- Tests strategy robustness to different rebalancing frequencies

### Performance Metrics
- Comprehensive set of standard metrics
- Both return-based and risk-adjusted measures
- Easy comparison between strategies

## Future Enhancements

Potential additions:
- More alpha models (RSI, MACD, Bollinger Bands, etc.)
- Multiple asset portfolio support
- Position sizing and risk management
- Parameter optimization framework
- Monte Carlo simulation
- More sophisticated HMM features
- Alternative regime detection methods
- Visualization tools

## Contributing

To add a new alpha model:
1. Create a new file in `AlphaModels/`
2. Inherit from `AlphaModel` base class
3. Implement `calculate_indicators()` method
4. Add to `AlphaModels/__init__.py`
5. Add unit tests

## License

This project is for research and educational purposes.

## Author

Created for backtesting research on regime-based trading strategies.

## Acknowledgments

- hmmlearn library for HMM implementation
- yfinance for market data
- scikit-learn for feature scaling

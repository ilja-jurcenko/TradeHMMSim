# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run Examples

### 1. Simple Example (Recommended for First Time)

```bash
python example.py
```

This will:
- Load SPY data (2020-2024)
- Run SMA strategy without HMM
- Run SMA strategy with HMM
- Compare results

### 2. Full Comparison (All Models)

```bash
python run_comparison.py
```

This will test all 7 alpha models with 4 different strategies:
- Alpha Only
- HMM Only  
- Alpha + HMM Filter
- Alpha + HMM Combine

**Note**: This takes ~15-30 minutes to complete.

### 2a. Save Results and Plots

Save organized output with timestamped directory:
```bash
python run_comparison.py SPY 2018-01-01 2024-12-31 --save-plots
```

This creates:
- `results_YYYYMMDD_HHMMSS/` directory
- CSV file with all results
- PNG plots for visualization

Custom output directory:
```bash
python run_comparison.py SPY 2018-01-01 2024-12-31 --output-dir my_experiment --save-plots
```

### 3. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Or with unittest
python -m unittest discover tests -v
```

## Custom Backtest

Create a new file `my_backtest.py`:

```python
from portfolio import Portfolio
from backtest import BacktestEngine
from AlphaModels import SMA
from SignalFilter import HMMRegimeFilter

# Load data
portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Create strategy
alpha_model = SMA(short_window=20, long_window=50)
hmm_filter = HMMRegimeFilter(n_states=3)

# Run backtest
engine = BacktestEngine(close, alpha_model, hmm_filter=hmm_filter)
results = engine.run(
    strategy_mode='alpha_hmm_combine',
    walk_forward=True,
    rebalance_frequency=5,
    transaction_cost=0.001
)

# Print results
engine.print_results()
```

## Key Parameters to Tune

### Alpha Model Windows
```python
SMA(short_window=10, long_window=30)  # Fast
SMA(short_window=50, long_window=200) # Slow
```

### Rebalancing Frequency
```python
rebalance_frequency=1   # Daily
rebalance_frequency=5   # Weekly
rebalance_frequency=21  # Monthly
```

### HMM Settings
```python
HMMRegimeFilter(
    n_states=3,           # Bull/Neutral/Bear
    covariance_type='diag',
    n_iter=100
)

# Walk-forward settings
train_window=252,  # 1 year
refit_every=21     # Monthly
```

### Transaction Costs
```python
transaction_cost=0.0     # No costs
transaction_cost=0.001   # 0.1% per trade
transaction_cost=0.005   # 0.5% per trade
```

## Strategy Modes Explained

### 1. alpha_only
Pure alpha model signals. No HMM filtering.
```python
results = engine.run(strategy_mode='alpha_only')
```

### 2. hmm_only
Only use HMM regime detection. Ignore alpha model.
```python
results = engine.run(strategy_mode='hmm_only', walk_forward=True)
```

### 3. alpha_hmm_filter
Alpha model generates signals, HMM blocks trades during bear markets.
```python
results = engine.run(strategy_mode='alpha_hmm_filter', walk_forward=True)
```

### 4. alpha_hmm_combine (Recommended)
Combine both alpha and HMM signals - take position when either indicates bullish conditions.
```python
results = engine.run(strategy_mode='alpha_hmm_combine', walk_forward=True)
```

## Understanding Results

```
STRATEGY PERFORMANCE
============================================================
Total Return:        42.15%      # Total profit/loss
Annual Return:       8.43%       # Annualized return
Volatility:          15.23%      # Annual standard deviation
Sharpe Ratio:        0.55        # Risk-adjusted return
Sortino Ratio:       0.78        # Downside risk-adjusted
Max Drawdown:        -18.34%     # Worst peak-to-trough
Profit Factor:       1.45        # Gross profit / gross loss
Win Rate:            52.30%      # % of winning trades
Calmar Ratio:        0.46        # Annual return / max DD
```

**Good Performance Indicators:**
- Sharpe Ratio > 1.0
- Sortino Ratio > 1.0  
- Max Drawdown < -20%
- Profit Factor > 1.5
- Win Rate > 50%

## Common Issues

### Issue: "No module named 'AlphaModels'"
**Solution**: Run from project root directory
```bash
cd /Users/user/Dev/TradeHMMSim
python example.py
```

### Issue: "Failed to download data"
**Solution**: Check internet connection and Yahoo Finance availability
```python
portfolio.load_data(progress=True)  # Show download progress
```

### Issue: Tests failing
**Solution**: Install test dependencies
```bash
pip install pytest
```

## Next Steps

1. ✅ Run `example.py` to verify installation
2. ✅ Experiment with different alpha models
3. ✅ Try different strategy modes
4. ✅ Tune parameters (windows, rebalancing, costs)
5. ✅ Run full comparison with `run_comparison.py`
6. ✅ Analyze results in generated CSV files

## File Outputs

The framework generates:
- `backtest_comparison_*.csv` - Full comparison results
- Console output with detailed metrics
- Easy to export to Excel for analysis

## Tips

- Start with shorter date ranges for faster testing
- Use `walk_forward=False` for quick initial tests
- Enable `walk_forward=True` for realistic results
- Compare alpha_only vs alpha_hmm_override to see HMM impact
- Test on different market periods (bull, bear, sideways)

## Support

Check `README.md` for detailed documentation.

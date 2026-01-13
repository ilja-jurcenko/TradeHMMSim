# Bollinger Bands Analysis Script

## Overview

Standalone analysis tool for evaluating Bollinger Bands mean-reversion strategy performance over a one-year period with comprehensive financial metrics.

## Location

`utilities/analyze_bollinger_bands.py`

## Features

✅ **One-Year Analysis** - Defaults to most recent 12 months  
✅ **Complete Metrics** - Return, Sharpe, Sortino, Max Drawdown, Win Rate  
✅ **Benchmark Comparison** - Compare against Buy & Hold  
✅ **Band Statistics** - Band width, touches, coverage  
✅ **Visual Charts** - Price with bands, equity curves, drawdowns  
✅ **Flexible Parameters** - Customize period, std dev, dates

## Usage

### Basic Usage (Last Year)

```bash
python utilities/analyze_bollinger_bands.py
```

### Specify Year

```bash
python utilities/analyze_bollinger_bands.py --start 2024-01-01 --end 2024-12-31
```

### Custom Parameters

```bash
# Wider bands (3 std dev)
python utilities/analyze_bollinger_bands.py --std 3

# Shorter period (10 days)
python utilities/analyze_bollinger_bands.py --period 10

# Different ticker
python utilities/analyze_bollinger_bands.py --ticker QQQ

# Higher transaction costs
python utilities/analyze_bollinger_bands.py --cost 0.002
```

### Disable Plots

```bash
python utilities/analyze_bollinger_bands.py --no-plot
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ticker` | Ticker symbol | SPY |
| `--period` | Bollinger Bands period | 20 |
| `--std` | Standard deviations | 2 |
| `--start` | Start date (YYYY-MM-DD) | 1 year ago |
| `--end` | End date (YYYY-MM-DD) | Today |
| `--cost` | Transaction cost | 0.001 |
| `--no-plot` | Disable chart display | False |

## Output

### Console Output

```
================================================================================
BOLLINGER BANDS STRATEGY ANALYSIS
================================================================================

Ticker: SPY
Period: 2024-01-01 to 2024-12-31
Bollinger Bands Parameters: Period=20, Std Dev=2
Transaction Cost: 0.100%

================================================================================

📊 BOLLINGER BANDS STRATEGY
----------------------------------------
Total Return:             12.27%
Annualized Return:        12.27%
Sharpe Ratio:              1.34
Sortino Ratio:             1.03
Max Drawdown:             -6.07%
Profit Factor:             1.42
Win Rate:                 21.83%
Number of Trades:             7
Time in Market:           12.30%

📈 BUY & HOLD BENCHMARK
----------------------------------------
Total Return:             25.59%
Sharpe Ratio:              1.88
Max Drawdown:             -8.41%

📊 STRATEGY vs BENCHMARK
----------------------------------------
Return Difference:       -13.31%  ❌
Sharpe Difference:        -0.54  ❌
Drawdown Reduction:        2.34%  ✅

💼 TRADING STATISTICS
----------------------------------------
Total Signals:               12
Long Periods:                31 (12.3%)
Flat Periods:               143 (56.7%)

📉 BOLLINGER BANDS STATISTICS
----------------------------------------
Avg Band Width:            5.57%
Max Band Width:           11.62%
Min Band Width:            2.92%
Lower Band Touches:          11
Upper Band Touches:          12
```

### Charts Generated

Three-panel chart saved as PNG:

1. **Price with Bollinger Bands**
   - Price line with upper/middle/lower bands
   - Buy signals (green triangles)
   - Sell signals (red triangles)
   - Shaded band area

2. **Equity Curve Comparison**
   - Bollinger Bands strategy equity
   - Buy & Hold benchmark equity
   - Initial capital line

3. **Drawdown Chart**
   - Strategy drawdown over time
   - Visualizes risk periods

**Filename:** `bollinger_bands_analysis_{ticker}_{start}_{end}.png`

## Example Use Cases

### 1. Analyze 2024 Performance

```bash
python utilities/analyze_bollinger_bands.py --start 2024-01-01 --end 2024-12-31
```

### 2. Test Different Band Widths

```bash
# Conservative (wider bands, fewer signals)
python utilities/analyze_bollinger_bands.py --std 3

# Aggressive (narrower bands, more signals)
python utilities/analyze_bollinger_bands.py --std 1.5
```

### 3. Compare Different Tickers

```bash
python utilities/analyze_bollinger_bands.py --ticker QQQ
python utilities/analyze_bollinger_bands.py --ticker IWM
python utilities/analyze_bollinger_bands.py --ticker DIA
```

### 4. Optimize Parameters

```bash
# Test different periods
for period in 10 15 20 25 30; do
    echo "Testing period=$period"
    python utilities/analyze_bollinger_bands.py --period $period --no-plot
done
```

## Interpretation Guide

### When Bollinger Bands Work Well

✅ **Mean-Reverting Markets** - Sideways/ranging conditions  
✅ **High Volatility** - More band touches = more opportunities  
✅ **Bear Markets** - Catches oversold bounces  
✅ **Short Time Frames** - Quick mean reversion

### When Bollinger Bands Struggle

❌ **Strong Trends** - Price stays outside bands  
❌ **Low Volatility** - Narrow bands, fewer signals  
❌ **Bull Markets** - Misses extended upside  
❌ **Trend-Following Periods** - Cuts winners short

### Key Metrics to Watch

- **Time in Market < 30%** - Conservative, selective entries
- **Win Rate < 50%** - Mean-reversion often has lower win rate but good risk/reward
- **Max Drawdown** - Should be lower than buy & hold in volatile markets
- **Band Width** - Higher = more volatility = more opportunities

## Integration with Framework

This script uses the same models as the main framework:

```python
from alpha_models import BollingerBands
from backtest import BacktestEngine

# Same model used in regime-adaptive strategy
bb_model = BollingerBands(short_window=20, long_window=2)
```

Can be compared directly with results from:
- `run_comparison.py` - Multi-strategy comparison
- Regime-Adaptive Alpha strategy (bear market component)

## Tips for Best Results

1. **Use in Bear/Sideways Markets** - Bollinger Bands excel in mean-reversion conditions
2. **Combine with Regime Detection** - See regime-adaptive strategy
3. **Adjust for Volatility** - Higher vol → wider bands, lower vol → narrower bands
4. **Monitor Band Width** - Narrowing bands often precede volatility expansion
5. **Transaction Costs Matter** - More trades = higher costs, adjust accordingly

## Troubleshooting

### No Data Loaded
- Check ticker symbol is valid
- Ensure date range has trading days
- Verify internet connection for data download

### Few/No Signals
- Try narrower bands (`--std 1.5`)
- Use shorter period (`--period 10`)
- Check if market is strongly trending (not ideal for BB)

### Poor Performance
- Bollinger Bands work best in ranging markets
- Consider using only in bear regimes (see regime-adaptive strategy)
- Adjust parameters for market conditions

## Related Scripts

- `utilities/check_hmm_only.py` - Analyze HMM regime detection
- `run_comparison.py` - Compare all strategies including regime-adaptive

## Technical Details

**Strategy Logic:**
1. Calculate 20-period SMA (middle band)
2. Calculate 20-period standard deviation
3. Upper band = SMA + (2 × std dev)
4. Lower band = SMA - (2 × std dev)
5. Buy when price crosses below lower band
6. Sell when price reaches middle band

**Position Sizing:** 100% of capital (all-in/all-out)

**Rebalancing:** Every period (daily if using daily data)

**Performance Attribution:** Includes slippage via transaction costs

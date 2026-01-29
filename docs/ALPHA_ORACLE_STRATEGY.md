# Alpha Oracle Strategy - ZigZag Timing Labels

## Overview

The **Alpha Oracle** strategy is a timing benchmark that identifies ideal entry and exit points based on local price extrema using a ZigZag-like algorithm. This is **not a tradable strategy** - it's designed to evaluate how well your actual trading strategies capture major market swings.

## Purpose

The alpha oracle answers key timing questions:
- Did you enter before a major rally?
- Did you exit before a major drawdown?
- Did you miss the swing entirely?

It provides a reference for **timing quality**, not profitability forecasts.

## Algorithm

The strategy uses a state machine to identify local minima (BUY signals) and maxima (SELL signals):

### State: Searching for BUY
1. Track the lowest price seen so far
2. If price drops below the current low, update the low
3. If price rises ≥ `min_move_pct` (default 5%) from the low:
   - Signal **BUY** at the low point
   - Switch to "searching for SELL" state

### State: Searching for SELL
1. Track the highest price seen so far
2. If price rises above the current high, update the high
3. If price drops ≥ `min_move_pct` from the high:
   - Signal **SELL** at the high point
   - Switch to "searching for BUY" state

### Key Properties
- Signals alternate: BUY → SELL → BUY
- Only one position at a time (binary 0/1)
- No lookahead bias (signals only when move threshold is confirmed)
- Does not reward overtrading
- Transaction costs apply (realistic benchmark)

## Parameters

```python
AlphaOracleStrategy(min_move_pct=0.05)
```

- **min_move_pct**: Minimum percentage move (as decimal) to identify a pivot
  - Default: 0.05 (5% move required)
  - Lower values → More signals, more noise
  - Higher values → Fewer signals, major swings only

## Usage

### In run_comparison.py

```bash
# Run only alpha oracle
python run_comparison.py SPY --strategies alpha_oracle

# Compare with other strategies
python run_comparison.py SPY --strategies alpha_only,alpha_oracle
```

### Standalone Example

```python
from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import SMA

# Load data
portfolio = Portfolio(['SPY'], '2020-01-01', '2023-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Create engine (alpha model not used by oracle but required)
alpha_model = SMA(short_window=10, long_window=30)
engine = BacktestEngine(close, alpha_model)

# Run oracle
results = engine.run(
    strategy_mode='alpha_oracle',
    rebalance_frequency=1,
    transaction_cost=0.001
)

print(f"Total Return: {results['metrics']['total_return']*100:.2f}%")
print(f"Number of Trades: {results['num_trades']}")
```

## Interpreting Results

### Pivot Analysis
The strategy prints detailed pivot information:
```
ZigZag Pivot Analysis:
  Min Move Threshold: 5.00%
  Total Pivots Identified: 24
  BUY Signals: 12
  SELL Signals: 12
  Average Swing Size: 8.50%
```

### Timing Deviation
Compare your strategy's performance to the oracle:
- **Higher return than oracle**: Unlikely with realistic costs (check for errors)
- **Close to oracle return**: Excellent timing quality
- **Moderate gap (5-15%)**: Good timing, room for improvement
- **Large gap (>20%)**: Poor timing or different risk profile

### What Oracle Does NOT Tell You
- Whether your alpha signal is predictive
- If your strategy is tradable with realistic costs
- How to improve your signals
- Future performance expectations

## Example Output

```
Running Alpha Oracle Strategy (5% ZigZag)
============================================================

ZigZag Pivot Analysis:
  Min Move Threshold: 5.00%
  Total Pivots Identified: 18
  BUY Signals: 9
  SELL Signals: 9
  Average Swing Size: 7.85%

RESULTS
============================================================
Total Return: 42.15%
Annualized Return: 12.35%
Sharpe Ratio: 1.85
Max Drawdown: -8.50%
Number of Trades: 18
Win Rate: 88.9%
```

## Comparison with Other Strategies

| Strategy | Purpose | Uses HMM | Lookahead | Timing Quality |
|----------|---------|----------|-----------|----------------|
| Alpha Only | Real alpha signal | No | No | Variable |
| Alpha Oracle | Ideal timing | No | No | Best possible |
| HMM Only | Regime timing | Yes | No | Good |
| Oracle (HMM) | Regime upper bound | Yes | Yes | Unrealistic |
| Alpha+HMM Filter | Filtered alpha | Yes | No | Better than Alpha Only |
| Alpha+HMM Combine | Combined signal | Yes | No | Better than Filter |

## Adjusting min_move_pct

### Higher Threshold (10%+)
- Captures only major trends
- Fewer trades
- Misses smaller opportunities
- Good for **trend evaluation**

### Lower Threshold (2-3%)
- Captures minor swings
- More trades
- Higher noise sensitivity
- Good for **swing trading evaluation**

### Default (5%)
- Balanced for most use cases
- Identifies significant moves
- Reasonable trade frequency

## Limitations

1. **Not predictive**: Labels are based on realized moves
2. **Hindsight knowledge**: You wouldn't know the low until price rose 5%
3. **Transaction costs**: Real costs included in backtest
4. **Slippage ignored**: Assumes perfect fills at pivot prices
5. **Market microstructure**: Ignores intraday volatility

## When to Use Alpha Oracle

✅ **Use for:**
- Benchmarking timing quality of strategies
- Evaluating signal delay (how late are entries?)
- Understanding maximum theoretical performance
- Identifying missed opportunities

❌ **Don't use for:**
- Forward-looking predictions
- Strategy generation (will overfit)
- Performance expectations (unrealistically high)
- Real trading decisions

## Files

- **Strategy**: `strategies/alpha_oracle.py`
- **Example**: `examples/example_alpha_oracle.py`
- **Tests**: Run with `python examples/example_alpha_oracle.py`

## See Also

- `docs/STRATEGY_FILTERING.md` - How to run specific strategies
- `strategies/oracle.py` - HMM oracle (different concept)
- `strategies/alpha_only.py` - Real alpha signal baseline

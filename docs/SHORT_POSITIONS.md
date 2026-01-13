# Short Position Support

## Overview
The backtest engine now fully supports short positions (-1), allowing alpha models to profit from falling prices.

## Position Values
- **1**: Long position (profit when price rises)
- **0**: Flat/no position
- **-1**: Short position (profit when price falls)

## Key Features

### 1. Return Calculation
Short positions correctly inverse returns:
```python
strategy_returns = positions.shift(1).fillna(0) * price_returns
```
- When short (-1) and price falls (negative return), strategy profits (positive return)
- When short (-1) and price rises (positive return), strategy loses (negative return)

### 2. Time in Market
Fixed metric to count both long and short positions:
```python
time_in_market = float(np.sum(positions != 0) / len(positions))
```
Previously only counted long positions (`positions > 0`).

### 3. Transaction Costs
Apply to all position changes using absolute value:
```python
position_changes = positions.diff().fillna(0).abs()
costs = position_changes * transaction_cost
```
Charges for both long↔short and short↔flat transitions.

### 4. Trade Counting
Uses absolute position changes:
```python
num_trades = int(np.sum(np.abs(positions.diff().fillna(0))) / 2)
```
Correctly counts short trades same as long trades.

## Creating Short-Enabled Alpha Models

### Example: Trend-Following with Shorts
```python
class TrendShortModel(AlphaModel):
    def generate_signals(self, close: pd.Series) -> pd.Series:
        short_ma, long_ma = self.calculate_indicators(close)
        
        signals = pd.Series(0, index=close.index)
        signals[short_ma > long_ma] = 1   # Long in uptrend
        signals[short_ma < long_ma] = -1  # Short in downtrend
        
        return signals
```

### Example: Mean-Reversion (Long-Only)
```python
class BollingerBands(AlphaModel):
    def generate_signals(self, close: pd.Series) -> pd.Series:
        # Only generates 1 (long) or 0 (flat)
        # Never generates -1 (short)
        signals = pd.Series(0, index=close.index)
        
        if price_crosses_below_lower_band:
            signals[i] = 1  # Long on oversold
        elif price_returns_to_middle:
            signals[i] = 0  # Exit to flat
            
        return signals
```

## Benefits of Short Positions

1. **Profit in Down Markets**: Generate positive returns when prices fall
2. **Better Risk Management**: Stay active during bear markets instead of sitting flat
3. **Higher Time in Market**: Can be positioned during both bull and bear regimes
4. **Improved Sharpe Ratios**: Better risk-adjusted returns in range-bound markets

## Test Coverage

Comprehensive test suite with 13 tests covering:
- ✅ Short position acceptance
- ✅ Correct return calculations
- ✅ Time in market metric
- ✅ Transaction costs
- ✅ Trade counting
- ✅ Long↔short transitions
- ✅ Mixed position types
- ✅ Rebalancing with shorts
- ✅ Return alignment validation

All tests passing: **13/13 ✅**

## Example Usage

See [examples/example_short_positions.py](../examples/example_short_positions.py) for a complete demonstration comparing long+short vs long-only strategies.

## Performance Comparison

Using example data (100 days up, 100 days down):

| Metric | Long+Short | Long-Only | Improvement |
|--------|------------|-----------|-------------|
| Total Return | 51.60% | 18.34% | +33.25% |
| Sharpe Ratio | 3.14 | 1.91 | +1.23 |
| Time in Market | 85.5% | 42.0% | +43.5% |

Short positions provided **33% additional return** by profiting during the downtrend.

## Notes

- The backtest engine has always supported short positions mathematically
- This update fixes the `time_in_market` metric to properly count shorts
- Bollinger Bands model updated to only generate long signals (mean-reversion strategy)
- All existing tests continue to pass (223 total tests)

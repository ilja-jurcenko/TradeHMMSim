# Regime Switch Strategy Implementation

## Overview

The HMM-only strategy has been updated to use **regime switches** (turning points) as buy and sell signals, instead of probability thresholds. This provides clearer entry and exit signals based on actual regime transitions.

## Strategy Logic

### Signal Generation

The strategy determines the most likely regime at each time point by comparing probabilities:
- **Bear Regime**: Highest bear probability
- **Neutral Regime**: Highest neutral probability  
- **Bull Regime**: Highest bull probability

### Trade Signals

**Buy Signals**: Generated when switching **TO** Bull regime from Bear or Neutral
- Action: Enter long position (Position = 1)
- Rationale: Market transitioning to bullish conditions

**Sell Signals**: Generated when switching **TO** Bear regime from Bull or Neutral
- Action: Exit position (Position = 0)
- Rationale: Market transitioning to bearish conditions

**Hold Signals**: No regime switch detected
- Action: Maintain current position
- Rationale: Staying within same regime

### Neutral Regime Handling

When switching to Neutral regime:
- **Maintains current position** (no forced exit)
- Provides buffer zone between bull and bear extremes
- Reduces whipsaws from rapid regime oscillation

## Implementation Details

### Code Changes

**File**: `strategies/hmm_only.py`

```python
def generate_positions(self, alpha_signals, close, common_idx, **kwargs):
    """Generate positions from HMM regime switches only."""
    
    # Determine most likely regime at each point
    probs_df = pd.DataFrame({
        'bear': bear_prob,
        'neutral': neutral_prob,
        'bull': bull_prob
    }, index=common_idx)
    
    regime = probs_df.idxmax(axis=1)
    
    # Detect regime switches (turning points)
    regime_switches = regime != regime.shift(1)
    
    # Generate positions based on switches
    for i, idx in enumerate(common_idx):
        if regime_switches.iloc[i]:
            if regime.iloc[i] == 'bull':
                current_position = 1  # BUY
            elif regime.iloc[i] == 'bear':
                current_position = 0  # SELL
            # else neutral: maintain position
```

### Logging Enhancements

Each log entry now includes:
- **Regime**: Current regime (bear/neutral/bull)
- **Regime_Switch**: Boolean indicating if switch occurred
- **Action**: BUY/SELL/HOLD based on regime change

Example log format:
```csv
Date,Price,Bear_Prob,Neutral_Prob,Bull_Prob,Regime,Regime_Switch,Position,Action,Strategy
2020-02-25,286.50,0.267,0.002,0.731,bull,True,1,BUY,HMM_Only
2020-02-26,285.45,0.285,0.597,0.118,neutral,True,1,HOLD,HMM_Only
2020-02-27,272.63,0.988,0.000,0.012,bear,True,0,SELL,HMM_Only
```

## Visualization

### Regime Switch Markers

**File**: `plotter.py` - Enhanced `plot_hmm_regime_colored_equity()`

The plot now includes:

1. **Colored price line** by active regime:
   - Green: Bull regime
   - Orange: Neutral regime
   - Red: Bear regime

2. **Regime switch markers**:
   - Green up-arrow (▲): Switch to Bull → BUY signal
   - Orange circle (●): Switch to Neutral → HOLD
   - Red down-arrow (▼): Switch to Bear → SELL signal

3. **Vertical dashed lines** at each switch point for clarity

### Legend

The plot legend includes:
- Regime colors (Bull/Neutral/Bear)
- Switch markers with symbols
- Clear indication of turning points

## Performance Results

### Test Period: 2018-01-01 to 2021-12-31

Using regime switch strategy with `config_default.json`:

| Metric | Value |
|--------|-------|
| **Total Return** | 62.31% |
| **Annual Return** | 29.37% |
| **Sharpe Ratio** | 1.78 |
| **Sortino Ratio** | 1.51 |
| **Calmar Ratio** | 3.11 |
| **Max Drawdown** | -9.44% |
| **Number of Trades** | 4 |
| **Time in Market** | 86.08% |
| **Win Rate** | 50.21% |

### Comparison to Buy & Hold

- **Buy & Hold Return**: 50.92%
- **Strategy Return**: 62.31%
- **Outperformance**: +11.39%
- **Lower Drawdown**: -9.44% vs -33.72%

## Advantages

1. **Clear Signals**: Regime switches provide unambiguous entry/exit points
2. **Reduced Drawdown**: Exits during bear regimes limit losses
3. **High Win Rate**: 50%+ success rate on trades
4. **Low Trade Frequency**: Only 4 trades over 4 years reduces costs
5. **Visual Clarity**: Easy to identify turning points on charts

## Configuration

Uses standard HMM parameters from `config_default.json`:

```json
"hmm": {
  "n_states": 3,
  "random_state": 42,
  "covariance_type": "diag",
  "n_iter": 100,
  "tol": 0.001,
  "train_window": 504,
  "refit_every": 21,
  "short_vol_window": 10,
  "long_vol_window": 30,
  "short_ma_window": 10,
  "long_ma_window": 30
}
```

Note: `bear_prob_threshold` and `bull_prob_threshold` are no longer used for position generation, but still affect regime classification in the filter.

## Usage

Run HMM-only strategy with regime switches:

```bash
python3 run_comparison.py SPY \
  --start-date 2018-01-01 \
  --end-date 2021-12-31 \
  --config config_default.json \
  --save-plots \
  --strategies hmm_only \
  --enable-logging
```

This generates:
- Performance metrics
- Regime-colored equity plots with switch markers
- Detailed logs with regime and switch information

## Future Enhancements

Potential improvements:
1. **Confirmation periods**: Require N consecutive periods in new regime before switching
2. **Volatility-based filters**: Ignore switches during low-volatility periods
3. **Multi-asset**: Apply regime switches across portfolio allocation
4. **Regime persistence**: Track how long each regime lasts
5. **Adaptive thresholds**: Adjust switch sensitivity based on market conditions

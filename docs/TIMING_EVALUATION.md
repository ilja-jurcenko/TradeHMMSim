# Timing Quality Evaluation System

## Overview

The timing quality evaluation system compares every trading strategy against the **Alpha Oracle** baseline to measure how well each strategy captures optimal market timing. This provides objective metrics for strategy evaluation beyond just returns and Sharpe ratios.

## How It Works

### Step 1: Establish Oracle Baseline

The system first runs the Alpha Oracle strategy (ZigZag-based timing) to identify ideal entry and exit points:

```
Alpha Oracle Signals:
BUY  at local minima (after 5% drop from high)
SELL at local maxima (after 5% rise from low)
```

These signals become the **reference baseline** for all other strategies.

### Step 2: Compare Each Strategy

For every strategy tested, the system calculates timing metrics by comparing positions to oracle positions.

## Timing Metrics

### 1. Entry Timing Score (bars)

**Definition**: Average bars early (negative) or late (positive) for entries

**Calculation**: For each oracle BUY signal, find the closest strategy entry and measure the time difference.

**Interpretation**:
- **Negative value** = Early entry (entered before oracle BUY)
  - Example: `-5 bars` means strategy entered 5 periods before oracle
  - Can be good (early trend capture) or bad (premature entry)
- **Positive value** = Late entry (entered after oracle BUY)
  - Example: `+10 bars` means strategy missed optimal entry by 10 periods
  - Always suboptimal (missed gains)
- **Zero** = Perfect timing

**Best Case**: Small negative value (slightly early) or zero

### 2. Exit Timing Score (bars)

**Definition**: Average bars early (negative) or late (positive) for exits

**Calculation**: For each oracle SELL signal, find the closest strategy exit and measure the time difference.

**Interpretation**:
- **Negative value** = Early exit (exited before oracle SELL)
  - Example: `-3 bars` means strategy exited 3 periods before oracle
  - Can be good (risk management) or bad (missed gains)
- **Positive value** = Late exit (exited after oracle SELL)
  - Example: `+8 bars` means strategy held 8 periods past optimal exit
  - Always suboptimal (gave back gains or increased losses)
- **Zero** = Perfect timing

**Best Case**: Small negative value (slightly early) or zero

### 3. Coverage (%)

**Definition**: Percentage of oracle bull periods where strategy was also long

**Calculation**:
```
Coverage = (Periods where both Strategy=1 AND Oracle=1) / (Total Oracle=1 periods) * 100
```

**Interpretation**:
- **100%** = Perfect coverage (captured all oracle opportunities)
- **80-99%** = Excellent (captured most opportunities)
- **60-79%** = Good (captured majority)
- **40-59%** = Fair (missed many opportunities)
- **<40%** = Poor (missed most opportunities)

**Best Case**: Close to 100%

### 4. False Entries

**Definition**: Number of times strategy entered when oracle was in bear (cash) position

**Calculation**: Count strategy entries (0→1) when oracle position = 0

**Interpretation**:
- **0** = Perfect (no premature entries)
- **Low (1-3)** = Excellent discipline
- **Medium (4-10)** = Acceptable false signal rate
- **High (>10)** = Poor signal quality or overtrading

**Best Case**: 0 (but 1-3 is excellent)

### 5. False Exits

**Definition**: Number of times strategy exited when oracle was still in bull position

**Calculation**: Count strategy exits (1→0) when oracle position = 1

**Interpretation**:
- **0** = Perfect (no premature exits)
- **Low (1-3)** = Excellent trend following
- **Medium (4-10)** = Acceptable exit discipline
- **High (>10)** = Poor trend following or excessive caution

**Best Case**: 0 (but 1-3 is excellent)

## Example Output

```
TIMING QUALITY ANALYSIS (vs Alpha Oracle)
================================================================================

Best Entry Timing (closest to oracle entries, negative = early):
  Model  Strategy              Entry Timing (bars)  Coverage (%)  False Entries
  SMA    Alpha + HMM Combine              -2.3            87.5              2
  EMA    Alpha Oracle                      0.0           100.0              0
  WMA    Alpha + HMM Filter               -4.1            82.3              3
  ...

Best Exit Timing (closest to oracle exits, negative = early):
  Model  Strategy              Exit Timing (bars)  False Exits
  EMA    Alpha Oracle                     0.0              0
  SMA    Alpha + HMM Combine             -1.8              1
  HMA    Regime-Adaptive Alpha           -3.2              2
  ...

Best Coverage (% of oracle bull periods captured):
  Model  Strategy              Coverage (%)  Entry Timing (bars)  Total Return (%)
  EMA    Alpha Oracle                100.0                 0.0              45.2
  SMA    Alpha + HMM Combine          87.5                -2.3              38.7
  WMA    Alpha + HMM Filter           82.3                -4.1              35.1
  ...

Fewest False Signals:
  Model  Strategy              False Entries  False Exits  Total False  Sharpe Ratio
  EMA    Alpha Oracle                     0            0            0          1.85
  SMA    Alpha + HMM Combine              2            1            3          1.52
  HMA    Regime-Adaptive Alpha            3            2            5          1.48
  ...
```

## Use Cases

### 1. Strategy Development
- Identify if strategy is too early or too late
- Measure improvement as you refine entry/exit rules
- Quantify timing quality independent of market regime

### 2. Parameter Optimization
- Don't just optimize for returns
- Optimize for timing quality metrics
- Balance coverage vs false signals

### 3. Strategy Comparison
- Compare strategies on timing dimensions
- Identify which strategies have best entry vs exit timing
- Find strategies with high coverage and low false signals

### 4. Risk Management
- High false exits = choppy, nervous strategy
- High false entries = overtrading, lack of discipline
- Late exits = giving back profits

## Interpreting Combinations

### Ideal Strategy Profile
```
Entry Timing:   -1 to +2 bars (slightly early to on-time)
Exit Timing:    -2 to 0 bars (slightly early to on-time)
Coverage:       >80%
False Entries:  <3
False Exits:    <3
```

### Strategy Archetypes

#### **Aggressive Early Entry**
- Entry Timing: -5 to -10 bars
- Coverage: >90%
- False Entries: High (5-10)
- **Character**: Catches trends early but suffers false starts

#### **Conservative Late Entry**
- Entry Timing: +3 to +8 bars
- Coverage: 60-75%
- False Entries: Low (0-2)
- **Character**: Waits for confirmation, misses early gains

#### **Nervous Exit Strategy**
- Exit Timing: -5 to -10 bars
- False Exits: High (5-10)
- **Character**: Exits too early, leaves money on table

#### **Stubborn Hold Strategy**
- Exit Timing: +5 to +15 bars
- False Exits: Low (0-2)
- **Character**: Holds too long, gives back profits

#### **Balanced Timing**
- Entry: -2 to +1 bars
- Exit: -1 to 0 bars
- Coverage: 80-90%
- False Signals: <3 each
- **Character**: Well-calibrated, disciplined execution

## Limitations

### 1. Oracle is Hindsight-Based
- Oracle uses realized price moves
- You wouldn't know the low until price rose 5%
- Oracle timing is theoretical maximum, not achievable in real-time

### 2. Doesn't Account for Risk
- Early entry has more downside risk
- Late exit risks drawdown
- Coverage doesn't consider position sizing

### 3. Transaction Costs
- More entries/exits = more costs
- Perfect timing with 50 trades may underperform good timing with 10 trades

### 4. Market Regime Dependent
- Metrics vary by market conditions
- Trending markets favor different timing than choppy markets
- Test across multiple time periods

## Best Practices

### ✅ Do:
- Use timing metrics alongside returns/Sharpe
- Compare strategies across multiple alpha models
- Look for consistent timing patterns
- Balance coverage with false signal rate
- Test on out-of-sample data

### ❌ Don't:
- Optimize solely for timing metrics (will overfit)
- Expect to match oracle perfectly (unrealistic)
- Ignore transaction costs
- Use as sole evaluation criterion
- Trust single-period results

## Integration with run_comparison.py

The timing evaluation runs automatically:

```bash
# Run with timing evaluation (automatic)
python run_comparison.py SPY --strategies alpha_only,alpha_oracle,alpha_hmm_combine

# View results in CSV
# Columns: Entry Timing (bars), Exit Timing (bars), Coverage (%), False Entries, False Exits

# View in markdown report
# Section: "Timing Quality Analysis (vs Alpha Oracle)"
```

## Files

- **Implementation**: `run_comparison.py` (`calculate_timing_metrics()` function)
- **Oracle Strategy**: `strategies/alpha_oracle.py`
- **Documentation**: `docs/TIMING_EVALUATION.md` (this file)
- **Example Output**: `results/run_<timestamp>/ANALYSIS.md`

## See Also

- [Alpha Oracle Strategy](ALPHA_ORACLE_STRATEGY.md) - How oracle identifies pivots
- [Strategy Filtering](STRATEGY_FILTERING.md) - Running specific strategies
- [Config Guide](CONFIG_GUIDE.md) - Configuration parameters

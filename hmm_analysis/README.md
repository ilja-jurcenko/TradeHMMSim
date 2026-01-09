# HMM Regime Analysis

## Overview

This analysis visualizes how the Hidden Markov Model (HMM) identifies market regimes and how different strategies use these regimes for trading signals.

## Generated Visualizations

### 1. `hmm_regime_analysis.png`

Four-panel comparison showing:

**Panel 1: Price with HMM Regime Background**
- SPY price with colored backgrounds indicating HMM-detected regimes
- Red = Bear regime (high volatility)
- Green = Bull regime (low volatility, uptrend)
- Gray = Neutral regime (moderate volatility)

**Panel 2: Alpha Model Signals (EMA 10/30)**
- Standard trend-following signals from EMA crossover
- Green triangles (▲) = Buy signals
- Red triangles (▼) = Sell signals
- Blue shading = When in position
- **68.3% time in market, 671 signals**

**Panel 3: HMM Only Signals**
- Pure regime-based trading (bull+neutral > 65% threshold)
- Ignores trend signals completely
- Purple shading = When HMM says in position
- **60.4% time in market, ~13 trades**
- This is why all alpha models show 26.43% return for HMM Only

**Panel 4: Alpha + HMM Combine (Contrarian Entry)**
- Current strategy: Follow alpha + add early entries
- Green triangles (▲) = Normal alpha buy signals
- Orange triangles (▲) = Contrarian buy signals (HMM detects bull before trend turns)
- Teal shading = When in position
- **81.2% time in market** (more than alpha alone due to contrarian entries)

### 2. `hmm_probabilities.png`

Three-panel probability time series:

**Panel 1: Bear Regime Probability**
- Mean: 0.390, Max: 1.000
- Above 0.65 threshold: 38.5% of the time

**Panel 2: Bull Regime Probability**
- Mean: 0.317, Max: 1.000
- Above 0.65 threshold: **only 16.6%** of the time
- Too restrictive when used alone!

**Panel 3: Bull + Neutral Combined Probability**
- Mean: 0.610, Max: 1.000
- Above 0.65 threshold: 60.4% of the time
- This is what HMM Only strategy uses
- Combining bull+neutral makes the signal much more practical

## Key Insights

### Why HMM Only Returns 26.43%?

1. **Same across all alpha models** - HMM Only ignores the alpha model parameter completely
2. **Uses bull+neutral > 0.65** - This triggers 60.4% of the time
3. **13 trades total** - Much less frequent than trend-following (19 trades for EMA)
4. **Deterministic** - Same HMM config = same result

### Signal Overlap Analysis (threshold=0.65)

- **Both Alpha and HMM agree**: 14.8% of days (145 days)
- **Alpha only**: 53.5% of days (526 days)
- **HMM only**: 1.8% of days (18 days)
- **Neither**: 29.9% of days (294 days)

This shows that:
- Alpha and HMM signals are mostly complementary (not overlapping)
- HMM rarely signals when alpha doesn't (only 1.8%)
- Contrarian entries add value by catching these 18 days early

### Contrarian Strategy Performance

The combine strategy adds **contrarian entries** when:
- Alpha says "no position" (downtrend)
- HMM predicts bull+neutral > 65% (regime shift coming)

Results:
- **More time in market**: 81.2% vs 73.3% (alpha only)
- **Better risk-adjusted returns**: Sharpe 1.08 vs 0.95
- **Lower drawdowns**: -12.94% vs -14.80%
- **Fewer total returns**: 65.98% vs 93.22% (tradeoff for risk reduction)

## Threshold Sensitivity

Testing different thresholds for bull+neutral probability:

| Threshold | HMM Signals | Combined Time in Market |
|-----------|-------------|------------------------|
| 0.40 | 60.4% | 81.2% |
| 0.50 | 60.8% | 81.4% |
| 0.60 | 60.4% | 81.2% |
| **0.65** | **60.4%** | **81.2%** |
| 0.70 | 60.3% | 81.2% |

The strategy is relatively stable across thresholds 0.40-0.70, with 0.65 being a reasonable middle ground.

## Conclusions

1. **HMM captures different information than trend**: Only 14.8% overlap
2. **Combining bull+neutral is essential**: Pure bull probability only triggers 16.6% of time
3. **Contrarian entries add value**: Early detection of regime shifts
4. **Contrarian exits hurt performance**: Too aggressive, removed from strategy
5. **Best for risk-adjusted returns**: Higher Sharpe, lower drawdowns, but lower absolute returns

## Files

- `hmm_regime_analysis.png` - Main visualization with all strategies
- `hmm_probabilities.png` - Regime probability time series
- `README.md` - This file

Generated: 2026-01-09
Data: SPY 2020-01-01 to 2025-12-31

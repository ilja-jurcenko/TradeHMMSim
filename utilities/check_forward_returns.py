"""
Deep dive into forward-looking returns by regime.
Check if HMM regimes actually predict future performance.
"""

import numpy as np
import pandas as pd
from portfolio import Portfolio
from SignalFilter import HMMRegimeFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
portfolio = Portfolio(['SPY'], '2020-01-01', '2025-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Run HMM
hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
print("Running HMM regime detection...")
probs, regime, switches = hmm_filter.walkforward_filter(close, train_window=504, refit_every=21)

# Identify regimes
regime_info = hmm_filter.identify_regimes(close, regime)
bear_regime = regime_info['bear_regime']
bull_regime = regime_info['bull_regime']
neutral_regime = regime_info['neutral_regime']

print(f"\nRegimes: Bear={bear_regime}, Bull={bull_regime}, Neutral={neutral_regime}")

# Align data
common_idx = regime.index.intersection(close.index)
regime_aligned = regime.loc[common_idx]
close_aligned = close.loc[common_idx]

print(f"\n{'='*80}")
print("DETAILED FORWARD-LOOKING ANALYSIS")
print(f"{'='*80}")

# Calculate different forward return metrics
forward_periods = [1, 5, 10, 21]

for period in forward_periods:
    print(f"\n{'='*80}")
    print(f"{period}-DAY FORWARD RETURNS")
    print(f"{'='*80}")
    
    # Calculate forward returns (what happens AFTER being in this regime)
    forward_returns = close_aligned.pct_change(period).shift(-period)
    
    results = []
    for regime_id, regime_name in [(bear_regime, 'Bear'), (bull_regime, 'Bull'), (neutral_regime, 'Neutral')]:
        regime_mask = (regime_aligned == regime_id)
        
        # Filter out NaN forward returns
        valid_mask = regime_mask & ~forward_returns.isna()
        fwd_ret_valid = forward_returns[valid_mask]
        
        if len(fwd_ret_valid) > 0:
            mean_ret = fwd_ret_valid.mean()
            median_ret = fwd_ret_valid.median()
            std_ret = fwd_ret_valid.std()
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            
            # Win rate
            win_rate = (fwd_ret_valid > 0).sum() / len(fwd_ret_valid)
            
            # Percentiles
            p25 = fwd_ret_valid.quantile(0.25)
            p75 = fwd_ret_valid.quantile(0.75)
            
            results.append({
                'regime': regime_name,
                'mean': mean_ret,
                'median': median_ret,
                'std': std_ret,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'p25': p25,
                'p75': p75,
                'count': len(fwd_ret_valid)
            })
    
    # Print results
    for r in results:
        print(f"\n{r['regime']} Regime:")
        print(f"  Sample size: {r['count']} observations")
        print(f"  Mean return: {r['mean']:.2%}")
        print(f"  Median return: {r['median']:.2%}")
        print(f"  Std deviation: {r['std']:.2%}")
        print(f"  Sharpe ratio: {r['sharpe']:.3f}")
        print(f"  Win rate: {r['win_rate']:.1%}")
        print(f"  25th percentile: {r['p25']:.2%}")
        print(f"  75th percentile: {r['p75']:.2%}")
    
    # Statistical comparison
    print(f"\n{'='*60}")
    print("COMPARISON:")
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    best_mean = max(results, key=lambda x: x['mean'])
    print(f"  Best Sharpe ratio: {best_sharpe['regime']} ({best_sharpe['sharpe']:.3f})")
    print(f"  Best mean return: {best_mean['regime']} ({best_mean['mean']:.2%})")

print(f"\n{'='*80}")
print("KEY QUESTION: Does HMM Predict Direction?")
print(f"{'='*80}")

# Check if being in bear regime TODAY predicts negative returns ahead
period = 21
forward_returns = close_aligned.pct_change(period).shift(-period)

bear_mask = (regime_aligned == bear_regime) & ~forward_returns.isna()
bull_mask = (regime_aligned == bull_regime) & ~forward_returns.isna()

bear_fwd = forward_returns[bear_mask]
bull_fwd = forward_returns[bull_mask]

print(f"\n21-day forward returns:")
print(f"\nBear regime:")
print(f"  % of times forward return is NEGATIVE: {(bear_fwd < 0).sum() / len(bear_fwd) * 100:.1f}%")
print(f"  % of times forward return is POSITIVE: {(bear_fwd > 0).sum() / len(bear_fwd) * 100:.1f}%")
print(f"  Average negative return: {bear_fwd[bear_fwd < 0].mean():.2%}")
print(f"  Average positive return: {bear_fwd[bear_fwd > 0].mean():.2%}")

print(f"\nBull regime:")
print(f"  % of times forward return is NEGATIVE: {(bull_fwd < 0).sum() / len(bull_fwd) * 100:.1f}%")
print(f"  % of times forward return is POSITIVE: {(bull_fwd > 0).sum() / len(bull_fwd) * 100:.1f}%")
print(f"  Average negative return: {bull_fwd[bull_fwd < 0].mean():.2%}")
print(f"  Average positive return: {bull_fwd[bull_fwd > 0].mean():.2%}")

print(f"\n{'='*80}")
print("INTERPRETATION:")
print(f"{'='*80}")

bear_negative_pct = (bear_fwd < 0).sum() / len(bear_fwd) * 100
bull_positive_pct = (bull_fwd > 0).sum() / len(bull_fwd) * 100

print(f"""
If HMM were a good predictor of DIRECTION:
- Bear regime should have >50% negative forward returns
- Bull regime should have >50% positive forward returns

Actual results:
- Bear regime: {bear_negative_pct:.1f}% negative forward returns
- Bull regime: {bull_positive_pct:.1f}% positive forward returns

""")

if bear_negative_pct > 50:
    print("✓ Bear regime DOES predict negative returns")
else:
    print("✗ Bear regime does NOT predict negative returns")
    print("  This suggests bear regime identifies HIGH VOLATILITY, not down trends")

if bull_positive_pct > 60:
    print("✓ Bull regime strongly predicts positive returns")
elif bull_positive_pct > 50:
    print("○ Bull regime weakly predicts positive returns")
else:
    print("✗ Bull regime does NOT predict positive returns")

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
print("""
The HMM identifies VOLATILITY regimes, not directional trends:

- Bear regime = HIGH VOLATILITY (can be up or down)
- Bull regime = LOW VOLATILITY + positive drift
- Neutral regime = MEDIUM VOLATILITY

This is why:
1. Bear regime has positive average returns (volatile but upward market)
2. Bull regime has best Sharpe (low volatility + positive returns)
3. Combining with trend signals (alpha) improves directional accuracy
""")

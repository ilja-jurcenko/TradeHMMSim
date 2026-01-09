"""
Measure HMM Regime Detection Accuracy

Different approaches to evaluate how well the HMM identifies market regimes.
"""

import numpy as np
import pandas as pd
from portfolio import Portfolio
from signal_filter import HMMRegimeFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy import stats

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

print(f"\nRegimes identified:")
print(f"  Bear: {bear_regime} (volatility: {regime_info['regime_volatilities'][bear_regime]:.4f})")
print(f"  Bull: {bull_regime} (volatility: {regime_info['regime_volatilities'][bull_regime]:.4f})")
print(f"  Neutral: {neutral_regime} (volatility: {regime_info['regime_volatilities'][neutral_regime]:.4f})")

# Align data
common_idx = regime.index.intersection(close.index)
regime_aligned = regime.loc[common_idx]
close_aligned = close.loc[common_idx]

# Calculate actual market metrics for validation
returns = close_aligned.pct_change().fillna(0)
rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
rolling_ret = close_aligned.pct_change(21)  # 21-day returns

print(f"\n{'='*80}")
print("METHOD 1: VALIDATE AGAINST ACTUAL VOLATILITY")
print(f"{'='*80}")
print("\nDoes HMM correctly identify high/low volatility periods?")

# For each regime, calculate average realized volatility
for regime_id, regime_name in [(bear_regime, 'Bear'), (bull_regime, 'Bull'), (neutral_regime, 'Neutral')]:
    regime_mask = (regime_aligned == regime_id)
    regime_vol = rolling_vol[regime_mask].mean()
    regime_return = rolling_ret[regime_mask].mean()
    days_in_regime = regime_mask.sum()
    
    print(f"\n{regime_name} Regime (ID={regime_id}):")
    print(f"  Days: {days_in_regime} ({days_in_regime/len(regime_aligned)*100:.1f}%)")
    print(f"  Avg realized volatility: {regime_vol:.2%}")
    print(f"  Avg 21-day return: {regime_return:.2%}")

# Statistical test: Are volatilities significantly different?
print(f"\n{'='*80}")
print("Statistical Significance Test (ANOVA)")
print(f"{'='*80}")

vol_by_regime = [
    rolling_vol[regime_aligned == bear_regime].dropna(),
    rolling_vol[regime_aligned == bull_regime].dropna(),
    rolling_vol[regime_aligned == neutral_regime].dropna()
]

f_stat, p_value = stats.f_oneway(*vol_by_regime)
print(f"\nF-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.6f}")
if p_value < 0.001:
    print("✓ Regimes have SIGNIFICANTLY different volatilities (p < 0.001)")
elif p_value < 0.05:
    print("✓ Regimes have significantly different volatilities (p < 0.05)")
else:
    print("✗ Regimes do NOT have significantly different volatilities")

print(f"\n{'='*80}")
print("METHOD 2: FORWARD-LOOKING PERFORMANCE")
print(f"{'='*80}")
print("\nDo regime predictions predict future returns?")

# For each regime today, what are forward returns?
forward_periods = [5, 10, 21]  # 1 week, 2 weeks, 1 month

for period in forward_periods:
    print(f"\n{period}-day forward returns:")
    forward_returns = close_aligned.pct_change(period).shift(-period)
    
    for regime_id, regime_name in [(bear_regime, 'Bear'), (bull_regime, 'Bull'), (neutral_regime, 'Neutral')]:
        regime_mask = (regime_aligned == regime_id)
        fwd_ret = forward_returns[regime_mask].mean()
        fwd_std = forward_returns[regime_mask].std()
        fwd_sharpe = fwd_ret / fwd_std if fwd_std > 0 else 0
        
        print(f"  {regime_name:8s}: mean={fwd_ret:7.2%}, std={fwd_std:7.2%}, sharpe={fwd_sharpe:.3f}")

print(f"\n{'='*80}")
print("METHOD 3: REGIME PERSISTENCE")
print(f"{'='*80}")
print("\nHow stable are regime predictions?")

# Calculate average regime duration
regime_changes = regime_aligned.diff().fillna(0) != 0
regime_runs = []
current_regime = regime_aligned.iloc[0]
current_run_length = 1

for i in range(1, len(regime_aligned)):
    if regime_aligned.iloc[i] == current_regime:
        current_run_length += 1
    else:
        regime_runs.append((current_regime, current_run_length))
        current_regime = regime_aligned.iloc[i]
        current_run_length = 1
regime_runs.append((current_regime, current_run_length))

# Average duration by regime
for regime_id, regime_name in [(bear_regime, 'Bear'), (bull_regime, 'Bull'), (neutral_regime, 'Neutral')]:
    regime_durations = [length for r, length in regime_runs if r == regime_id]
    if regime_durations:
        avg_duration = np.mean(regime_durations)
        med_duration = np.median(regime_durations)
        print(f"\n{regime_name} Regime:")
        print(f"  Occurrences: {len(regime_durations)}")
        print(f"  Avg duration: {avg_duration:.1f} days")
        print(f"  Median duration: {med_duration:.0f} days")

print(f"\n{'='*80}")
print("METHOD 4: CONFUSION MATRIX (vs Market Drawdown)")
print(f"{'='*80}")
print("\nCompare HMM regime to actual market drawdowns")

# Calculate running drawdown
cumulative_returns = (1 + returns).cumprod()
running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max

# Define "true" market states based on drawdown
drawdown_threshold_bear = -0.10  # 10% drawdown = bear
drawdown_threshold_neutral = -0.05  # 5% drawdown = neutral

true_bear = (drawdown < drawdown_threshold_bear).astype(int)
true_neutral = ((drawdown >= drawdown_threshold_bear) & (drawdown < drawdown_threshold_neutral)).astype(int)
true_bull = (drawdown >= drawdown_threshold_neutral).astype(int)

# Create confusion matrix
hmm_bear = (regime_aligned == bear_regime).astype(int)
hmm_bull = (regime_aligned == bull_regime).astype(int)

print("\nConfusion Matrix (Bear regime detection):")
print(f"  True Bear days: {true_bear.sum()} | HMM predicted: {hmm_bear.sum()}")
print(f"  True Positives: {(true_bear & hmm_bear).sum()}")
print(f"  False Positives: {(~true_bear.astype(bool) & hmm_bear.astype(bool)).sum()}")
print(f"  False Negatives: {(true_bear.astype(bool) & ~hmm_bear.astype(bool)).sum()}")

if (hmm_bear.sum() > 0):
    precision = (true_bear & hmm_bear).sum() / hmm_bear.sum()
    recall = (true_bear & hmm_bear).sum() / true_bear.sum() if true_bear.sum() > 0 else 0
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")

print(f"\n{'='*80}")
print("METHOD 5: PREDICTIVE POWER (ROC-AUC)")
print(f"{'='*80}")
print("\nCan regime probabilities predict market declines?")

from sklearn.metrics import roc_auc_score, roc_curve

# Target: Next 21 days has negative return
future_returns = close_aligned.pct_change(21).shift(-21)
bear_market_ahead = (future_returns < -0.05).astype(int)  # 5% decline ahead

# Remove NaN values
valid_idx = ~future_returns.isna()
bear_market_ahead = bear_market_ahead[valid_idx]

# Predictor: Bear probability
bear_prob = probs[bear_regime].loc[common_idx][valid_idx]

if bear_market_ahead.sum() > 0 and bear_market_ahead.sum() < len(bear_market_ahead):
    auc_score = roc_auc_score(bear_market_ahead, bear_prob)
    print(f"\nROC-AUC Score: {auc_score:.3f}")
    if auc_score > 0.7:
        print("✓ STRONG predictive power (AUC > 0.70)")
    elif auc_score > 0.6:
        print("✓ MODERATE predictive power (AUC > 0.60)")
    elif auc_score > 0.5:
        print("○ WEAK predictive power (AUC > 0.50)")
    else:
        print("✗ NO predictive power (AUC ≤ 0.50)")
else:
    print("Cannot calculate AUC (insufficient data)")

print(f"\n{'='*80}")
print("SUMMARY & RECOMMENDATIONS")
print(f"{'='*80}")
print("""
Best practices for validating HMM regime detection:

1. VOLATILITY VALIDATION (Method 1) ⭐
   - Check if regimes have significantly different realized volatility
   - Use statistical tests (ANOVA, t-tests)
   - This is the PRIMARY validation since HMM uses volatility

2. FORWARD-LOOKING RETURNS (Method 2) ⭐⭐
   - Most important for trading: do regimes predict future performance?
   - Check Sharpe ratios by regime
   - Bull regime should have better forward returns

3. REGIME PERSISTENCE (Method 3)
   - Regimes should persist for reasonable periods (not flickering)
   - Average duration > 5-10 days is reasonable

4. CONFUSION MATRIX (Method 4)
   - Compare to "ground truth" (market drawdowns, VIX, etc.)
   - Useful but subjective (what IS the true regime?)

5. PREDICTIVE POWER (Method 5) ⭐⭐⭐
   - ROC-AUC measures if probabilities predict future events
   - Most objective measure of usefulness
   - AUC > 0.6 is good, > 0.7 is excellent

For trading strategies, Methods 2 & 5 are most important.
""")

print("\nAnalysis complete!")

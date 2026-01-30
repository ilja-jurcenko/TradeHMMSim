"""
Find the best HMM random_state by comparing agreement with statistical regime categorization.
This script iterates through different random_state values and measures how well each
HMM model aligns with simple statistical regime detection.
"""

import sys
from pathlib import Path
# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from portfolio import Portfolio
from signal_filter import HMMRegimeFilter

# Try to import tqdm for progress bar, fall back to simple progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not installed. Using simple progress indicator.")
    print("Install with: pip install tqdm (for better progress bars)\n")

print("="*80)
print("FINDING BEST HMM RANDOM STATE FOR STATISTICAL AGREEMENT")
print("="*80)

# Load data
print("\nLoading data...")
portfolio = Portfolio(['SPY'], '2020-01-01', '2025-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Calculate returns
returns = close.pct_change().fillna(0)
print(f"Data loaded: {len(close)} days")

# Calculate statistical regime categorization (same as analyze_statistical_vs_hmm.py)
print("\nCalculating statistical regime categorization...")
window = 21  # 1-month rolling window

# Calculate rolling statistics
rolling_mean = returns.rolling(window=window, min_periods=window).mean() * 252  # Annualized
rolling_std = returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)  # Annualized

# Calculate overall statistics for thresholds
overall_mean = returns.mean() * 252
overall_std = returns.std() * np.sqrt(252)

print(f"\nOverall statistics:")
print(f"  Mean return (annualized): {overall_mean*100:.2f}%")
print(f"  Volatility (annualized): {overall_std*100:.2f}%")

# Define statistical regimes
statistical_regime = pd.Series(index=returns.index, dtype=int)

for i in range(len(returns)):
    if pd.isna(rolling_mean.iloc[i]) or pd.isna(rolling_std.iloc[i]):
        statistical_regime.iloc[i] = -1  # Unknown
        continue
    
    mean_val = rolling_mean.iloc[i]
    std_val = rolling_std.iloc[i]
    
    # Categorization logic:
    # Bear: mean < 0 OR (mean < overall_mean*0.5 AND std > overall_std*1.2)
    # Bull: mean > overall_mean*0.8 AND std < overall_std*1.1
    # Neutral: everything else
    
    if mean_val < 0 or (mean_val < overall_mean * 0.5 and std_val > overall_std * 1.2):
        statistical_regime.iloc[i] = 0  # Bear
    elif mean_val > overall_mean * 0.8 and std_val < overall_std * 1.1:
        statistical_regime.iloc[i] = 2  # Bull
    else:
        statistical_regime.iloc[i] = 1  # Neutral

# Remove unknown periods
valid_mask = statistical_regime >= 0
statistical_regime = statistical_regime[valid_mask]

print(f"\nStatistical regime distribution:")
for regime_id, regime_name in [(0, 'Bear'), (1, 'Neutral'), (2, 'Bull')]:
    count = (statistical_regime == regime_id).sum()
    pct = count / len(statistical_regime) * 100
    print(f"  {regime_name}: {count} days ({pct:.1f}%)")

# Configuration for HMM testing
print(f"\n{'='*80}")
print("TESTING DIFFERENT RANDOM STATES")
print(f"{'='*80}")

random_state_range = range(0, 100)  # Test random_state from 0 to 99
train_window = 1000
refit_every = 26
short_vol_window = 10
long_vol_window = 60
short_ma_window = 10
long_ma_window = 60

results = []

print(f"\nTesting random_state values from {random_state_range.start} to {random_state_range.stop-1}...")
print(f"HMM Configuration:")
print(f"  train_window: {train_window}")
print(f"  refit_every: {refit_every}")
print(f"  short_vol_window: {short_vol_window}")
print(f"  long_vol_window: {long_vol_window}")
print(f"  short_ma_window: {short_ma_window}")
print(f"  long_ma_window: {long_ma_window}")
print()

# Test each random_state
iterator = tqdm(random_state_range, desc="Testing random states") if HAS_TQDM else random_state_range
for i, random_state in enumerate(iterator):
    if not HAS_TQDM and i % 10 == 0:
        print(f"Progress: {i}/{len(random_state_range)} ({i/len(random_state_range)*100:.0f}%)")
    
    try:
        # Create HMM filter with current random_state
        hmm_filter = HMMRegimeFilter(
            n_states=3, 
            random_state=random_state,
            short_vol_window=short_vol_window, 
            long_vol_window=long_vol_window,
            short_ma_window=short_ma_window, 
            long_ma_window=long_ma_window,
            covariance_type='full', 
            n_iter=100
        )
        
        # Run walkforward filtering
        probs, regime, switches = hmm_filter.walkforward_filter(
            close, 
            train_window=train_window,
            refit_every=refit_every
        )
        
        # Identify HMM regimes
        regime_info = hmm_filter.identify_regimes(close, regime)
        bear_regime = regime_info['bear_regime']
        bull_regime = regime_info['bull_regime']
        neutral_regime = regime_info['neutral_regime']
        
        # Align data
        common_idx = statistical_regime.index.intersection(regime.index)
        stat_aligned = statistical_regime.loc[common_idx]
        regime_aligned = regime.loc[common_idx]
        
        # Map HMM regimes to consistent labels: 0=Bear, 1=Neutral, 2=Bull
        regime_mapping = {bear_regime: 0, bull_regime: 2}
        if neutral_regime is not None:
            regime_mapping[neutral_regime] = 1
        else:
            # If no neutral, assign based on whether it's bear or bull
            remaining = [x for x in [0, 1, 2] if x not in [bear_regime, bull_regime]]
            if remaining:
                regime_mapping[remaining[0]] = 1
        
        hmm_regime_normalized = regime_aligned.map(regime_mapping)
        
        # Calculate agreement
        agreement = (stat_aligned == hmm_regime_normalized).sum()
        agreement_pct = agreement / len(stat_aligned) * 100
        
        # Calculate per-regime statistics
        regime_stats = {}
        for regime_id, regime_name in [(0, 'Bear'), (1, 'Neutral'), (2, 'Bull')]:
            stat_count = (stat_aligned == regime_id).sum()
            hmm_count = (hmm_regime_normalized == regime_id).sum()
            
            # Calculate overlap
            both_mask = (stat_aligned == regime_id) & (hmm_regime_normalized == regime_id)
            overlap = both_mask.sum()
            
            regime_stats[regime_name] = {
                'stat_count': stat_count,
                'hmm_count': hmm_count,
                'overlap': overlap
            }
        
        # Store results
        results.append({
            'random_state': random_state,
            'agreement_pct': agreement_pct,
            'agreement_days': agreement,
            'total_days': len(stat_aligned),
            'bear_stat': regime_stats['Bear']['stat_count'],
            'bear_hmm': regime_stats['Bear']['hmm_count'],
            'bear_overlap': regime_stats['Bear']['overlap'],
            'neutral_stat': regime_stats['Neutral']['stat_count'],
            'neutral_hmm': regime_stats['Neutral']['hmm_count'],
            'neutral_overlap': regime_stats['Neutral']['overlap'],
            'bull_stat': regime_stats['Bull']['stat_count'],
            'bull_hmm': regime_stats['Bull']['hmm_count'],
            'bull_overlap': regime_stats['Bull']['overlap'],
            'bear_regime_id': bear_regime,
            'bull_regime_id': bull_regime,
            'neutral_regime_id': neutral_regime
        })
        
    except Exception as e:
        print(f"\n✗ Error with random_state={random_state}: {e}")
        continue

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by agreement percentage
results_df = results_df.sort_values('agreement_pct', ascending=False)

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print(f"{'='*80}")

print(f"\nTested {len(results_df)} random_state values")
print(f"\nTop 10 Best Random States (by agreement %):")
print("-" * 80)

for idx, row in results_df.head(10).iterrows():
    print(f"\n#{results_df.index.get_loc(idx)+1}. random_state = {int(row['random_state'])}")
    print(f"   Agreement: {row['agreement_pct']:.2f}% ({int(row['agreement_days'])}/{int(row['total_days'])} days)")
    print(f"   Regime mapping: Bear={int(row['bear_regime_id'])}, Neutral={int(row['neutral_regime_id']) if pd.notna(row['neutral_regime_id']) else 'None'}, Bull={int(row['bull_regime_id'])}")
    print(f"   Bear overlap: {int(row['bear_overlap'])}/{int(row['bear_stat'])} stat days, {int(row['bear_hmm'])} HMM days")
    print(f"   Neutral overlap: {int(row['neutral_overlap'])}/{int(row['neutral_stat'])} stat days, {int(row['neutral_hmm'])} HMM days")
    print(f"   Bull overlap: {int(row['bull_overlap'])}/{int(row['bull_stat'])} stat days, {int(row['bull_hmm'])} HMM days")

print(f"\n{'='*80}")
print(f"Bottom 5 Worst Random States:")
print("-" * 80)

for idx, row in results_df.tail(5).iterrows():
    print(f"\nrandom_state = {int(row['random_state'])}: {row['agreement_pct']:.2f}% agreement")

# Save results to CSV
output_file = 'hmm_analysis/random_state_comparison_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Full results saved to: {output_file}")

# Print statistics
print(f"\n{'='*80}")
print("STATISTICS")
print(f"{'='*80}")
print(f"\nAgreement Percentage Statistics:")
print(f"  Mean: {results_df['agreement_pct'].mean():.2f}%")
print(f"  Median: {results_df['agreement_pct'].median():.2f}%")
print(f"  Std Dev: {results_df['agreement_pct'].std():.2f}%")
print(f"  Min: {results_df['agreement_pct'].min():.2f}% (random_state={int(results_df.loc[results_df['agreement_pct'].idxmin(), 'random_state'])})")
print(f"  Max: {results_df['agreement_pct'].max():.2f}% (random_state={int(results_df.loc[results_df['agreement_pct'].idxmax(), 'random_state'])})")

# Recommendation
best_random_state = int(results_df.iloc[0]['random_state'])
best_agreement = results_df.iloc[0]['agreement_pct']

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")
print(f"\n✓ BEST random_state: {best_random_state}")
print(f"  Agreement with statistical method: {best_agreement:.2f}%")
print(f"  This random_state produces an HMM model most aligned with")
print(f"  simple statistical regime categorization.")
print(f"\n  To use this in your scripts, update:")
print(f"    random_state={best_random_state}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

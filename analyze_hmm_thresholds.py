"""
Analyze HMM probability distributions and threshold effectiveness.
"""

import numpy as np
import pandas as pd
from portfolio import Portfolio
from AlphaModels import EMA
from SignalFilter import HMMRegimeFilter

# Load data
portfolio = Portfolio(['SPY'], '2020-01-01', '2025-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Setup models
alpha_model = EMA(short_window=10, long_window=30)
hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)

# Generate alpha signals
alpha_signals = alpha_model.generate_signals(close)

# Run HMM
print("Running HMM walk-forward filter...")
probs, regime, switches = hmm_filter.walkforward_filter(
    close, 
    train_window=504,
    refit_every=21
)

# Identify regimes
regime_info = hmm_filter.identify_regimes(close, regime)
bear_regime = regime_info['bear_regime']
bull_regime = regime_info['bull_regime']
neutral_regime = regime_info['neutral_regime']

print(f"\nRegime Identification:")
print(f"  Bear: {bear_regime}, Bull: {bull_regime}, Neutral: {neutral_regime}")

# Align indices
common_idx = alpha_signals.index.intersection(probs.index)
alpha_signals_aligned = alpha_signals.loc[common_idx]

# Get probabilities
bear_prob = probs[bear_regime].loc[common_idx]
bull_prob = probs[bull_regime].loc[common_idx]
if neutral_regime is not None:
    neutral_prob = probs[neutral_regime].loc[common_idx]
    bull_prob_combined = bull_prob + neutral_prob
else:
    bull_prob_combined = bull_prob
    neutral_prob = pd.Series(0, index=common_idx)

# Analyze probability distributions
print(f"\n{'='*80}")
print("PROBABILITY DISTRIBUTIONS")
print(f"{'='*80}")
print(f"\nBear Probability:")
print(f"  Mean: {bear_prob.mean():.3f}")
print(f"  Median: {bear_prob.median():.3f}")
print(f"  Std: {bear_prob.std():.3f}")
print(f"  Max: {bear_prob.max():.3f}")
print(f"  % > 0.65: {(bear_prob > 0.65).sum() / len(bear_prob) * 100:.1f}%")

print(f"\nBull Probability:")
print(f"  Mean: {bull_prob.mean():.3f}")
print(f"  Median: {bull_prob.median():.3f}")
print(f"  Std: {bull_prob.std():.3f}")
print(f"  Max: {bull_prob.max():.3f}")
print(f"  % > 0.65: {(bull_prob > 0.65).sum() / len(bull_prob) * 100:.1f}%")

print(f"\nBull + Neutral Probability:")
print(f"  Mean: {bull_prob_combined.mean():.3f}")
print(f"  Median: {bull_prob_combined.median():.3f}")
print(f"  Std: {bull_prob_combined.std():.3f}")
print(f"  Max: {bull_prob_combined.max():.3f}")
print(f"  % > 0.65: {(bull_prob_combined > 0.65).sum() / len(bull_prob_combined) * 100:.1f}%")

if neutral_regime is not None:
    print(f"\nNeutral Probability:")
    print(f"  Mean: {neutral_prob.mean():.3f}")
    print(f"  Median: {neutral_prob.median():.3f}")
    print(f"  Std: {neutral_prob.std():.3f}")
    print(f"  Max: {neutral_prob.max():.3f}")

# Test different threshold combinations
print(f"\n{'='*80}")
print("HMM SIGNAL GENERATION WITH DIFFERENT THRESHOLDS")
print(f"{'='*80}")

thresholds_to_test = [0.40, 0.50, 0.60, 0.65, 0.70]

for threshold in thresholds_to_test:
    # Current logic: (bull_prob > threshold) AND (bear_prob < threshold)
    hmm_signal_current = ((bull_prob > threshold) & (bear_prob < threshold)).astype(int)
    
    # Alternative: use combined bull+neutral probability
    hmm_signal_combined = ((bull_prob_combined > threshold) & (bear_prob < threshold)).astype(int)
    
    # Combined positions
    positions_current = (alpha_signals_aligned | hmm_signal_current).astype(int)
    positions_combined = (alpha_signals_aligned | hmm_signal_combined).astype(int)
    
    print(f"\nThreshold: {threshold}")
    print(f"  Current (bull only > {threshold}):")
    print(f"    HMM signals: {hmm_signal_current.sum()} / {len(hmm_signal_current)} ({hmm_signal_current.sum()/len(hmm_signal_current)*100:.1f}%)")
    print(f"    Combined time in market: {positions_current.sum() / len(positions_current) * 100:.1f}%")
    print(f"  Alternative (bull+neutral > {threshold}):")
    print(f"    HMM signals: {hmm_signal_combined.sum()} / {len(hmm_signal_combined)} ({hmm_signal_combined.sum()/len(hmm_signal_combined)*100:.1f}%)")
    print(f"    Combined time in market: {positions_combined.sum() / len(positions_combined) * 100:.1f}%")

# Alpha signals analysis
print(f"\n{'='*80}")
print("ALPHA SIGNALS ANALYSIS")
print(f"{'='*80}")
print(f"  Alpha signals: {alpha_signals_aligned.sum()} / {len(alpha_signals_aligned)} ({alpha_signals_aligned.sum()/len(alpha_signals_aligned)*100:.1f}%)")

# Check overlap
hmm_only = ((bull_prob > 0.65) & (bear_prob < 0.65)).astype(int)
both_signal = (alpha_signals_aligned & hmm_only).astype(int)
alpha_only_signal = (alpha_signals_aligned & ~hmm_only.astype(bool)).astype(int)
hmm_only_signal = (~alpha_signals_aligned.astype(bool) & hmm_only.astype(bool)).astype(int)

print(f"\n  Signal Overlap (threshold=0.65):")
print(f"    Both Alpha and HMM: {both_signal.sum()} days ({both_signal.sum()/len(common_idx)*100:.1f}%)")
print(f"    Alpha only: {alpha_only_signal.sum()} days ({alpha_only_signal.sum()/len(common_idx)*100:.1f}%)")
print(f"    HMM only: {hmm_only_signal.sum()} days ({hmm_only_signal.sum()/len(common_idx)*100:.1f}%)")
print(f"    Neither: {(~(alpha_signals_aligned.astype(bool) | hmm_only.astype(bool))).sum()} days")

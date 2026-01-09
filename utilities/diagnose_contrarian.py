"""Diagnose contrarian strategy behavior"""
from portfolio import Portfolio
from alpha_models import EMA
from signal_filter import HMMRegimeFilter
import pandas as pd

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
print("Running HMM...")
probs, regime, switches = hmm_filter.walkforward_filter(close, train_window=504, refit_every=21)

# Identify regimes
regime_info = hmm_filter.identify_regimes(close, regime)
bear_regime = regime_info['bear_regime']
bull_regime = regime_info['bull_regime']
neutral_regime = regime_info['neutral_regime']

print(f"\nRegimes: Bear={bear_regime}, Bull={bull_regime}, Neutral={neutral_regime}")

# Align
common_idx = alpha_signals.index.intersection(probs.index)
alpha_signals_aligned = alpha_signals.loc[common_idx]

# Get probabilities
bear_prob = probs[bear_regime].loc[common_idx]
bull_prob = probs[bull_regime].loc[common_idx]
bull_prob_combined = bull_prob.copy()
if neutral_regime is not None:
    bull_prob_combined = bull_prob + probs[neutral_regime].loc[common_idx]

# Apply contrarian logic
threshold = 0.65
hmm_bull_signal = (bull_prob_combined > threshold).astype(bool)
hmm_bear_signal = (bear_prob > threshold).astype(bool)

positions_alpha = alpha_signals_aligned.astype(bool).copy()

# Detect contrarian signals
contrarian_entry = (~alpha_signals_aligned.astype(bool)) & hmm_bull_signal
contrarian_exit = (alpha_signals_aligned.astype(bool)) & hmm_bear_signal

positions_contrarian = positions_alpha.copy()
positions_contrarian[contrarian_entry] = True
positions_contrarian[contrarian_exit] = False

print(f"\n{'='*80}")
print("SIGNAL ANALYSIS")
print(f"{'='*80}")
print(f"Total days: {len(common_idx)}")
print(f"\nAlpha signals: {alpha_signals_aligned.sum()} days ({alpha_signals_aligned.sum()/len(common_idx)*100:.1f}%)")
print(f"HMM bull signals: {hmm_bull_signal.sum()} days ({hmm_bull_signal.sum()/len(common_idx)*100:.1f}%)")
print(f"HMM bear signals: {hmm_bear_signal.sum()} days ({hmm_bear_signal.sum()/len(common_idx)*100:.1f}%)")

print(f"\n{'='*80}")
print("CONTRARIAN SIGNALS")
print(f"{'='*80}")
print(f"Contrarian entries: {contrarian_entry.sum()} days ({contrarian_entry.sum()/len(common_idx)*100:.1f}%)")
print(f"  (Alpha=0, HMM bull=1)")
print(f"\nContrarian exits: {contrarian_exit.sum()} days ({contrarian_exit.sum()/len(common_idx)*100:.1f}%)")
print(f"  (Alpha=1, HMM bear=1)")

print(f"\n{'='*80}")
print("POSITION COMPARISON")
print(f"{'='*80}")
print(f"Alpha only time in market: {positions_alpha.sum()} days ({positions_alpha.sum()/len(common_idx)*100:.1f}%)")
print(f"Contrarian time in market: {positions_contrarian.sum()} days ({positions_contrarian.sum()/len(common_idx)*100:.1f}%)")

# Show some examples
print(f"\n{'='*80}")
print("SAMPLE CONTRARIAN ENTRIES (first 5)")
print(f"{'='*80}")
entry_dates = common_idx[contrarian_entry][:5]
for date in entry_dates:
    print(f"{date.date()}: Price=${close.loc[date]:.2f}, Alpha={alpha_signals_aligned.loc[date]}, "
          f"Bull+Neutral Prob={bull_prob_combined.loc[date]:.3f}, Bear Prob={bear_prob.loc[date]:.3f}")

print(f"\n{'='*80}")
print("SAMPLE CONTRARIAN EXITS (first 5)")
print(f"{'='*80}")
exit_dates = common_idx[contrarian_exit][:5]
for date in exit_dates:
    print(f"{date.date()}: Price=${close.loc[date]:.2f}, Alpha={alpha_signals_aligned.loc[date]}, "
          f"Bull+Neutral Prob={bull_prob_combined.loc[date]:.3f}, Bear Prob={bear_prob.loc[date]:.3f}")

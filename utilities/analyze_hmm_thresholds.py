"""
Analyze HMM probability distributions and threshold effectiveness.
"""

import numpy as np
import pandas as pd
from portfolio import Portfolio
from alpha_models import EMA
from signal_filter import HMMRegimeFilter

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

# Create visualization
print(f"\n{'='*80}")
print("GENERATING HMM REGIME VISUALIZATION")
print(f"{'='*80}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import os

# Create output directory
os.makedirs('hmm_analysis', exist_ok=True)

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle('HMM Regime Analysis with Trading Signals', fontsize=16, fontweight='bold')

# Align data
plot_idx = common_idx
close_plot = close.loc[plot_idx]
regime_plot = regime.loc[plot_idx]

# Plot 1: Price with regime coloring
ax1 = axes[0]
ax1.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax1.set_title('SPY Price with HMM Regime Background', fontsize=11, fontweight='bold')

# Plot price line
ax1.plot(plot_idx, close_plot, color='black', linewidth=1.5, label='SPY Price', zorder=3)

# Color background by regime
for regime_id in [bear_regime, bull_regime, neutral_regime]:
    if regime_id is None:
        continue
    regime_mask = (regime_plot == regime_id)
    regime_dates = plot_idx[regime_mask]
    
    if regime_id == bear_regime:
        color, label = 'red', 'Bear Regime'
        alpha = 0.2
    elif regime_id == bull_regime:
        color, label = 'green', 'Bull Regime'
        alpha = 0.2
    else:
        color, label = 'gray', 'Neutral Regime'
        alpha = 0.15
    
    # Fill between for regime periods
    for i in range(len(regime_dates)):
        if i == 0 or (regime_dates[i] - regime_dates[i-1]).days > 2:
            start_idx = regime_dates[i]
            end_idx = regime_dates[i]
        else:
            end_idx = regime_dates[i]
        
        if i == len(regime_dates) - 1 or (i < len(regime_dates) - 1 and (regime_dates[i+1] - regime_dates[i]).days > 2):
            y_min, y_max = ax1.get_ylim()
            ax1.axvspan(start_idx, end_idx, alpha=alpha, color=color, zorder=1)

ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 2: Price with Alpha signals (entry/exit markers)
ax2 = axes[1]
ax2.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax2.set_title('Alpha Model Signals (EMA 10/30)', fontsize=11, fontweight='bold')
ax2.plot(plot_idx, close_plot, color='black', linewidth=1.2, alpha=0.7, label='SPY Price')

# Mark entries and exits
signal_changes = alpha_signals_aligned.diff()
entries = plot_idx[signal_changes == 1]
exits = plot_idx[signal_changes == -1]

if len(entries) > 0:
    ax2.scatter(entries, close.loc[entries], color='green', marker='^', s=100, 
               label=f'Buy Signal ({len(entries)})', zorder=5, edgecolors='darkgreen', linewidths=1.5)
if len(exits) > 0:
    ax2.scatter(exits, close.loc[exits], color='red', marker='v', s=100, 
               label=f'Sell Signal ({len(exits)})', zorder=5, edgecolors='darkred', linewidths=1.5)

# Shade when in position
in_position = alpha_signals_aligned.astype(bool)
ax2.fill_between(plot_idx, close_plot.min() * 0.98, close_plot.max() * 1.02, 
                 where=in_position, alpha=0.15, color='blue', label='In Position', interpolate=True)

ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 3: HMM Only signals
ax3 = axes[2]
ax3.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax3.set_title('HMM Only Signals (Bull+Neutral > 0.65)', fontsize=11, fontweight='bold')
ax3.plot(plot_idx, close_plot, color='black', linewidth=1.2, alpha=0.7, label='SPY Price')

# HMM signal
hmm_signal = (bull_prob_combined > 0.65).astype(int)
hmm_changes = hmm_signal.diff()
hmm_entries = plot_idx[hmm_changes == 1]
hmm_exits = plot_idx[hmm_changes == -1]

if len(hmm_entries) > 0:
    ax3.scatter(hmm_entries, close.loc[hmm_entries], color='green', marker='^', s=100, 
               label=f'HMM Buy ({len(hmm_entries)})', zorder=5, edgecolors='darkgreen', linewidths=1.5)
if len(hmm_exits) > 0:
    ax3.scatter(hmm_exits, close.loc[hmm_exits], color='red', marker='v', s=100, 
               label=f'HMM Sell ({len(hmm_exits)})', zorder=5, edgecolors='darkred', linewidths=1.5)

# Shade when HMM says in position
hmm_in_position = hmm_signal.astype(bool)
ax3.fill_between(plot_idx, close_plot.min() * 0.98, close_plot.max() * 1.02, 
                 where=hmm_in_position, alpha=0.15, color='purple', label='HMM In Position', interpolate=True)

ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 4: Combined (Contrarian) strategy
ax4 = axes[3]
ax4.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax4.set_title('Alpha + HMM Combine (Contrarian Entry)', fontsize=11, fontweight='bold')
ax4.plot(plot_idx, close_plot, color='black', linewidth=1.2, alpha=0.7, label='SPY Price')

# Combined positions (contrarian entry logic)
combined_positions = alpha_signals_aligned.astype(bool).copy()
contrarian_entry = (~alpha_signals_aligned.astype(bool)) & (bull_prob_combined > 0.65).astype(bool)
combined_positions[contrarian_entry] = True

combined_changes = combined_positions.astype(int).diff()
combined_entries = plot_idx[combined_changes == 1]
combined_exits = plot_idx[combined_changes == -1]

# Mark contrarian entries differently
contrarian_entry_dates = plot_idx[contrarian_entry & (combined_changes == 1)]
normal_entries = [d for d in combined_entries if d not in contrarian_entry_dates]

if len(normal_entries) > 0:
    ax4.scatter(normal_entries, close.loc[normal_entries], color='green', marker='^', s=100, 
               label=f'Alpha Buy ({len(normal_entries)})', zorder=5, edgecolors='darkgreen', linewidths=1.5)
if len(contrarian_entry_dates) > 0:
    ax4.scatter(contrarian_entry_dates, close.loc[contrarian_entry_dates], color='orange', marker='^', s=120, 
               label=f'Contrarian Buy ({len(contrarian_entry_dates)})', zorder=6, edgecolors='darkorange', linewidths=2)
if len(combined_exits) > 0:
    ax4.scatter(combined_exits, close.loc[combined_exits], color='red', marker='v', s=100, 
               label=f'Sell ({len(combined_exits)})', zorder=5, edgecolors='darkred', linewidths=1.5)

# Shade when in position
ax4.fill_between(plot_idx, close_plot.min() * 0.98, close_plot.max() * 1.02, 
                 where=combined_positions, alpha=0.15, color='teal', label='In Position', interpolate=True)

ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax4.set_xlabel('Date', fontsize=10, fontweight='bold')

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

# Save figure
output_path = 'hmm_analysis/hmm_regime_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")
plt.close()

# Create regime probabilities plot
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig2.suptitle('HMM Regime Probabilities Over Time', fontsize=16, fontweight='bold')

# Plot probabilities
axes2[0].plot(plot_idx, bear_prob, color='red', linewidth=1.5, label='Bear Probability')
axes2[0].axhline(y=0.65, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.65)')
axes2[0].fill_between(plot_idx, 0, bear_prob, alpha=0.3, color='red')
axes2[0].set_ylabel('Probability', fontsize=10, fontweight='bold')
axes2[0].set_title('Bear Regime Probability', fontsize=11, fontweight='bold')
axes2[0].legend(loc='upper left', fontsize=9)
axes2[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
axes2[0].set_ylim([0, 1])

axes2[1].plot(plot_idx, bull_prob, color='green', linewidth=1.5, label='Bull Probability')
axes2[1].axhline(y=0.65, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.65)')
axes2[1].fill_between(plot_idx, 0, bull_prob, alpha=0.3, color='green')
axes2[1].set_ylabel('Probability', fontsize=10, fontweight='bold')
axes2[1].set_title('Bull Regime Probability', fontsize=11, fontweight='bold')
axes2[1].legend(loc='upper left', fontsize=9)
axes2[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
axes2[1].set_ylim([0, 1])

axes2[2].plot(plot_idx, bull_prob_combined, color='blue', linewidth=1.5, label='Bull+Neutral Probability')
axes2[2].axhline(y=0.65, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.65)')
axes2[2].fill_between(plot_idx, 0, bull_prob_combined, alpha=0.3, color='blue')
axes2[2].set_ylabel('Probability', fontsize=10, fontweight='bold')
axes2[2].set_title('Bull + Neutral Combined Probability (Used for HMM Only)', fontsize=11, fontweight='bold')
axes2[2].legend(loc='upper left', fontsize=9)
axes2[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
axes2[2].set_ylim([0, 1])
axes2[2].set_xlabel('Date', fontsize=10, fontweight='bold')

# Format x-axis
for ax in axes2:
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

output_path2 = 'hmm_analysis/hmm_probabilities.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Probabilities plot saved to: {output_path2}")
plt.close()

print(f"\nAnalysis complete! Check the 'hmm_analysis/' directory for visualizations.")

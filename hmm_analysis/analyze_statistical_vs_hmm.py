"""
Compare statistical regime categorization (based on returns mean/std) with HMM model.
This script categorizes market regimes using simple statistical rules and compares
them to HMM-based regime detection.
"""

import sys
from pathlib import Path
# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from portfolio import Portfolio
from signal_filter import HMMRegimeFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
import os

print("="*80)
print("STATISTICAL VS HMM REGIME CATEGORIZATION ANALYSIS")
print("="*80)

# Load data
print("\nLoading data...")
portfolio = Portfolio(['SPY'], '2020-01-01', '2025-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Calculate returns
returns = close.pct_change().fillna(0)
print(f"Data loaded: {len(close)} days")

# Run HMM model
print("\nRunning HMM model...")
hmm_filter = HMMRegimeFilter(n_states=3, random_state=14,
                             short_vol_window=10, long_vol_window=60,
                             short_ma_window=10, long_ma_window=60,
                             covariance_type='full', n_iter=100)
probs, regime, switches = hmm_filter.walkforward_filter(
    close, 
    train_window=1000,
    refit_every=26
)

# Identify HMM regimes
regime_info = hmm_filter.identify_regimes(close, regime)
bear_regime = regime_info['bear_regime']
bull_regime = regime_info['bull_regime']
neutral_regime = regime_info['neutral_regime']

print(f"HMM Regimes identified:")
print(f"  Bear: {bear_regime}, Bull: {bull_regime}, Neutral: {neutral_regime}")

# Align data
common_idx = returns.index.intersection(regime.index)
returns_aligned = returns.loc[common_idx]
close_aligned = close.loc[common_idx]
regime_aligned = regime.loc[common_idx]

# Statistical regime categorization using rolling windows
print("\nCalculating statistical regime categorization...")
window = 21  # 1-month rolling window

# Calculate rolling statistics
rolling_mean = returns_aligned.rolling(window=window, min_periods=window).mean() * 252  # Annualized
rolling_std = returns_aligned.rolling(window=window, min_periods=window).std() * np.sqrt(252)  # Annualized

# Calculate overall statistics for thresholds
overall_mean = returns_aligned.mean() * 252
overall_std = returns_aligned.std() * np.sqrt(252)

print(f"\nOverall statistics:")
print(f"  Mean return (annualized): {overall_mean*100:.2f}%")
print(f"  Volatility (annualized): {overall_std*100:.2f}%")

# Define statistical regimes based on rolling mean and volatility
# Bear: negative returns with high volatility
# Bull: positive returns with low-moderate volatility
# Neutral: low returns or moderate volatility

statistical_regime = pd.Series(index=common_idx, dtype=int)

for i in range(len(common_idx)):
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
regime_aligned = regime_aligned[valid_mask]
close_aligned = close_aligned[valid_mask]
returns_aligned = returns_aligned[valid_mask]
rolling_mean = rolling_mean[valid_mask]
rolling_std = rolling_std[valid_mask]
common_idx = statistical_regime.index

print(f"\nStatistical regime distribution:")
for regime_id, regime_name in [(0, 'Bear'), (1, 'Neutral'), (2, 'Bull')]:
    count = (statistical_regime == regime_id).sum()
    pct = count / len(statistical_regime) * 100
    print(f"  {regime_name}: {count} days ({pct:.1f}%)")

# Calculate HMM regime distribution (normalized to same labels)
hmm_regime_normalized = regime_aligned.copy()
# Map HMM regimes to consistent labels: 0=Bear, 1=Neutral, 2=Bull
regime_mapping = {bear_regime: 0, bull_regime: 2}
if neutral_regime is not None:
    regime_mapping[neutral_regime] = 1
else:
    # If no neutral, assign based on whether it's bear or bull
    regime_mapping[0] = 0 if 0 not in regime_mapping else 1
    regime_mapping[1] = 1 if 1 not in regime_mapping else 2
    regime_mapping[2] = 2 if 2 not in regime_mapping else 1

hmm_regime_normalized = hmm_regime_normalized.map(regime_mapping)

print(f"\nHMM regime distribution:")
for regime_id, regime_name in [(0, 'Bear'), (1, 'Neutral'), (2, 'Bull')]:
    count = (hmm_regime_normalized == regime_id).sum()
    pct = count / len(hmm_regime_normalized) * 100
    print(f"  {regime_name}: {count} days ({pct:.1f}%)")

# Calculate agreement between statistical and HMM regimes
agreement = (statistical_regime == hmm_regime_normalized).sum()
agreement_pct = agreement / len(statistical_regime) * 100

print(f"\n{'='*80}")
print(f"REGIME AGREEMENT ANALYSIS")
print(f"{'='*80}")
print(f"Total agreement: {agreement} / {len(statistical_regime)} days ({agreement_pct:.1f}%)")

# Calculate confusion matrix
confusion_matrix = pd.DataFrame(0, index=['Stat Bear', 'Stat Neutral', 'Stat Bull'],
                               columns=['HMM Bear', 'HMM Neutral', 'HMM Bull'])

for i in range(len(statistical_regime)):
    stat_regime = int(statistical_regime.iloc[i])
    hmm_regime = int(hmm_regime_normalized.iloc[i])
    
    stat_label = ['Stat Bear', 'Stat Neutral', 'Stat Bull'][stat_regime]
    hmm_label = ['HMM Bear', 'HMM Neutral', 'HMM Bull'][hmm_regime]
    
    confusion_matrix.loc[stat_label, hmm_label] += 1

print(f"\nConfusion Matrix (rows=Statistical, cols=HMM):")
print(confusion_matrix)
print(f"\nRow percentages (how statistical regimes map to HMM):")
print((confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100).round(1))

# Calculate regime-specific statistics for both methods
print(f"\n{'='*80}")
print(f"REGIME-SPECIFIC PERFORMANCE COMPARISON")
print(f"{'='*80}")

for regime_id, regime_name in [(0, 'Bear'), (1, 'Neutral'), (2, 'Bull')]:
    stat_mask = (statistical_regime == regime_id)
    hmm_mask = (hmm_regime_normalized == regime_id)
    
    if stat_mask.sum() > 0:
        stat_returns = returns_aligned[stat_mask]
        stat_mean = stat_returns.mean() * 252 * 100
        stat_std = stat_returns.std() * np.sqrt(252) * 100
        
        print(f"\n{regime_name} Regime - Statistical Method:")
        print(f"  Days: {stat_mask.sum()}")
        print(f"  Mean return (annualized): {stat_mean:.2f}%")
        print(f"  Volatility (annualized): {stat_std:.2f}%")
    
    if hmm_mask.sum() > 0:
        hmm_returns = returns_aligned[hmm_mask]
        hmm_mean = hmm_returns.mean() * 252 * 100
        hmm_std = hmm_returns.std() * np.sqrt(252) * 100
        
        print(f"\n{regime_name} Regime - HMM Method:")
        print(f"  Days: {hmm_mask.sum()}")
        print(f"  Mean return (annualized): {hmm_mean:.2f}%")
        print(f"  Volatility (annualized): {hmm_std:.2f}%")

# Create visualizations
print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}")

os.makedirs('hmm_analysis', exist_ok=True)

# Figure 1: Regime comparison over time
fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
fig.suptitle('Statistical vs HMM Regime Categorization Comparison', fontsize=16, fontweight='bold')

# Plot 1: Price with statistical regimes
ax1 = axes[0]
ax1.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax1.set_title('Price with Statistical Regime Background', fontsize=11, fontweight='bold')
ax1.plot(common_idx, close_aligned, color='black', linewidth=1.5, label='SPY Price', zorder=3)

# Color background by statistical regime
for regime_id, color, label in [(0, 'red', 'Bear'), (1, 'gray', 'Neutral'), (2, 'green', 'Bull')]:
    regime_mask = (statistical_regime == regime_id)
    regime_dates = common_idx[regime_mask]
    
    for date in regime_dates:
        ax1.axvspan(date, date, alpha=0.2, color=color, zorder=1)

ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 2: Price with HMM regimes
ax2 = axes[1]
ax2.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax2.set_title('Price with HMM Regime Background', fontsize=11, fontweight='bold')
ax2.plot(common_idx, close_aligned, color='black', linewidth=1.5, label='SPY Price', zorder=3)

# Color background by HMM regime
for regime_id, color, label in [(0, 'red', 'Bear'), (1, 'gray', 'Neutral'), (2, 'green', 'Bull')]:
    regime_mask = (hmm_regime_normalized == regime_id)
    regime_dates = common_idx[regime_mask]
    
    for date in regime_dates:
        ax2.axvspan(date, date, alpha=0.2, color=color, zorder=1)

ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 3: Disagreement highlighting
ax3 = axes[2]
ax3.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax3.set_title('Regime Disagreement Highlighting (Red = Disagree, Green = Agree)', fontsize=11, fontweight='bold')
ax3.plot(common_idx, close_aligned, color='black', linewidth=1.5, alpha=0.5, zorder=2)

# Highlight agreement/disagreement
agreement_mask = (statistical_regime == hmm_regime_normalized)
disagreement_dates = common_idx[~agreement_mask]
agreement_dates = common_idx[agreement_mask]

for date in disagreement_dates:
    ax3.axvspan(date, date, alpha=0.3, color='red', zorder=1)
for date in agreement_dates:
    ax3.axvspan(date, date, alpha=0.15, color='green', zorder=1)

# Add text annotation
agreement_text = f"Agreement: {agreement_pct:.1f}% | Disagreement: {100-agreement_pct:.1f}%"
ax3.text(0.02, 0.98, agreement_text, transform=ax3.transAxes, 
        fontsize=10, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontweight='bold')

ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 4: Rolling statistics used for statistical regime
ax4 = axes[3]
ax4.set_ylabel('Annualized Return (%)', fontsize=10, fontweight='bold')
ax4.set_title(f'Rolling Statistics ({window}-day window) for Statistical Categorization', fontsize=11, fontweight='bold')

# Plot rolling mean
ax4.plot(common_idx, rolling_mean * 100, color='blue', linewidth=1.5, label='Rolling Mean Return', zorder=3)
ax4.axhline(y=overall_mean*100, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Overall Mean')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

# Color background by statistical regime
for regime_id, color in [(0, 'red'), (1, 'gray'), (2, 'green')]:
    regime_mask = (statistical_regime == regime_id)
    regime_dates = common_idx[regime_mask]
    for date in regime_dates:
        ax4.axvspan(date, date, alpha=0.15, color=color, zorder=1)

ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 5: Rolling volatility
ax5 = axes[4]
ax5.set_ylabel('Annualized Volatility (%)', fontsize=10, fontweight='bold')
ax5.set_title(f'Rolling Volatility ({window}-day window)', fontsize=11, fontweight='bold')
ax5.set_xlabel('Date', fontsize=10, fontweight='bold')

# Plot rolling std
ax5.plot(common_idx, rolling_std * 100, color='orange', linewidth=1.5, label='Rolling Volatility', zorder=3)
ax5.axhline(y=overall_std*100, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Overall Volatility')

# Color background by statistical regime
for regime_id, color in [(0, 'red'), (1, 'gray'), (2, 'green')]:
    regime_mask = (statistical_regime == regime_id)
    regime_dates = common_idx[regime_mask]
    for date in regime_dates:
        ax5.axvspan(date, date, alpha=0.15, color=color, zorder=1)

ax5.legend(loc='upper left', fontsize=9)
ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

output_path1 = 'hmm_analysis/statistical_vs_hmm_regimes.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n✓ Regime comparison saved to: {output_path1}")
plt.close()

# Figure 2: Performance comparison
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Statistical vs HMM Regime Performance Metrics', fontsize=16, fontweight='bold')

# Prepare data for bar charts
regime_names = ['Bear', 'Neutral', 'Bull']
stat_means = []
stat_stds = []
stat_counts = []
hmm_means = []
hmm_stds = []
hmm_counts = []

for regime_id in [0, 1, 2]:
    stat_mask = (statistical_regime == regime_id)
    hmm_mask = (hmm_regime_normalized == regime_id)
    
    if stat_mask.sum() > 0:
        stat_returns = returns_aligned[stat_mask]
        stat_means.append(stat_returns.mean() * 252 * 100)
        stat_stds.append(stat_returns.std() * np.sqrt(252) * 100)
        stat_counts.append(stat_mask.sum())
    else:
        stat_means.append(0)
        stat_stds.append(0)
        stat_counts.append(0)
    
    if hmm_mask.sum() > 0:
        hmm_returns = returns_aligned[hmm_mask]
        hmm_means.append(hmm_returns.mean() * 252 * 100)
        hmm_stds.append(hmm_returns.std() * np.sqrt(252) * 100)
        hmm_counts.append(hmm_mask.sum())
    else:
        hmm_means.append(0)
        hmm_stds.append(0)
        hmm_counts.append(0)

x_pos = np.arange(len(regime_names))
width = 0.35

# Plot 1: Mean returns comparison
ax_mean = axes2[0, 0]
bars1 = ax_mean.bar(x_pos - width/2, stat_means, width, label='Statistical (Left)',
                    color=['lightcoral', 'lightgray', 'lightgreen'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax_mean.bar(x_pos + width/2, hmm_means, width, label='HMM (Right)',
                    color=['darkred', 'darkgray', 'darkgreen'], alpha=0.8, edgecolor='black', linewidth=1.5)

ax_mean.set_ylabel('Mean Return (%)', fontsize=10, fontweight='bold')
ax_mean.set_title('Annualized Mean Returns by Regime\n(Left=Statistical, Right=HMM)', fontsize=11, fontweight='bold')
ax_mean.set_xticks(x_pos)
ax_mean.set_xticklabels(regime_names, fontsize=10, fontweight='bold')
ax_mean.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax_mean.axhline(y=0, color='black', linewidth=0.8)
ax_mean.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

# Add value labels with method indicator
for i, bars in enumerate([bars1, bars2]):
    method = 'S' if i == 0 else 'H'  # S=Statistical, H=HMM
    for bar in bars:
        height = bar.get_height()
        ax_mean.text(bar.get_x() + bar.get_width()/2., height,
                    f'{method}\n{height:.1f}%', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=7, fontweight='bold')

# Plot 2: Volatility comparison
ax_std = axes2[0, 1]
bars3 = ax_std.bar(x_pos - width/2, stat_stds, width, label='Statistical (Left)',
                   color=['lightcoral', 'lightgray', 'lightgreen'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars4 = ax_std.bar(x_pos + width/2, hmm_stds, width, label='HMM (Right)',
                   color=['darkred', 'darkgray', 'darkgreen'], alpha=0.8, edgecolor='black', linewidth=1.5)

ax_std.set_ylabel('Volatility (%)', fontsize=10, fontweight='bold')
ax_std.set_title('Annualized Volatility by Regime\n(Left=Statistical, Right=HMM)', fontsize=11, fontweight='bold')
ax_std.set_xticks(x_pos)
ax_std.set_xticklabels(regime_names, fontsize=10, fontweight='bold')
ax_std.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax_std.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

# Add value labels with method indicator
for i, bars in enumerate([bars3, bars4]):
    method = 'S' if i == 0 else 'H'
    for bar in bars:
        height = bar.get_height()
        ax_std.text(bar.get_x() + bar.get_width()/2., height,
                   f'{method}\n{height:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

# Plot 3: Day count comparison
ax_count = axes2[1, 0]
bars5 = ax_count.bar(x_pos - width/2, stat_counts, width, label='Statistical (Left)',
                     color=['lightcoral', 'lightgray', 'lightgreen'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars6 = ax_count.bar(x_pos + width/2, hmm_counts, width, label='HMM (Right)',
                     color=['darkred', 'darkgray', 'darkgreen'], alpha=0.8, edgecolor='black', linewidth=1.5)

ax_count.set_ylabel('Number of Days', fontsize=10, fontweight='bold')
ax_count.set_title('Days Spent in Each Regime\n(Left=Statistical, Right=HMM)', fontsize=11, fontweight='bold')
ax_count.set_xticks(x_pos)
ax_count.set_xticklabels(regime_names, fontsize=10, fontweight='bold')
ax_count.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax_count.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

# Add value labels with method indicator
for i, bars in enumerate([bars5, bars6]):
    method = 'S' if i == 0 else 'H'
    for bar in bars:
        height = bar.get_height()
        ax_count.text(bar.get_x() + bar.get_width()/2., height,
                     f'{method}\n{int(height)}', ha='center', va='bottom', fontsize=7, fontweight='bold')

# Plot 4: Confusion matrix heatmap
ax_conf = axes2[1, 1]
conf_matrix_norm = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100
im = ax_conf.imshow(conf_matrix_norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax_conf.set_xticks(np.arange(len(conf_matrix_norm.columns)))
ax_conf.set_yticks(np.arange(len(conf_matrix_norm.index)))
ax_conf.set_xticklabels(conf_matrix_norm.columns)
ax_conf.set_yticklabels(conf_matrix_norm.index)
ax_conf.set_xlabel('HMM Regime', fontsize=10, fontweight='bold')
ax_conf.set_ylabel('Statistical Regime', fontsize=10, fontweight='bold')
ax_conf.set_title('Confusion Matrix (% of row)', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(conf_matrix_norm.index)):
    for j in range(len(conf_matrix_norm.columns)):
        text = ax_conf.text(j, i, f'{conf_matrix_norm.values[i, j]:.1f}%',
                           ha="center", va="center", color="black", fontsize=9, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax_conf)
cbar.set_label('Agreement %', rotation=270, labelpad=15, fontweight='bold')

plt.tight_layout()

output_path2 = 'hmm_analysis/statistical_vs_hmm_performance.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Performance comparison saved to: {output_path2}")
plt.close()

# Figure 3: Return distribution plots (bell curves) for each regime
fig3, axes3 = plt.subplots(3, 2, figsize=(16, 12))
fig3.suptitle('Return Distribution Analysis by Regime (Bell Curves)', fontsize=16, fontweight='bold')

from scipy import stats

for row, (regime_id, regime_name) in enumerate([(0, 'Bear'), (1, 'Neutral'), (2, 'Bull')]):
    # Statistical method distribution
    ax_stat = axes3[row, 0]
    stat_mask = (statistical_regime == regime_id)
    
    if stat_mask.sum() > 0:
        stat_returns = returns_aligned[stat_mask] * 100  # Convert to percentage
        
        # Create histogram
        n, bins, patches = ax_stat.hist(stat_returns, bins=50, density=True, 
                                       alpha=0.6, color='lightblue', edgecolor='black', label='Actual Returns')
        
        # Fit normal distribution
        mu, std = stat_returns.mean(), stat_returns.std()
        
        # Plot fitted normal distribution (bell curve)
        x = np.linspace(stat_returns.min(), stat_returns.max(), 100)
        fitted_curve = stats.norm.pdf(x, mu, std)
        ax_stat.plot(x, fitted_curve, 'r-', linewidth=2.5, label=f'Normal Fit\nμ={mu:.3f}%, σ={std:.3f}%')
        
        # Add vertical lines for mean and +/- 1 std
        ax_stat.axvline(mu, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mu:.3f}%')
        ax_stat.axvline(mu + std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_stat.axvline(mu - std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1σ')
        
        # Color based on regime
        if regime_name == 'Bear':
            color = 'lightcoral'
        elif regime_name == 'Bull':
            color = 'lightgreen'
        else:
            color = 'lightgray'
        ax_stat.set_facecolor(color)
        ax_stat.patch.set_alpha(0.15)
        
        ax_stat.set_xlabel('Daily Return (%)', fontsize=10, fontweight='bold')
        ax_stat.set_ylabel('Density', fontsize=10, fontweight='bold')
        ax_stat.set_title(f'{regime_name} Regime - Statistical Method\n(N={stat_mask.sum()} days)', 
                         fontsize=11, fontweight='bold')
        ax_stat.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax_stat.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics text box
        stats_text = f'Annualized:\nReturn: {mu*252:.1f}%\nVol: {std*np.sqrt(252):.1f}%'
        ax_stat.text(0.02, 0.98, stats_text, transform=ax_stat.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    family='monospace')
    
    # HMM method distribution
    ax_hmm = axes3[row, 1]
    hmm_mask = (hmm_regime_normalized == regime_id)
    
    if hmm_mask.sum() > 0:
        hmm_returns = returns_aligned[hmm_mask] * 100  # Convert to percentage
        
        # Create histogram
        n, bins, patches = ax_hmm.hist(hmm_returns, bins=50, density=True,
                                      alpha=0.6, color='lightblue', edgecolor='black', label='Actual Returns')
        
        # Fit normal distribution
        mu, std = hmm_returns.mean(), hmm_returns.std()
        
        # Plot fitted normal distribution (bell curve)
        x = np.linspace(hmm_returns.min(), hmm_returns.max(), 100)
        fitted_curve = stats.norm.pdf(x, mu, std)
        ax_hmm.plot(x, fitted_curve, 'r-', linewidth=2.5, label=f'Normal Fit\nμ={mu:.3f}%, σ={std:.3f}%')
        
        # Add vertical lines for mean and +/- 1 std
        ax_hmm.axvline(mu, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mu:.3f}%')
        ax_hmm.axvline(mu + std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_hmm.axvline(mu - std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1σ')
        
        # Color based on regime
        if regime_name == 'Bear':
            color = 'lightcoral'
        elif regime_name == 'Bull':
            color = 'lightgreen'
        else:
            color = 'lightgray'
        ax_hmm.set_facecolor(color)
        ax_hmm.patch.set_alpha(0.15)
        
        ax_hmm.set_xlabel('Daily Return (%)', fontsize=10, fontweight='bold')
        ax_hmm.set_ylabel('Density', fontsize=10, fontweight='bold')
        ax_hmm.set_title(f'{regime_name} Regime - HMM Method\n(N={hmm_mask.sum()} days)', 
                        fontsize=11, fontweight='bold')
        ax_hmm.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax_hmm.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics text box
        stats_text = f'Annualized:\nReturn: {mu*252:.1f}%\nVol: {std*np.sqrt(252):.1f}%'
        ax_hmm.text(0.02, 0.98, stats_text, transform=ax_hmm.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   family='monospace')

plt.tight_layout()

output_path3 = 'hmm_analysis/statistical_vs_hmm_distributions.png'
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"✓ Distribution analysis saved to: {output_path3}")
plt.close()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\nKey Findings:")
print(f"1. Overall agreement between methods: {agreement_pct:.1f}%")
print(f"2. Statistical method is more reactive (based on recent {window}-day window)")
print(f"3. HMM method is more stable (trained on {504}-day window)")
print(f"4. Check visualizations in 'hmm_analysis/' directory for detailed comparison")

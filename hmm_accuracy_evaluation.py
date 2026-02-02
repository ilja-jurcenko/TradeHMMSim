"""
HMM Accuracy Evaluation - Supervised turning point detection evaluation.

This script evaluates HMM regime detection accuracy by:
1. Labeling true turning points using forward-looking reversal logic
2. Running HMM walkforward regime detection
3. Comparing HMM regime switches with labeled turning points
4. Calculating accuracy metrics
"""

import pandas as pd
import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from portfolio import Portfolio
from signal_filter import HMMRegimeFilter
from config_loader import ConfigLoader


def calculate_turning_point_accuracy(labeled_turns: pd.DataFrame, 
                                     regime_switches: pd.Series,
                                     tolerance_window: int = 5) -> dict:
    """
    Calculate accuracy metrics between labeled turning points and HMM regime switches.
    
    Parameters:
    -----------
    labeled_turns : pd.DataFrame
        DataFrame with 'is_top' and 'is_bottom' columns from label_turning_points()
    regime_switches : pd.Series
        Series of regime switches from HMM (contains regime values at switch points)
    tolerance_window : int
        Number of bars to consider a "match" (switch within N bars of true turning point)
        
    Returns:
    --------
    dict
        Dictionary with accuracy metrics:
        - true_positives: Number of true turning points detected by HMM
        - false_positives: Number of HMM switches that aren't true turning points
        - false_negatives: Number of true turning points missed by HMM
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1_score: 2 * (precision * recall) / (precision + recall)
        - total_labeled_turns: Total number of labeled turning points
        - total_hmm_switches: Total number of HMM regime switches
    """
    # Align indices
    common_idx = labeled_turns.index.intersection(regime_switches.index)
    
    # Get true turning points (either top or bottom)
    true_turns = labeled_turns.loc[common_idx]
    true_turn_dates = true_turns[(true_turns['is_top'] == 1) | (true_turns['is_bottom'] == 1)].index
    
    # Get HMM switch dates
    hmm_switch_dates = regime_switches.index
    
    # Calculate matches within tolerance window
    true_positives = 0
    matched_hmm_switches = set()
    matched_true_turns = set()
    
    # For each true turning point, check if there's an HMM switch nearby
    for true_turn_date in true_turn_dates:
        for hmm_switch_date in hmm_switch_dates:
            if hmm_switch_date in matched_hmm_switches:
                continue
            # Calculate days difference
            days_diff = abs((hmm_switch_date - true_turn_date).days)
            if days_diff <= tolerance_window:
                true_positives += 1
                matched_hmm_switches.add(hmm_switch_date)
                matched_true_turns.add(true_turn_date)
                break
    
    # Calculate metrics
    false_positives = len(hmm_switch_dates) - true_positives
    false_negatives = len(true_turn_dates) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_labeled_turns': len(true_turn_dates),
        'total_hmm_switches': len(hmm_switch_dates),
        'matched_true_turns': len(matched_true_turns),
        'matched_hmm_switches': len(matched_hmm_switches)
    }


def plot_labeled_turns(close: pd.Series,
                       labeled_turns: pd.DataFrame,
                       output_path: str):
    """
    Plot price with labeled turning points.
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    labeled_turns : pd.DataFrame
        Labeled turning points with 'is_top' and 'is_bottom'
    output_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price line
    ax.plot(close.index, close.values, 'k-', linewidth=1.5, alpha=0.7, label='Price')
    
    # Mark labeled tops (green outline, transparent fill)
    tops = labeled_turns[labeled_turns['is_top'] == 1]
    if len(tops) > 0:
        ax.scatter(tops.index, close.loc[tops.index], 
                   facecolors='none', marker='o', s=80, zorder=5, 
                   edgecolors='green', linewidths=1.2,
                   label=f'Labeled Tops ({len(tops)})')
    
    # Mark labeled bottoms (red outline, transparent fill)
    bottoms = labeled_turns[labeled_turns['is_bottom'] == 1]
    if len(bottoms) > 0:
        ax.scatter(bottoms.index, close.loc[bottoms.index], 
                   facecolors='none', marker='o', s=80, zorder=5,
                   edgecolors='red', linewidths=1.2,
                   label=f'Labeled Bottoms ({len(bottoms)})')
    
    # Calculate total turning points
    total_turns = len(tops) + len(bottoms)
    
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f'Labeled Turning Points (Forward-Looking Reversal) - Total: {total_turns}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved to: {output_path}")


def plot_labeled_turns_with_regimes(close: pd.Series,
                                     labeled_turns: pd.DataFrame,
                                     regime_labels: pd.Series,
                                     output_path: str):
    """
    Plot price with labeled turning points and regime coloring.
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    labeled_turns : pd.DataFrame
        Labeled turning points with 'is_top' and 'is_bottom'
    regime_labels : pd.Series
        Regime labels ('bullish', 'bearish', 'neutral')
    output_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price line with regime coloring
    regime_colors = {
        'bullish': 'green',
        'bearish': 'red',
        'neutral': 'gray'
    }
    
    # Plot base price line first (thin, light) to ensure continuity
    ax.plot(close.index, close.values, color='lightgray', linewidth=0.5, alpha=0.5, zorder=0)
    
    # Find regime changes
    regime_changes = [0]
    for i in range(1, len(regime_labels)):
        if regime_labels.iloc[i] != regime_labels.iloc[i-1]:
            regime_changes.append(i)
    regime_changes.append(len(regime_labels))
    
    # Plot each regime segment as lines only, ensuring no gaps
    for j in range(len(regime_changes) - 1):
        start_i = regime_changes[j]
        end_i = regime_changes[j+1]
        curr_regime = regime_labels.iloc[start_i]
        color = regime_colors.get(curr_regime, 'gray')
        # Use end_i+1 to ensure segments overlap at boundaries (no visual gaps)
        slice_end = min(end_i + 1, len(close))
        ax.plot(close.index[start_i:slice_end], close.values[start_i:slice_end], 
                color=color, linewidth=1.5, alpha=0.7, zorder=1)
    
    # Mark labeled tops (green dots, filled)
    tops = labeled_turns[labeled_turns['is_top'] == 1]
    if len(tops) > 0:
        ax.scatter(tops.index, close.loc[tops.index], 
                   color='green', marker='o', s=20, zorder=5, 
                   edgecolors='darkgreen', linewidths=0.5,
                   label=f'Labeled Tops ({len(tops)})')
    
    # Mark labeled bottoms (red dots, filled)
    bottoms = labeled_turns[labeled_turns['is_bottom'] == 1]
    if len(bottoms) > 0:
        ax.scatter(bottoms.index, close.loc[bottoms.index], 
                   color='red', marker='o', s=20, zorder=5,
                   edgecolors='darkred', linewidths=0.5,
                   label=f'Labeled Bottoms ({len(bottoms)})')
    
    # Calculate total turning points and regime stats
    total_turns = len(tops) + len(bottoms)
    num_bullish = (regime_labels == 'bullish').sum()
    num_bearish = (regime_labels == 'bearish').sum()
    pct_bullish = num_bullish / len(regime_labels) * 100
    pct_bearish = num_bearish / len(regime_labels) * 100
    
    # Calculate average regime duration
    regime_changes_mask = regime_labels != regime_labels.shift(1)
    regime_periods = regime_labels.groupby((regime_changes_mask).cumsum()).size()
    regime_types = regime_labels.groupby((regime_changes_mask).cumsum()).first()
    
    bullish_periods = regime_periods[regime_types == 'bullish']
    bearish_periods = regime_periods[regime_types == 'bearish']
    
    avg_bullish_days = bullish_periods.mean() if len(bullish_periods) > 0 else 0
    avg_bearish_days = bearish_periods.mean() if len(bearish_periods) > 0 else 0
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, 
              label=f'Bullish Regime ({pct_bullish:.1f}%, avg {avg_bullish_days:.1f} days)'),
        Patch(facecolor='red', alpha=0.7, 
              label=f'Bearish Regime ({pct_bearish:.1f}%, avg {avg_bearish_days:.1f} days)'),
    ]
    
    if len(tops) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='green', markeredgecolor='darkgreen',
                                         markeredgewidth=0.5, markersize=5, 
                                         label=f'Tops ({len(tops)})', linewidth=0))
    if len(bottoms) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='red', markeredgecolor='darkred',
                                         markeredgewidth=0.5, markersize=5, 
                                         label=f'Bottoms ({len(bottoms)})', linewidth=0))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f'Labeled Turning Points with Regime Classification - Total: {total_turns}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved to: {output_path}")


def plot_comparison_with_hmm(close: pd.Series,
                             labeled_turns: pd.DataFrame,
                             regime_labels: pd.Series,
                             hmm_regime: pd.Series,
                             hmm_states_raw: pd.Series,
                             switch_prob: pd.Series,
                             entropy: pd.Series,
                             switch_prob_delta: pd.Series,
                             alert_score: pd.Series,
                             output_path: str):
    """
    Plot comparison of labeled turning points vs HMM detected regimes.
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    labeled_turns : pd.DataFrame
        Labeled turning points with 'is_top' and 'is_bottom'
    regime_labels : pd.Series
        Labeled regime labels ('bullish', 'bearish', 'neutral')
    hmm_regime : pd.Series
        HMM detected regime labels
    hmm_states_raw : pd.Series
        Raw HMM state predictions (numeric) from walkforward
    switch_prob : pd.Series
        Probability of switching states at next step
    entropy : pd.Series
        Normalized state entropy (uncertainty)
    switch_prob_delta : pd.Series
        Normalized positive change in switch probability
    alert_score : pd.Series
        Turn alert score combining switch_prob, entropy, and delta
    output_path : str
        Path to save the plot
    """
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(16, 34), sharex=True)
    
    regime_colors = {
        'bullish': 'green',
        'bearish': 'red',
        'neutral': 'gray'
    }
    
    # ===== TOP PANEL: Labeled Turning Points and Regimes =====
    ax1.set_title('Labeled Turning Points & Regimes (Zigzag)', fontsize=14, fontweight='bold')
    
    # Plot base price line
    ax1.plot(close.index, close.values, color='lightgray', linewidth=0.5, alpha=0.5, zorder=0)
    
    # Find regime changes for labeled data
    regime_changes = [0]
    for i in range(1, len(regime_labels)):
        if regime_labels.iloc[i] != regime_labels.iloc[i-1]:
            regime_changes.append(i)
    regime_changes.append(len(regime_labels))
    
    # Plot labeled regime segments
    for j in range(len(regime_changes) - 1):
        start_i = regime_changes[j]
        end_i = regime_changes[j+1]
        curr_regime = regime_labels.iloc[start_i]
        color = regime_colors.get(curr_regime, 'gray')
        slice_end = min(end_i + 1, len(close))
        ax1.plot(close.index[start_i:slice_end], close.values[start_i:slice_end], 
                color=color, linewidth=1.5, alpha=0.7, zorder=1)
    
    # Mark labeled tops
    tops = labeled_turns[labeled_turns['is_top'] == 1]
    if len(tops) > 0:
        ax1.scatter(tops.index, close.loc[tops.index], 
                   color='green', marker='o', s=20, zorder=5, 
                   edgecolors='darkgreen', linewidths=0.5,
                   label=f'Labeled Tops ({len(tops)})')
    
    # Mark labeled bottoms
    bottoms = labeled_turns[labeled_turns['is_bottom'] == 1]
    if len(bottoms) > 0:
        ax1.scatter(bottoms.index, close.loc[bottoms.index], 
                   color='red', marker='o', s=20, zorder=5,
                   edgecolors='darkred', linewidths=0.5,
                   label=f'Labeled Bottoms ({len(bottoms)})')
    
    # Calculate regime stats
    num_bullish = (regime_labels == 'bullish').sum()
    num_bearish = (regime_labels == 'bearish').sum()
    pct_bullish = num_bullish / len(regime_labels) * 100
    pct_bearish = num_bearish / len(regime_labels) * 100
    
    # Calculate average regime duration for labeled
    regime_changes_mask = regime_labels != regime_labels.shift(1)
    regime_periods = regime_labels.groupby((regime_changes_mask).cumsum()).size()
    regime_types = regime_labels.groupby((regime_changes_mask).cumsum()).first()
    
    bullish_periods = regime_periods[regime_types == 'bullish']
    bearish_periods = regime_periods[regime_types == 'bearish']
    
    avg_bullish_days = bullish_periods.mean() if len(bullish_periods) > 0 else 0
    avg_bearish_days = bearish_periods.mean() if len(bearish_periods) > 0 else 0
    
    # Create legend for top panel
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, 
              label=f'Bullish ({pct_bullish:.1f}%, avg {avg_bullish_days:.1f}d)'),
        Patch(facecolor='red', alpha=0.7, 
              label=f'Bearish ({pct_bearish:.1f}%, avg {avg_bearish_days:.1f}d)'),
    ]
    
    if len(tops) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='green', markeredgecolor='darkgreen',
                                         markeredgewidth=0.5, markersize=5, 
                                         label=f'Tops ({len(tops)})', linewidth=0))
    if len(bottoms) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='red', markeredgecolor='darkred',
                                         markeredgewidth=0.5, markersize=5, 
                                         label=f'Bottoms ({len(bottoms)})', linewidth=0))
    
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # ===== BOTTOM PANEL: HMM Detected Regimes =====
    ax2.set_title('HMM Walkforward Detected Regimes', fontsize=14, fontweight='bold')
    
    # Plot base price line
    ax2.plot(close.index, close.values, color='lightgray', linewidth=0.5, alpha=0.5, zorder=0)
    
    # Find regime changes for HMM data
    hmm_changes = [0]
    for i in range(1, len(hmm_regime)):
        if hmm_regime.iloc[i] != hmm_regime.iloc[i-1]:
            hmm_changes.append(i)
    hmm_changes.append(len(hmm_regime))
    
    # Plot HMM regime segments
    for j in range(len(hmm_changes) - 1):
        start_i = hmm_changes[j]
        end_i = hmm_changes[j+1]
        curr_regime = hmm_regime.iloc[start_i]
        color = regime_colors.get(curr_regime, 'gray')
        slice_end = min(end_i + 1, len(close))
        ax2.plot(close.index[start_i:slice_end], close.values[start_i:slice_end], 
                color=color, linewidth=1.5, alpha=0.7, zorder=1)
    
    # Mark HMM regime switches
    hmm_switches = hmm_regime[hmm_regime.ne(hmm_regime.shift(1))].dropna()
    if len(hmm_switches) > 0:
        ax2.scatter(hmm_switches.index, close.loc[hmm_switches.index], 
                   color='blue', marker='v', s=30, zorder=5, 
                   edgecolors='darkblue', linewidths=0.5,
                   label=f'HMM Switches ({len(hmm_switches)})')
    
    # Calculate HMM regime stats
    hmm_bullish = (hmm_regime == 'bullish').sum()
    hmm_bearish = (hmm_regime == 'bearish').sum()
    hmm_neutral = (hmm_regime == 'neutral').sum()
    hmm_pct_bullish = hmm_bullish / len(hmm_regime) * 100
    hmm_pct_bearish = hmm_bearish / len(hmm_regime) * 100
    
    # Calculate average regime duration for HMM
    hmm_changes_mask = hmm_regime != hmm_regime.shift(1)
    hmm_periods = hmm_regime.groupby((hmm_changes_mask).cumsum()).size()
    hmm_types = hmm_regime.groupby((hmm_changes_mask).cumsum()).first()
    
    hmm_bullish_periods = hmm_periods[hmm_types == 'bullish']
    hmm_bearish_periods = hmm_periods[hmm_types == 'bearish']
    hmm_neutral_periods = hmm_periods[hmm_types == 'neutral']
    
    hmm_avg_bullish_days = hmm_bullish_periods.mean() if len(hmm_bullish_periods) > 0 else 0
    hmm_avg_bearish_days = hmm_bearish_periods.mean() if len(hmm_bearish_periods) > 0 else 0
    hmm_avg_neutral_days = hmm_neutral_periods.mean() if len(hmm_neutral_periods) > 0 else 0
    
    # Create legend for bottom panel
    legend_elements2 = [
        Patch(facecolor='green', alpha=0.7, 
              label=f'Bullish ({hmm_pct_bullish:.1f}%, avg {hmm_avg_bullish_days:.1f}d)'),
        Patch(facecolor='red', alpha=0.7, 
              label=f'Bearish ({hmm_pct_bearish:.1f}%, avg {hmm_avg_bearish_days:.1f}d)'),
    ]
    
    if hmm_neutral > 0:
        legend_elements2.append(Patch(facecolor='gray', alpha=0.7, 
              label=f'Neutral ({hmm_neutral/len(hmm_regime)*100:.1f}%, avg {hmm_avg_neutral_days:.1f}d)'))
    
    if len(hmm_switches) > 0:
        legend_elements2.append(plt.Line2D([0], [0], marker='v', color='w', 
                                         markerfacecolor='blue', markeredgecolor='darkblue',
                                         markeredgewidth=0.5, markersize=5, 
                                         label=f'Switches ({len(hmm_switches)})', linewidth=0))
    
    ax2.legend(handles=legend_elements2, loc='upper left', fontsize=10)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # ===== THIRD PANEL: Price Colored by HMM Predicted States =====
    ax3.set_title('Price Colored by HMM Predicted States (Walkforward)', fontsize=14, fontweight='bold')
    
    # Use raw states from walkforward filter
    # Align with close prices
    common_idx = close.index.intersection(hmm_states_raw.index)
    close_aligned = close.loc[common_idx]
    states_aligned = hmm_states_raw.loc[common_idx]
    
    # State colors (numeric states)
    state_color_map = {}
    unique_states = sorted(states_aligned.unique())
    if len(unique_states) == 2:
        state_color_map = {unique_states[0]: 'red', unique_states[1]: 'green'}  # bear, bull
    elif len(unique_states) == 3:
        state_color_map = {unique_states[0]: 'red', unique_states[1]: 'gray', unique_states[2]: 'green'}  # bear, neutral, bull
    
    # Plot base price line
    ax3.plot(close_aligned.index, close_aligned.values, color='lightgray', linewidth=0.5, alpha=0.3, zorder=0)
    
    # Plot each state as scatter dots
    for state in unique_states:
        mask = states_aligned == state
        color = state_color_map.get(state, 'gray')
        ax3.scatter(close_aligned.index[mask], close_aligned.values[mask],
                   c=color, s=3, alpha=0.7, zorder=1, label=f'State {state}')
    
    ax3.legend(loc='upper left', fontsize=10)
    ax3.set_ylabel('Price', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # ===== FOURTH PANEL: Switch Probability =====
    ax4.set_title('HMM Regime Switch Probability P(switch|data)', fontsize=14, fontweight='bold')
    
    # Align switch probability with close prices
    common_idx_switch = close.index.intersection(switch_prob.index)
    switch_prob_aligned = switch_prob.loc[common_idx_switch]
    
    # Plot switch probability as area chart
    ax4.fill_between(switch_prob_aligned.index, 0, switch_prob_aligned.values,
                     color='orange', alpha=0.3, label='Switch Probability')
    ax4.plot(switch_prob_aligned.index, switch_prob_aligned.values,
            color='darkorange', linewidth=1.5, alpha=0.8)
    
    # Mark labeled turning points
    tops_idx = labeled_turns[labeled_turns['is_top'] == 1].index.intersection(switch_prob_aligned.index)
    bottoms_idx = labeled_turns[labeled_turns['is_bottom'] == 1].index.intersection(switch_prob_aligned.index)
    
    if len(tops_idx) > 0:
        ax4.scatter(tops_idx, switch_prob_aligned.loc[tops_idx],
                   color='green', marker='v', s=40, zorder=5,
                   edgecolors='darkgreen', linewidths=0.8,
                   label=f'Labeled Tops ({len(tops_idx)})')
    
    if len(bottoms_idx) > 0:
        ax4.scatter(bottoms_idx, switch_prob_aligned.loc[bottoms_idx],
                   color='red', marker='^', s=40, zorder=5,
                   edgecolors='darkred', linewidths=0.8,
                   label=f'Labeled Bottoms ({len(bottoms_idx)})')
    
    # Add horizontal line at 0.5
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    
    # Calculate statistics
    avg_switch_prob = switch_prob_aligned.mean()
    high_switch_events = (switch_prob_aligned > 0.5).sum()
    pct_high_switch = high_switch_events / len(switch_prob_aligned) * 100
    
    # Add text box with statistics
    stats_text = f'Avg: {avg_switch_prob:.3f}\n>50%: {pct_high_switch:.1f}% ({high_switch_events} days)'
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    ax4.legend(loc='upper right', fontsize=10)
    ax4.set_ylabel('Switch Probability', fontsize=12)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)
    
    # ===== FIFTH PANEL: State Entropy (Uncertainty) =====
    ax5.set_title('State Entropy (Uncertainty) H_norm(t) = -Σ p_t(i)log(p_t(i)) / log(K)', fontsize=14, fontweight='bold')
    
    # Align entropy with close prices
    common_idx_entropy = close.index.intersection(entropy.index)
    entropy_aligned = entropy.loc[common_idx_entropy]
    
    # Plot entropy as area chart
    ax5.fill_between(entropy_aligned.index, 0, entropy_aligned.values,
                     color='teal', alpha=0.3, label='State Entropy')
    ax5.plot(entropy_aligned.index, entropy_aligned.values,
            color='darkcyan', linewidth=1.5, alpha=0.8)
    
    # Mark labeled turning points
    tops_idx_entropy = labeled_turns[labeled_turns['is_top'] == 1].index.intersection(entropy_aligned.index)
    bottoms_idx_entropy = labeled_turns[labeled_turns['is_bottom'] == 1].index.intersection(entropy_aligned.index)
    
    if len(tops_idx_entropy) > 0:
        ax5.scatter(tops_idx_entropy, entropy_aligned.loc[tops_idx_entropy],
                   color='green', marker='v', s=40, zorder=5,
                   edgecolors='darkgreen', linewidths=0.8,
                   label=f'Labeled Tops ({len(tops_idx_entropy)})')
    
    if len(bottoms_idx_entropy) > 0:
        ax5.scatter(bottoms_idx_entropy, entropy_aligned.loc[bottoms_idx_entropy],
                   color='red', marker='^', s=40, zorder=5,
                   edgecolors='darkred', linewidths=0.8,
                   label=f'Labeled Bottoms ({len(bottoms_idx_entropy)})')
    
    # Add horizontal line at 0.5 (mid-range uncertainty)
    ax5.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Mid uncertainty (0.5)')
    
    # Calculate statistics
    avg_entropy = entropy_aligned.mean()
    high_entropy_events = (entropy_aligned > 0.5).sum()
    pct_high_entropy = high_entropy_events / len(entropy_aligned) * 100
    
    # Add text box with statistics
    stats_text_entropy = f'Avg: {avg_entropy:.3f}\n>0.5: {pct_high_entropy:.1f}% ({high_entropy_events} days)'
    ax5.text(0.02, 0.98, stats_text_entropy, transform=ax5.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
            fontsize=10)
    
    ax5.legend(loc='upper right', fontsize=10)
    ax5.set_ylabel('Entropy (0=certain, 1=max uncertainty)', fontsize=12)
    ax5.set_ylim(-0.05, 1.05)
    ax5.grid(True, alpha=0.3)
    
    # ===== SIXTH PANEL: Switch Probability Delta =====
    ax6.set_title('Switch Probability Delta (Rising probability of regime change)', fontsize=14, fontweight='bold')
    
    # Align switch prob delta with close prices
    common_idx_delta = close.index.intersection(switch_prob_delta.index)
    delta_aligned = switch_prob_delta.loc[common_idx_delta]
    
    # Plot delta as impulse/stem plot
    ax6.fill_between(delta_aligned.index, 0, delta_aligned.values,
                     color='orange', alpha=0.4, label='P_switch increase')
    ax6.plot(delta_aligned.index, delta_aligned.values,
            color='darkorange', linewidth=1.2, alpha=0.8)
    
    # Mark labeled turning points
    tops_idx_delta = labeled_turns[labeled_turns['is_top'] == 1].index.intersection(delta_aligned.index)
    bottoms_idx_delta = labeled_turns[labeled_turns['is_bottom'] == 1].index.intersection(delta_aligned.index)
    
    if len(tops_idx_delta) > 0:
        ax6.scatter(tops_idx_delta, delta_aligned.loc[tops_idx_delta],
                   color='green', marker='v', s=50, zorder=5,
                   edgecolors='darkgreen', linewidths=1.0,
                   label=f'Labeled Tops ({len(tops_idx_delta)})')
    
    if len(bottoms_idx_delta) > 0:
        ax6.scatter(bottoms_idx_delta, delta_aligned.loc[bottoms_idx_delta],
                   color='red', marker='^', s=50, zorder=5,
                   edgecolors='darkred', linewidths=1.0,
                   label=f'Labeled Bottoms ({len(bottoms_idx_delta)})')
    
    # Calculate statistics
    avg_delta = delta_aligned.mean()
    max_delta = delta_aligned.max()
    nonzero_days = (delta_aligned > 0).sum()
    
    # Add text box with statistics
    stats_text_delta = f'Mean: {avg_delta:.4f}\nMax: {max_delta:.4f}\nNon-zero: {nonzero_days} ({nonzero_days/len(delta_aligned)*100:.1f}%)'
    ax6.text(0.02, 0.98, stats_text_delta, transform=ax6.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            fontsize=10)
    
    ax6.legend(loc='upper right', fontsize=10)
    ax6.set_ylabel('ΔP_switch (normalized)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # ===== SEVENTH PANEL: Turn Alert Score =====
    ax7.set_title('Turn Alert Score = 0.5×P_switch(t) + 0.2×H_norm(t) + 0.3×ΔP_switch(t)', fontsize=14, fontweight='bold')
    
    # Align alert score with close prices
    common_idx_alert = close.index.intersection(alert_score.index)
    alert_aligned = alert_score.loc[common_idx_alert]
    
    # Plot alert score as area chart with gradient effect
    ax7.fill_between(alert_aligned.index, 0, alert_aligned.values,
                     color='crimson', alpha=0.3, label='Turn Alert Score')
    ax7.plot(alert_aligned.index, alert_aligned.values,
            color='darkred', linewidth=1.5, alpha=0.8)
    
    # Mark labeled turning points
    tops_idx_alert = labeled_turns[labeled_turns['is_top'] == 1].index.intersection(alert_aligned.index)
    bottoms_idx_alert = labeled_turns[labeled_turns['is_bottom'] == 1].index.intersection(alert_aligned.index)
    
    if len(tops_idx_alert) > 0:
        ax7.scatter(tops_idx_alert, alert_aligned.loc[tops_idx_alert],
                   color='green', marker='v', s=50, zorder=5,
                   edgecolors='darkgreen', linewidths=1.0,
                   label=f'Labeled Tops ({len(tops_idx_alert)})')
    
    if len(bottoms_idx_alert) > 0:
        ax7.scatter(bottoms_idx_alert, alert_aligned.loc[bottoms_idx_alert],
                   color='red', marker='^', s=50, zorder=5,
                   edgecolors='darkred', linewidths=1.0,
                   label=f'Labeled Bottoms ({len(bottoms_idx_alert)})')
    
    # Find and mark top alert scores
    if len(alert_aligned) > 0:
        alert_threshold = alert_aligned.quantile(0.95)  # Top 5%
        high_alert_dates = alert_aligned[alert_aligned >= alert_threshold].index
        ax7.scatter(high_alert_dates, alert_aligned.loc[high_alert_dates],
                   color='yellow', marker='*', s=100, zorder=4,
                   edgecolors='orange', linewidths=0.5, alpha=0.7,
                   label=f'High Alert (top 5%)')
    
    # Calculate statistics
    avg_alert = alert_aligned.mean()
    max_alert = alert_aligned.max()
    alert_95 = alert_aligned.quantile(0.95)
    high_alert_events = (alert_aligned > alert_95).sum()
    
    # Add text box with statistics
    stats_text_alert = f'Mean: {avg_alert:.4f}\nMax: {max_alert:.4f}\n95th: {alert_95:.4f}\nHigh alerts: {high_alert_events}'
    ax7.text(0.02, 0.98, stats_text_alert, transform=ax7.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            fontsize=10)
    
    ax7.legend(loc='upper right', fontsize=10)
    ax7.set_ylabel('Alert Score', fontsize=12)
    ax7.set_xlabel('Date', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison plot saved to: {output_path}")


def evaluate_hmm_accuracy(ticker: str,
                         start_date: str,
                         end_date: str,
                         config_path: str = None,
                         threshold_pct: float = 0.05,
                         tolerance_window: int = 5):
    """
    Evaluate HMM accuracy against labeled turning points using zigzag algorithm.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol to test
    start_date : str
        Start date for evaluation
    end_date : str
        End date for evaluation
    config_path : str, optional
        Path to configuration file
    threshold_pct : float
        Minimum percentage move to confirm a pivot (e.g., 0.05 = 5%)
    tolerance_window : int
        Number of bars to consider HMM switch matching labeled turn
        
    Returns:
    --------
    dict
        Evaluation results with accuracy metrics
    """
    print("\n" + "="*80)
    print("ZIGZAG PIVOT LABELING EVALUATION")
    print("="*80)
    print(f"Ticker: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Labeling parameters: threshold_pct={threshold_pct*100:.1f}%")
    print(f"Matching tolerance: {tolerance_window} bars")
    
    # Load configuration
    if config_path:
        print(f"\nLoading configuration from: {config_path}")
        config = ConfigLoader.load_config(config_path)
    else:
        # Use defaults
        config = {
            'hmm': {
                'n_states': 3,
                'train_window': 504,
                'refit_every': 21,
                'short_vol_window': 10,
                'long_vol_window': 30,
                'short_ma_window': 10,
                'long_ma_window': 30,
                'covariance_type': 'diag',
                'n_iter': 100,
                'tol': 1e-3
            }
        }
    
    # Extract HMM parameters
    hmm_config = config.get('hmm', {})
    n_states = hmm_config.get('n_states', 3)
    train_window = hmm_config.get('train_window', 504)
    refit_every = hmm_config.get('refit_every', 21)
    short_vol_window = hmm_config.get('short_vol_window', 10)
    long_vol_window = hmm_config.get('long_vol_window', 30)
    short_ma_window = hmm_config.get('short_ma_window', 10)
    long_ma_window = hmm_config.get('long_ma_window', 30)
    covariance_type = hmm_config.get('covariance_type', 'diag')
    n_iter = hmm_config.get('n_iter', 100)
    tol = hmm_config.get('tol', 1e-3)
    
    # Load data
    print("\nLoading data...")
    print(f"Note: Loading extra data before start_date for HMM training window")
    
    # Calculate how much prior data we need
    # Need train_window + some buffer for feature calculation
    buffer_days = train_window + 100  # Extra buffer for MA and volatility calculations
    
    # Load data with buffer
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    extended_start_dt = start_dt - timedelta(days=int(buffer_days * 1.5))  # 1.5x for weekends/holidays
    extended_start = extended_start_dt.strftime('%Y-%m-%d')
    
    print(f"  Evaluation period: {start_date} to {end_date}")
    print(f"  Loading data from: {extended_start} (for training)")
    
    portfolio = Portfolio([ticker], extended_start, end_date)
    portfolio.load_data()
    close_full = portfolio.get_close_prices(ticker)
    print(f"  Loaded {len(close_full)} total data points")
    
    # Get the actual evaluation period data
    close_eval = close_full.loc[start_date:end_date]
    print(f"  Evaluation period: {len(close_eval)} data points ({close_eval.index[0]} to {close_eval.index[-1]})")
    
    # Initialize HMM filter
    print("\nInitializing HMM regime filter...")
    print(f"  Number of states: {n_states}")
    print(f"  Train window: {train_window}, Refit every: {refit_every}")
    hmm_filter = HMMRegimeFilter(
        n_states=n_states,
        random_state=42,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        short_vol_window=short_vol_window,
        long_vol_window=long_vol_window,
        short_ma_window=short_ma_window,
        long_ma_window=long_ma_window
    )
    
    # Step 1: Label pivots and regimes using zigzag algorithm
    print("\n" + "="*80)
    print("STEP 1: LABELING PIVOTS AND REGIMES (ZIGZAG)")
    print("="*80)
    print(f"Labeling on full dataset")
    labeled_turns_full = hmm_filter.label_zigzag_pivots(
        close_full,
        threshold_pct=threshold_pct
    )
    
    # Filter to evaluation period only
    labeled_turns = labeled_turns_full.loc[start_date:end_date]
    
    num_tops = (labeled_turns['is_top'] == 1).sum()
    num_bottoms = (labeled_turns['is_bottom'] == 1).sum()
    total_turns = num_tops + num_bottoms
    
    print(f"\nLabeled turning points in evaluation period ({start_date} to {end_date}):")
    print(f"  Tops: {num_tops}")
    print(f"  Bottoms: {num_bottoms}")
    print(f"  Total: {total_turns}")
    
    if total_turns == 0:
        print("\nWARNING: No turning points labeled. Try adjusting parameters.")
        return None
    
    # Extract regime labels (already computed by zigzag algorithm)
    print("\n" + "="*80)
    print("STEP 2: REGIME DISTRIBUTION (LABELED)")
    print("="*80)
    
    # Regimes are already in labeled_turns DataFrame from zigzag algorithm
    regime_labels = labeled_turns['regime_label']
    
    # Count regime periods
    num_bullish = (regime_labels == 'bullish').sum()
    num_bearish = (regime_labels == 'bearish').sum()
    num_neutral = (regime_labels == 'neutral').sum()
    
    # Calculate average regime duration
    regime_changes_mask = regime_labels != regime_labels.shift(1)
    regime_periods = regime_labels.groupby((regime_changes_mask).cumsum()).size()
    regime_types = regime_labels.groupby((regime_changes_mask).cumsum()).first()
    
    bullish_periods = regime_periods[regime_types == 'bullish']
    bearish_periods = regime_periods[regime_types == 'bearish']
    neutral_periods = regime_periods[regime_types == 'neutral']
    
    avg_bullish_days = bullish_periods.mean() if len(bullish_periods) > 0 else 0
    avg_bearish_days = bearish_periods.mean() if len(bearish_periods) > 0 else 0
    avg_neutral_days = neutral_periods.mean() if len(neutral_periods) > 0 else 0
    
    print(f"\nLabeled regime distribution:")
    print(f"  Bullish: {num_bullish} days ({num_bullish/len(regime_labels)*100:.1f}%) - Avg duration: {avg_bullish_days:.1f} days ({len(bullish_periods)} periods)")
    print(f"  Bearish: {num_bearish} days ({num_bearish/len(regime_labels)*100:.1f}%) - Avg duration: {avg_bearish_days:.1f} days ({len(bearish_periods)} periods)")
    if num_neutral > 0:
        print(f"  Neutral: {num_neutral} days ({num_neutral/len(regime_labels)*100:.1f}%) - Avg duration: {avg_neutral_days:.1f} days ({len(neutral_periods)} periods)")
    
    # Step 3: Run HMM walkforward test
    print("\n" + "="*80)
    print("STEP 3: HMM WALKFORWARD REGIME DETECTION")
    print("="*80)
    
    probs, hmm_regime, switches, switch_prob, entropy, switch_prob_delta, alert_score = hmm_filter.walkforward_filter(
        close_full,
        train_window=train_window,
        refit_every=refit_every,
        short_vol_window=short_vol_window,
        long_vol_window=long_vol_window,
        short_ma_window=short_ma_window,
        long_ma_window=long_ma_window,
        return_switch_prob=True,
        return_entropy=True,
        return_switch_prob_delta=True,
        return_alert_score=True
    )
    
    # Identify HMM regimes (bear/bull/neutral)
    regime_info = hmm_filter.identify_regimes(close_full, hmm_regime)
    bear_regime = regime_info['bear_regime']
    bull_regime = regime_info['bull_regime']
    neutral_regime = regime_info.get('neutral_regime', None)
    
    print(f"\nHMM regime mapping:")
    print(f"  State {bear_regime} = Bearish (mean return: {regime_info['regime_returns'][bear_regime]:.4f})")
    print(f"  State {bull_regime} = Bullish (mean return: {regime_info['regime_returns'][bull_regime]:.4f})")
    if neutral_regime is not None:
        print(f"  State {neutral_regime} = Neutral (mean return: {regime_info['regime_returns'][neutral_regime]:.4f})")
    
    # Map HMM states to regime labels
    hmm_regime_labels = hmm_regime.map({
        bear_regime: 'bearish',
        bull_regime: 'bullish',
        neutral_regime: 'neutral' if neutral_regime is not None else None
    })
    
    # Filter to evaluation period
    hmm_regime_eval = hmm_regime_labels.loc[start_date:end_date]
    switch_prob_eval = switch_prob.loc[start_date:end_date]
    entropy_eval = entropy.loc[start_date:end_date]
    switch_prob_delta_eval = switch_prob_delta.loc[start_date:end_date]
    alert_score_eval = alert_score.loc[start_date:end_date]
    
    print(f"\nHMM detected {len(switches)} regime switches")
    print(f"HMM regime distribution (evaluation period):")
    hmm_bullish = (hmm_regime_eval == 'bullish').sum()
    hmm_bearish = (hmm_regime_eval == 'bearish').sum()
    hmm_neutral = (hmm_regime_eval == 'neutral').sum()
    
    # Calculate average regime duration for HMM
    hmm_changes_mask = hmm_regime_eval != hmm_regime_eval.shift(1)
    hmm_periods = hmm_regime_eval.groupby((hmm_changes_mask).cumsum()).size()
    hmm_types = hmm_regime_eval.groupby((hmm_changes_mask).cumsum()).first()
    
    hmm_bullish_periods = hmm_periods[hmm_types == 'bullish']
    hmm_bearish_periods = hmm_periods[hmm_types == 'bearish']
    hmm_neutral_periods = hmm_periods[hmm_types == 'neutral']
    
    hmm_avg_bullish_days = hmm_bullish_periods.mean() if len(hmm_bullish_periods) > 0 else 0
    hmm_avg_bearish_days = hmm_bearish_periods.mean() if len(hmm_bearish_periods) > 0 else 0
    hmm_avg_neutral_days = hmm_neutral_periods.mean() if len(hmm_neutral_periods) > 0 else 0
    
    print(f"  Bullish: {hmm_bullish} days ({hmm_bullish/len(hmm_regime_eval)*100:.1f}%) - Avg duration: {hmm_avg_bullish_days:.1f} days ({len(hmm_bullish_periods)} periods)")
    print(f"  Bearish: {hmm_bearish} days ({hmm_bearish/len(hmm_regime_eval)*100:.1f}%) - Avg duration: {hmm_avg_bearish_days:.1f} days ({len(hmm_bearish_periods)} periods)")
    if hmm_neutral > 0:
        print(f"  Neutral: {hmm_neutral} days ({hmm_neutral/len(hmm_regime_eval)*100:.1f}%) - Avg duration: {hmm_avg_neutral_days:.1f} days ({len(hmm_neutral_periods)} periods)")
    
    # Print switch probability statistics
    print("\nSwitch Probability Statistics:")
    print(f"  Mean: {switch_prob_eval.mean():.3f}")
    print(f"  Median: {switch_prob_eval.median():.3f}")
    print(f"  Std: {switch_prob_eval.std():.3f}")
    print(f"  Min: {switch_prob_eval.min():.3f}")
    print(f"  Max: {switch_prob_eval.max():.3f}")
    high_switch_days = (switch_prob_eval > 0.5).sum()
    print(f"  Days with >50% switch probability: {high_switch_days} ({high_switch_days/len(switch_prob_eval)*100:.1f}%)")
    
    # Correlation between switch probability and labeled turning points
    labeled_turns_binary = (labeled_turns['is_top'] | labeled_turns['is_bottom']).astype(float)
    common_idx_corr = labeled_turns_binary.index.intersection(switch_prob_eval.index)
    if len(common_idx_corr) > 0:
        corr = np.corrcoef(labeled_turns_binary.loc[common_idx_corr], 
                          switch_prob_eval.loc[common_idx_corr])[0, 1]
        print(f"  Correlation with labeled turning points: {corr:.3f}")
    
    # Print entropy statistics
    print("\nState Entropy Statistics:")
    print(f"  Mean: {entropy_eval.mean():.3f}")
    print(f"  Median: {entropy_eval.median():.3f}")
    print(f"  Std: {entropy_eval.std():.3f}")
    print(f"  Min: {entropy_eval.min():.3f}")
    print(f"  Max: {entropy_eval.max():.3f}")
    high_entropy_days = (entropy_eval > 0.5).sum()
    print(f"  Days with >50% entropy (high uncertainty): {high_entropy_days} ({high_entropy_days/len(entropy_eval)*100:.1f}%)")
    
    # Correlation between entropy and labeled turning points
    common_idx_entropy = labeled_turns_binary.index.intersection(entropy_eval.index)
    if len(common_idx_entropy) > 0:
        corr_entropy = np.corrcoef(labeled_turns_binary.loc[common_idx_entropy], 
                                  entropy_eval.loc[common_idx_entropy])[0, 1]
        print(f"  Correlation with labeled turning points: {corr_entropy:.3f}")
    
    # Print switch probability delta statistics
    print("\nSwitch Probability Delta Statistics (rising switch probability):")
    print(f"  Mean: {switch_prob_delta_eval.mean():.4f}")
    print(f"  Median: {switch_prob_delta_eval.median():.4f}")
    print(f"  Std: {switch_prob_delta_eval.std():.4f}")
    print(f"  Min: {switch_prob_delta_eval.min():.4f}")
    print(f"  Max: {switch_prob_delta_eval.max():.4f}")
    nonzero_delta = (switch_prob_delta_eval > 0).sum()
    print(f"  Non-zero delta days: {nonzero_delta} ({nonzero_delta/len(switch_prob_delta_eval)*100:.1f}%)")
    
    # Correlation between switch prob delta and labeled turning points
    common_idx_delta = labeled_turns_binary.index.intersection(switch_prob_delta_eval.index)
    if len(common_idx_delta) > 0:
        corr_delta = np.corrcoef(labeled_turns_binary.loc[common_idx_delta], 
                                switch_prob_delta_eval.loc[common_idx_delta])[0, 1]
        print(f"  Correlation with labeled turning points: {corr_delta:.3f}")
    
    # Print alert score statistics
    print("\nTurn Alert Score Statistics (0.5×switch_prob + 0.2×entropy + 0.3×delta):")
    print(f"  Mean: {alert_score_eval.mean():.4f}")
    print(f"  Median: {alert_score_eval.median():.4f}")
    print(f"  Std: {alert_score_eval.std():.4f}")
    print(f"  Min: {alert_score_eval.min():.4f}")
    print(f"  Max: {alert_score_eval.max():.4f}")
    print(f"  95th percentile: {alert_score_eval.quantile(0.95):.4f}")
    print(f"  99th percentile: {alert_score_eval.quantile(0.99):.4f}")
    
    # Correlation between alert score and labeled turning points
    common_idx_alert = labeled_turns_binary.index.intersection(alert_score_eval.index)
    if len(common_idx_alert) > 0:
        corr_alert = np.corrcoef(labeled_turns_binary.loc[common_idx_alert], 
                                alert_score_eval.loc[common_idx_alert])[0, 1]
        print(f"  Correlation with labeled turning points: {corr_alert:.3f}")
        
        # Find how many turning points had high alert scores
        alert_threshold = alert_score_eval.quantile(0.95)
        turns_with_high_alert = labeled_turns_binary.loc[common_idx_alert]
        alert_at_turns = alert_score_eval.loc[common_idx_alert]
        turns_captured = ((turns_with_high_alert == 1) & (alert_at_turns >= alert_threshold)).sum()
        total_turns = turns_with_high_alert.sum()
        print(f"  Turning points with high alert (top 5%): {turns_captured}/{int(total_turns)} ({turns_captured/total_turns*100:.1f}%)")
    
    # Step 4: Calculate accuracy metrics
    print("\n" + "="*80)
    print("STEP 4: ACCURACY EVALUATION")
    print("="*80)
    
    # Align indices
    common_idx = regime_labels.index.intersection(hmm_regime_eval.index)
    labeled = regime_labels.loc[common_idx]
    predicted = hmm_regime_eval.loc[common_idx]
    
    # Calculate overall accuracy
    correct = (labeled == predicted).sum()
    total = len(common_idx)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate balanced accuracy (average of per-class recall) to handle class imbalance
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
    balanced_acc = balanced_accuracy_score(labeled, predicted)
    kappa = cohen_kappa_score(labeled, predicted)
    
    # Calculate naive baseline (always predict most frequent class)
    most_frequent = labeled.mode()[0]
    naive_accuracy = (labeled == most_frequent).sum() / len(labeled)
    
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total} days)")
    print(f"  WARNING: Overall accuracy can be misleading with class imbalance!")
    print(f"  Naive baseline (always predict '{most_frequent}'): {naive_accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_acc:.2%} (average of per-class recall - better for imbalanced data)")
    print(f"Cohen's Kappa: {kappa:.3f} (agreement above chance, -1 to 1, >0.6 is good)")
    
    # Calculate per-regime metrics
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Get unique labels
    unique_labels = sorted(set(labeled.unique()) | set(predicted.unique()))
    
    # Confusion matrix
    cm = confusion_matrix(labeled, predicted, labels=unique_labels)
    
    print("\nConfusion Matrix:")
    print(f"{'':>12}", end='')
    for label in unique_labels:
        print(f"{label:>12}", end='')
    print()
    for i, label in enumerate(unique_labels):
        print(f"{label:>12}", end='')
        for j in range(len(unique_labels)):
            print(f"{cm[i][j]:>12}", end='')
        print()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(labeled, predicted, labels=unique_labels, zero_division=0))
    
    # Calculate regime switch accuracy (how well HMM detects turning points)
    labeled_switches = labeled[labeled.ne(labeled.shift(1))].dropna()
    hmm_switches_eval = predicted[predicted.ne(predicted.shift(1))].dropna()
    
    print(f"\nRegime Switch Detection:")
    print(f"  Labeled switches: {len(labeled_switches)}")
    print(f"  HMM switches: {len(hmm_switches_eval)}")
    
    # Find matching switches within tolerance window
    matches = 0
    for switch_date in labeled_switches.index:
        for hmm_switch_date in hmm_switches_eval.index:
            days_diff = abs((switch_date - hmm_switch_date).days)
            if days_diff <= tolerance_window:
                matches += 1
                break
    
    switch_precision = matches / len(hmm_switches_eval) if len(hmm_switches_eval) > 0 else 0
    switch_recall = matches / len(labeled_switches) if len(labeled_switches) > 0 else 0
    
    print(f"  Matched switches (within {tolerance_window} days): {matches}")
    print(f"  Switch Precision: {switch_precision:.2%}")
    print(f"  Switch Recall: {switch_recall:.2%}")
    
    # Prepare results
    results = {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'labeling_params': {
            'threshold_pct': threshold_pct
        },
        'labeled_turns': labeled_turns,
        'regime_labels': regime_labels,
        'hmm_regime': hmm_regime_eval,
        'hmm_probs': probs,
        'hmm_switches': switches,
        'accuracy_metrics': {
            'overall_accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'cohens_kappa': kappa,
            'naive_baseline': naive_accuracy,
            'correct_days': correct,
            'total_days': total,
            'confusion_matrix': cm,
            'switch_precision': switch_precision,
            'switch_recall': switch_recall,
            'labeled_switches': len(labeled_switches),
            'hmm_switches': len(hmm_switches_eval),
            'matched_switches': matches
        }
    }
    
    # Generate plot
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', f'labeled_turns_{ticker}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Use evaluation period data for plotting
    close = close_eval
    
    # Generate labeled plot
    plot_path = os.path.join(results_dir, f'labeled_turns_{ticker}.png')
    plot_labeled_turns_with_regimes(close, labeled_turns, regime_labels, plot_path)
    
    # Generate comparison plot with HMM
    # Get raw numeric states for third panel (before regime mapping)
    hmm_states_raw_eval = hmm_regime.loc[start_date:end_date]
    
    comparison_path = os.path.join(results_dir, f'comparison_hmm_{ticker}.png')
    plot_comparison_with_hmm(close, labeled_turns, regime_labels, hmm_regime_eval, 
                            hmm_states_raw_eval, switch_prob_eval, entropy_eval,
                            switch_prob_delta_eval, alert_score_eval, comparison_path)
    
    # Save labeling summary to file
    summary_path = os.path.join(results_dir, 'labeling_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ZIGZAG LABELING & HMM ACCURACY EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Labeling Parameters (Zigzag):\n")
        f.write(f"  Threshold: {threshold_pct*100:.1f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("LABELED TURNING POINTS\n")
        f.write("="*80 + "\n")
        f.write(f"Tops:    {num_tops:>6}\n")
        f.write(f"Bottoms: {num_bottoms:>6}\n")
        f.write(f"Total:   {total_turns:>6}\n\n")
        
        f.write("="*80 + "\n")
        f.write("HMM ACCURACY METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total} days)\n")
        f.write(f"  WARNING: Can be misleading with class imbalance!\n")
        f.write(f"  Naive baseline (always predict most frequent): {naive_accuracy:.2%}\n\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.2%} (average of per-class recall)\n")
        f.write(f"Cohen's Kappa: {kappa:.3f} (agreement above chance, >0.6 is good)\n\n")
        
        f.write("Regime Distribution Comparison:\n")
        f.write(f"  {'Regime':<12} {'Labeled':<25} {'HMM Detected':<25}\n")
        f.write(f"  {'-'*12} {'-'*25} {'-'*25}\n")
        f.write(f"  {'Bullish':<12} {num_bullish:>6} ({num_bullish/len(regime_labels)*100:>5.1f}%) {avg_bullish_days:>5.1f}d  {hmm_bullish:>6} ({hmm_bullish/len(hmm_regime_eval)*100:>5.1f}%) {hmm_avg_bullish_days:>5.1f}d\n")
        f.write(f"  {'Bearish':<12} {num_bearish:>6} ({num_bearish/len(regime_labels)*100:>5.1f}%) {avg_bearish_days:>5.1f}d  {hmm_bearish:>6} ({hmm_bearish/len(hmm_regime_eval)*100:>5.1f}%) {hmm_avg_bearish_days:>5.1f}d\n")
        if num_neutral > 0 or hmm_neutral > 0:
            f.write(f"  {'Neutral':<12} {num_neutral:>6} ({num_neutral/len(regime_labels)*100:>5.1f}%) {avg_neutral_days:>5.1f}d  {hmm_neutral:>6} ({hmm_neutral/len(hmm_regime_eval)*100:>5.1f}%) {hmm_avg_neutral_days:>5.1f}d\n")
        
        f.write("\nAverage Regime Duration:\n")
        f.write(f"  {'Regime':<12} {'Labeled (days)':<20} {'HMM Detected (days)':<20}\n")
        f.write(f"  {'-'*12} {'-'*20} {'-'*20}\n")
        f.write(f"  {'Bullish':<12} {avg_bullish_days:>8.1f} ({len(bullish_periods):>3} periods)  {hmm_avg_bullish_days:>8.1f} ({len(hmm_bullish_periods):>3} periods)\n")
        f.write(f"  {'Bearish':<12} {avg_bearish_days:>8.1f} ({len(bearish_periods):>3} periods)  {hmm_avg_bearish_days:>8.1f} ({len(hmm_bearish_periods):>3} periods)\n")
        if num_neutral > 0 or hmm_neutral > 0:
            f.write(f"  {'Neutral':<12} {avg_neutral_days:>8.1f} ({len(neutral_periods):>3} periods)  {hmm_avg_neutral_days:>8.1f} ({len(hmm_neutral_periods):>3} periods)\n")
        
        f.write("\nRegime Switch Detection:\n")
        f.write(f"  Labeled switches: {len(labeled_switches)}\n")
        f.write(f"  HMM switches: {len(hmm_switches_eval)}\n")
        f.write(f"  Matched switches: {matches}\n")
        f.write(f"  Switch Precision: {switch_precision:.2%}\n")
        f.write(f"  Switch Recall: {switch_recall:.2%}\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    
    # Export labeled data to CSV
    print("\nExporting labeled data to CSV...")
    
    # Create dataframe with price, turning points, and regimes
    export_data = pd.DataFrame(index=close.index)
    export_data['close'] = close
    
    # Add regime labels
    export_data['regime_label'] = regime_labels
    
    # Add HMM regime predictions
    export_data['hmm_regime'] = hmm_regime_eval
    
    # Add switch probability
    export_data['switch_prob'] = switch_prob_eval
    
    # Add entropy
    export_data['entropy'] = entropy_eval
    
    # Add switch probability delta
    export_data['switch_prob_delta'] = switch_prob_delta_eval
    
    # Add alert score
    export_data['alert_score'] = alert_score_eval
    
    # Add agreement indicator
    export_data['regime_match'] = (regime_labels == hmm_regime_eval).astype(int)
    
    # Add turning point labels
    export_data['is_top'] = 0
    export_data['is_bottom'] = 0
    export_data['is_labeled_turn'] = 0
    
    for idx in labeled_turns.index:
        if idx in export_data.index:
            export_data.loc[idx, 'is_top'] = labeled_turns.loc[idx, 'is_top']
            export_data.loc[idx, 'is_bottom'] = labeled_turns.loc[idx, 'is_bottom']
            if labeled_turns.loc[idx, 'is_top'] == 1 or labeled_turns.loc[idx, 'is_bottom'] == 1:
                export_data.loc[idx, 'is_labeled_turn'] = 1
    
    # Save to CSV
    csv_path = os.path.join(results_dir, f'labeled_data_{ticker}.csv')
    export_data.to_csv(csv_path, index=True, index_label='date')
    
    print(f"✓ Labeled data exported to: {csv_path}")
    print(f"  Columns: {', '.join(export_data.columns.tolist())}")
    print(f"  Total rows: {len(export_data)}")
    print(f"  Date range: {export_data.index[0]} to {export_data.index[-1]}")
    
    print(f"\n✓ Results directory: {results_dir}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate HMM regime detection accuracy against labeled turning points.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default parameters
  python hmm_accuracy_evaluation.py SPY --start-date 2020-01-01 --end-date 2024-12-31
  
  # Evaluate with config file
  python hmm_accuracy_evaluation.py SPY --start-date 2020-01-01 --end-date 2024-12-31 --config config_optimal.json
  
  # Evaluate with custom threshold (fewer pivots with higher threshold)
  python hmm_accuracy_evaluation.py SPY --start-date 2020-01-01 --end-date 2024-12-31 --threshold-pct 0.08
        """
    )
    
    parser.add_argument('ticker', type=str, help='Ticker symbol to evaluate')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--threshold-pct', type=float, default=0.05, help='Zigzag threshold percentage (default: 0.05 = 5%%)')
    parser.add_argument('--tolerance', type=int, default=10, help='Matching tolerance window in bars (default: 10)')
    
    args = parser.parse_args()
    
    results = evaluate_hmm_accuracy(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config,
        threshold_pct=args.threshold_pct,
        tolerance_window=args.tolerance
    )

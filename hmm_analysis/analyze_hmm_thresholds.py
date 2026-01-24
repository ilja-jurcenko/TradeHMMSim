"""
Analyze HMM probability distributions and threshold effectiveness.
Tests 3 different HMM configurations:
1. Baseline (504, 21) - Current production parameters  
2. Optimal (252, 42) - Best from parameter search
3. Stable (252, 63) - Most stable configuration

Compares two implementations:
- Forward Algorithm (manual): Optimized, no lookahead
- Predict Proba: Uses library function, simpler but slower
"""

import numpy as np
import pandas as pd
from portfolio import Portfolio
from alpha_models import EMA
from signal_filter import HMMRegimeFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
import os
import time

os.makedirs("hmm_analysis", exist_ok=True)

portfolio = Portfolio(["SPY"], "2020-01-01", "2025-12-31")
portfolio.load_data()
close = portfolio.get_close_prices("SPY")

alpha_model = EMA(short_window=10, long_window=30)
alpha_signals = alpha_model.generate_signals(close)

hmm_configs = [
    {"name": "Baseline", "train_window": 504, "refit_every": 21, "description": "Current production parameters"},
    {"name": "Optimal", "train_window": 252, "refit_every": 42, "description": "Best from parameter search (57.7% accuracy)"},
    {"name": "Stable", "train_window": 252, "refit_every": 63, "description": "Most stable (16.7 switches/year)"},
    {"name": "Accurate", "train_window": 756, "refit_every": 21, "description": "Most accurate (61.7% accuracy)"},
]

def calculate_performance_metrics(positions, close_prices):
    """Calculate Sharpe ratio and total returns."""
    returns = close_prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0:
        return 0.0, 0.0
    
    total_return = (1 + strategy_returns).prod() - 1
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe = 0.0 if (std_return == 0 or np.isnan(std_return)) else (mean_return / std_return) * np.sqrt(252)
    
    return sharpe, total_return

def analyze_hmm_config(config, alpha_signals, close, method="forward"):
    """Run analysis for a single HMM configuration.
    
    Parameters:
    -----------
    method : str
        Either "forward" (manual forward algorithm) or "predict_proba" (library function)
    """
    method_name = "Forward Algorithm" if method == "forward" else "Predict Proba"
    print(f"\n{'='*80}")
    print(f"ANALYZING: {config['name']} Configuration - {method_name}")
    print(f"Parameters: train_window={config['train_window']}, refit_every={config['refit_every']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
    
    start_time = time.time()
    if method == "forward":
        probs, regime, switches = hmm_filter.walkforward_filter(
            close, train_window=config["train_window"], refit_every=config["refit_every"]
        )
    else:  # predict_proba
        probs, regime, switches = hmm_filter.walkforward_filter_predict_proba(
            close, train_window=config["train_window"], refit_every=config["refit_every"]
        )
    elapsed_time = time.time() - start_time
    
    regime_info = hmm_filter.identify_regimes(close, regime)
    
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Regime Identification: Bear={regime_info['bear_regime']}, Bull={regime_info['bull_regime']}, Neutral={regime_info['neutral_regime']}")
    print(f"Regime switches: {len(switches)}")
    
    return {"probs": probs, "regime": regime, "switches": switches, "elapsed_time": elapsed_time, **regime_info}

all_results = {}
for config in hmm_configs:
    # Test both methods
    for method in ["forward", "predict_proba"]:
        result = analyze_hmm_config(config, alpha_signals, close, method=method)
        method_suffix = "_forward" if method == "forward" else "_predict_proba"
        all_results[config["name"] + method_suffix] = {**config, "method": method, **result}

print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}")

for config_name, result in all_results.items():
    print(f"\nProcessing: {config_name}")
    
    probs, regime = result["probs"], result["regime"]
    bear_regime, bull_regime, neutral_regime = result["bear_regime"], result["bull_regime"], result["neutral_regime"]
    
    common_idx = alpha_signals.index.intersection(probs.index)
    alpha_signals_aligned = alpha_signals.loc[common_idx]
    
    bear_prob = probs[bear_regime].loc[common_idx]
    bull_prob = probs[bull_regime].loc[common_idx]
    bull_prob_combined = (bull_prob + probs[neutral_regime].loc[common_idx]) if neutral_regime is not None else bull_prob
    
    combined_positions = alpha_signals_aligned.astype(bool).copy()
    contrarian_entry = (~alpha_signals_aligned.astype(bool)) & (bull_prob_combined > 0.65)
    combined_positions[contrarian_entry] = True
    
    sharpe, total_return = calculate_performance_metrics(combined_positions.astype(int), close.loc[common_idx])
    print(f"  Sharpe: {sharpe:.3f}, Total Return: {total_return*100:.2f}%, Switches: {len(result['switches'])}")
    
    result.update({"sharpe": sharpe, "total_return": total_return, "combined_positions": combined_positions, "switches_count": len(result['switches'])})
    
    # Generate plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    method_label = result.get('method', 'forward').replace('_', ' ').title()
    time_label = f" | Time: {result.get('elapsed_time', 0):.1f}s"
    fig.suptitle(f"HMM Analysis - {config_name} | {method_label} | Sharpe: {sharpe:.3f} | Return: {total_return*100:.2f}% | Switches: {len(result['switches'])}{time_label}", fontsize=14, fontweight="bold")
    
    close_plot = close.loc[common_idx]
    regime_plot = regime.loc[common_idx]
    
    # Plot 1: Price with regime coloring
    ax1 = axes[0]
    ax1.plot(common_idx, close_plot, color="black", linewidth=1.5, label="SPY Price", zorder=3)
    ax1.set_ylabel("Price ($)", fontsize=10, fontweight="bold")
    ax1.set_title("SPY with HMM Regime Background", fontsize=11, fontweight="bold")
    
    for regime_id, color, alpha_val in [(bear_regime, "red", 0.2), (bull_regime, "green", 0.2), (neutral_regime, "gray", 0.15)]:
        if regime_id is None:
            continue
        regime_dates = common_idx[regime_plot == regime_id]
        for i in range(len(regime_dates)):
            if i == 0 or (regime_dates[i] - regime_dates[i-1]).days > 2:
                start_idx = regime_dates[i]
            end_idx = regime_dates[i]
            if i == len(regime_dates) - 1 or (i < len(regime_dates) - 1 and (regime_dates[i+1] - regime_dates[i]).days > 2):
                ax1.axvspan(start_idx, end_idx, alpha=alpha_val, color=color, zorder=1)
    
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined strategy
    ax2 = axes[1]
    ax2.plot(common_idx, close_plot, color="black", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Price ($)", fontsize=10, fontweight="bold")
    ax2.set_title("Alpha + HMM Combine (Contrarian Entry)", fontsize=11, fontweight="bold")
    
    combined_changes = combined_positions.astype(int).diff()
    combined_entries = common_idx[combined_changes == 1]
    combined_exits = common_idx[combined_changes == -1]
    contrarian_entry_dates = common_idx[contrarian_entry & (combined_changes == 1)]
    normal_entries = [d for d in combined_entries if d not in contrarian_entry_dates]
    
    if len(normal_entries) > 0:
        ax2.scatter(normal_entries, close.loc[normal_entries], color="green", marker="^", s=100, label=f"Alpha Buy ({len(normal_entries)})", zorder=5, edgecolors="darkgreen", linewidths=1.5)
    if len(contrarian_entry_dates) > 0:
        ax2.scatter(contrarian_entry_dates, close.loc[contrarian_entry_dates], color="orange", marker="^", s=120, label=f"Contrarian Buy ({len(contrarian_entry_dates)})", zorder=6, edgecolors="darkorange", linewidths=2)
    if len(combined_exits) > 0:
        ax2.scatter(combined_exits, close.loc[combined_exits], color="red", marker="v", s=100, label=f"Sell ({len(combined_exits)})", zorder=5, edgecolors="darkred", linewidths=1.5)
    
    ax2.fill_between(common_idx, close_plot.min() * 0.98, close_plot.max() * 1.02, where=combined_positions, alpha=0.15, color="teal", label="In Position")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Position with regime coloring
    ax3 = axes[2]
    ax3.set_ylabel("Price ($)", fontsize=10, fontweight="bold")
    ax3.set_title("Position Entry/Exit by Regime", fontsize=11, fontweight="bold")
    
    for i in range(len(common_idx) - 1):
        if combined_positions.iloc[i]:
            current_regime = regime_plot.iloc[i]
            if current_regime == bear_regime:
                color = "red"
            elif current_regime == bull_regime:
                color = "green"
            elif current_regime == neutral_regime:
                color = "gray"
            else:
                color = "black"
            ax3.plot([common_idx[i], common_idx[i+1]], [close_plot.iloc[i], close_plot.iloc[i+1]], color=color, linewidth=2, alpha=0.8)
        else:
            ax3.plot([common_idx[i], common_idx[i+1]], [close_plot.iloc[i], close_plot.iloc[i+1]], color="lightgray", linewidth=1, alpha=0.5)
    
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="In Position - Bull"),
        Line2D([0], [0], color="gray", lw=2, label="In Position - Neutral"),
        Line2D([0], [0], color="red", lw=2, label="In Position - Bear"),
        Line2D([0], [0], color="lightgray", lw=1, alpha=0.5, label="Out of Position"),
    ]
    ax3.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Date", fontsize=10, fontweight="bold")
    
    for ax in axes:
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    output_path = f"hmm_analysis/hmm_regime_analysis_{config_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'Config':<30} {'Method':<15} {'Time(s)':<10} {'Sharpe':<10} {'Return':<12} {'Switches':<10}")
print(f"{'-'*100}")
for config_name, result in all_results.items():
    base_name = config_name.replace('_forward', '').replace('_predict_proba', '')
    method = result.get('method', 'forward').replace('_', ' ').title()
    time_val = result.get('elapsed_time', 0)
    print(f"{base_name:<30} {method:<15} {time_val:<10.2f} {result['sharpe']:<10.3f} {result['total_return']*100:<12.2f}% {result['switches_count']:<10}")

print(f"\n{'='*80}")
print("COMPARISON: Forward vs Predict Proba")
print(f"{'='*80}")
# Group by base config name
configs_base = set(c.replace('_forward', '').replace('_predict_proba', '') for c in all_results.keys())
for base_config in sorted(configs_base):
    forward_key = base_config + "_forward"
    predict_key = base_config + "_predict_proba"
    
    if forward_key in all_results and predict_key in all_results:
        forward_result = all_results[forward_key]
        predict_result = all_results[predict_key]
        
        time_diff = predict_result['elapsed_time'] - forward_result['elapsed_time']
        time_ratio = predict_result['elapsed_time'] / forward_result['elapsed_time'] if forward_result['elapsed_time'] > 0 else 0
        sharpe_diff = predict_result['sharpe'] - forward_result['sharpe']
        
        print(f"\n{base_config}:")
        print(f"  Time: Forward={forward_result['elapsed_time']:.2f}s, Predict={predict_result['elapsed_time']:.2f}s (Δ={time_diff:+.2f}s, {time_ratio:.2f}x)")
        print(f"  Sharpe: Forward={forward_result['sharpe']:.3f}, Predict={predict_result['sharpe']:.3f} (Δ={sharpe_diff:+.3f})")
        print(f"  Switches: Forward={forward_result['switches_count']}, Predict={predict_result['switches_count']}")

for config_name, result in all_results.items():
    print(f"{config_name}: Sharpe={result['sharpe']:.3f}, Return={result['total_return']*100:.2f}%, Switches={result['switches_count']}")
print(f"\nFiles saved to: hmm_analysis/")

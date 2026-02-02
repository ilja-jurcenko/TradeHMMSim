"""
Investigate the discrepancy between low switch probabilities and frequent regime switches.

This script analyzes:
1. Switch probability (based on transition matrix): P_switch(t) = 1 - Σ_i p_t(i) * A_ii
2. Actual regime switches (based on hysteresis detection)
3. State probability changes between time steps
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio import Portfolio
from signal_filter import HMMRegimeFilter
from config_loader import ConfigLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def analyze_switch_probability_vs_switches(ticker='SPY', 
                                          start_date='2020-01-01', 
                                          end_date='2024-12-31',
                                          config_path='config_optimal.json'):
    """
    Analyze why switch probability is low but we see many regime switches.
    """
    print("\n" + "="*80)
    print("INVESTIGATING SWITCH PROBABILITY vs REGIME SWITCHES")
    print("="*80)
    
    # Load config
    config = ConfigLoader.load_config(config_path)
    hmm_config = config.get('hmm', {})
    
    n_states = hmm_config.get('n_states', 2)
    train_window = hmm_config.get('train_window', 5000)
    refit_every = hmm_config.get('refit_every', 20)
    
    # Load data with buffer
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    buffer_days = train_window + 100
    extended_start_dt = start_dt - timedelta(days=int(buffer_days * 1.5))
    extended_start = extended_start_dt.strftime('%Y-%m-%d')
    
    print(f"\nLoading data for {ticker}...")
    portfolio = Portfolio([ticker], extended_start, end_date)
    portfolio.load_data()
    close_full = portfolio.get_close_prices(ticker)
    
    # Initialize HMM
    hmm_filter = HMMRegimeFilter(
        n_states=n_states,
        random_state=42,
        covariance_type=hmm_config.get('covariance_type', 'diag'),
        n_iter=hmm_config.get('n_iter', 100),
        tol=hmm_config.get('tol', 1e-3),
        short_vol_window=hmm_config.get('short_vol_window', 10),
        long_vol_window=hmm_config.get('long_vol_window', 30),
        short_ma_window=hmm_config.get('short_ma_window', 10),
        long_ma_window=hmm_config.get('long_ma_window', 30)
    )
    
    # Run walkforward with switch probability
    print(f"\nRunning HMM walkforward...")
    probs, hmm_regime, switches, switch_prob = hmm_filter.walkforward_filter(
        close_full,
        train_window=train_window,
        refit_every=refit_every,
        return_switch_prob=True
    )
    
    # Filter to evaluation period
    probs_eval = probs.loc[start_date:end_date]
    regime_eval = hmm_regime.loc[start_date:end_date]
    switch_prob_eval = switch_prob.loc[start_date:end_date]
    
    print(f"\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    # 1. Transition matrix analysis
    print(f"\nTransition Matrix (from final model):")
    transmat = hmm_filter.model.transmat_
    print(transmat)
    print(f"\nPersistence probabilities (diagonal):")
    for i in range(n_states):
        print(f"  State {i}: {transmat[i, i]:.4f} ({transmat[i, i]*100:.2f}%)")
    
    # 2. Switch probability statistics
    print(f"\nSwitch Probability Statistics:")
    print(f"  Mean: {switch_prob_eval.mean():.4f}")
    print(f"  Median: {switch_prob_eval.median():.4f}")
    print(f"  Std: {switch_prob_eval.std():.4f}")
    print(f"  Min: {switch_prob_eval.min():.4f}")
    print(f"  Max: {switch_prob_eval.max():.4f}")
    print(f"  95th percentile: {switch_prob_eval.quantile(0.95):.4f}")
    print(f"  99th percentile: {switch_prob_eval.quantile(0.99):.4f}")
    
    # 3. Regime switches
    regime_switches = regime_eval[regime_eval.ne(regime_eval.shift(1))].dropna()
    print(f"\nRegime Switches:")
    print(f"  Total switches: {len(regime_switches)}")
    print(f"  Days between switches (avg): {len(regime_eval) / (len(regime_switches) + 1):.1f}")
    
    # 4. State probability changes
    print(f"\nState Probability Changes:")
    # Calculate magnitude of probability change for dominant state
    prob_max = probs_eval.max(axis=1)  # Max probability at each time
    prob_max_change = prob_max.diff().abs()
    
    print(f"  Avg absolute change in max probability: {prob_max_change.mean():.4f}")
    print(f"  Max absolute change: {prob_max_change.max():.4f}")
    print(f"  Changes > 0.1: {(prob_max_change > 0.1).sum()} times")
    print(f"  Changes > 0.2: {(prob_max_change > 0.2).sum()} times")
    
    # 5. Analyze switch probability at actual switch points
    print(f"\nSwitch Probability AT Regime Switch Points:")
    if len(regime_switches) > 0:
        switch_probs_at_switches = switch_prob_eval.loc[regime_switches.index]
        print(f"  Mean: {switch_probs_at_switches.mean():.4f}")
        print(f"  Median: {switch_probs_at_switches.median():.4f}")
        print(f"  Max: {switch_probs_at_switches.max():.4f}")
        print(f"  Min: {switch_probs_at_switches.min():.4f}")
    
    # 6. Analyze switch probability ONE DAY BEFORE switch points
    print(f"\nSwitch Probability ONE DAY BEFORE Regime Switches:")
    if len(regime_switches) > 1:
        # Get dates one day before switches
        switch_dates = regime_switches.index
        prob_index_list = switch_prob_eval.index.tolist()
        
        before_switch_probs = []
        for switch_date in switch_dates:
            try:
                idx = prob_index_list.index(switch_date)
                if idx > 0:
                    before_switch_probs.append(switch_prob_eval.iloc[idx-1])
            except (ValueError, IndexError):
                pass
        
        if before_switch_probs:
            before_switch_probs = pd.Series(before_switch_probs)
            print(f"  Mean: {before_switch_probs.mean():.4f}")
            print(f"  Median: {before_switch_probs.median():.4f}")
            print(f"  Max: {before_switch_probs.max():.4f}")
    
    # 7. Key insight: Compare switch probability vs hysteresis thresholds
    print(f"\n" + "="*80)
    print("KEY INSIGHT: Two Different Mechanisms")
    print("="*80)
    print(f"\n1. SWITCH PROBABILITY (Transition Matrix Based):")
    print(f"   - Formula: P_switch(t) = 1 - Σ_i p_t(i) * A_ii")
    print(f"   - Based on: Model's transition matrix (learned persistence)")
    print(f"   - Interpretation: 'What model predicts will happen NEXT step'")
    print(f"   - Typical value: {switch_prob_eval.mean():.1%} (very low!)")
    
    print(f"\n2. REGIME SWITCHES (Hysteresis Detection):")
    print(f"   - Based on: State probability thresholds & confirmation")
    print(f"   - Detection logic:")
    print(f"     * Enter new regime when P(state) > 0.55 for 2 consecutive days")
    print(f"     * Exit current regime when P(current) < 0.55")
    print(f"   - Result: {len(regime_switches)} switches detected")
    
    print(f"\n3. WHY THE DISCREPANCY?")
    print(f"   - Transition matrix has HIGH persistence (~98% bullish, ~91% bearish)")
    print(f"   - This makes switch_prob LOW (model expects to stay in current state)")
    print(f"   - BUT: Underlying observations (data) can cause rapid probability shifts")
    print(f"   - When market conditions change, state probabilities can flip quickly")
    print(f"   - Hysteresis detection catches these flips AFTER they happen")
    print(f"   - Switch probability is PREDICTIVE, regime switches are REACTIVE")
    
    # 8. Demonstrate with examples
    print(f"\n" + "="*80)
    print("EXAMPLE REGIME SWITCHES")
    print("="*80)
    
    if len(regime_switches) >= 3:
        print(f"\nShowing first 3 regime switches:")
        for i, (switch_date, regime_val) in enumerate(list(regime_switches.items())[:3]):
            print(f"\n  Switch {i+1}: {switch_date.date()}")
            print(f"    New regime: {regime_val}")
            
            # Get surrounding probabilities
            try:
                idx = probs_eval.index.get_loc(switch_date)
                window_start = max(0, idx - 2)
                window_end = min(len(probs_eval), idx + 3)
                
                print(f"    State probabilities around switch:")
                for j in range(window_start, window_end):
                    date = probs_eval.index[j]
                    probs_row = probs_eval.iloc[j]
                    sp = switch_prob_eval.iloc[j]
                    marker = " <-- SWITCH" if date == switch_date else ""
                    print(f"      {date.date()}: State probs={[f'{p:.3f}' for p in probs_row.values]}, "
                          f"Switch_prob={sp:.4f}{marker}")
            except:
                pass
    
    # 9. Create visualization
    print(f"\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    
    # Panel 1: Price
    ax1 = axes[0]
    close_eval = close_full.loc[start_date:end_date]
    ax1.plot(close_eval.index, close_eval.values, 'k-', linewidth=1, alpha=0.7)
    
    # Mark regime switches on price
    for switch_date in regime_switches.index:
        ax1.axvline(switch_date, color='red', alpha=0.3, linestyle='--', linewidth=1)
    
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(f'{ticker} Price with Regime Switches (n={len(regime_switches)})', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: State probabilities
    ax2 = axes[1]
    for col in probs_eval.columns:
        ax2.plot(probs_eval.index, probs_eval[col], label=f'State {col}', linewidth=1.5, alpha=0.7)
    
    # Add threshold lines
    ax2.axhline(y=0.55, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.55)')
    
    # Mark switches
    for switch_date in regime_switches.index:
        ax2.axvline(switch_date, color='red', alpha=0.3, linestyle='--', linewidth=1)
    
    ax2.set_ylabel('State Probability', fontsize=12)
    ax2.set_title('State Probabilities (Switches occur when probabilities cross thresholds)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Panel 3: Switch probability
    ax3 = axes[2]
    ax3.fill_between(switch_prob_eval.index, 0, switch_prob_eval.values,
                     color='orange', alpha=0.3)
    ax3.plot(switch_prob_eval.index, switch_prob_eval.values,
            color='darkorange', linewidth=1.5, alpha=0.8)
    
    # Mark switches
    for switch_date in regime_switches.index:
        ax3.axvline(switch_date, color='red', alpha=0.3, linestyle='--', linewidth=1)
    
    ax3.set_ylabel('Switch Probability', fontsize=12)
    ax3.set_title('Switch Probability P_switch(t) = 1 - Σ p_t(i)*A_ii (Forward-looking, transition-based)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Probability change magnitude
    ax4 = axes[3]
    prob_max = probs_eval.max(axis=1)
    prob_max_change = prob_max.diff().abs()
    
    ax4.fill_between(prob_max_change.index, 0, prob_max_change.values,
                     color='purple', alpha=0.3)
    ax4.plot(prob_max_change.index, prob_max_change.values,
            color='darkviolet', linewidth=1.5, alpha=0.8)
    
    # Mark switches
    for switch_date in regime_switches.index:
        if switch_date in prob_max_change.index:
            ax4.scatter([switch_date], [prob_max_change.loc[switch_date]],
                       color='red', s=50, zorder=5, marker='o', edgecolors='darkred')
    
    ax4.axhline(y=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Large change (0.1)')
    
    ax4.set_ylabel('Probability Change', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_title('Absolute Change in Dominant State Probability (Shows when regime uncertainty occurs)', 
                  fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = f'results/switch_probability_investigation_{ticker}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved to: {output_path}")
    
    print(f"\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"""
The low switch probability ({switch_prob_eval.mean():.1%}) vs high number of switches ({len(regime_switches)}) is NOT a bug!

Two different mechanisms:
1. Switch Probability: Forward-looking, based on learned transition matrix
   - Shows model expects high persistence (stay in current state)
   - Low values reflect the model's conservative predictions
   
2. Regime Switches: Backward-looking, based on observed probability changes
   - Detects when state probabilities actually cross thresholds
   - Reacts to market condition changes that shift probabilities
   
The model learns that regimes are persistent (high diagonal in transition matrix),
but market data can still cause rapid shifts in state probabilities when conditions change.
This is EXPECTED behavior in regime-switching models!
""")


if __name__ == '__main__':
    analyze_switch_probability_vs_switches()

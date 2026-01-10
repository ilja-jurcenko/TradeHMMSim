"""
Grid search for optimal HMM parameters (train_window and refit_every).
Evaluates parameter combinations based on regime prediction accuracy and forward returns.
"""

import numpy as np
import pandas as pd
from portfolio import Portfolio
from signal_filter import HMMRegimeFilter
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def calculate_forward_returns(close: pd.Series, horizons: list = [5, 10, 21]) -> pd.DataFrame:
    """Calculate forward returns at multiple horizons."""
    returns = {}
    for h in horizons:
        returns[f'fwd_{h}d'] = close.shift(-h) / close - 1
    return pd.DataFrame(returns, index=close.index)


def evaluate_hmm_params(close: pd.Series, train_window: int, refit_every: int, 
                       n_states: int = 3, random_state: int = 42) -> dict:
    """
    Evaluate HMM with given parameters.
    
    Returns metrics:
    - Regime accuracy based on forward returns
    - Regime stability (fewer switches is generally better)
    - Coverage (how much data is usable)
    """
    hmm_filter = HMMRegimeFilter(n_states=n_states, random_state=random_state)
    
    try:
        # Run HMM
        probs, regime, switches = hmm_filter.walkforward_filter(
            close, 
            train_window=train_window,
            refit_every=refit_every
        )
        
        # Identify regimes
        regime_info = hmm_filter.identify_regimes(close, regime)
        bear_regime = regime_info['bear_regime']
        bull_regime = regime_info['bull_regime']
        neutral_regime = regime_info['neutral_regime']
        
        # Calculate forward returns
        fwd_returns = calculate_forward_returns(close, horizons=[5, 10, 21])
        
        # Align indices
        common_idx = regime.index.intersection(fwd_returns.index)
        regime_aligned = regime.loc[common_idx]
        fwd_aligned = fwd_returns.loc[common_idx]
        
        # Calculate accuracy: Does the regime predict forward returns correctly?
        # Bull/Neutral should have positive forward returns, Bear should have negative
        correct_predictions = {}
        regime_returns = {}
        
        for horizon in [5, 10, 21]:
            col = f'fwd_{horizon}d'
            fwd = fwd_aligned[col].dropna()
            common = regime_aligned.index.intersection(fwd.index)
            
            if len(common) == 0:
                continue
            
            regime_sub = regime_aligned.loc[common]
            fwd_sub = fwd.loc[common]
            
            # Count correct predictions
            bull_correct = ((regime_sub == bull_regime) & (fwd_sub > 0)).sum()
            bear_correct = ((regime_sub == bear_regime) & (fwd_sub < 0)).sum()
            if neutral_regime is not None:
                neutral_correct = ((regime_sub == neutral_regime) & (fwd_sub > 0)).sum()
            else:
                neutral_correct = 0
            
            total_bull = (regime_sub == bull_regime).sum()
            total_bear = (regime_sub == bear_regime).sum()
            total_neutral = (regime_sub == neutral_regime).sum() if neutral_regime is not None else 0
            
            # Calculate accuracy
            correct = bull_correct + bear_correct + neutral_correct
            total = len(regime_sub)
            accuracy = correct / total if total > 0 else 0
            
            correct_predictions[horizon] = accuracy
            
            # Calculate mean returns by regime
            regime_returns[f'{horizon}d_bull'] = fwd_sub[regime_sub == bull_regime].mean()
            regime_returns[f'{horizon}d_bear'] = fwd_sub[regime_sub == bear_regime].mean()
            if neutral_regime is not None:
                regime_returns[f'{horizon}d_neutral'] = fwd_sub[regime_sub == neutral_regime].mean()
        
        # Calculate regime distribution
        regime_counts = regime_aligned.value_counts()
        regime_pct = regime_counts / len(regime_aligned) * 100
        
        # Count regime switches
        num_switches = (regime_aligned.diff() != 0).sum()
        
        # Calculate regime volatilities
        returns = close.pct_change()
        common_vol = regime_aligned.index.intersection(returns.index)
        regime_vol = regime_aligned.loc[common_vol]
        returns_vol = returns.loc[common_vol]
        
        vol_by_regime = {}
        for r in regime_aligned.unique():
            mask = regime_vol == r
            vol_by_regime[r] = returns_vol[mask].std() * np.sqrt(252) * 100  # Annualized %
        
        return {
            'train_window': train_window,
            'refit_every': refit_every,
            'coverage': len(common_idx) / len(close) * 100,
            'num_switches': num_switches,
            'switches_per_year': num_switches / (len(regime_aligned) / 252),
            'bull_regime': bull_regime,
            'bear_regime': bear_regime,
            'neutral_regime': neutral_regime,
            'bear_pct': regime_pct.get(bear_regime, 0),
            'bull_pct': regime_pct.get(bull_regime, 0),
            'neutral_pct': regime_pct.get(neutral_regime, 0) if neutral_regime is not None else 0,
            'bear_vol': vol_by_regime.get(bear_regime, np.nan),
            'bull_vol': vol_by_regime.get(bull_regime, np.nan),
            'neutral_vol': vol_by_regime.get(neutral_regime, np.nan) if neutral_regime is not None else np.nan,
            'accuracy_5d': correct_predictions.get(5, 0),
            'accuracy_10d': correct_predictions.get(10, 0),
            'accuracy_21d': correct_predictions.get(21, 0),
            'bear_return_5d': regime_returns.get('5d_bear', 0),
            'bull_return_5d': regime_returns.get('5d_bull', 0),
            'neutral_return_5d': regime_returns.get('5d_neutral', 0),
            'bear_return_21d': regime_returns.get('21d_bear', 0),
            'bull_return_21d': regime_returns.get('21d_bull', 0),
            'neutral_return_21d': regime_returns.get('21d_neutral', 0),
            'success': True
        }
        
    except Exception as e:
        return {
            'train_window': train_window,
            'refit_every': refit_every,
            'success': False,
            'error': str(e)
        }


def run_parameter_search(ticker: str = 'SPY', 
                        start_date: str = '2020-01-01', 
                        end_date: str = '2025-12-31',
                        train_windows: list = None,
                        refit_intervals: list = None,
                        n_states: int = 3,
                        random_state: int = 42) -> pd.DataFrame:
    """
    Run grid search over HMM parameters.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol to analyze
    start_date, end_date : str
        Date range for analysis
    train_windows : list
        List of training window sizes to test (in days)
    refit_intervals : list
        List of refit frequencies to test (in days)
    n_states : int
        Number of HMM states
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Results for all parameter combinations
    """
    # Default parameter ranges
    if train_windows is None:
        train_windows = [252, 378, 504, 630, 756]  # 1yr, 1.5yr, 2yr, 2.5yr, 3yr
    
    if refit_intervals is None:
        refit_intervals = [5, 10, 21, 42, 63]  # 1wk, 2wk, 1mo, 2mo, 3mo
    
    print(f"{'='*80}")
    print(f"HMM PARAMETER GRID SEARCH")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Train windows: {train_windows}")
    print(f"Refit intervals: {refit_intervals}")
    print(f"Total combinations: {len(train_windows) * len(refit_intervals)}")
    print(f"{'='*80}\n")
    
    # Load data
    portfolio = Portfolio([ticker], start_date, end_date)
    portfolio.load_data()
    close = portfolio.get_close_prices(ticker)
    
    # Run grid search
    results = []
    total = len(train_windows) * len(refit_intervals)
    count = 0
    
    for train_window, refit_every in product(train_windows, refit_intervals):
        count += 1
        print(f"[{count}/{total}] Testing train_window={train_window}, refit_every={refit_every}...", end=' ')
        
        result = evaluate_hmm_params(close, train_window, refit_every, n_states, random_state)
        results.append(result)
        
        if result['success']:
            print(f"✓ Accuracy(21d): {result['accuracy_21d']*100:.1f}%, Switches/yr: {result['switches_per_year']:.1f}")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Filter successful runs
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("\n⚠️  No successful parameter combinations found!")
        return df
    
    # Calculate composite score
    # Higher accuracy and lower switch frequency is better
    # Normalize metrics to 0-1 scale
    df_success['accuracy_avg'] = (df_success['accuracy_5d'] + df_success['accuracy_10d'] + df_success['accuracy_21d']) / 3
    
    # Normalize (higher is better)
    max_acc = df_success['accuracy_avg'].max()
    min_acc = df_success['accuracy_avg'].min()
    df_success['accuracy_norm'] = (df_success['accuracy_avg'] - min_acc) / (max_acc - min_acc) if max_acc > min_acc else 1.0
    
    # Normalize switches (lower is better, so invert)
    max_sw = df_success['switches_per_year'].max()
    min_sw = df_success['switches_per_year'].min()
    df_success['switches_norm'] = 1.0 - (df_success['switches_per_year'] - min_sw) / (max_sw - min_sw) if max_sw > min_sw else 1.0
    
    # Composite score: 70% accuracy, 30% stability
    df_success['composite_score'] = 0.7 * df_success['accuracy_norm'] + 0.3 * df_success['switches_norm']
    
    # Sort by composite score
    df_success = df_success.sort_values('composite_score', ascending=False)
    
    # Update original dataframe
    df.loc[df['success'] == True, 'composite_score'] = df_success['composite_score']
    df.loc[df['success'] == True, 'accuracy_avg'] = df_success['accuracy_avg']
    
    return df


def print_summary(df: pd.DataFrame, top_n: int = 10):
    """Print summary of parameter search results."""
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("\nNo successful runs to summarize.")
        return
    
    print(f"\n{'='*80}")
    print(f"TOP {top_n} PARAMETER COMBINATIONS")
    print(f"{'='*80}\n")
    
    # Sort by composite score
    df_top = df_success.sort_values('composite_score', ascending=False).head(top_n)
    
    for idx, (i, row) in enumerate(df_top.iterrows(), 1):
        print(f"Rank #{idx}")
        print(f"  Train Window: {row['train_window']} days ({row['train_window']/252:.1f} years)")
        print(f"  Refit Every: {row['refit_every']} days ({row['refit_every']/21:.1f} months)")
        print(f"  Composite Score: {row['composite_score']:.3f}")
        print(f"  Accuracy: 5d={row['accuracy_5d']*100:.1f}%, 10d={row['accuracy_10d']*100:.1f}%, 21d={row['accuracy_21d']*100:.1f}% (avg={row['accuracy_avg']*100:.1f}%)")
        print(f"  Regime Switches: {row['num_switches']} total ({row['switches_per_year']:.1f} per year)")
        print(f"  Regime Distribution: Bear={row['bear_pct']:.1f}%, Bull={row['bull_pct']:.1f}%, Neutral={row['neutral_pct']:.1f}%")
        print(f"  21d Forward Returns: Bear={row['bear_return_21d']*100:.2f}%, Bull={row['bull_return_21d']*100:.2f}%, Neutral={row['neutral_return_21d']*100:.2f}%")
        print()
    
    # Current baseline (504, 21)
    baseline = df_success[(df_success['train_window'] == 504) & (df_success['refit_every'] == 21)]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        rank = (df_success['composite_score'] > baseline['composite_score']).sum() + 1
        print(f"{'='*80}")
        print(f"CURRENT BASELINE COMPARISON (train_window=504, refit_every=21)")
        print(f"{'='*80}")
        print(f"  Rank: #{rank} out of {len(df_success)}")
        print(f"  Composite Score: {baseline['composite_score']:.3f}")
        print(f"  Accuracy (avg): {baseline['accuracy_avg']*100:.1f}%")
        print(f"  Switches per year: {baseline['switches_per_year']:.1f}")
        print()
    
    # Best by accuracy only
    best_acc = df_success.loc[df_success['accuracy_avg'].idxmax()]
    print(f"{'='*80}")
    print(f"BEST BY ACCURACY ALONE")
    print(f"{'='*80}")
    print(f"  Train Window: {best_acc['train_window']} days, Refit Every: {best_acc['refit_every']} days")
    print(f"  Accuracy (avg): {best_acc['accuracy_avg']*100:.1f}%")
    print(f"  Switches per year: {best_acc['switches_per_year']:.1f}")
    print()
    
    # Best by stability
    best_stable = df_success.loc[df_success['switches_per_year'].idxmin()]
    print(f"{'='*80}")
    print(f"MOST STABLE (Fewest Switches)")
    print(f"{'='*80}")
    print(f"  Train Window: {best_stable['train_window']} days, Refit Every: {best_stable['refit_every']} days")
    print(f"  Switches per year: {best_stable['switches_per_year']:.1f}")
    print(f"  Accuracy (avg): {best_stable['accuracy_avg']*100:.1f}%")
    print()


def save_results(df: pd.DataFrame, output_path: str = 'hmm_analysis/parameter_search_results.csv'):
    """Save results to CSV."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    # Run parameter search
    results = run_parameter_search(
        ticker='SPY',
        start_date='2020-01-01',
        end_date='2025-12-31',
        train_windows=[252, 378, 504, 630, 756],  # 1yr to 3yr
        refit_intervals=[5, 10, 21, 42, 63],  # 1wk to 3mo
        n_states=3,
        random_state=42
    )
    
    # Print summary
    print_summary(results, top_n=10)
    
    # Save results
    save_results(results)

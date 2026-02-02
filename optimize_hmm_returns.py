"""
Parameter search for HMM configuration using Bayesian Optimization.

Optimizes HMM parameters for highest returns across multiple time periods
using HMM-only trading strategy.
"""
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args


# Test periods for robustness
TEST_PERIODS = [
    ("2005-01-01", "2010-12-31"),
    ("2010-01-01", "2015-12-31"),
    ("2015-01-01", "2020-12-31"),
    ("2020-01-01", "2024-12-31"),
]


def run_backtest_period(config_path: str, start_date: str, end_date: str):
    """
    Run backtest using run_comparison.py and extract performance metrics.
    
    Returns:
    --------
    dict with metrics or None if failed
    """
    cmd = [
        'python3', 'run_comparison.py',
        '--config', config_path,
        '--start-date', start_date,
        '--end-date', end_date,
        '--strategies', 'hmm_only'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        # Extract metrics using regex
        metrics = {}
        
        # Look for HMM-only strategy metrics
        # Total Return:        45.23%
        match = re.search(r'HMM[-_\s]?Only.*?Total Return:\s+([-\d.]+)%', output, re.DOTALL | re.IGNORECASE)
        if not match:
            match = re.search(r'Total Return:\s+([-\d.]+)%', output)
        if match:
            metrics['total_return'] = float(match.group(1))
        
        # Annualized Return:   8.12%
        match = re.search(r'Annualized Return:\s+([-\d.]+)%', output)
        if match:
            metrics['annual_return'] = float(match.group(1))
        
        # Sharpe Ratio:        1.23
        match = re.search(r'Sharpe Ratio:\s+([-\d.]+)', output)
        if match:
            metrics['sharpe_ratio'] = float(match.group(1))
        
        # Max Drawdown:        -15.4%
        match = re.search(r'Max Drawdown:\s+([-\d.]+)%', output)
        if match:
            metrics['max_drawdown'] = float(match.group(1))
        
        # Volatility:          12.3%
        match = re.search(r'Volatility:\s+([-\d.]+)%', output)
        if match:
            metrics['volatility'] = float(match.group(1))
        
        if len(metrics) == 0:
            print("❌ Failed to extract metrics")
            return None
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def calculate_composite_score(period_results: list) -> float:
    """
    Calculate composite score across multiple periods.
    
    Strategy: Optimize for consistent performance across all periods
    - Average annualized return (50%)
    - Minimum return across periods (30%) - robustness
    - Average Sharpe ratio (20%)
    """
    if not period_results or len(period_results) == 0:
        return -np.inf
    
    annual_returns = [r.get('annual_return', -100) for r in period_results]
    sharpe_ratios = [r.get('sharpe_ratio', -10) for r in period_results]
    
    # Average performance
    avg_return = np.mean(annual_returns)
    avg_sharpe = np.mean(sharpe_ratios)
    
    # Worst period (robustness)
    min_return = np.min(annual_returns)
    
    # Weighted composite score
    score = (
        0.50 * avg_return +          # Average return
        0.30 * min_return +           # Worst period (robustness)
        0.20 * avg_sharpe * 10        # Sharpe ratio scaled to return units
    )
    
    return score


def parameter_search(base_config_path: str, n_calls: int = 50):
    """
    Bayesian optimization over HMM parameters for maximum returns.
    
    Tests parameters across multiple 5-year periods and optimizes for
    consistent high returns.
    
    Parameters:
    -----------
    base_config_path : str
        Path to base configuration file
    n_calls : int
        Number of evaluations (budget)
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Force HMM-only strategy
    base_config['strategy']['strategy_mode'] = 'hmm_only'
    
    # Define parameter search space
    space = [
        Integer(200, 1000, name='train_window'),      # 0.8 to 4 years
        Integer(10, 126, name='refit_every'),         # 2 weeks to semi-annual
        Integer(5, 30, name='short_vol_window'),      # Short volatility window
        Integer(20, 100, name='long_vol_window'),     # Long volatility window
        Integer(5, 30, name='short_ma_window'),       # Short MA window
        Integer(20, 100, name='long_ma_window'),      # Long MA window
        Real(0.50, 0.80, name='bear_prob_threshold'), # Bear regime threshold
        Real(0.50, 0.80, name='bull_prob_threshold'), # Bull regime threshold
    ]
    
    print("="*80)
    print("HMM RETURNS OPTIMIZATION")
    print("="*80)
    print(f"Strategy: HMM-only (regime-based positioning)")
    print(f"Optimization budget: {n_calls} evaluations")
    print(f"\nTest Periods:")
    for start, end in TEST_PERIODS:
        print(f"  {start} to {end}")
    print(f"\nParameter Space:")
    for dim in space:
        print(f"  {dim.name}: [{dim.bounds[0]}, {dim.bounds[1]}]")
    print("="*80)
    
    # Create temporary config file
    temp_config_path = "config_temp_returns_search.json"
    
    # Store all results
    all_results = []
    iteration = [0]
    
    @use_named_args(space)
    def objective(**params):
        """Objective function to minimize (negative composite score)."""
        iteration[0] += 1
        
        print(f"\n[{iteration[0]}/{n_calls}] Testing parameters:")
        print(f"  train_window={params['train_window']}, refit_every={params['refit_every']}")
        print(f"  vol=[{params['short_vol_window']},{params['long_vol_window']}], "
              f"ma=[{params['short_ma_window']},{params['long_ma_window']}]")
        print(f"  thresholds=[{params['bear_prob_threshold']:.3f},{params['bull_prob_threshold']:.3f}]")
        
        # Update config with new parameters (convert to native Python types)
        config = base_config.copy()
        config['hmm']['train_window'] = int(params['train_window'])
        config['hmm']['refit_every'] = int(params['refit_every'])
        config['hmm']['short_vol_window'] = int(params['short_vol_window'])
        config['hmm']['long_vol_window'] = int(params['long_vol_window'])
        config['hmm']['short_ma_window'] = int(params['short_ma_window'])
        config['hmm']['long_ma_window'] = int(params['long_ma_window'])
        config['hmm']['bear_prob_threshold'] = float(params['bear_prob_threshold'])
        config['hmm']['bull_prob_threshold'] = float(params['bull_prob_threshold'])
        
        # Test across all periods
        period_results = []
        for period_idx, (start_date, end_date) in enumerate(TEST_PERIODS, 1):
            print(f"    Period {period_idx} ({start_date[:4]}-{end_date[:4]})...", end=" ")
            
            # Save temporary config
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Run backtest
            metrics = run_backtest_period(temp_config_path, start_date, end_date)
            
            if metrics:
                period_results.append(metrics)
                print(f"✓ Return: {metrics.get('annual_return', 0):6.2f}%, Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
            else:
                print("✗ Failed")
                return 1000.0  # Bad score
        
        if len(period_results) < len(TEST_PERIODS):
            print(f"  ⚠️  Only {len(period_results)}/{len(TEST_PERIODS)} periods succeeded")
            return 1000.0
        
        # Calculate composite score
        score = calculate_composite_score(period_results)
        
        # Summary statistics
        avg_return = np.mean([r['annual_return'] for r in period_results])
        min_return = np.min([r['annual_return'] for r in period_results])
        max_return = np.max([r['annual_return'] for r in period_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in period_results])
        
        print(f"  📊 Avg={avg_return:6.2f}%, Min={min_return:6.2f}%, "
              f"Max={max_return:6.2f}%, Sharpe={avg_sharpe:.2f}, Score={score:.2f}")
        
        # Store results
        result = {
            **params,
            'composite_score': score,
            'avg_annual_return': avg_return,
            'min_annual_return': min_return,
            'max_annual_return': max_return,
            'avg_sharpe_ratio': avg_sharpe,
        }
        
        # Add per-period metrics
        for idx, period_metrics in enumerate(period_results, 1):
            for key, value in period_metrics.items():
                result[f'period{idx}_{key}'] = value
        
        all_results.append(result)
        
        # Return negative score (optimizer minimizes)
        return -score
    
    # Run Bayesian Optimization
    print("\nStarting Bayesian Optimization...\n")
    
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=min(10, n_calls // 2),
        random_state=42,
        verbose=False,
        n_jobs=1
    )
    
    # Clean up temp config
    Path(temp_config_path).unlink(missing_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("\n❌ No successful evaluations!")
        return None
    
    # Sort by composite score
    df = df.sort_values('composite_score', ascending=False)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nCompleted {len(df)} successful evaluations")
    print(f"Best composite score: {df.iloc[0]['composite_score']:.2f}")
    
    print("\nTop 5 Parameter Combinations:")
    print("-"*80)
    for idx, (i, row) in enumerate(df.head(5).iterrows(), 1):
        print(f"\n{idx}. Composite Score: {row['composite_score']:.2f}")
        print(f"   Parameters:")
        print(f"     train_window={int(row['train_window'])}, refit_every={int(row['refit_every'])}")
        print(f"     vol=[{int(row['short_vol_window'])},{int(row['long_vol_window'])}], "
              f"ma=[{int(row['short_ma_window'])},{int(row['long_ma_window'])}]")
        print(f"     thresholds=[{row['bear_prob_threshold']:.3f},{row['bull_prob_threshold']:.3f}]")
        print(f"   Performance:")
        print(f"     Avg Annual Return: {row['avg_annual_return']:6.2f}%")
        print(f"     Min Return:        {row['min_annual_return']:6.2f}% (robustness)")
        print(f"     Max Return:        {row['max_annual_return']:6.2f}%")
        print(f"     Avg Sharpe Ratio:  {row['avg_sharpe_ratio']:.2f}")
        print(f"   Per-Period Returns:")
        for period_idx, (start, end) in enumerate(TEST_PERIODS, 1):
            ret = row.get(f'period{period_idx}_annual_return', 0)
            sharpe = row.get(f'period{period_idx}_sharpe_ratio', 0)
            print(f"     {start[:4]}-{end[:4]}: {ret:6.2f}% (Sharpe: {sharpe:5.2f})")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/hmm_returns_search_{timestamp}.csv"
    Path("results").mkdir(exist_ok=True)
    df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Update config file with best parameters
    best = df.iloc[0]
    print("\n" + "="*80)
    print("BEST PARAMETERS FOR RETURNS")
    print("="*80)
    print(f"train_window: {int(best['train_window'])}")
    print(f"refit_every: {int(best['refit_every'])}")
    print(f"short_vol_window: {int(best['short_vol_window'])}")
    print(f"long_vol_window: {int(best['long_vol_window'])}")
    print(f"short_ma_window: {int(best['short_ma_window'])}")
    print(f"long_ma_window: {int(best['long_ma_window'])}")
    print(f"bear_prob_threshold: {best['bear_prob_threshold']:.3f}")
    print(f"bull_prob_threshold: {best['bull_prob_threshold']:.3f}")
    print(f"\nExpected Performance:")
    print(f"  Average Annual Return: {best['avg_annual_return']:6.2f}%")
    print(f"  Worst Period Return:   {best['min_annual_return']:6.2f}%")
    print(f"  Best Period Return:    {best['max_annual_return']:6.2f}%")
    print(f"  Average Sharpe Ratio:  {best['avg_sharpe_ratio']:.2f}")
    
    # Create optimized config file
    optimized_config_path = "config_returns_optimal.json"
    print(f"\n✓ Creating: {optimized_config_path}")
    
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    config['strategy']['strategy_mode'] = 'hmm_only'
    config['hmm']['train_window'] = int(best['train_window'])
    config['hmm']['refit_every'] = int(best['refit_every'])
    config['hmm']['short_vol_window'] = int(best['short_vol_window'])
    config['hmm']['long_vol_window'] = int(best['long_vol_window'])
    config['hmm']['short_ma_window'] = int(best['short_ma_window'])
    config['hmm']['long_ma_window'] = int(best['long_ma_window'])
    config['hmm']['bear_prob_threshold'] = float(best['bear_prob_threshold'])
    config['hmm']['bull_prob_threshold'] = float(best['bull_prob_threshold'])
    
    with open(optimized_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Configuration saved!")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize HMM parameters for maximum returns')
    parser.add_argument('--config', default='config_optimal.json', 
                       help='Base configuration file')
    parser.add_argument('--n-calls', type=int, default=30,
                       help='Number of optimization evaluations (budget)')
    
    args = parser.parse_args()
    
    results_df = parameter_search(
        args.config,
        args.n_calls
    )

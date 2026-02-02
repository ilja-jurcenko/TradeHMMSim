"""
Parameter search for HMM configuration using Bayesian Optimization.

Optimizes all HMM parameters to maximize accuracy metrics using
intelligent gradient-free search (Bayesian Optimization).
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
from skopt import dump, load

def run_evaluation(config_path: str, start_date: str, end_date: str, threshold_pct: float = 0.05):
    """
    Run hmm_accuracy_evaluation and extract metrics.
    
    Returns:
    --------
    dict with metrics or None if failed
    """
    cmd = [
        'python3', 'hmm_accuracy_evaluation.py', 'SPY',
        '--start-date', start_date,
        '--end-date', end_date,
        '--threshold-pct', str(threshold_pct),
        '--config', config_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        # Extract metrics using regex
        metrics = {}
        
        # Overall Accuracy: 60.02% (755/1258 days)
        match = re.search(r'Overall Accuracy:\s+([\d.]+)%', output)
        if match:
            metrics['overall_accuracy'] = float(match.group(1))
        
        # Balanced Accuracy: 60.98% (average of per-class recall - better for imbalanced data)
        match = re.search(r'Balanced Accuracy:\s+([\d.]+)%', output)
        if match:
            metrics['balanced_accuracy'] = float(match.group(1))
        
        # Cohen's Kappa: 0.174 (agreement above chance, -1 to 1, >0.6 is good)
        match = re.search(r"Cohen's Kappa:\s+([\d.-]+)", output)
        if match:
            metrics['kappa'] = float(match.group(1))
        
        # Switch Precision: 21.95%
        match = re.search(r'Switch Precision:\s+([\d.]+)%', output)
        if match:
            metrics['switch_precision'] = float(match.group(1))
        
        # Switch Recall: 20.00%
        match = re.search(r'Switch Recall:\s+([\d.]+)%', output)
        if match:
            metrics['switch_recall'] = float(match.group(1))
        
        # HMM switches
        match = re.search(r'HMM switches:\s+(\d+)', output)
        if match:
            metrics['hmm_switches'] = int(match.group(1))
        
        # Labeled switches
        match = re.search(r'Labeled switches:\s+(\d+)', output)
        if match:
            metrics['labeled_switches'] = int(match.group(1))
        
        if len(metrics) == 0:
            print("    ❌ Failed to extract metrics from output")
            return None
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print("    ❌ Timeout (300s)")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def calculate_composite_score(metrics: dict) -> float:
    """
    Calculate composite score from multiple metrics.
    
    Weighted combination:
    - Balanced Accuracy (40%)
    - Cohen's Kappa (30%) - scaled to 0-100
    - Switch Precision (15%)
    - Switch Recall (15%)
    """
    if metrics is None:
        return -np.inf
    
    balanced_acc = metrics.get('balanced_accuracy', 0)
    kappa = metrics.get('kappa', 0)
    switch_prec = metrics.get('switch_precision', 0)
    switch_rec = metrics.get('switch_recall', 0)
    
    # Scale kappa from [-1, 1] to [0, 100]
    kappa_scaled = (kappa + 1) * 50
    
    # Weighted sum
    score = (
        0.40 * balanced_acc +
        0.30 * kappa_scaled +
        0.15 * switch_prec +
        0.15 * switch_rec
    )
    
    return score

def parameter_search(base_config_path: str, 
                    start_date: str = "2020-01-01",
                    end_date: str = "2024-12-31",
                    threshold_pct: float = 0.05,
                    n_calls: int = 50):
    """
    Bayesian optimization over all HMM parameters.
    
    Uses Gaussian Process based optimization to intelligently search
    the parameter space, focusing on promising regions.
    
    Parameters:
    -----------
    base_config_path : str
        Path to base configuration file
    start_date : str
        Evaluation start date
    end_date : str
        Evaluation end date
    threshold_pct : float
        Zigzag threshold percentage
    n_calls : int
        Number of evaluations (budget)
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Define parameter search space
    space = [
        Integer(300, 1500, name='train_window'),      # 1.2 to 6 years
        Integer(5, 126, name='refit_every'),          # Weekly to semi-annual
        Integer(5, 30, name='short_vol_window'),      # Short volatility window
        Integer(20, 100, name='long_vol_window'),     # Long volatility window
        Integer(5, 30, name='short_ma_window'),       # Short MA window
        Integer(20, 100, name='long_ma_window'),      # Long MA window
        Real(0.50, 0.80, name='bear_prob_threshold'), # Bear regime threshold
        Real(0.50, 0.80, name='bull_prob_threshold'), # Bull regime threshold
    ]
    
    print("="*80)
    print("HMM BAYESIAN OPTIMIZATION")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Threshold: {threshold_pct*100}%")
    print(f"Optimization budget: {n_calls} evaluations")
    print(f"\nParameter Space:")
    for dim in space:
        print(f"  {dim.name}: [{dim.bounds[0]}, {dim.bounds[1]}]")
    print("="*80)
    
    # Create temporary config file
    temp_config_path = "config_temp_search.json"
    
    # Store all results
    all_results = []
    iteration = [0]  # Use list to allow modification in nested function
    
    @use_named_args(space)
    def objective(**params):
        """
        Objective function to minimize (negative composite score).
        
        Returns negative score because optimizer minimizes.
        """
        iteration[0] += 1
        
        print(f"\n[{iteration[0]}/{n_calls}] Testing parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
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
        
        # Save temporary config
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run evaluation
        metrics = run_evaluation(temp_config_path, start_date, end_date, threshold_pct)
        
        if metrics:
            # Calculate composite score
            score = calculate_composite_score(metrics)
            
            print(f"  ✓ Balanced Acc: {metrics.get('balanced_accuracy', 0):.2f}%, "
                  f"Kappa: {metrics.get('kappa', 0):.3f}, "
                  f"Switch P/R: {metrics.get('switch_precision', 0):.1f}%/{metrics.get('switch_recall', 0):.1f}%, "
                  f"Score: {score:.2f}")
            
            # Store results
            result = {**params, 'composite_score': score, **metrics}
            all_results.append(result)
            
            # Return negative score (optimizer minimizes)
            return -score
        else:
            print(f"  ✗ Failed - returning worst score")
            # Return large positive value (bad score when minimizing)
            return 1000.0
    
    # Run Bayesian Optimization
    print("\nStarting Bayesian Optimization...\n")
    
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=10,  # Random exploration first
        random_state=42,
        verbose=False,
        n_jobs=1  # Sequential execution for stability
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
        print(f"   train_window={int(row['train_window'])}, refit_every={int(row['refit_every'])}")
        print(f"   short_vol_window={int(row['short_vol_window'])}, long_vol_window={int(row['long_vol_window'])}")
        print(f"   short_ma_window={int(row['short_ma_window'])}, long_ma_window={int(row['long_ma_window'])}")
        print(f"   bear_prob_threshold={row['bear_prob_threshold']:.2f}, bull_prob_threshold={row['bull_prob_threshold']:.2f}")
        print(f"   Metrics: Balanced Acc={row['balanced_accuracy']:.2f}%, Kappa={row['kappa']:.3f}, "
              f"Switch P/R={row['switch_precision']:.1f}%/{row['switch_recall']:.1f}%")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/hmm_bayesian_search_{timestamp}.csv"
    Path("results").mkdir(exist_ok=True)
    df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save optimization object (optional, may fail due to pickling issues)
    opt_file = f"results/hmm_bayesian_opt_{timestamp}.pkl"
    try:
        dump(result, opt_file)
        print(f"✓ Optimization state saved to: {opt_file}")
    except Exception as e:
        print(f"⚠ Could not save optimization state (CSV contains all results): {type(e).__name__}")
    
    # Update config file with best parameters
    best = df.iloc[0]
    print("\n" + "="*80)
    print("BEST PARAMETERS")
    print("="*80)
    print(f"train_window: {int(best['train_window'])}")
    print(f"refit_every: {int(best['refit_every'])}")
    print(f"short_vol_window: {int(best['short_vol_window'])}")
    print(f"long_vol_window: {int(best['long_vol_window'])}")
    print(f"short_ma_window: {int(best['short_ma_window'])}")
    print(f"long_ma_window: {int(best['long_ma_window'])}")
    print(f"bear_prob_threshold: {best['bear_prob_threshold']:.3f}")
    print(f"bull_prob_threshold: {best['bull_prob_threshold']:.3f}")
    
    # Auto-update configuration with best parameters
    print(f"\nUpdating {base_config_path} with best parameters...")
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    config['hmm']['train_window'] = int(best['train_window'])
    config['hmm']['refit_every'] = int(best['refit_every'])
    config['hmm']['short_vol_window'] = int(best['short_vol_window'])
    config['hmm']['long_vol_window'] = int(best['long_vol_window'])
    config['hmm']['short_ma_window'] = int(best['short_ma_window'])
    config['hmm']['long_ma_window'] = int(best['long_ma_window'])
    config['hmm']['bear_prob_threshold'] = float(best['bear_prob_threshold'])
    config['hmm']['bull_prob_threshold'] = float(best['bull_prob_threshold'])
    
    with open(base_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Configuration updated!")
    
    return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize HMM parameters using Bayesian Optimization')
    parser.add_argument('--config', default='config_optimal.json', 
                       help='Base configuration file')
    parser.add_argument('--start-date', default='2020-01-01',
                       help='Start date for evaluation')
    parser.add_argument('--end-date', default='2024-12-31',
                       help='End date for evaluation')
    parser.add_argument('--threshold-pct', type=float, default=0.05,
                       help='Threshold percentage for zigzag labeling')
    parser.add_argument('--n-calls', type=int, default=50,
                       help='Number of optimization evaluations (budget)')
    
    args = parser.parse_args()
    
    results_df = parameter_search(
        args.config,
        args.start_date,
        args.end_date,
        args.threshold_pct,
        args.n_calls
    )


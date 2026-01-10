"""
Example: Testing Multiple Configurations

This script demonstrates how to run backtests with different configurations
and compare the results.
"""

from run_comparison import run_comparison
import pandas as pd


def test_multiple_configs():
    """Test multiple configuration files and compare results."""
    
    configs = [
        ('config_default.json', 'Baseline (504,21)'),
        ('config_optimal.json', 'Optimal (252,42)'),
        ('config_accurate.json', 'Accurate (756,21)')
    ]
    
    all_results = []
    
    print("\n" + "="*80)
    print("TESTING MULTIPLE CONFIGURATIONS")
    print("="*80)
    
    for config_file, config_name in configs:
        print(f"\n{'='*80}")
        print(f"Running: {config_name}")
        print(f"Config: {config_file}")
        print(f"{'='*80}")
        
        try:
            # Run comparison with config
            results_df, output_dir = run_comparison(config_path=config_file)
            
            # Add configuration name to results
            results_df['Configuration'] = config_name
            all_results.append(results_df)
            
            print(f"\n✓ {config_name} completed")
            print(f"  Output: {output_dir}")
            
        except Exception as e:
            print(f"\n✗ Error with {config_name}: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Print comparison summary
        print("\n" + "="*80)
        print("CONFIGURATION COMPARISON SUMMARY")
        print("="*80)
        
        # Group by configuration and strategy
        summary = combined_results.groupby(['Configuration', 'Strategy']).agg({
            'Total Return (%)': 'mean',
            'Sharpe Ratio': 'mean',
            'Max Drawdown (%)': 'mean',
            'Num Trades': 'mean'
        }).round(2)
        
        print("\n" + summary.to_string())
        
        # Save combined results
        output_file = 'results/config_comparison.csv'
        combined_results.to_csv(output_file, index=False)
        print(f"\n✓ Combined results saved to: {output_file}")
        
        return combined_results
    else:
        print("\n✗ No results to compare")
        return None


if __name__ == '__main__':
    results = test_multiple_configs()
    
    if results is not None:
        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("="*80)
        print(f"Total runs: {len(results)}")
        print(f"Configurations tested: {results['Configuration'].nunique()}")
        print(f"Strategies tested: {results['Strategy'].nunique()}")
        print(f"Models tested: {results['Model'].nunique()}")

"""
HMM Impact Analysis on Trend-Following Strategies

Compares 3 HMM configurations against standalone alpha models to quantify
the impact of HMM regime detection on trend-following performance.

Focus metrics: Total Returns and Sharpe Ratio
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA
from signal_filter import HMMRegimeFilter
from statistics import Statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_hmm_impact_analysis(
    ticker: str = 'SPY',
    start_date: str = '2020-01-01',
    end_date: str = '2025-12-31',
    output_dir: str = None
):
    """
    Run comprehensive HMM impact analysis.
    
    Tests 3 HMM configurations:
    1. Baseline (504, 21) - Original parameters
    2. Optimal (252, 42) - Best total returns
    3. Accurate (756, 21) - Best Sharpe ratio
    
    For each alpha model, compares:
    - Alpha Only (baseline trend-following)
    - Alpha + HMM Combine (with regime detection)
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol to test
    start_date : str
        Start date for analysis
    end_date : str
        End date for analysis
    output_dir : str, optional
        Output directory (default: reports/hmm_impact_TIMESTAMP)
    
    Returns:
    --------
    tuple
        (summary_df, detailed_df, output_directory)
    """
    
    print("\n" + "="*80)
    print("HMM IMPACT ANALYSIS ON TREND-FOLLOWING STRATEGIES")
    print("="*80)
    print(f"Ticker: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"\nResearch Question: What is the impact of HMM regime detection")
    print(f"                   on trend-following strategy performance?")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('reports', f'hmm_impact_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # HMM configurations to test
    hmm_configs = [
        {'name': 'Baseline', 'train_window': 504, 'refit_every': 21, 'desc': 'Original parameters'},
        {'name': 'Optimal', 'train_window': 252, 'refit_every': 42, 'desc': 'Best total returns'},
        {'name': 'Accurate', 'train_window': 756, 'refit_every': 21, 'desc': 'Best Sharpe ratio'}
    ]
    
    # Alpha models to test
    alpha_models = [SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA]
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print("="*80)
    portfolio = Portfolio([ticker], start_date, end_date)
    portfolio.load_data()
    portfolio.summary()
    close = portfolio.get_close_prices(ticker)
    
    # Calculate benchmark
    print("\nCalculating Buy & Hold benchmark...")
    returns = close.pct_change().fillna(0)
    benchmark_metrics = Statistics.calculate_all_metrics(returns)
    print(f"  Total Return: {benchmark_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.2f}")
    
    # Storage for results
    detailed_results = []
    summary_results = []
    
    # Test each alpha model
    for model_class in alpha_models:
        model_name = model_class.__name__
        
        print(f"\n{'='*80}")
        print(f"TESTING: {model_name}")
        print("="*80)
        
        model = model_class(short_window=10, long_window=30)
        
        # Strategy 1: Alpha Only (baseline)
        print(f"\n[Baseline] Running {model_name} - Alpha Only...")
        engine_alpha = BacktestEngine(close, model)
        results_alpha = engine_alpha.run(
            strategy_mode='alpha_only',
            rebalance_frequency=1,
            transaction_cost=0.001
        )
        
        alpha_return = results_alpha['metrics']['total_return'] * 100
        alpha_sharpe = results_alpha['metrics']['sharpe_ratio']
        alpha_drawdown = results_alpha['metrics']['max_drawdown'] * 100
        alpha_trades = results_alpha['num_trades']
        
        print(f"  Return: {alpha_return:.2f}%, Sharpe: {alpha_sharpe:.3f}, DD: {alpha_drawdown:.2f}%")
        
        detailed_results.append({
            'Model': model_name,
            'Configuration': 'Alpha Only',
            'Strategy': 'Alpha Only',
            'Total Return (%)': alpha_return,
            'Sharpe Ratio': alpha_sharpe,
            'Max Drawdown (%)': alpha_drawdown,
            'Num Trades': alpha_trades,
            'HMM Params': 'N/A'
        })
        
        # Test each HMM configuration
        hmm_results_list = []
        
        for config in hmm_configs:
            config_name = config['name']
            train_window = config['train_window']
            refit_every = config['refit_every']
            
            print(f"\n[{config_name}] Running {model_name} - Alpha + HMM Combine...")
            print(f"  HMM params: train_window={train_window}, refit_every={refit_every}")
            
            hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
            engine_hmm = BacktestEngine(close, model, hmm_filter=hmm_filter)
            results_hmm = engine_hmm.run(
                strategy_mode='alpha_hmm_combine',
                walk_forward=True,
                train_window=train_window,
                refit_every=refit_every,
                bear_prob_threshold=0.65,
                bull_prob_threshold=0.65,
                rebalance_frequency=1,
                transaction_cost=0.001
            )
            
            hmm_return = results_hmm['metrics']['total_return'] * 100
            hmm_sharpe = results_hmm['metrics']['sharpe_ratio']
            hmm_drawdown = results_hmm['metrics']['max_drawdown'] * 100
            hmm_trades = results_hmm['num_trades']
            
            return_improvement = hmm_return - alpha_return
            sharpe_improvement = hmm_sharpe - alpha_sharpe
            
            print(f"  Return: {hmm_return:.2f}% ({return_improvement:+.2f}%)")
            print(f"  Sharpe: {hmm_sharpe:.3f} ({sharpe_improvement:+.3f})")
            
            detailed_results.append({
                'Model': model_name,
                'Configuration': config_name,
                'Strategy': 'Alpha + HMM Combine',
                'Total Return (%)': hmm_return,
                'Sharpe Ratio': hmm_sharpe,
                'Max Drawdown (%)': hmm_drawdown,
                'Num Trades': hmm_trades,
                'HMM Params': f"({train_window},{refit_every})"
            })
            
            hmm_results_list.append({
                'config': config_name,
                'return': hmm_return,
                'sharpe': hmm_sharpe,
                'return_improvement': return_improvement,
                'sharpe_improvement': sharpe_improvement
            })
        
        # Create summary for this model
        best_return_config = max(hmm_results_list, key=lambda x: x['return'])
        best_sharpe_config = max(hmm_results_list, key=lambda x: x['sharpe'])
        avg_return_improvement = np.mean([x['return_improvement'] for x in hmm_results_list])
        avg_sharpe_improvement = np.mean([x['sharpe_improvement'] for x in hmm_results_list])
        
        summary_results.append({
            'Model': model_name,
            'Alpha Only Return (%)': alpha_return,
            'Alpha Only Sharpe': alpha_sharpe,
            'Best HMM Return (%)': best_return_config['return'],
            'Best HMM Config (Return)': best_return_config['config'],
            'Return Improvement (%)': best_return_config['return_improvement'],
            'Best HMM Sharpe': best_sharpe_config['sharpe'],
            'Best HMM Config (Sharpe)': best_sharpe_config['config'],
            'Sharpe Improvement': best_sharpe_config['sharpe_improvement'],
            'Avg Return Improvement (%)': avg_return_improvement,
            'Avg Sharpe Improvement': avg_sharpe_improvement
        })
    
    # Create DataFrames
    detailed_df = pd.DataFrame(detailed_results)
    summary_df = pd.DataFrame(summary_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: HMM IMPACT ANALYSIS")
    print("="*80)
    print("\nBest Return Improvements:")
    print(summary_df.nlargest(5, 'Return Improvement (%)')[
        ['Model', 'Best HMM Config (Return)', 'Return Improvement (%)']
    ].to_string(index=False))
    
    print("\nBest Sharpe Improvements:")
    print(summary_df.nlargest(5, 'Sharpe Improvement')[
        ['Model', 'Best HMM Config (Sharpe)', 'Sharpe Improvement']
    ].to_string(index=False))
    
    print("\nAverage Impact Across All Configurations:")
    print(f"  Avg Return Improvement: {summary_df['Avg Return Improvement (%)'].mean():.2f}%")
    print(f"  Avg Sharpe Improvement: {summary_df['Avg Sharpe Improvement'].mean():.3f}")
    
    # Save results
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    summary_df.to_csv(os.path.join(output_dir, 'summary_results.csv'), index=False)
    
    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Plot 1: Return comparison by model and configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Returns comparison
    ax1 = axes[0, 0]
    pivot_return = detailed_df.pivot_table(
        values='Total Return (%)',
        index='Model',
        columns='Configuration',
        aggfunc='first'
    )
    pivot_return.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Total Returns: Alpha Only vs HMM Configurations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Return (%)', fontsize=12)
    ax1.set_xlabel('Alpha Model', fontsize=12)
    ax1.legend(title='Configuration', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=benchmark_metrics['total_return']*100, color='red', linestyle='--', 
                linewidth=2, label='Buy & Hold', alpha=0.7)
    
    # Sharpe comparison
    ax2 = axes[0, 1]
    pivot_sharpe = detailed_df.pivot_table(
        values='Sharpe Ratio',
        index='Model',
        columns='Configuration',
        aggfunc='first'
    )
    pivot_sharpe.plot(kind='bar', ax=ax2, width=0.8, colormap='viridis')
    ax2.set_title('Sharpe Ratio: Alpha Only vs HMM Configurations', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.set_xlabel('Alpha Model', fontsize=12)
    ax2.legend(title='Configuration', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=benchmark_metrics['sharpe_ratio'], color='red', linestyle='--',
                linewidth=2, label='Buy & Hold', alpha=0.7)
    
    # Return improvement by configuration
    ax3 = axes[1, 0]
    improvement_data = []
    for config in hmm_configs:
        config_name = config['name']
        config_data = detailed_df[detailed_df['Configuration'] == config_name]
        alpha_data = detailed_df[detailed_df['Configuration'] == 'Alpha Only']
        
        improvements = []
        for model in alpha_data['Model'].unique():
            alpha_return = alpha_data[alpha_data['Model'] == model]['Total Return (%)'].values[0]
            hmm_return = config_data[config_data['Model'] == model]['Total Return (%)'].values[0]
            improvements.append(hmm_return - alpha_return)
        
        improvement_data.append({
            'Configuration': config_name,
            'Avg Improvement': np.mean(improvements)
        })
    
    improvement_df = pd.DataFrame(improvement_data)
    improvement_df.plot(x='Configuration', y='Avg Improvement', kind='bar', ax=ax3, legend=False, color='steelblue')
    ax3.set_title('Average Return Improvement by HMM Configuration', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Return Improvement (%)', fontsize=12)
    ax3.set_xlabel('HMM Configuration', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Sharpe improvement by configuration
    ax4 = axes[1, 1]
    sharpe_improvement_data = []
    for config in hmm_configs:
        config_name = config['name']
        config_data = detailed_df[detailed_df['Configuration'] == config_name]
        alpha_data = detailed_df[detailed_df['Configuration'] == 'Alpha Only']
        
        improvements = []
        for model in alpha_data['Model'].unique():
            alpha_sharpe = alpha_data[alpha_data['Model'] == model]['Sharpe Ratio'].values[0]
            hmm_sharpe = config_data[config_data['Model'] == model]['Sharpe Ratio'].values[0]
            improvements.append(hmm_sharpe - alpha_sharpe)
        
        sharpe_improvement_data.append({
            'Configuration': config_name,
            'Avg Improvement': np.mean(improvements)
        })
    
    sharpe_improvement_df = pd.DataFrame(sharpe_improvement_data)
    sharpe_improvement_df.plot(x='Configuration', y='Avg Improvement', kind='bar', ax=ax4, legend=False, color='coral')
    ax4.set_title('Average Sharpe Improvement by HMM Configuration', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Sharpe Improvement', fontsize=12)
    ax4.set_xlabel('HMM Configuration', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_impact_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {os.path.join(output_dir, 'hmm_impact_analysis.png')}")
    
    # Generate markdown report
    report_file = os.path.join(output_dir, 'HMM_IMPACT_REPORT.md')
    with open(report_file, 'w') as f:
        f.write(f"# HMM Impact Analysis on Trend-Following Strategies\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Ticker:** {ticker}\n\n")
        f.write(f"**Period:** {start_date} to {end_date}\n\n")
        f.write(f"---\n\n")
        
        f.write(f"## Research Question\n\n")
        f.write(f"**What is the impact of HMM regime detection on trend-following strategy performance?**\n\n")
        f.write(f"This analysis compares standalone alpha model performance (pure trend-following) ")
        f.write(f"against alpha models enhanced with HMM regime detection (Alpha + HMM Combine strategy). ")
        f.write(f"Three HMM configurations are tested to assess robustness and identify optimal parameters.\n\n")
        f.write(f"---\n\n")
        
        f.write(f"## Benchmark Performance (Buy & Hold)\n\n")
        f.write(f"- **Total Return:** {benchmark_metrics['total_return']*100:.2f}%\n")
        f.write(f"- **Sharpe Ratio:** {benchmark_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- **Max Drawdown:** {benchmark_metrics['max_drawdown']*100:.2f}%\n\n")
        f.write(f"---\n\n")
        
        f.write(f"## HMM Configurations Tested\n\n")
        f.write(f"| Configuration | train_window | refit_every | Description |\n")
        f.write(f"|---------------|--------------|-------------|-------------|\n")
        for config in hmm_configs:
            f.write(f"| {config['name']} | {config['train_window']} | {config['refit_every']} | {config['desc']} |\n")
        f.write(f"\n---\n\n")
        
        f.write(f"## Executive Summary\n\n")
        avg_return_imp = summary_df['Avg Return Improvement (%)'].mean()
        avg_sharpe_imp = summary_df['Avg Sharpe Improvement'].mean()
        best_overall_return = summary_df['Return Improvement (%)'].max()
        best_overall_sharpe = summary_df['Sharpe Improvement'].max()
        models_improved_return = (summary_df['Avg Return Improvement (%)'] > 0).sum()
        models_improved_sharpe = (summary_df['Avg Sharpe Improvement'] > 0).sum()
        
        f.write(f"### Key Findings\n\n")
        f.write(f"1. **Average Return Improvement:** {avg_return_imp:+.2f}%\n")
        f.write(f"2. **Average Sharpe Improvement:** {avg_sharpe_imp:+.3f}\n")
        f.write(f"3. **Best Return Improvement:** {best_overall_return:+.2f}%\n")
        f.write(f"4. **Best Sharpe Improvement:** {best_overall_sharpe:+.3f}\n")
        f.write(f"5. **Models with Positive Return Impact:** {models_improved_return}/{len(summary_df)}\n")
        f.write(f"6. **Models with Positive Sharpe Impact:** {models_improved_sharpe}/{len(summary_df)}\n\n")
        
        if avg_return_imp > 0 and avg_sharpe_imp > 0:
            f.write(f"**Conclusion:** HMM regime detection provides **positive impact** on trend-following strategies, ")
            f.write(f"improving both returns and risk-adjusted performance on average.\n\n")
        elif avg_return_imp > 0:
            f.write(f"**Conclusion:** HMM regime detection provides **positive return impact** but mixed risk-adjusted results.\n\n")
        else:
            f.write(f"**Conclusion:** HMM regime detection shows **mixed results** - careful configuration selection is critical.\n\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Detailed Results by Alpha Model\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"### {row['Model']}\n\n")
            f.write(f"**Baseline (Alpha Only):**\n")
            f.write(f"- Total Return: {row['Alpha Only Return (%)']:.2f}%\n")
            f.write(f"- Sharpe Ratio: {row['Alpha Only Sharpe']:.3f}\n\n")
            
            f.write(f"**Best HMM Configuration (Return):**\n")
            f.write(f"- Configuration: {row['Best HMM Config (Return)']}\n")
            f.write(f"- Total Return: {row['Best HMM Return (%)']:.2f}%\n")
            f.write(f"- Improvement: {row['Return Improvement (%)']:+.2f}%\n\n")
            
            f.write(f"**Best HMM Configuration (Sharpe):**\n")
            f.write(f"- Configuration: {row['Best HMM Config (Sharpe)']}\n")
            f.write(f"- Sharpe Ratio: {row['Best HMM Sharpe']:.3f}\n")
            f.write(f"- Improvement: {row['Sharpe Improvement']:+.3f}\n\n")
            
            f.write(f"**Average Impact (all HMM configs):**\n")
            f.write(f"- Avg Return Improvement: {row['Avg Return Improvement (%)']:+.2f}%\n")
            f.write(f"- Avg Sharpe Improvement: {row['Avg Sharpe Improvement']:+.3f}\n\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Top Performers\n\n")
        f.write(f"### Best Return Improvements\n\n")
        f.write(f"| Model | Best Config | Return Improvement (%) |\n")
        f.write(f"|-------|-------------|------------------------|\n")
        for _, row in summary_df.nlargest(5, 'Return Improvement (%)').iterrows():
            f.write(f"| {row['Model']} | {row['Best HMM Config (Return)']} | {row['Return Improvement (%)']:+.2f}% |\n")
        f.write(f"\n")
        
        f.write(f"### Best Sharpe Improvements\n\n")
        f.write(f"| Model | Best Config | Sharpe Improvement |\n")
        f.write(f"|-------|-------------|--------------------|\n")
        for _, row in summary_df.nlargest(5, 'Sharpe Improvement').iterrows():
            f.write(f"| {row['Model']} | {row['Best HMM Config (Sharpe)']} | {row['Sharpe Improvement']:+.3f} |\n")
        f.write(f"\n---\n\n")
        
        f.write(f"## Configuration Comparison\n\n")
        for config in hmm_configs:
            config_name = config['name']
            config_data = detailed_df[detailed_df['Configuration'] == config_name]
            alpha_data = detailed_df[detailed_df['Configuration'] == 'Alpha Only']
            
            return_improvements = []
            sharpe_improvements = []
            
            for model in alpha_data['Model'].unique():
                alpha_return = alpha_data[alpha_data['Model'] == model]['Total Return (%)'].values[0]
                alpha_sharpe = alpha_data[alpha_data['Model'] == model]['Sharpe Ratio'].values[0]
                hmm_return = config_data[config_data['Model'] == model]['Total Return (%)'].values[0]
                hmm_sharpe = config_data[config_data['Model'] == model]['Sharpe Ratio'].values[0]
                
                return_improvements.append(hmm_return - alpha_return)
                sharpe_improvements.append(hmm_sharpe - alpha_sharpe)
            
            f.write(f"### {config_name} Configuration ({config['train_window']}, {config['refit_every']})\n\n")
            f.write(f"- **Description:** {config['desc']}\n")
            f.write(f"- **Avg Return Improvement:** {np.mean(return_improvements):+.2f}%\n")
            f.write(f"- **Avg Sharpe Improvement:** {np.mean(sharpe_improvements):+.3f}\n")
            f.write(f"- **Models Improved (Return):** {sum(1 for x in return_improvements if x > 0)}/{len(return_improvements)}\n")
            f.write(f"- **Models Improved (Sharpe):** {sum(1 for x in sharpe_improvements if x > 0)}/{len(sharpe_improvements)}\n\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Recommendations\n\n")
        best_config_return = summary_df.groupby('Best HMM Config (Return)').size().idxmax()
        best_config_sharpe = summary_df.groupby('Best HMM Config (Sharpe)').size().idxmax()
        
        f.write(f"1. **For Maximum Returns:** Use **{best_config_return}** configuration\n")
        f.write(f"   - Most frequently delivers best returns across alpha models\n\n")
        f.write(f"2. **For Risk-Adjusted Returns:** Use **{best_config_sharpe}** configuration\n")
        f.write(f"   - Most frequently delivers best Sharpe ratios across alpha models\n\n")
        f.write(f"3. **Model-Specific Optimization:** Consider configuration on per-model basis\n")
        f.write(f"   - Different alpha models may benefit from different HMM parameters\n\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Files Generated\n\n")
        f.write(f"- **Summary Results:** `summary_results.csv`\n")
        f.write(f"- **Detailed Results:** `detailed_results.csv`\n")
        f.write(f"- **Visualization:** `hmm_impact_analysis.png`\n")
        f.write(f"- **This Report:** `HMM_IMPACT_REPORT.md`\n")
    
    print(f"✓ Report saved to: {report_file}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"\nKey findings:")
    print(f"  Average Return Improvement: {avg_return_imp:+.2f}%")
    print(f"  Average Sharpe Improvement: {avg_sharpe_imp:+.3f}")
    print(f"  Best Return Improvement: {best_overall_return:+.2f}%")
    print(f"  Best Sharpe Improvement: {best_overall_sharpe:+.3f}")
    
    return summary_df, detailed_df, output_dir


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'SPY'
    start_date = sys.argv[2] if len(sys.argv) > 2 else '2020-01-01'
    end_date = sys.argv[3] if len(sys.argv) > 3 else '2025-12-31'
    
    # Run analysis
    summary, detailed, output_dir = run_hmm_impact_analysis(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )

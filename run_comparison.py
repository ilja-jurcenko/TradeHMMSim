"""
Main script to compare the impact of HMM model with AlphaModels.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA
from alpha_model_factory import AlphaModelFactory
from signal_filter import HMMRegimeFilter
from statistics import Statistics
from plotter import BacktestPlotter
from plotter import BacktestPlotter
from config_loader import ConfigLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt


def run_comparison(ticker: str = 'SPY', 
                   start_date: str = '2018-01-01',
                   end_date: str = '2024-12-31',
                   alpha_models: list = None,
                   short_window: int = 10,
                   long_window: int = 30,
                   rebalance_frequency: int = 1,
                   transaction_cost: float = 0.001,
                   output_dir: str = None,
                   save_plots: bool = False,
                   config_path: str = None,
                   train_window: int = 504,
                   refit_every: int = 21,
                   bear_prob_threshold: float = 0.65,
                   bull_prob_threshold: float = 0.65):
    """
    Run comprehensive comparison of AlphaModels with and without HMM filtering.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol to test
    start_date : str
        Start date for backtest
    end_date : str
        End date for backtest
    alpha_models : list
        List of alpha model classes to test (default: all)
    short_window : int
        Short MA window
    long_window : int
        Long MA window
    rebalance_frequency : int
        Rebalancing frequency in days
    transaction_cost : float
        Transaction cost per trade
    output_dir : str, optional
        Directory to save results. If None, creates timestamped directory
    save_plots : bool
        Whether to generate and save plots
    config_path : str, optional
        Path to configuration JSON file. If provided, overrides other parameters.
    train_window : int
        Training window for HMM walk-forward
    refit_every : int
        Refit HMM every N periods
    bear_prob_threshold : float
        Bear regime probability threshold
    bull_prob_threshold : float
        Bull regime probability threshold
        
    Returns:
    --------
    tuple
        (results_df, output_directory)
    """
    # Load configuration from file if provided
    if config_path is not None:
        print(f"\nLoading configuration from: {config_path}")
        config = ConfigLoader.load_config(config_path)
        ConfigLoader.print_config(config)
        
        # Override parameters with config values
        ticker = config.get('data', {}).get('ticker', ticker)
        start_date = config.get('data', {}).get('start_date', start_date)
        end_date = config.get('data', {}).get('end_date', end_date)
        
        # Check for alpha_models list (new format) or alpha_model (legacy format)
        if 'alpha_models' in config:
            # New format: list of alpha model configurations
            alpha_model_configs = config['alpha_models']
            alpha_models = []
            model_params = {}  # Store params for each model
            
            print(f"\nLoading {len(alpha_model_configs)} alpha model(s) from config...")
            for model_config in alpha_model_configs:
                model_type = model_config.get('type')
                params = model_config.get('parameters', {})
                model_class = AlphaModelFactory._MODELS.get(model_type)
                
                if model_class is None:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                alpha_models.append(model_class)
                model_params[model_type] = params
                print(f"  - {model_type}: short={params.get('short_window', short_window)}, long={params.get('long_window', long_window)}")
            
            # Store model params for later use
            config['_model_params'] = model_params
            
        elif 'alpha_model' in config:
            # Legacy format: single alpha model
            print("\nUsing legacy single alpha_model config format")
            short_window = config.get('alpha_model', {}).get('parameters', {}).get('short_window', short_window)
            long_window = config.get('alpha_model', {}).get('parameters', {}).get('long_window', long_window)
        
        rebalance_frequency = config.get('backtest', {}).get('rebalance_frequency', rebalance_frequency)
        transaction_cost = config.get('backtest', {}).get('transaction_cost', transaction_cost)
        save_plots = config.get('output', {}).get('save_plots', save_plots)
        output_dir_cfg = config.get('output', {}).get('output_dir')
        if output_dir_cfg is not None:
            output_dir = output_dir_cfg
        train_window = config.get('hmm', {}).get('train_window', train_window)
        refit_every = config.get('hmm', {}).get('refit_every', refit_every)
        bear_prob_threshold = config.get('hmm', {}).get('bear_prob_threshold', bear_prob_threshold)
        bull_prob_threshold = config.get('hmm', {}).get('bull_prob_threshold', bull_prob_threshold)
    else:
        config = None
    
    print("\n" + "="*80)
    print("BACKTESTING FRAMEWORK - ALPHA MODELS VS HMM COMPARISON")
    print("="*80)
    
    # Create output directory with timestamp in results folder
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('results', f'run_{timestamp}')
    else:
        # If custom dir specified, still put it under results/
        if not output_dir.startswith('results/'):
            output_dir = os.path.join('results', output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Create subdirectory for individual plots
    if save_plots:
        plots_dir = os.path.join(output_dir, 'individual_plots')
        os.makedirs(plots_dir, exist_ok=True)
    
    # Default models
    if alpha_models is None:
        alpha_models = [SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA]
    
    # Load data
    print(f"\nLoading data for {ticker}...")
    portfolio = Portfolio([ticker], start_date, end_date)
    portfolio.load_data()
    portfolio.summary()
    
    close = portfolio.get_close_prices(ticker)
    
    # Initialize HMM filter
    print("\nInitializing HMM regime filter...")
    hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
    
    # Storage for results
    results_list = []
    
    # Test each alpha model with different strategies
    for model_class in alpha_models:
        model_name = model_class.__name__
        print(f"\n{'='*80}")
        print(f"TESTING: {model_name}")
        print(f"{'='*80}")
        
        # Get model-specific params from config if available
        if config and '_model_params' in config and model_name in config['_model_params']:
            model_params = config['_model_params'][model_name]
            model_short = model_params.get('short_window', short_window)
            model_long = model_params.get('long_window', long_window)
            print(f"Using config params: short={model_short}, long={model_long}")
        else:
            model_short = short_window
            model_long = long_window
        
        model = model_class(short_window=model_short, long_window=model_long)
        
        # Strategy 1: Alpha only
        print(f"\n[1/4] Running {model_name} - Alpha Only...")
        engine_alpha = BacktestEngine(close, model)
        results_alpha = engine_alpha.run(
            strategy_mode='alpha_only',
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost
        )
        
        # Save individual plot
        if save_plots:
            plot_file = os.path.join(plots_dir, f'{model_name}_Alpha_Only.png')
            BacktestPlotter.plot_results(results_alpha, close, save_path=plot_file)
            plt.close('all')
        
        results_list.append({
            'Model': model_name,
            'Strategy': 'Alpha Only',
            'Total Return (%)': results_alpha['metrics']['total_return'] * 100,
            'Annual Return (%)': results_alpha['metrics']['annualized_return'] * 100,
            'Sharpe Ratio': results_alpha['metrics']['sharpe_ratio'],
            'Sortino Ratio': results_alpha['metrics']['sortino_ratio'],
            'Max Drawdown (%)': results_alpha['metrics']['max_drawdown'] * 100,
            'Profit Factor': results_alpha['metrics']['profit_factor'],
            'Win Rate (%)': results_alpha['metrics']['win_rate'] * 100,
            'Num Trades': results_alpha['num_trades'],
            'Time in Market (%)': results_alpha['time_in_market'] * 100
        })
        
        # Strategy 2: HMM only
        print(f"\n[2/4] Running {model_name} - HMM Only...")
        hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42)
        engine_hmm = BacktestEngine(close, model, hmm_filter=hmm_filter_new)
        results_hmm = engine_hmm.run(
            strategy_mode='hmm_only',
            walk_forward=True,
            train_window=train_window,
            refit_every=refit_every,
            bear_prob_threshold=bear_prob_threshold,
            bull_prob_threshold=bull_prob_threshold,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost
        )
        
        # Save individual plot
        if save_plots:
            plot_file = os.path.join(plots_dir, f'{model_name}_HMM_Only.png')
            BacktestPlotter.plot_results(results_hmm, close, save_path=plot_file)
            plt.close('all')
        
        results_list.append({
            'Model': model_name,
            'Strategy': 'HMM Only',
            'Total Return (%)': results_hmm['metrics']['total_return'] * 100,
            'Annual Return (%)': results_hmm['metrics']['annualized_return'] * 100,
            'Sharpe Ratio': results_hmm['metrics']['sharpe_ratio'],
            'Sortino Ratio': results_hmm['metrics']['sortino_ratio'],
            'Max Drawdown (%)': results_hmm['metrics']['max_drawdown'] * 100,
            'Profit Factor': results_hmm['metrics']['profit_factor'],
            'Win Rate (%)': results_hmm['metrics']['win_rate'] * 100,
            'Num Trades': results_hmm['num_trades'],
            'Time in Market (%)': results_hmm['time_in_market'] * 100
        })
        
        # Strategy 3: Alpha + HMM Filter
        print(f"\n[3/4] Running {model_name} - Alpha + HMM Filter...")
        hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42)
        engine_filter = BacktestEngine(close, model, hmm_filter=hmm_filter_new)
        results_filter = engine_filter.run(
            strategy_mode='alpha_hmm_filter',
            walk_forward=True,
            train_window=train_window,
            refit_every=refit_every,
            bear_prob_threshold=bear_prob_threshold,
            bull_prob_threshold=bull_prob_threshold,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost
        )
        
        # Save individual plot
        if save_plots:
            plot_file = os.path.join(plots_dir, f'{model_name}_Alpha_HMM_Filter.png')
            BacktestPlotter.plot_results(results_filter, close, save_path=plot_file)
            plt.close('all')
        
        results_list.append({
            'Model': model_name,
            'Strategy': 'Alpha + HMM Filter',
            'Total Return (%)': results_filter['metrics']['total_return'] * 100,
            'Annual Return (%)': results_filter['metrics']['annualized_return'] * 100,
            'Sharpe Ratio': results_filter['metrics']['sharpe_ratio'],
            'Sortino Ratio': results_filter['metrics']['sortino_ratio'],
            'Max Drawdown (%)': results_filter['metrics']['max_drawdown'] * 100,
            'Profit Factor': results_filter['metrics']['profit_factor'],
            'Win Rate (%)': results_filter['metrics']['win_rate'] * 100,
            'Num Trades': results_filter['num_trades'],
            'Time in Market (%)': results_filter['time_in_market'] * 100
        })
        
        # Strategy 4: Alpha + HMM Combine
        print(f"\n[4/4] Running {model_name} - Alpha + HMM Combine...")
        hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42)
        engine_combine = BacktestEngine(close, model, hmm_filter=hmm_filter_new)
        results_combine = engine_combine.run(
            strategy_mode='alpha_hmm_combine',
            walk_forward=True,
            train_window=train_window,
            refit_every=refit_every,
            bear_prob_threshold=bear_prob_threshold,
            bull_prob_threshold=bull_prob_threshold,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost
        )
        
        # Save individual plot
        if save_plots:
            plot_file = os.path.join(plots_dir, f'{model_name}_Alpha_HMM_Combine.png')
            BacktestPlotter.plot_results(results_combine, close, save_path=plot_file)
            plt.close('all')
        
        results_list.append({
            'Model': model_name,
            'Strategy': 'Alpha + HMM Combine',
            'Total Return (%)': results_combine['metrics']['total_return'] * 100,
            'Annual Return (%)': results_combine['metrics']['annualized_return'] * 100,
            'Sharpe Ratio': results_combine['metrics']['sharpe_ratio'],
            'Sortino Ratio': results_combine['metrics']['sortino_ratio'],
            'Max Drawdown (%)': results_combine['metrics']['max_drawdown'] * 100,
            'Profit Factor': results_combine['metrics']['profit_factor'],
            'Win Rate (%)': results_combine['metrics']['win_rate'] * 100,
            'Num Trades': results_combine['num_trades'],
            'Time in Market (%)': results_combine['time_in_market'] * 100
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Calculate benchmark (Buy & Hold)
    print("\n" + "="*80)
    print("CALCULATING BUY & HOLD BENCHMARK")
    print("="*80)
    returns = close.pct_change().fillna(0)
    benchmark_metrics = Statistics.calculate_all_metrics(returns)
    
    print("\nBuy & Hold Performance:")
    print(f"  Total Return: {benchmark_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {benchmark_metrics['max_drawdown']*100:.2f}%")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print("\nTop 10 Strategies by Total Return:")
    print(results_df.sort_values('Total Return (%)', ascending=False).head(10).to_string(index=False))
    
    print("\nTop 10 Strategies by Sharpe Ratio:")
    print(results_df.sort_values('Sharpe Ratio', ascending=False).head(10)[
        ['Model', 'Strategy', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
    ].to_string(index=False))
    
    # Calculate average performance by strategy type
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE BY STRATEGY TYPE")
    print("="*80)
    avg_by_strategy = results_df.groupby('Strategy').agg({
        'Total Return (%)': 'mean',
        'Sharpe Ratio': 'mean',
        'Max Drawdown (%)': 'mean',
        'Num Trades': 'mean',
        'Time in Market (%)': 'mean'
    }).round(2)
    print(avg_by_strategy.to_string())
    
    # Calculate HMM impact
    print("\n" + "="*80)
    print("HMM IMPACT ANALYSIS")
    print("="*80)
    
    for model_name in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model_name]
        
        alpha_only = model_data[model_data['Strategy'] == 'Alpha Only'].iloc[0]
        hmm_only = model_data[model_data['Strategy'] == 'HMM Only'].iloc[0]
        alpha_filter = model_data[model_data['Strategy'] == 'Alpha + HMM Filter'].iloc[0]
        alpha_combine = model_data[model_data['Strategy'] == 'Alpha + HMM Combine'].iloc[0]
        
        print(f"\n{model_name}:")
        print(f"  Alpha Only Return: {alpha_only['Total Return (%)']:.2f}%")
        print(f"  HMM Only Return: {hmm_only['Total Return (%)']:.2f}%")
        print(f"  Alpha + Filter Return: {alpha_filter['Total Return (%)']:.2f}%")
        print(f"  Alpha + Combine Return: {alpha_combine['Total Return (%)']:.2f}%")
        print(f"  HMM Filter Impact: {alpha_filter['Total Return (%)'] - alpha_only['Total Return (%)']:.2f}%")
        print(f"  HMM Combine Impact: {alpha_combine['Total Return (%)'] - alpha_only['Total Return (%)']:.2f}%")
    
    # Save results to output directory
    output_file = os.path.join(output_dir, f'comparison_{ticker}_{start_date}_{end_date}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Create markdown analysis report
    md_file = os.path.join(output_dir, 'ANALYSIS.md')
    with open(md_file, 'w') as f:
        f.write(f"# Backtest Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Ticker:** {ticker}\n\n")
        f.write(f"**Period:** {start_date} to {end_date}\n\n")
        f.write(f"---\n\n")
        
        # Configuration section
        f.write(f"## Configuration\n\n")
        if config_path:
            f.write(f"**Config File:** `{config_path}`\n\n")
        f.write(f"### Data Parameters\n")
        f.write(f"- **Ticker:** {ticker}\n")
        f.write(f"- **Start Date:** {start_date}\n")
        f.write(f"- **End Date:** {end_date}\n\n")
        
        f.write(f"### Alpha Model Parameters\n")
        f.write(f"- **Short Window:** {short_window}\n")
        f.write(f"- **Long Window:** {long_window}\n\n")
        
        f.write(f"### Backtest Parameters\n")
        f.write(f"- **Rebalance Frequency:** {rebalance_frequency} period(s)\n")
        f.write(f"- **Transaction Cost:** {transaction_cost*100:.3f}%\n\n")
        
        f.write(f"### HMM Parameters\n")
        f.write(f"- **Training Window:** {train_window} periods\n")
        f.write(f"- **Refit Every:** {refit_every} periods\n")
        f.write(f"- **Bear Probability Threshold:** {bear_prob_threshold}\n")
        f.write(f"- **Bull Probability Threshold:** {bull_prob_threshold}\n\n")
        
        f.write(f"### Output Settings\n")
        f.write(f"- **Save Plots:** {save_plots}\n")
        f.write(f"- **Output Directory:** `{output_dir}`\n\n")
        f.write(f"---\n\n")
        
        # Benchmark performance
        f.write(f"## Benchmark Performance (Buy & Hold)\n\n")
        f.write(f"- **Total Return:** {benchmark_metrics['total_return']*100:.2f}%\n")
        f.write(f"- **Sharpe Ratio:** {benchmark_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- **Max Drawdown:** {benchmark_metrics['max_drawdown']*100:.2f}%\n\n")
        f.write(f"---\n\n")
        
        # Top strategies
        f.write(f"## Top 10 Strategies by Total Return\n\n")
        f.write("| Model | Strategy | Total Return (%) | Sharpe Ratio | Max Drawdown (%) | Num Trades |\n")
        f.write("|-------|----------|------------------|--------------|------------------|------------|\n")
        for _, row in results_df.sort_values('Total Return (%)', ascending=False).head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['Total Return (%)']:.2f} | {row['Sharpe Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Num Trades']:.0f} |\n")
        f.write("\n")
        
        # Top by Sharpe
        f.write(f"## Top 10 Strategies by Sharpe Ratio\n\n")
        f.write("| Model | Strategy | Total Return (%) | Sharpe Ratio | Max Drawdown (%) | Num Trades |\n")
        f.write("|-------|----------|------------------|--------------|------------------|------------|\n")
        for _, row in results_df.sort_values('Sharpe Ratio', ascending=False).head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['Total Return (%)']:.2f} | {row['Sharpe Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Num Trades']:.0f} |\n")
        f.write("\n")
        
        # Average performance by strategy type
        f.write(f"## Average Performance by Strategy Type\n\n")
        f.write("| Strategy | Avg Total Return (%) | Avg Sharpe Ratio | Avg Max Drawdown (%) | Avg Num Trades | Avg Time in Market (%) |\n")
        f.write("|----------|----------------------|------------------|----------------------|----------------|------------------------|\n")
        for strategy, row in avg_by_strategy.iterrows():
            f.write(f"| {strategy} | {row['Total Return (%)']:.2f} | {row['Sharpe Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Num Trades']:.2f} | {row['Time in Market (%)']:.2f} |\n")
        f.write("\n")
        
        # HMM Impact Analysis
        f.write(f"## HMM Impact Analysis\n\n")
        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name]
            alpha_only = model_data[model_data['Strategy'] == 'Alpha Only'].iloc[0]
            hmm_only = model_data[model_data['Strategy'] == 'HMM Only'].iloc[0]
            alpha_filter = model_data[model_data['Strategy'] == 'Alpha + HMM Filter'].iloc[0]
            alpha_combine = model_data[model_data['Strategy'] == 'Alpha + HMM Combine'].iloc[0]
            
            f.write(f"### {model_name}\n\n")
            f.write(f"- **Alpha Only Return:** {alpha_only['Total Return (%)']:.2f}%\n")
            f.write(f"- **HMM Only Return:** {hmm_only['Total Return (%)']:.2f}%\n")
            f.write(f"- **Alpha + Filter Return:** {alpha_filter['Total Return (%)']:.2f}%\n")
            f.write(f"- **Alpha + Combine Return:** {alpha_combine['Total Return (%)']:.2f}%\n")
            f.write(f"- **HMM Filter Impact:** {alpha_filter['Total Return (%)'] - alpha_only['Total Return (%)']:.2f}%\n")
            f.write(f"- **HMM Combine Impact:** {alpha_combine['Total Return (%)'] - alpha_only['Total Return (%)']:.2f}%\n\n")
        
        # Files generated
        f.write(f"---\n\n")
        f.write(f"## Generated Files\n\n")
        f.write(f"- **CSV Results:** `comparison_{ticker}_{start_date}_{end_date}.csv`\n")
        if save_plots:
            f.write(f"- **Summary Plots:** `comparison_plots_{ticker}.png`\n")
            f.write(f"- **Individual Plots:** `individual_plots/` (28 plots)\n")
        f.write(f"\n")
    
    print(f"✓ Analysis report saved to {md_file}")
    
    # Generate and save plots if requested
    if save_plots:
        print("\nGenerating plots...")
        
        # Initialize plotter with dummy data (we'll rerun backtests for plotting)
        # Get top 3 strategies by Sharpe ratio
        top_strategies = results_df.nlargest(3, 'Sharpe Ratio')
        print(f"\nTop 3 strategies by Sharpe Ratio:")
        for idx, row in top_strategies.iterrows():
            print(f"  {row['Model']} - {row['Strategy']}: Sharpe={row['Sharpe Ratio']:.2f}, Return={row['Total Return (%)']:.2f}%")
        
        # Create comparison plot
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Returns comparison
        ax1 = plt.subplot(2, 2, 1)
        model_returns = results_df.groupby('Model')['Total Return (%)'].mean().sort_values(ascending=False)
        model_returns.plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Average Total Return by Model')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sharpe Ratio comparison
        ax2 = plt.subplot(2, 2, 2)
        model_sharpe = results_df.groupby('Model')['Sharpe Ratio'].mean().sort_values(ascending=False)
        model_sharpe.plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Average Sharpe Ratio by Model')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Max Drawdown comparison
        ax3 = plt.subplot(2, 2, 3)
        model_dd = results_df.groupby('Model')['Max Drawdown (%)'].mean().sort_values(ascending=True)
        model_dd.plot(kind='bar', ax=ax3, color='indianred')
        ax3.set_title('Average Max Drawdown by Model')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Strategy comparison (scatter)
        ax4 = plt.subplot(2, 2, 4)
        for strategy in results_df['Strategy'].unique():
            strategy_data = results_df[results_df['Strategy'] == strategy]
            ax4.scatter(strategy_data['Max Drawdown (%)'], strategy_data['Total Return (%)'], 
                       label=strategy, alpha=0.6, s=100)
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Total Return (%)')
        ax4.set_title('Risk-Return Profile by Strategy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'comparison_plots_{ticker}.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Summary plots saved to {plot_file}")
        
        # Count individual plots
        num_models = len(alpha_models)
        num_plots = num_models * 4  # 4 strategies per model
        print(f"✓ Individual plots saved to {os.path.join(output_dir, 'individual_plots/')}")
        print(f"  Total: {num_plots} plots ({num_models} models × 4 strategies)")
    
    return results_df, output_dir


if __name__ == '__main__':
    import sys
    
    # Check for config file argument first
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    
    # Parse command line arguments (config overrides these)
    ticker = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'SPY'
    start_date = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else '2018-01-01'
    end_date = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else '2024-12-31'
    show_plots = '--plot' in sys.argv or '-p' in sys.argv
    
    # Check for output directory argument
    output_dir = None
    save_plots_flag = False
    for i, arg in enumerate(sys.argv):
        if arg == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
        if arg == '--save-plots':
            save_plots_flag = True
    
    # Run comparison
    results, output_directory = run_comparison(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        short_window=10,
        long_window=30,
        rebalance_frequency=1,
        transaction_cost=0.001,
        output_dir=output_dir,
        save_plots=save_plots_flag or show_plots,
        config_path=config_path
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_directory}")
    print("="*80)


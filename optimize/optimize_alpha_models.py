"""
Alpha Model Parameter Optimization
Performs grid search optimization for all alpha models.
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_model_factory import AlphaModelFactory
from backtest import BacktestEngine
from statistics import Statistics


class AlphaModelOptimizer:
    """Optimize alpha model parameters using grid search."""
    
    def __init__(self, close: pd.Series, initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001):
        """
        Initialize optimizer.
        
        Parameters:
        -----------
        close : pd.Series
            Close price series
        initial_capital : float
            Initial capital
        transaction_cost : float
            Transaction cost per trade
        """
        self.close = close
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def optimize_model(self, model_type: str, 
                      short_window_min: int = 10,
                      short_window_max: int = 50,
                      long_window_min: int = 30,
                      long_window_max: int = 200,
                      optimize_for: str = 'profit_factor') -> Dict[str, Any]:
        """
        Optimize parameters for a single model using iterative search.
        Only keeps improving parameter combinations.
        
        Parameters:
        -----------
        model_type : str
            Type of alpha model (e.g., 'SMA', 'EMA')
        short_window_min : int
            Minimum short window value
        short_window_max : int
            Maximum short window value
        long_window_min : int
            Minimum long window value
        long_window_max : int
            Maximum long window value
        optimize_for : str
            Metric to optimize (default: 'profit_factor')
            
        Returns:
        --------
        Dict[str, Any]
            Optimization results including best parameters and all results
        """
        print(f"\n{'='*60}")
        print(f"Optimizing {model_type}")
        print(f"{'='*60}")
        print(f"Short window range: [{short_window_min}, {short_window_max}]")
        print(f"Long window range: [{long_window_min}, {long_window_max}]")
        print(f"Optimize for: {optimize_for}")
        
        results = []
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        # Iterative search - only remember improving runs
        combo_count = 0
        total_tested = 0
        
        for short_window in range(short_window_min, short_window_max + 1):
            for long_window in range(long_window_min, long_window_max + 1):
                combo_count += 1
                
                # Skip invalid combinations
                if short_window >= long_window:
                    continue
                
                total_tested += 1
                
                # Progress indicator every 20 combinations
                if total_tested % 20 == 0:
                    print(f"  Progress: Tested {total_tested} combinations, best {optimize_for}: {best_score:.4f}")
                
                try:
                    # Create alpha config
                    alpha_config = {
                        'type': model_type,
                        'parameters': {
                            'short_window': short_window,
                            'long_window': long_window
                        }
                    }
                    
                    # Run backtest
                    engine = BacktestEngine(
                        close=self.close,
                        alpha_config=alpha_config,
                        initial_capital=self.initial_capital
                    )
                    
                    # Suppress backtest output
                    import io
                    import contextlib
                    
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = engine.run(
                            strategy_mode='alpha_only',
                            transaction_cost=self.transaction_cost
                        )
                    
                    metrics = result['metrics']
                    
                    # Get current score
                    current_score = metrics[optimize_for]
                    
                    # Only remember if this improves upon previous best
                    if current_score > best_score:
                        best_score = current_score
                        best_params = (short_window, long_window)
                        
                        # Store result for improving runs only
                        result_dict = {
                            'short_window': short_window,
                            'long_window': long_window,
                            'total_return': metrics['total_return'],
                            'annualized_return': metrics['annualized_return'],
                            'sharpe_ratio': metrics['sharpe_ratio'],
                            'sortino_ratio': metrics['sortino_ratio'],
                            'max_drawdown': metrics['max_drawdown'],
                            'profit_factor': metrics['profit_factor'],
                            'win_rate': metrics['win_rate'],
                            'volatility': metrics['volatility'],
                            'calmar_ratio': metrics['calmar_ratio'],
                            'num_trades': result['num_trades'],
                            'time_in_market': result['time_in_market']
                        }
                        results.append(result_dict)
                        best_metrics = result_dict
                        
                        print(f"  ✓ New best: ({short_window}, {long_window}) -> {optimize_for}={current_score:.4f}")
                
                except Exception as e:
                    print(f"  Error with params ({short_window}, {long_window}): {str(e)}")
                    continue
        
        print(f"\nOptimization complete!")
        print(f"Total combinations tested: {total_tested}")
        print(f"Improving combinations found: {len(results)}")
        
        if best_params is None:
            print(f"WARNING: No valid parameter combinations found for {model_type}")
            # Return empty result
            return {
                'model_type': model_type,
                'best_params': None,
                'best_metrics': None,
                'all_results': results,
                'optimize_for': optimize_for,
                'total_tested': total_tested
            }
        
        print(f"Best parameters: short={best_params[0]}, long={best_params[1]}")
        print(f"Best {optimize_for}: {best_score:.4f}")
        
        return {
            'model_type': model_type,
            'best_params': best_params,
            'best_metrics': best_metrics,
            'all_results': results,
            'optimize_for': optimize_for,
            'total_tested': total_tested
        }
    
    def optimize_all_models(self, 
                           short_window_min: int = 10,
                           short_window_max: int = 50,
                           long_window_min: int = 30,
                           long_window_max: int = 200,
                           optimize_for: str = 'profit_factor') -> Dict[str, Any]:
        """
        Optimize all available alpha models using iterative search.
        
        Parameters:
        -----------
        short_window_min : int
            Minimum short window value
        short_window_max : int
            Maximum short window value
        long_window_min : int
            Minimum long window value
        long_window_max : int
            Maximum long window value
        optimize_for : str
            Metric to optimize (default: 'profit_factor')
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with results for each model
        """
        models = AlphaModelFactory.get_available_models()
        all_results = {}
        
        print(f"\n{'='*60}")
        print(f"ALPHA MODEL PARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Models to optimize: {', '.join(models)}")
        print(f"Optimization metric: {optimize_for}")
        print(f"Short window range: [{short_window_min}, {short_window_max}]")
        print(f"Long window range: [{long_window_min}, {long_window_max}]")
        print(f"Date range: {self.close.index[0].date()} to {self.close.index[-1].date()}")
        print(f"Number of periods: {len(self.close)}")
        
        for model_type in models:
            result = self.optimize_model(
                model_type=model_type,
                short_window_min=short_window_min,
                short_window_max=short_window_max,
                long_window_min=long_window_min,
                long_window_max=long_window_max,
                optimize_for=optimize_for
            )
            all_results[model_type] = result
        
        return all_results


def calculate_benchmark(close: pd.Series) -> Dict[str, float]:
    """
    Calculate buy-and-hold benchmark metrics.
    
    Parameters:
    -----------
    close : pd.Series
        Close price series
        
    Returns:
    --------
    Dict[str, float]
        Benchmark metrics
    """
    returns = close.pct_change().fillna(0).values  # Convert to numpy array
    metrics = Statistics.calculate_all_metrics(returns)
    return metrics


def generate_optimization_report(results: Dict[str, Any], 
                                 benchmark_metrics: Dict[str, float],
                                 ticker: str,
                                 start_date: str,
                                 end_date: str,
                                 optimize_for: str,
                                 output_path: str) -> None:
    """
    Generate markdown report for optimization results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Optimization results for all models
    benchmark_metrics : Dict[str, float]
        Benchmark (buy-and-hold) metrics
    ticker : str
        Ticker symbol
    start_date : str
        Start date
    end_date : str
        End date
    optimize_for : str
        Optimization metric used
    output_path : str
        Path to save the report
    """
    report = []
    
    # Header
    report.append("# Alpha Model Parameter Optimization Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Timeline
    report.append("## Timeline")
    report.append("")
    report.append(f"- **Ticker:** {ticker}")
    report.append(f"- **Start Date:** {start_date}")
    report.append(f"- **End Date:** {end_date}")
    report.append(f"- **Optimization Metric:** {optimize_for.replace('_', ' ').title()}")
    report.append("")
    
    # Benchmark
    report.append("## Benchmark Performance (Buy & Hold)")
    report.append("")
    report.append(f"- **Total Return:** {benchmark_metrics['total_return']*100:.2f}%")
    report.append(f"- **Annualized Return:** {benchmark_metrics['annualized_return']*100:.2f}%")
    report.append(f"- **Sharpe Ratio:** {benchmark_metrics['sharpe_ratio']:.3f}")
    report.append(f"- **Max Drawdown:** {benchmark_metrics['max_drawdown']*100:.2f}%")
    report.append(f"- **Profit Factor:** {benchmark_metrics['profit_factor']:.3f}")
    report.append(f"- **Volatility:** {benchmark_metrics['volatility']*100:.2f}%")
    report.append("")
    
    # Summary table
    report.append("## Optimization Results Summary")
    report.append("")
    report.append("| Model | Short | Long | Profit Factor | Total Return | Sharpe | Max DD | Trades | Tested |")
    report.append("|-------|-------|------|---------------|--------------|--------|--------|--------|--------|")
    
    for model_type, result in sorted(results.items()):
        best = result['best_metrics']
        total_tested = result.get('total_tested', 'N/A')
        if best is None:
            report.append(f"| {model_type:<5} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | {total_tested} |")
            continue
        report.append(
            f"| {model_type:<5} | {best['short_window']:>5} | {best['long_window']:>4} | "
            f"{best['profit_factor']:>13.3f} | {best['total_return']*100:>11.2f}% | "
            f"{best['sharpe_ratio']:>6.3f} | {best['max_drawdown']*100:>6.2f}% | "
            f"{best['num_trades']:>6} | {total_tested:>6} |"
        )
    
    report.append("")
    report.append("*Note: 'Tested' shows total parameter combinations tested for each model.*")
    report.append("")
    
    # Detailed results for each model
    report.append("## Detailed Model Results")
    report.append("")
    
    for model_type, result in sorted(results.items()):
        best = result['best_metrics']
        
        if best is None:
            report.append(f"### {model_type}")
            report.append("")
            report.append("**Status:** No valid parameter combinations found")
            report.append("")
            continue
        
        report.append(f"### {model_type}")
        report.append("")
        report.append("**Model Description:**")
        
        descriptions = {
            'SMA': 'Simple Moving Average - Traditional crossover strategy using arithmetic mean',
            'EMA': 'Exponential Moving Average - Weights recent prices more heavily',
            'WMA': 'Weighted Moving Average - Linear weighting favoring recent data',
            'HMA': 'Hull Moving Average - Reduces lag using weighted moving averages',
            'KAMA': "Kaufman's Adaptive Moving Average - Adjusts to market volatility",
            'TEMA': 'Triple Exponential Moving Average - Further reduces lag',
            'ZLEMA': 'Zero-Lag Exponential Moving Average - Minimizes lag in trend detection'
        }
        
        report.append(f"- {descriptions.get(model_type, 'N/A')}")
        report.append("")
        
        report.append("**Optimal Parameters:**")
        report.append(f"- Short Window: {best['short_window']}")
        report.append(f"- Long Window: {best['long_window']}")
        report.append(f"- Total Combinations Tested: {result.get('total_tested', 'N/A')}")
        report.append(f"- Improving Combinations Found: {len(result['all_results'])}")
        report.append("")
        
        report.append("**Performance Metrics:**")
        report.append(f"- Total Return: {best['total_return']*100:.2f}%")
        report.append(f"- Annualized Return: {best['annualized_return']*100:.2f}%")
        report.append(f"- Sharpe Ratio: {best['sharpe_ratio']:.3f}")
        report.append(f"- Sortino Ratio: {best['sortino_ratio']:.3f}")
        report.append(f"- Max Drawdown: {best['max_drawdown']*100:.2f}%")
        report.append(f"- Profit Factor: {best['profit_factor']:.3f}")
        report.append(f"- Win Rate: {best['win_rate']*100:.2f}%")
        report.append(f"- Volatility: {best['volatility']*100:.2f}%")
        report.append(f"- Calmar Ratio: {best['calmar_ratio']:.3f}")
        report.append(f"- Number of Trades: {best['num_trades']}")
        report.append(f"- Time in Market: {best['time_in_market']*100:.1f}%")
        report.append("")
        
        report.append("**Comparison to Benchmark:**")
        report.append(f"- Return Difference: {(best['total_return'] - benchmark_metrics['total_return'])*100:+.2f}%")
        report.append(f"- Sharpe Difference: {best['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']:+.3f}")
        report.append(f"- Profit Factor Difference: {best['profit_factor'] - benchmark_metrics['profit_factor']:+.3f}")
        report.append("")
    
    # Ranking
    report.append("## Model Rankings")
    report.append("")
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v['best_metrics'] is not None}
    
    if len(valid_results) == 0:
        report.append("No valid optimization results found.")
        report.append("")
    else:
        # Sort by total return
        report.append("### By Total Return")
        sorted_by_return = sorted(
            valid_results.items(), 
            key=lambda x: x[1]['best_metrics']['total_return'], 
            reverse=True
        )
        report.append("")
        for i, (model, result) in enumerate(sorted_by_return, 1):
            ret = result['best_metrics']['total_return']
            report.append(f"{i}. **{model}**: {ret*100:.2f}%")
        report.append("")
        
        # Sort by profit factor
        report.append("### By Profit Factor")
        sorted_by_pf = sorted(
            valid_results.items(),
            key=lambda x: x[1]['best_metrics']['profit_factor'],
            reverse=True
        )
        report.append("")
        for i, (model, result) in enumerate(sorted_by_pf, 1):
            pf = result['best_metrics']['profit_factor']
            report.append(f"{i}. **{model}**: {pf:.3f}")
        report.append("")
        
        # Sort by Sharpe ratio
        report.append("### By Sharpe Ratio")
        sorted_by_sharpe = sorted(
            valid_results.items(),
            key=lambda x: x[1]['best_metrics']['sharpe_ratio'],
            reverse=True
        )
        report.append("")
        for i, (model, result) in enumerate(sorted_by_sharpe, 1):
            sharpe = result['best_metrics']['sharpe_ratio']
            report.append(f"{i}. **{model}**: {sharpe:.3f}")
        report.append("")    # Conclusion
    report.append("## Conclusion")
    report.append("")
    
    if len(valid_results) == 0:
        report.append("No models completed optimization successfully. Please check data and parameters.")
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\nReport saved to: {output_path}")
        return
    
    # Find best models
    best_return_model = sorted_by_return[0][0]
    best_return_value = sorted_by_return[0][1]['best_metrics']['total_return']
    
    best_pf_model = sorted_by_pf[0][0]
    best_pf_value = sorted_by_pf[0][1]['best_metrics']['profit_factor']
    
    best_sharpe_model = sorted_by_sharpe[0][0]
    best_sharpe_value = sorted_by_sharpe[0][1]['best_metrics']['sharpe_ratio']
    
    # Count models beating benchmark
    models_beating_benchmark = sum(
        1 for result in valid_results.values()
        if result['best_metrics']['total_return'] > benchmark_metrics['total_return']
    )
    
    report.append(f"### Key Findings")
    report.append("")
    report.append(f"1. **Best Total Return:** {best_return_model} achieved {best_return_value*100:.2f}%, "
                 f"{'outperforming' if best_return_value > benchmark_metrics['total_return'] else 'underperforming'} "
                 f"the buy-and-hold benchmark by {(best_return_value - benchmark_metrics['total_return'])*100:+.2f}%")
    report.append("")
    report.append(f"2. **Best Profit Factor:** {best_pf_model} with {best_pf_value:.3f}, indicating "
                 f"{'strong' if best_pf_value > 1.5 else 'moderate' if best_pf_value > 1.0 else 'weak'} "
                 f"profitability per unit of risk")
    report.append("")
    report.append(f"3. **Best Risk-Adjusted Return:** {best_sharpe_model} with Sharpe ratio of {best_sharpe_value:.3f}")
    report.append("")
    report.append(f"4. **Benchmark Comparison:** {models_beating_benchmark} out of {len(valid_results)} models "
                 f"({models_beating_benchmark/len(valid_results)*100:.0f}%) outperformed the buy-and-hold strategy")
    report.append("")
    
    # Parameter insights
    avg_short = np.mean([r['best_metrics']['short_window'] for r in valid_results.values()])
    avg_long = np.mean([r['best_metrics']['long_window'] for r in valid_results.values()])
    
    report.append(f"5. **Parameter Patterns:** Average optimal short window was {avg_short:.0f} and "
                 f"long window was {avg_long:.0f}")
    report.append("")
    
    # Trading frequency
    avg_trades = np.mean([r['best_metrics']['num_trades'] for r in valid_results.values()])
    report.append(f"6. **Trading Frequency:** Models averaged {avg_trades:.0f} trades, with {best_return_model} "
                 f"executing {valid_results[best_return_model]['best_metrics']['num_trades']} trades")
    report.append("")
    
    report.append("### Recommendations")
    report.append("")
    
    if models_beating_benchmark > len(valid_results) / 2:
        report.append("- **Overall:** Trend-following strategies show promise, with majority outperforming buy-and-hold")
    else:
        report.append("- **Overall:** Buy-and-hold remains competitive; trend-following requires careful model selection")
    
    report.append(f"- **For Maximum Returns:** Use {best_return_model} with parameters "
                 f"({valid_results[best_return_model]['best_metrics']['short_window']}, "
                 f"{valid_results[best_return_model]['best_metrics']['long_window']})")
    
    report.append(f"- **For Risk-Adjusted Returns:** Use {best_sharpe_model} with parameters "
                 f"({valid_results[best_sharpe_model]['best_metrics']['short_window']}, "
                 f"{valid_results[best_sharpe_model]['best_metrics']['long_window']})")
    
    report.append(f"- **For Consistent Profitability:** Use {best_pf_model} with parameters "
                 f"({valid_results[best_pf_model]['best_metrics']['short_window']}, "
                 f"{valid_results[best_pf_model]['best_metrics']['long_window']})")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Note: Past performance does not guarantee future results. "
                 "These optimizations are based on historical data and should be validated "
                 "on out-of-sample periods before deployment.*")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to: {output_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize alpha model parameters')
    parser.add_argument('ticker', type=str, help='Ticker symbol (e.g., SPY)')
    parser.add_argument('start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--optimize-for', type=str, default='profit_factor',
                       choices=['total_return', 'profit_factor', 'sharpe_ratio'],
                       help='Metric to optimize (default: profit_factor)')
    parser.add_argument('--short-min', type=int, default=10,
                       help='Minimum short window (default: 10)')
    parser.add_argument('--short-max', type=int, default=50,
                       help='Maximum short window (default: 50)')
    parser.add_argument('--long-min', type=int, default=30,
                       help='Minimum long window (default: 30)')
    parser.add_argument('--long-max', type=int, default=200,
                       help='Maximum long window (default: 200)')
    parser.add_argument('--output-dir', type=str, default='optimize',
                       help='Output directory (default: optimize)')
    
    args = parser.parse_args()
    
    # Download data
    print(f"\nDownloading {args.ticker} data from {args.start_date} to {args.end_date}...")
    data = yf.download(args.ticker, start=args.start_date, end=args.end_date, progress=False)
    close = data['Close']
    
    # Ensure close is a Series (yfinance may return DataFrame for single ticker)
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    
    print(f"Downloaded {len(close)} data points")
    
    # Calculate benchmark
    print("\nCalculating benchmark metrics...")
    benchmark_metrics = calculate_benchmark(close)
    print(f"Benchmark Total Return: {benchmark_metrics['total_return']*100:.2f}%")
    print(f"Benchmark Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.3f}")
    
    # Run optimization
    optimizer = AlphaModelOptimizer(close, transaction_cost=0.001)
    results = optimizer.optimize_all_models(
        short_window_min=args.short_min,
        short_window_max=args.short_max,
        long_window_min=args.long_min,
        long_window_max=args.long_max,
        optimize_for=args.optimize_for
    )
    
    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.output_dir, f'optimization_report_{timestamp}.md')
    
    generate_optimization_report(
        results=results,
        benchmark_metrics=benchmark_metrics,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        optimize_for=args.optimize_for,
        output_path=output_path
    )
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

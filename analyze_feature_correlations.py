"""
Feature Correlation Analysis

Analyze correlations between HMM features to identify:
- Highly correlated features (multicollinearity)
- Redundant features
- Feature distributions and statistics
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib for heatmap")

from portfolio import Portfolio
from signal_filter import HMMRegimeFilter


def analyze_feature_correlations(ticker: str,
                                 start_date: str,
                                 end_date: str,
                                 short_vol_window: int = 10,
                                 long_vol_window: int = 80,
                                 short_ma_window: int = 10,
                                 long_ma_window: int = 80):
    """
    Analyze feature correlations and statistics.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    short_vol_window : int
        Short volatility window
    long_vol_window : int
        Long volatility window
    short_ma_window : int
        Short MA window
    long_ma_window : int
        Long MA window
    """
    print("\n" + "="*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*80)
    print(f"Ticker: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Volatility windows: {short_vol_window}, {long_vol_window}")
    print(f"MA windows: {short_ma_window}, {long_ma_window}")
    
    # Load data
    print("\nLoading data...")
    portfolio = Portfolio([ticker], start_date, end_date)
    portfolio.load_data()
    close = portfolio.get_close_prices(ticker)
    print(f"  Loaded {len(close)} data points")
    
    # Create features
    print("\nCreating features...")
    hmm_filter = HMMRegimeFilter(
        n_states=2,
        short_vol_window=short_vol_window,
        long_vol_window=long_vol_window,
        short_ma_window=short_ma_window,
        long_ma_window=long_ma_window
    )
    
    features = hmm_filter.make_features(close)
    print(f"  Created {len(features)} feature observations with {len(features.columns)} features")
    print(f"  Features: {', '.join(features.columns)}")
    
    # Basic statistics
    print("\n" + "="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    print("\nDescriptive Statistics:")
    print(features.describe())
    
    # Check for NaN/Inf
    print("\nData Quality Check:")
    nan_counts = features.isna().sum()
    inf_counts = np.isinf(features).sum()
    if nan_counts.sum() > 0:
        print(f"  WARNING: Found NaN values:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"    {col}: {nan_counts[col]} NaN values")
    else:
        print("  ✓ No NaN values found")
    
    if inf_counts.sum() > 0:
        print(f"  WARNING: Found Inf values:")
        for col in inf_counts[inf_counts > 0].index:
            print(f"    {col}: {inf_counts[col]} Inf values")
    else:
        print("  ✓ No Inf values found")
    
    # Correlation analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Compute correlation matrix
    corr_matrix = features.corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Find highly correlated pairs (>0.8 or <-0.8)
    print("\nHighly Correlated Feature Pairs (|r| > 0.8):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        print(f"  Found {len(high_corr_pairs)} highly correlated pairs:")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"    {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print("  No highly correlated pairs found")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', f'feature_analysis_{ticker}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Visualization 1: Correlation heatmap
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5, 
                    cbar_kws={"shrink": 0.8}, ax=ax,
                    vmin=-1, vmax=1)
    else:
        # Fallback to matplotlib
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'Feature Correlation Matrix - {ticker}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, f'correlation_heatmap_{ticker}.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation heatmap saved to: {heatmap_path}")
    
    # Visualization 2: Feature distributions
    n_features = len(features.columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(features.columns):
        ax = axes[idx]
        data = features[col].dropna()
        
        # Plot histogram
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{col}\nMean: {data.mean():.4f}, Std: {data.std():.4f}', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for mean and median
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1, label='Mean')
        ax.axvline(data.median(), color='blue', linestyle='--', linewidth=1, label='Median')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Feature Distributions - {ticker}', fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    dist_path = os.path.join(results_dir, f'feature_distributions_{ticker}.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature distributions saved to: {dist_path}")
    
    # Visualization 3: Feature time series
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(features.columns):
        ax = axes[idx]
        ax.plot(features.index, features[col], linewidth=0.5, alpha=0.7)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Feature Time Series - {ticker}', fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    timeseries_path = os.path.join(results_dir, f'feature_timeseries_{ticker}.png')
    plt.savefig(timeseries_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature time series saved to: {timeseries_path}")
    
    # Save correlation matrix to CSV
    corr_path = os.path.join(results_dir, f'correlation_matrix_{ticker}.csv')
    corr_matrix.to_csv(corr_path)
    print(f"✓ Correlation matrix saved to: {corr_path}")
    
    # Save feature statistics to CSV
    stats_path = os.path.join(results_dir, f'feature_statistics_{ticker}.csv')
    features.describe().to_csv(stats_path)
    print(f"✓ Feature statistics saved to: {stats_path}")
    
    # Save summary report
    summary_path = os.path.join(results_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE CORRELATION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Feature Configuration:\n")
        f.write(f"  Short volatility window: {short_vol_window}\n")
        f.write(f"  Long volatility window: {long_vol_window}\n")
        f.write(f"  Short MA window: {short_ma_window}\n")
        f.write(f"  Long MA window: {long_ma_window}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DATA SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total observations: {len(features)}\n")
        f.write(f"Number of features: {len(features.columns)}\n")
        f.write(f"Features: {', '.join(features.columns)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DATA QUALITY\n")
        f.write("="*80 + "\n")
        if nan_counts.sum() > 0:
            f.write("NaN values found:\n")
            for col in nan_counts[nan_counts > 0].index:
                f.write(f"  {col}: {nan_counts[col]}\n")
        else:
            f.write("No NaN values found\n")
        
        if inf_counts.sum() > 0:
            f.write("\nInf values found:\n")
            for col in inf_counts[inf_counts > 0].index:
                f.write(f"  {col}: {inf_counts[col]}\n")
        else:
            f.write("No Inf values found\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("HIGHLY CORRELATED PAIRS (|r| > 0.8)\n")
        f.write("="*80 + "\n")
        if high_corr_pairs:
            f.write(f"Found {len(high_corr_pairs)} highly correlated pairs:\n\n")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                f.write(f"  {feat1:20} <-> {feat2:20}: {corr:>6.3f}\n")
                
            f.write("\nRecommendations:\n")
            f.write("  - Consider removing one feature from each highly correlated pair\n")
            f.write("  - High correlation (>0.9) suggests redundancy\n")
            f.write("  - Moderate correlation (0.8-0.9) may still provide value\n")
        else:
            f.write("No highly correlated pairs found (all |r| <= 0.8)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FEATURE STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(features.describe().to_string())
    
    print(f"✓ Summary report saved to: {summary_path}")
    
    print(f"\n✓ Results directory: {results_dir}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'features': features,
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'results_dir': results_dir
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze HMM feature correlations and statistics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default parameters
  python analyze_feature_correlations.py SPY --start-date 2020-01-01 --end-date 2024-12-31
  
  # With custom windows
  python analyze_feature_correlations.py SPY --start-date 2020-01-01 --end-date 2024-12-31 \\
    --short-vol 10 --long-vol 80 --short-ma 10 --long-ma 80
        """
    )
    
    parser.add_argument('ticker', type=str, help='Ticker symbol to analyze')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--short-vol', type=int, default=10, help='Short volatility window (default: 10)')
    parser.add_argument('--long-vol', type=int, default=80, help='Long volatility window (default: 80)')
    parser.add_argument('--short-ma', type=int, default=10, help='Short MA window (default: 10)')
    parser.add_argument('--long-ma', type=int, default=80, help='Long MA window (default: 80)')
    
    args = parser.parse_args()
    
    results = analyze_feature_correlations(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        short_vol_window=args.short_vol,
        long_vol_window=args.long_vol,
        short_ma_window=args.short_ma,
        long_ma_window=args.long_ma
    )

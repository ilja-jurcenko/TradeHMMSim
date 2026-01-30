"""
Main script to compare the impact of HMM model with AlphaModels.
"""

import pandas as pd
import numpy as np
import os
import json
import shutil
from datetime import datetime
from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA, BollingerBands
from alpha_model_factory import AlphaModelFactory
from signal_filter import HMMRegimeFilter
from statistics import Statistics
from plotter import BacktestPlotter
from plotter import BacktestPlotter
from config_loader import ConfigLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt


def calculate_timing_metrics(strategy_positions: pd.Series, oracle_positions: pd.Series, 
                             close: pd.Series) -> dict:
    """
    Calculate timing quality metrics by comparing strategy signals to oracle signals.
    
    Parameters:
    -----------
    strategy_positions : pd.Series
        Strategy position series (0 or 1)
    oracle_positions : pd.Series
        Oracle position series (0 or 1)
    close : pd.Series
        Close price series
        
    Returns:
    --------
    dict
        Dictionary with timing metrics:
        - entry_timing_avg: Average bars early (negative) or late (positive) for entries
        - exit_timing_avg: Average bars early (negative) or late (positive) for exits
        - coverage_pct: % of oracle bull periods captured
        - false_entries: Number of entries during oracle bear periods
        - false_exits: Number of exits during oracle bull periods
        - entry_timing_std: Std dev of entry timing
        - exit_timing_std: Std dev of exit timing
    """
    # Align indices
    common_idx = strategy_positions.index.intersection(oracle_positions.index)
    strat_pos = strategy_positions.loc[common_idx]
    oracle_pos = oracle_positions.loc[common_idx]
    
    # Identify oracle entries (0 -> 1) and exits (1 -> 0)
    oracle_entries = []
    oracle_exits = []
    for i in range(1, len(oracle_pos)):
        if oracle_pos.iloc[i] == 1 and oracle_pos.iloc[i-1] == 0:
            oracle_entries.append(i)
        elif oracle_pos.iloc[i] == 0 and oracle_pos.iloc[i-1] == 1:
            oracle_exits.append(i)
    
    # Identify strategy entries and exits
    strategy_entries = []
    strategy_exits = []
    for i in range(1, len(strat_pos)):
        if strat_pos.iloc[i] == 1 and strat_pos.iloc[i-1] == 0:
            strategy_entries.append(i)
        elif strat_pos.iloc[i] == 0 and strat_pos.iloc[i-1] == 1:
            strategy_exits.append(i)
    
    # Calculate entry timing: for each oracle entry, find closest strategy entry
    entry_timings = []
    for oracle_entry_idx in oracle_entries:
        if not strategy_entries:
            continue
        # Find closest strategy entry
        distances = [s_idx - oracle_entry_idx for s_idx in strategy_entries]
        closest_idx = min(range(len(distances)), key=lambda i: abs(distances[i]))
        timing = distances[closest_idx]
        entry_timings.append(timing)
    
    # Calculate exit timing: for each oracle exit, find closest strategy exit
    exit_timings = []
    for oracle_exit_idx in oracle_exits:
        if not strategy_exits:
            continue
        # Find closest strategy exit
        distances = [s_idx - oracle_exit_idx for s_idx in strategy_exits]
        closest_idx = min(range(len(distances)), key=lambda i: abs(distances[i]))
        timing = distances[closest_idx]
        exit_timings.append(timing)
    
    # Calculate coverage: % of oracle bull periods where strategy was also long
    oracle_bull_periods = (oracle_pos == 1).sum()
    if oracle_bull_periods > 0:
        strategy_in_oracle_bull = ((oracle_pos == 1) & (strat_pos == 1)).sum()
        coverage_pct = (strategy_in_oracle_bull / oracle_bull_periods) * 100
    else:
        coverage_pct = 0.0
    
    # Calculate false entries: entries when oracle was in bear (0)
    false_entries = 0
    for s_idx in strategy_entries:
        if oracle_pos.iloc[s_idx] == 0:
            false_entries += 1
    
    # Calculate false exits: exits when oracle was in bull (1)
    false_exits = 0
    for s_idx in strategy_exits:
        if oracle_pos.iloc[s_idx] == 1:
            false_exits += 1
    
    return {
        'entry_timing_avg': np.mean(entry_timings) if entry_timings else 0.0,
        'entry_timing_std': np.std(entry_timings) if entry_timings else 0.0,
        'exit_timing_avg': np.mean(exit_timings) if exit_timings else 0.0,
        'exit_timing_std': np.std(exit_timings) if exit_timings else 0.0,
        'coverage_pct': coverage_pct,
        'false_entries': false_entries,
        'false_exits': false_exits,
        'oracle_entries': len(oracle_entries),
        'oracle_exits': len(oracle_exits),
        'strategy_entries': len(strategy_entries),
        'strategy_exits': len(strategy_exits)
    }


def run_comparison(ticker = None, 
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
                   bull_prob_threshold: float = 0.65,
                   use_regime_rebalancing: bool = True,
                   enable_logging: bool = False,
                   strategies: list = None):
    """
    Run comprehensive comparison of AlphaModels with and without HMM filtering.
    
    Parameters:
    -----------
    ticker : str or List[str]
        Ticker symbol(s) to test. Default is ['SPY', 'AGG'] for multi-asset regime-based allocation.
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
    use_regime_rebalancing : bool
        Whether to use regime-based portfolio rebalancing for multi-asset portfolios.
        When True with HMM strategies, automatically shifts between SPY (bull/neutral) and AGG (bear).
    enable_logging : bool
        Enable detailed CSV logging of trading decisions for each strategy
    strategies : list, optional
        List of strategy names to run. If None, runs all strategies.
        Valid values: ['alpha_only', 'hmm_only', 'oracle', 'alpha_hmm_filter', 
                       'alpha_hmm_combine', 'regime_adaptive_alpha', 'alpha_oracle']
        
    Returns:
    --------
    tuple
        (results_df, output_directory)
    """
    # Set default ticker to multi-asset portfolio if not provided
    if ticker is None:
        ticker = ['SPY', 'AGG']  # Default to two-asset portfolio with regime-based allocation
    
    # Ensure ticker is a list for consistent handling
    if isinstance(ticker, str):
        ticker = [ticker]
    
    # Load configuration from file if provided
    if config_path is not None:
        print(f"\nLoading configuration from: {config_path}")
        config = ConfigLoader.load_config(config_path)
        ConfigLoader.print_config(config)
        
        # Override parameters with config values (but command-line args take precedence)
        # Note: start_date and end_date should already be set correctly by caller if provided via CLI
        config_ticker = config.get('data', {}).get('ticker', ticker)
        if isinstance(config_ticker, str):
            ticker = [config_ticker]
        else:
            ticker = config_ticker
        
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
        use_regime_rebalancing = config.get('backtest', {}).get('use_regime_rebalancing', use_regime_rebalancing)
        save_plots = config.get('output', {}).get('save_plots', save_plots)
        output_dir_cfg = config.get('output', {}).get('output_dir')
        if output_dir_cfg is not None:
            output_dir = output_dir_cfg
        train_window = config.get('hmm', {}).get('train_window', train_window)
        refit_every = config.get('hmm', {}).get('refit_every', refit_every)
        short_vol_window = config.get('hmm', {}).get('short_vol_window', 10)
        long_vol_window = config.get('hmm', {}).get('long_vol_window', 30)
        short_ma_window = config.get('hmm', {}).get('short_ma_window', 10)
        long_ma_window = config.get('hmm', {}).get('long_ma_window', 30)
        covariance_type = config.get('hmm', {}).get('covariance_type', 'diag')
        n_iter = config.get('hmm', {}).get('n_iter', 100)
        tol = config.get('hmm', {}).get('tol', 1e-3)
        bear_prob_threshold = config.get('hmm', {}).get('bear_prob_threshold', bear_prob_threshold)
        bull_prob_threshold = config.get('hmm', {}).get('bull_prob_threshold', bull_prob_threshold)
    else:
        config = None
        # Set defaults for volatility and MA windows when no config provided
        short_vol_window = 10
        long_vol_window = 30
        short_ma_window = 10
        long_ma_window = 30
        covariance_type = 'diag'
        n_iter = 100
        tol = 1e-3
    
    print("\n" + "="*80)
    print("BACKTESTING FRAMEWORK - ALPHA MODELS VS HMM COMPARISON")
    print("="*80)
    
    # Store original user-specified dates
    user_start_date = start_date
    user_end_date = end_date
    
    # For HMM walk-forward testing, we need to load extra historical data
    # to ensure the first prediction can be made at user_start_date.
    # We need to account for:
    # 1. train_window: trading days needed for initial HMM training
    # 2. Feature windows: days needed for feature calculation (moving averages, volatility)
    # Since train_window is in trading days, we need to:
    # 1. Load data to find trading days
    # 2. Count back (train_window + max_feature_window) trading days from user_start_date
    # 3. Use that as data_load_start_date
    
    # Determine max feature window from config
    # Features use max(long_ma_window, long_vol_window) for longest lookback
    max_feature_window = max(long_ma_window, long_vol_window)
    total_lookback_days = train_window + max_feature_window
    
    print(f"\nCalculating data requirements:")
    print(f"  HMM train_window: {train_window} trading days")
    print(f"  Max feature window: {max_feature_window} trading days")
    print(f"  Total lookback needed: {total_lookback_days} trading days")
    
    # First, do a quick load to determine the correct data_load_start_date
    # We'll load a generous amount of historical data to find the right start
    from datetime import timedelta
    start_date_dt = pd.to_datetime(start_date)
    
    # Load extra calendar days (rough estimate: trading days × 1.5 for weekends/holidays)
    # Add extra buffer to be safe
    buffer_calendar_days = int(total_lookback_days * 1.6)
    temp_start = start_date_dt - timedelta(days=buffer_calendar_days)
    
    print(f"\nLoading data to calculate proper training window...")
    print(f"  Temporary load from: {temp_start.strftime('%Y-%m-%d')}")
    
    # Load data with buffer to find exact trading days
    from portfolio import Portfolio
    temp_portfolio = Portfolio(ticker, temp_start.strftime('%Y-%m-%d'), user_end_date)
    temp_portfolio.load_data()
    
    # Get close prices for the first ticker
    if isinstance(ticker, list):
        temp_close = temp_portfolio.get_close_prices(ticker[0])
    else:
        temp_close = temp_portfolio.get_close_prices(ticker)
    
    # Find the position of user_start_date in the data
    try:
        user_start_dt = pd.to_datetime(user_start_date)
        # Find the closest trading day on or after user_start_date
        valid_dates = temp_close.index[temp_close.index >= user_start_dt]
        if len(valid_dates) == 0:
            raise ValueError(f"No trading days found on or after {user_start_date}")
        actual_start_date = valid_dates[0]
        idx_start = temp_close.index.get_loc(actual_start_date)
        
        # Check if we have enough history (need train_window + max_feature_window)
        if idx_start < total_lookback_days:
            raise ValueError(
                f"Insufficient data: Need {total_lookback_days} trading days "
                f"({train_window} for HMM + {max_feature_window} for features) "
                f"before {user_start_date}, but only have {idx_start} trading days available. "
                f"Try using an earlier start date or loading more historical data."
            )
        
        # Calculate the date that is total_lookback_days trading days before user_start_date
        data_load_start_idx = idx_start - total_lookback_days
        data_load_start_date = temp_close.index[data_load_start_idx].strftime('%Y-%m-%d')
        extra_trading_days = idx_start - data_load_start_idx
        
        print(f"  Found {idx_start} trading days before {actual_start_date}")
        print(f"  Calculated data load start: {data_load_start_date} ({extra_trading_days} trading days before test start)")
        
    except Exception as e:
        print(f"  Warning: Could not calculate exact trading days: {e}")
        print(f"  Falling back to calendar day calculation")
        data_load_start_date = (start_date_dt - timedelta(days=total_lookback_days)).strftime('%Y-%m-%d')
        extra_trading_days = total_lookback_days
    
    print(f"\nDate range adjustment for HMM training:")
    print(f"  User-specified test period: {user_start_date} to {user_end_date}")
    print(f"  Data loading period: {data_load_start_date} to {user_end_date}")
    print(f"  Extra trading days for HMM training: {extra_trading_days}")
    
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
    
    # Save configuration files for reproducibility
    if config_path is not None:
        # Copy original config file
        config_filename = os.path.basename(config_path)
        config_copy_path = os.path.join(output_dir, f'original_{config_filename}')
        shutil.copy2(config_path, config_copy_path)
        print(f"✓ Original config saved to: {config_copy_path}")
    
    # Save runtime configuration (actual parameters used)
    runtime_config = {
        'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': {
            'ticker': ticker,
            'test_start_date': user_start_date,
            'test_end_date': user_end_date,
            'data_load_start_date': data_load_start_date,
            'data_load_end_date': user_end_date,
            'extra_training_days': extra_trading_days
        },
        'backtest': {
            'initial_capital': 100000.0,
            'rebalance_frequency': rebalance_frequency,
            'transaction_cost': transaction_cost,
            'use_regime_rebalancing': use_regime_rebalancing
        },
        'hmm': {
            'n_states': 3,
            'random_state': 42,
            'train_window': train_window,
            'refit_every': refit_every,
            'short_vol_window': short_vol_window,
            'long_vol_window': long_vol_window,
            'bear_prob_threshold': bear_prob_threshold,
            'bull_prob_threshold': bull_prob_threshold
        },
        'alpha_model': {
            'short_window': short_window,
            'long_window': long_window
        },
        'output': {
            'save_plots': save_plots,
            'output_dir': output_dir,
            'enable_logging': enable_logging
        },
        'strategies': strategies,
        'config_source': config_path if config_path else 'command_line_defaults'
    }
    
    # Add alpha models info if available
    if alpha_models is not None:
        runtime_config['alpha_models'] = [model.__name__ for model in alpha_models]
    
    runtime_config_path = os.path.join(output_dir, 'runtime_config.json')
    with open(runtime_config_path, 'w') as f:
        json.dump(runtime_config, f, indent=2)
    print(f"✓ Runtime config saved to: {runtime_config_path}")
    
    # Create subdirectory for logs if logging enabled
    log_dir = None
    if enable_logging:
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logging directory: {log_dir}")
    
    # Create subdirectory for individual plots
    if save_plots:
        plots_dir = os.path.join(output_dir, 'individual_plots')
        os.makedirs(plots_dir, exist_ok=True)
    
    # Default models
    if alpha_models is None:
        alpha_models = [SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA]
    
    # Default strategies
    if strategies is None:
        strategies = ['alpha_only', 'hmm_only', 'oracle', 'alpha_hmm_filter', 
                      'alpha_hmm_combine', 'regime_adaptive_alpha', 'alpha_oracle']
    
    # Validate strategies
    valid_strategies = ['alpha_only', 'hmm_only', 'oracle', 'alpha_hmm_filter', 
                        'alpha_hmm_combine', 'regime_adaptive_alpha', 'alpha_oracle']
    for strategy in strategies:
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Valid strategies: {valid_strategies}")
    
    # Determine if using multi-asset portfolio
    is_multi_asset = len(ticker) > 1
    ticker_str = ', '.join(ticker) if is_multi_asset else ticker[0]
    
    # Load data with adjusted start date for HMM training
    print(f"\nLoading data for {ticker_str}...")
    if is_multi_asset:
        print(f"Using multi-asset portfolio with regime-based rebalancing: {use_regime_rebalancing}")
    portfolio = Portfolio(ticker, data_load_start_date, user_end_date)
    portfolio.load_data()
    portfolio.summary()
    
    # Get close prices - use primary ticker (first in list) for alpha signal generation
    primary_ticker = ticker[0]
    close_full = portfolio.get_close_prices(primary_ticker)
    
    # Trim close prices to user-specified date range for final results
    # But keep full data for HMM training
    close_test_period = close_full[close_full.index >= user_start_date]
    print(f"\nData split:")
    print(f"  Full dataset for HMM training: {len(close_full)} days ({close_full.index[0]} to {close_full.index[-1]})")
    print(f"  Test period for backtest: {len(close_test_period)} days ({close_test_period.index[0]} to {close_test_period.index[-1]})")
    
    # Use full data for HMM training but will trim results later
    # Note: The HMM walkforward_filter will start predictions from train_window onwards,
    # which means predictions will start exactly at user_start_date since we loaded
    # train_window days of data before that date. This ensures:
    # 1. HMM has enough historical data for initial training
    # 2. Test period starts exactly on user-specified start_date
    # 3. No data leakage (only past data used for training at each point)
    close = close_full
    
    # Initialize HMM filter
    print("\nInitializing HMM regime filter...")
    hmm_filter = HMMRegimeFilter(n_states=3, random_state=42, 
                                 covariance_type=covariance_type,
                                 n_iter=n_iter,
                                 tol=tol,
                                 short_vol_window=short_vol_window,
                                 long_vol_window=long_vol_window,
                                 short_ma_window=short_ma_window,
                                 long_ma_window=long_ma_window)
    
    # Storage for results
    results_list = []
    
    # Step 1: Run Alpha Oracle FIRST to get reference signals for timing evaluation
    print("\n" + "="*80)
    print("STEP 1: COLLECTING ALPHA ORACLE REFERENCE SIGNALS")
    print("="*80)
    print("Running Oracle strategy to establish ideal timing baseline...")
    
    # Use first alpha model for oracle (doesn't matter which since oracle ignores alpha signals)
    oracle_model = alpha_models[0](short_window=short_window, long_window=long_window)
    oracle_engine = BacktestEngine(close, oracle_model)
    oracle_results = oracle_engine.run(
        strategy_mode='alpha_oracle',
        rebalance_frequency=rebalance_frequency,
        transaction_cost=transaction_cost,
        enable_logging=False
    )
    oracle_positions = oracle_results['positions']
    
    print(f"\n✓ Oracle baseline established:")
    print(f"  Total Return: {oracle_results['metrics']['total_return']*100:.2f}%")
    print(f"  Number of Trades: {oracle_results['num_trades']}")
    print(f"  Win Rate: {oracle_results['metrics']['win_rate']*100:.1f}%")
    
    # Step 2: Test each alpha model with different strategies
    print("\n" + "="*80)
    print("STEP 2: EVALUATING STRATEGIES AGAINST ORACLE BASELINE")
    print("="*80)
    
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
        if 'alpha_only' not in strategies:
            print(f"\n[1/6] Skipping {model_name} - Alpha Only...")
        else:
            print(f"\n[1/6] Running {model_name} - Alpha Only...")
            if is_multi_asset and use_regime_rebalancing:
                print("  Note: Multi-asset mode without regime rebalancing (static equal weights)")
            engine_alpha = BacktestEngine(close, model)
            results_alpha = engine_alpha.run(
                strategy_mode='alpha_only',
                rebalance_frequency=rebalance_frequency,
                transaction_cost=transaction_cost,
                enable_logging=enable_logging,
                log_dir=log_dir
            )
            
            # Save individual plot
            if save_plots:
                plot_file = os.path.join(plots_dir, f'{model_name}_Alpha_Only.png')
                BacktestPlotter.plot_results(results_alpha, close, save_path=plot_file)
                plt.close('all')
            
            # Calculate timing metrics vs oracle
            timing_metrics = calculate_timing_metrics(results_alpha['positions'], oracle_positions, close)
            
            results_list.append({
                'Model': model_name,
                'Strategy': 'Alpha Only',
                'Total Return (%)': results_alpha['metrics']['total_return'] * 100,
                'Annual Return (%)': results_alpha['metrics']['annualized_return'] * 100,
                'Sharpe Ratio': results_alpha['metrics']['sharpe_ratio'],
                'Sortino Ratio': results_alpha['metrics']['sortino_ratio'],
                'Calmar Ratio': results_alpha['metrics']['calmar_ratio'],
                'Max Drawdown (%)': results_alpha['metrics']['max_drawdown'] * 100,
                'Profit Factor': results_alpha['metrics']['profit_factor'],
                'Win Rate (%)': results_alpha['metrics']['win_rate'] * 100,
                'Num Trades': results_alpha['num_trades'],
                'Time in Market (%)': results_alpha['time_in_market'] * 100,
                'Entry Timing (bars)': timing_metrics['entry_timing_avg'],
                'Exit Timing (bars)': timing_metrics['exit_timing_avg'],
                'Coverage (%)': timing_metrics['coverage_pct'],
                'False Entries': timing_metrics['false_entries'],
                'False Exits': timing_metrics['false_exits']
            })
        
        # Strategy 2: HMM only
        if 'hmm_only' not in strategies:
            print(f"\n[2/6] Skipping {model_name} - HMM Only...")
        else:
            print(f"\n[2/6] Running {model_name} - HMM Only...")
            if is_multi_asset and use_regime_rebalancing:
                print("  Using regime-based rebalancing: Bull/Neutral → 100% SPY, Bear → 100% AGG")
            hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42,
                                             covariance_type=covariance_type,
                                             n_iter=n_iter,
                                             tol=tol,
                                             short_vol_window=short_vol_window,
                                             long_vol_window=long_vol_window,
                                             short_ma_window=short_ma_window,
                                             long_ma_window=long_ma_window)
            engine_hmm = BacktestEngine(close, model, hmm_filter=hmm_filter_new, 
                                        test_start_date=user_start_date)
            results_hmm = engine_hmm.run(
                strategy_mode='hmm_only',
                walk_forward=True,
                train_window=train_window,
                refit_every=refit_every,
                bear_prob_threshold=bear_prob_threshold,
                bull_prob_threshold=bull_prob_threshold,
                rebalance_frequency=rebalance_frequency,
                transaction_cost=transaction_cost,
                enable_logging=enable_logging,
                log_dir=log_dir
            )
            
            # Save individual plot
            if save_plots:
                plot_file = os.path.join(plots_dir, f'{model_name}_HMM_Only.png')
                BacktestPlotter.plot_results(results_hmm, close, save_path=plot_file)
                plt.close('all')
                
                # Save regime-colored equity plot
                plot_file_regime = os.path.join(plots_dir, f'{model_name}_HMM_Only_Regime_Colored.png')
                BacktestPlotter.plot_hmm_regime_colored_equity(
                    results_hmm, close, 
                    bear_threshold=bear_prob_threshold,
                    bull_threshold=bull_prob_threshold,
                    save_path=plot_file_regime
                )
                plt.close('all')
            
            # Calculate timing metrics vs oracle
            timing_metrics = calculate_timing_metrics(results_hmm['positions'], oracle_positions, close)
            
            results_list.append({
                'Model': model_name,
                'Strategy': 'HMM Only',
                'Total Return (%)': results_hmm['metrics']['total_return'] * 100,
                'Annual Return (%)': results_hmm['metrics']['annualized_return'] * 100,
                'Sharpe Ratio': results_hmm['metrics']['sharpe_ratio'],
                'Sortino Ratio': results_hmm['metrics']['sortino_ratio'],
                'Calmar Ratio': results_hmm['metrics']['calmar_ratio'],
                'Max Drawdown (%)': results_hmm['metrics']['max_drawdown'] * 100,
                'Profit Factor': results_hmm['metrics']['profit_factor'],
                'Win Rate (%)': results_hmm['metrics']['win_rate'] * 100,
                'Num Trades': results_hmm['num_trades'],
                'Time in Market (%)': results_hmm['time_in_market'] * 100,
                'Entry Timing (bars)': timing_metrics['entry_timing_avg'],
                'Exit Timing (bars)': timing_metrics['exit_timing_avg'],
                'Coverage (%)': timing_metrics['coverage_pct'],
                'False Entries': timing_metrics['false_entries'],
                'False Exits': timing_metrics['false_exits']
            })
        
        # Strategy 3: Oracle (HMM-Only with all data, no walk-forward)
        if 'oracle' not in strategies:
            print(f"\n[3/6] Skipping {model_name} - Oracle (HMM-Only)...")
        else:
            print(f"\n[3/6] Running {model_name} - Oracle (HMM-Only)...")
            print("  ⚠️  Oracle mode: Fits HMM on entire dataset (upper bound with future knowledge)")
            if is_multi_asset and use_regime_rebalancing:
                print("  Using regime-based rebalancing: Bull/Neutral → 100% SPY, Bear → 100% AGG")
            hmm_filter_oracle = HMMRegimeFilter(n_states=3, random_state=42,
                                                short_vol_window=short_vol_window,
                                                long_vol_window=long_vol_window,
                                                short_ma_window=short_ma_window,
                                                long_ma_window=long_ma_window,
                                                covariance_type=covariance_type,
                                                n_iter=n_iter,
                                                tol=tol)
            engine_oracle = BacktestEngine(close, model, hmm_filter=hmm_filter_oracle,
                                           test_start_date=user_start_date)
            results_oracle = engine_oracle.run(
                strategy_mode='oracle',
                bear_prob_threshold=bear_prob_threshold,
                bull_prob_threshold=bull_prob_threshold,
                rebalance_frequency=rebalance_frequency,
                transaction_cost=transaction_cost,
                enable_logging=enable_logging,
                log_dir=log_dir
            )
            
            # Save individual plot
            if save_plots:
                plot_file = os.path.join(plots_dir, f'{model_name}_Oracle.png')
                BacktestPlotter.plot_results(results_oracle, close, save_path=plot_file)
                plt.close('all')
                
                # Save regime-colored equity plot
                plot_file_regime = os.path.join(plots_dir, f'{model_name}_Oracle_Regime_Colored.png')
                BacktestPlotter.plot_hmm_regime_colored_equity(
                    results_oracle, close, 
                    bear_threshold=bear_prob_threshold,
                    bull_threshold=bull_prob_threshold,
                    save_path=plot_file_regime
                )
                plt.close('all')
            
            # Calculate timing metrics vs oracle
            timing_metrics = calculate_timing_metrics(results_oracle['positions'], oracle_positions, close)
            
            results_list.append({
                'Model': model_name,
                'Strategy': 'Oracle',
                'Total Return (%)': results_oracle['metrics']['total_return'] * 100,
                'Annual Return (%)': results_oracle['metrics']['annualized_return'] * 100,
                'Sharpe Ratio': results_oracle['metrics']['sharpe_ratio'],
                'Sortino Ratio': results_oracle['metrics']['sortino_ratio'],
                'Calmar Ratio': results_oracle['metrics']['calmar_ratio'],
                'Max Drawdown (%)': results_oracle['metrics']['max_drawdown'] * 100,
                'Profit Factor': results_oracle['metrics']['profit_factor'],
                'Win Rate (%)': results_oracle['metrics']['win_rate'] * 100,
                'Num Trades': results_oracle['num_trades'],
                'Time in Market (%)': results_oracle['time_in_market'] * 100,
                'Entry Timing (bars)': timing_metrics['entry_timing_avg'],
                'Exit Timing (bars)': timing_metrics['exit_timing_avg'],
                'Coverage (%)': timing_metrics['coverage_pct'],
                'False Entries': timing_metrics['false_entries'],
                'False Exits': timing_metrics['false_exits']
            })
        
        # Strategy 4: Alpha + HMM Filter
        if 'alpha_hmm_filter' not in strategies:
            print(f"\n[4/6] Skipping {model_name} - Alpha + HMM Filter...")
        else:
            print(f"\n[4/6] Running {model_name} - Alpha + HMM Filter...")
            if is_multi_asset and use_regime_rebalancing:
                print("  Using regime-based rebalancing: Bull/Neutral → 100% SPY, Bear → 100% AGG")
            hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42,
                                             covariance_type=covariance_type,
                                             n_iter=n_iter,
                                             tol=tol,
                                             short_vol_window=short_vol_window,
                                             long_vol_window=long_vol_window,
                                             short_ma_window=short_ma_window,
                                             long_ma_window=long_ma_window)
            engine_filter = BacktestEngine(close, model, hmm_filter=hmm_filter_new,
                                           test_start_date=user_start_date)
            results_filter = engine_filter.run(
                strategy_mode='alpha_hmm_filter',
                walk_forward=True,
                train_window=train_window,
                refit_every=refit_every,
                bear_prob_threshold=bear_prob_threshold,
                bull_prob_threshold=bull_prob_threshold,
                rebalance_frequency=rebalance_frequency,
                transaction_cost=transaction_cost,
                enable_logging=enable_logging,
                log_dir=log_dir
            )
            
            # Save individual plot
            if save_plots:
                plot_file = os.path.join(plots_dir, f'{model_name}_Alpha_HMM_Filter.png')
                BacktestPlotter.plot_results(results_filter, close, save_path=plot_file)
                plt.close('all')
            
            # Calculate timing metrics vs oracle
            timing_metrics = calculate_timing_metrics(results_filter['positions'], oracle_positions, close)
            
            results_list.append({
                'Model': model_name,
                'Strategy': 'Alpha + HMM Filter',
                'Total Return (%)': results_filter['metrics']['total_return'] * 100,
                'Annual Return (%)': results_filter['metrics']['annualized_return'] * 100,
                'Sharpe Ratio': results_filter['metrics']['sharpe_ratio'],
                'Sortino Ratio': results_filter['metrics']['sortino_ratio'],
                'Calmar Ratio': results_filter['metrics']['calmar_ratio'],
                'Max Drawdown (%)': results_filter['metrics']['max_drawdown'] * 100,
                'Profit Factor': results_filter['metrics']['profit_factor'],
                'Win Rate (%)': results_filter['metrics']['win_rate'] * 100,
                'Num Trades': results_filter['num_trades'],
                'Time in Market (%)': results_filter['time_in_market'] * 100,
                'Entry Timing (bars)': timing_metrics['entry_timing_avg'],
                'Exit Timing (bars)': timing_metrics['exit_timing_avg'],
                'Coverage (%)': timing_metrics['coverage_pct'],
                'False Entries': timing_metrics['false_entries'],
                'False Exits': timing_metrics['false_exits']
            })
        
        # Strategy 5: Alpha + HMM Combine
        if 'alpha_hmm_combine' not in strategies:
            print(f"\n[5/6] Skipping {model_name} - Alpha + HMM Combine...")
        else:
            print(f"\n[5/6] Running {model_name} - Alpha + HMM Combine...")
            if is_multi_asset and use_regime_rebalancing:
                print("  Using regime-based rebalancing: Bull/Neutral → 100% SPY, Bear → 100% AGG")
            hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42,
                                             covariance_type=covariance_type,
                                             n_iter=n_iter,
                                             tol=tol,
                                             short_vol_window=short_vol_window,
                                             long_vol_window=long_vol_window,
                                             short_ma_window=short_ma_window,
                                             long_ma_window=long_ma_window)
            engine_combine = BacktestEngine(close, model, hmm_filter=hmm_filter_new,
                                            test_start_date=user_start_date)
            results_combine = engine_combine.run(
                strategy_mode='alpha_hmm_combine',
                walk_forward=True,
                train_window=train_window,
                refit_every=refit_every,
                bear_prob_threshold=bear_prob_threshold,
                bull_prob_threshold=bull_prob_threshold,
                rebalance_frequency=rebalance_frequency,
                transaction_cost=transaction_cost,
                enable_logging=enable_logging,
                log_dir=log_dir
            )
            
            # Save individual plot
            if save_plots:
                plot_file = os.path.join(plots_dir, f'{model_name}_Alpha_HMM_Combine.png')
                BacktestPlotter.plot_results(results_combine, close, save_path=plot_file)
                plt.close('all')
                
                # Save 4-state strategy plot
                plot_file_4state = os.path.join(plots_dir, f'{model_name}_Alpha_HMM_Combine_4State.png')
                BacktestPlotter.plot_4state_strategy(results_combine, close, save_path=plot_file_4state)
                plt.close('all')
            
            # Calculate timing metrics vs oracle
            timing_metrics = calculate_timing_metrics(results_combine['positions'], oracle_positions, close)
            
            results_list.append({
                'Model': model_name,
                'Strategy': 'Alpha + HMM Combine',
                'Total Return (%)': results_combine['metrics']['total_return'] * 100,
                'Annual Return (%)': results_combine['metrics']['annualized_return'] * 100,
                'Sharpe Ratio': results_combine['metrics']['sharpe_ratio'],
                'Sortino Ratio': results_combine['metrics']['sortino_ratio'],
                'Calmar Ratio': results_combine['metrics']['calmar_ratio'],
                'Max Drawdown (%)': results_combine['metrics']['max_drawdown'] * 100,
                'Profit Factor': results_combine['metrics']['profit_factor'],
                'Win Rate (%)': results_combine['metrics']['win_rate'] * 100,
                'Num Trades': results_combine['num_trades'],
                'Time in Market (%)': results_combine['time_in_market'] * 100,
                'Entry Timing (bars)': timing_metrics['entry_timing_avg'],
                'Exit Timing (bars)': timing_metrics['exit_timing_avg'],
                'Coverage (%)': timing_metrics['coverage_pct'],
                'False Entries': timing_metrics['false_entries'],
                'False Exits': timing_metrics['false_exits']
            })
        
        # Strategy 6: Regime-Adaptive Alpha (Trend-following in bull/neutral, Bollinger Bands in bear)
        if 'regime_adaptive_alpha' not in strategies:
            print(f"\n[6/6] Skipping {model_name} - Regime-Adaptive Alpha...")
        else:
            print(f"\n[6/6] Running {model_name} - Regime-Adaptive Alpha...")
            print("  Bull/Neutral: Trend-following | Bear: Bollinger Bands mean-reversion")
            
            # Create Bollinger Bands model for bear markets
            bb_model = BollingerBands(short_window=20, long_window=2)
            
            hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42,
                                             covariance_type=covariance_type,
                                             n_iter=n_iter,
                                             tol=tol,
                                             short_vol_window=short_vol_window,
                                             long_vol_window=long_vol_window,
                                             short_ma_window=short_ma_window,
                                             long_ma_window=long_ma_window)
            engine_adaptive = BacktestEngine(close, model, hmm_filter=hmm_filter_new, 
                                            bear_alpha_model=bb_model,
                                            test_start_date=user_start_date)
            results_adaptive = engine_adaptive.run(
                strategy_mode='regime_adaptive_alpha',
                walk_forward=True,
                train_window=train_window,
                refit_every=refit_every,
                bear_prob_threshold=bear_prob_threshold,
                bull_prob_threshold=bull_prob_threshold,
                rebalance_frequency=rebalance_frequency,
                transaction_cost=transaction_cost,
                enable_logging=enable_logging,
                log_dir=log_dir
            )
            
            # Save individual plot
            if save_plots:
                plot_file = os.path.join(plots_dir, f'{model_name}_Regime_Adaptive.png')
                BacktestPlotter.plot_results(results_adaptive, close, save_path=plot_file)
                plt.close('all')
            
            # Calculate timing metrics vs oracle
            timing_metrics = calculate_timing_metrics(results_adaptive['positions'], oracle_positions, close)
            
            results_list.append({
                'Model': model_name,
                'Strategy': 'Regime-Adaptive Alpha',
                'Total Return (%)': results_adaptive['metrics']['total_return'] * 100,
                'Annual Return (%)': results_adaptive['metrics']['annualized_return'] * 100,
                'Sharpe Ratio': results_adaptive['metrics']['sharpe_ratio'],
                'Sortino Ratio': results_adaptive['metrics']['sortino_ratio'],
                'Calmar Ratio': results_adaptive['metrics']['calmar_ratio'],
                'Max Drawdown (%)': results_adaptive['metrics']['max_drawdown'] * 100,
                'Profit Factor': results_adaptive['metrics']['profit_factor'],
                'Win Rate (%)': results_adaptive['metrics']['win_rate'] * 100,
                'Num Trades': results_adaptive['num_trades'],
                'Time in Market (%)': results_adaptive['time_in_market'] * 100,
                'Entry Timing (bars)': timing_metrics['entry_timing_avg'],
                'Exit Timing (bars)': timing_metrics['exit_timing_avg'],
                'Coverage (%)': timing_metrics['coverage_pct'],
                'False Entries': timing_metrics['false_entries'],
                'False Exits': timing_metrics['false_exits']
            })
        
        # Strategy 7: Alpha Oracle (ZigZag Timing Labels)
        if 'alpha_oracle' not in strategies:
            print(f"\n[7/7] Skipping {model_name} - Alpha Oracle...")
        else:
            print(f"\n[7/7] Running {model_name} - Alpha Oracle (ZigZag Timing Labels)...")
            print("  Ideal timing benchmark based on local price extrema")
            
            engine_alpha_oracle = BacktestEngine(close, model)
            results_alpha_oracle = engine_alpha_oracle.run(
                strategy_mode='alpha_oracle',
                rebalance_frequency=rebalance_frequency,
                transaction_cost=transaction_cost,
                enable_logging=enable_logging,
                log_dir=log_dir
            )
            
            # Save individual plot
            if save_plots:
                plot_file = os.path.join(plots_dir, f'{model_name}_Alpha_Oracle.png')
                BacktestPlotter.plot_results(results_alpha_oracle, close, save_path=plot_file)
                plt.close('all')
            
            # Calculate timing metrics vs oracle (should be perfect)
            timing_metrics = calculate_timing_metrics(results_alpha_oracle['positions'], oracle_positions, close)
            
            results_list.append({
                'Model': model_name,
                'Strategy': 'Alpha Oracle',
                'Total Return (%)': results_alpha_oracle['metrics']['total_return'] * 100,
                'Annual Return (%)': results_alpha_oracle['metrics']['annualized_return'] * 100,
                'Sharpe Ratio': results_alpha_oracle['metrics']['sharpe_ratio'],
                'Sortino Ratio': results_alpha_oracle['metrics']['sortino_ratio'],
                'Calmar Ratio': results_alpha_oracle['metrics']['calmar_ratio'],
                'Max Drawdown (%)': results_alpha_oracle['metrics']['max_drawdown'] * 100,
                'Profit Factor': results_alpha_oracle['metrics']['profit_factor'],
                'Win Rate (%)': results_alpha_oracle['metrics']['win_rate'] * 100,
                'Num Trades': results_alpha_oracle['num_trades'],
                'Time in Market (%)': results_alpha_oracle['time_in_market'] * 100,
                'Entry Timing (bars)': timing_metrics['entry_timing_avg'],
                'Exit Timing (bars)': timing_metrics['exit_timing_avg'],
                'Coverage (%)': timing_metrics['coverage_pct'],
                'False Entries': timing_metrics['false_entries'],
                'False Exits': timing_metrics['false_exits']
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Calculate benchmark (Buy & Hold) on test period only
    print("\n" + "="*80)
    print("CALCULATING BUY & HOLD BENCHMARK")
    print("="*80)
    if is_multi_asset:
        # For multi-asset, use equal-weighted portfolio returns on test period
        print(f"Using equal-weighted portfolio: {ticker_str}")
        returns_full = portfolio.get_weighted_returns()
        # Trim to test period
        returns = returns_full[returns_full.index >= user_start_date]
    else:
        # Use test period close prices for benchmark
        returns = close_test_period.pct_change().fillna(0)
    benchmark_metrics = Statistics.calculate_all_metrics(returns)
    
    print("\nBuy & Hold Performance:")
    print(f"  Annual Return: {benchmark_metrics['annualized_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {benchmark_metrics['max_drawdown']*100:.2f}%")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Separate Oracle results from competitive strategies
    oracle_results = results_df[results_df['Strategy'] == 'Alpha Oracle']
    competitive_results = results_df[results_df['Strategy'] != 'Alpha Oracle']
    
    print("\nTop 10 Strategies by Total Return:")
    print(competitive_results.sort_values('Total Return (%)', ascending=False).head(10).to_string(index=False))
    
    print("\nTop 10 Strategies by Sharpe Ratio:")
    print(competitive_results.sort_values('Sharpe Ratio', ascending=False).head(10)[
        ['Model', 'Strategy', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
    ].to_string(index=False))
    
    print("\nTop 10 Strategies by Calmar Ratio:")
    print(competitive_results.sort_values('Calmar Ratio', ascending=False).head(10)[
        ['Model', 'Strategy', 'Total Return (%)', 'Calmar Ratio', 'Max Drawdown (%)']
    ].to_string(index=False))
    
    # Print timing quality analysis (exclude Oracle from comparisons)
    print("\n" + "="*80)
    print("TIMING QUALITY ANALYSIS (vs Alpha Oracle)")
    print("="*80)
    print("\nBest Entry Timing (closest to oracle entries, negative = early):")
    print(competitive_results.sort_values('Entry Timing (bars)', key=abs).head(10)[
        ['Model', 'Strategy', 'Entry Timing (bars)', 'Coverage (%)', 'False Entries']
    ].to_string(index=False))
    
    print("\nBest Exit Timing (closest to oracle exits, negative = early):")
    print(competitive_results.sort_values('Exit Timing (bars)', key=abs).head(10)[
        ['Model', 'Strategy', 'Exit Timing (bars)', 'False Exits']
    ].to_string(index=False))
    
    print("\nBest Coverage (% of oracle bull periods captured):")
    print(competitive_results.sort_values('Coverage (%)', ascending=False).head(10)[
        ['Model', 'Strategy', 'Coverage (%)', 'Entry Timing (bars)', 'Total Return (%)']
    ].to_string(index=False))
    
    print("\nFewest False Signals:")
    competitive_results_copy = competitive_results.copy()
    competitive_results_copy['Total False Signals'] = competitive_results_copy['False Entries'] + competitive_results_copy['False Exits']
    print(competitive_results_copy.sort_values('Total False Signals').head(10)[
        ['Model', 'Strategy', 'False Entries', 'False Exits', 'Total False Signals', 'Sharpe Ratio']
    ].to_string(index=False))
    
    # Calculate average performance by strategy type (exclude Oracle)
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE BY STRATEGY TYPE")
    print("="*80)
    avg_by_strategy = competitive_results.groupby('Strategy').agg({
        'Total Return (%)': 'mean',
        'Sharpe Ratio': 'mean',
        'Calmar Ratio': 'mean',
        'Max Drawdown (%)': 'mean',
        'Num Trades': 'mean',
        'Time in Market (%)': 'mean'
    }).round(2)
    print(avg_by_strategy.to_string())
    
    # Calculate HMM impact (only if required strategies exist)
    print("\n" + "="*80)
    print("HMM IMPACT ANALYSIS")
    print("="*80)
    
    # Check if we have the necessary strategies for impact analysis
    available_strategies = results_df['Strategy'].unique()
    required_strategies = ['Alpha Only', 'HMM Only', 'Alpha + HMM Filter', 'Alpha + HMM Combine']
    has_all_required = all(strategy in available_strategies for strategy in required_strategies)
    
    if not has_all_required:
        print(f"\nSkipping HMM Impact Analysis - requires all of: {', '.join(required_strategies)}")
        print(f"Available strategies: {', '.join(available_strategies)}")
    else:
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
    ticker_filename = '_'.join(ticker) if is_multi_asset else ticker[0]
    output_file = os.path.join(output_dir, f'comparison_{ticker_filename}_{user_start_date}_{user_end_date}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Create markdown analysis report
    md_file = os.path.join(output_dir, 'ANALYSIS.md')
    with open(md_file, 'w') as f:
        f.write(f"# Backtest Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if is_multi_asset:
            f.write(f"**Portfolio:** {ticker_str}\n\n")
            f.write(f"**Regime Rebalancing:** {use_regime_rebalancing}\n\n")
        else:
            f.write(f"**Ticker:** {ticker[0]}\n\n")
        f.write(f"**Test Period:** {user_start_date} to {user_end_date}\n\n")
        f.write(f"**Data Loaded:** {data_load_start_date} to {user_end_date} ({train_window} extra days for HMM training)\n\n")
        f.write(f"---\n\n")
        
        # Configuration section
        f.write(f"## Configuration\n\n")
        if config_path:
            f.write(f"**Config File:** `{config_path}`\n\n")
        f.write(f"### Data Parameters\n")
        if is_multi_asset:
            f.write(f"- **Portfolio:** {ticker_str}\n")
        else:
            f.write(f"- **Ticker:** {ticker[0]}\n")
        f.write(f"- **Test Start Date:** {user_start_date}\n")
        f.write(f"- **Test End Date:** {user_end_date}\n")
        f.write(f"- **Data Load Start:** {data_load_start_date} (includes {train_window} days for HMM training)\n\n")
        
        f.write(f"### Alpha Model Parameters\n")
        f.write(f"- **Short Window:** {short_window}\n")
        f.write(f"- **Long Window:** {long_window}\n\n")
        
        f.write(f"### Backtest Parameters\n")
        f.write(f"- **Rebalance Frequency:** {rebalance_frequency} period(s)\n")
        f.write(f"- **Transaction Cost:** {transaction_cost*100:.3f}%\n")
        if is_multi_asset:
            f.write(f"- **Regime-Based Rebalancing:** {use_regime_rebalancing}\n")
            if use_regime_rebalancing:
                f.write(f"  - Bull/Neutral → 100% {ticker[0]}, 0% {ticker[1]}\n")
                f.write(f"  - Bear → 0% {ticker[0]}, 100% {ticker[1]}\n")
        f.write(f"\n")
        
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
        f.write(f"- **Annual Return:** {benchmark_metrics['annualized_return']*100:.2f}%\n")
        f.write(f"- **Sharpe Ratio:** {benchmark_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- **Max Drawdown:** {benchmark_metrics['max_drawdown']*100:.2f}%\n\n")
        f.write(f"---\n\n")
        
        # Oracle reference performance
        if len(oracle_results) > 0:
            f.write(f"## Oracle Reference Performance\n\n")
            f.write(f"*Alpha Oracle represents ideal timing based on local price extrema (ZigZag). ")
            f.write(f"This is a theoretical benchmark, not a tradable strategy.*\n\n")
            
            # Show average oracle performance across models
            oracle_avg = oracle_results.agg({
                'Total Return (%)': 'mean',
                'Annual Return (%)': 'mean',
                'Sharpe Ratio': 'mean',
                'Max Drawdown (%)': 'mean',
                'Num Trades': 'mean',
                'Win Rate (%)': 'mean'
            })
            
            f.write(f"### Average Oracle Performance (across {len(oracle_results)} models)\n\n")
            f.write(f"- **Total Return:** {oracle_avg['Total Return (%)']:.2f}%\n")
            f.write(f"- **Annual Return:** {oracle_avg['Annual Return (%)']:.2f}%\n")
            f.write(f"- **Sharpe Ratio:** {oracle_avg['Sharpe Ratio']:.2f}\n")
            f.write(f"- **Max Drawdown:** {oracle_avg['Max Drawdown (%)']:.2f}%\n")
            f.write(f"- **Avg Trades:** {oracle_avg['Num Trades']:.0f}\n")
            f.write(f"- **Win Rate:** {oracle_avg['Win Rate (%)']:.1f}%\n\n")
            
            f.write(f"### Individual Oracle Results\n\n")
            f.write("| Model | Total Return (%) | Sharpe Ratio | Max Drawdown (%) | Num Trades | Win Rate (%) |\n")
            f.write("|-------|------------------|--------------|------------------|------------|--------------|\n")
            for _, row in oracle_results.iterrows():
                f.write(f"| {row['Model']} | {row['Total Return (%)']:.2f} | {row['Sharpe Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Num Trades']:.0f} | {row['Win Rate (%)']:.1f} |\n")
            f.write("\n")
        
        f.write(f"---\n\n")
        
        # Top strategies
        f.write(f"## Top 10 Strategies by Total Return\n\n")
        f.write("| Model | Strategy | Total Return (%) | Sharpe Ratio | Max Drawdown (%) | Num Trades |\n")
        f.write("|-------|----------|------------------|--------------|------------------|------------|\n")
        for _, row in competitive_results.sort_values('Total Return (%)', ascending=False).head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['Total Return (%)']:.2f} | {row['Sharpe Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Num Trades']:.0f} |\n")
        f.write("\n")
        
        # Top by Sharpe
        f.write(f"## Top 10 Strategies by Sharpe Ratio\n\n")
        f.write("| Model | Strategy | Total Return (%) | Sharpe Ratio | Max Drawdown (%) | Num Trades |\n")
        f.write("|-------|----------|------------------|--------------|------------------|------------|\n")
        for _, row in competitive_results.sort_values('Sharpe Ratio', ascending=False).head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['Total Return (%)']:.2f} | {row['Sharpe Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Num Trades']:.0f} |\n")
        f.write("\n")
        
        # Top by Calmar
        f.write(f"## Top 10 Strategies by Calmar Ratio\n\n")
        f.write("| Model | Strategy | Total Return (%) | Calmar Ratio | Max Drawdown (%) | Num Trades |\n")
        f.write("|-------|----------|------------------|--------------|------------------|------------|\n")
        for _, row in competitive_results.sort_values('Calmar Ratio', ascending=False).head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['Total Return (%)']:.2f} | {row['Calmar Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Num Trades']:.0f} |\n")
        f.write("\n")
        
        # Average performance by strategy type
        f.write(f"## Average Performance by Strategy Type\n\n")
        f.write("| Strategy | Avg Total Return (%) | Avg Sharpe Ratio | Avg Calmar Ratio | Avg Max Drawdown (%) | Avg Num Trades | Avg Time in Market (%) |\n")
        f.write("|----------|----------------------|------------------|------------------|----------------------|----------------|------------------------|\n")
        for strategy, row in avg_by_strategy.iterrows():
            f.write(f"| {strategy} | {row['Total Return (%)']:.2f} | {row['Sharpe Ratio']:.2f} | {row['Calmar Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Num Trades']:.2f} | {row['Time in Market (%)']:.2f} |\n")
        f.write("\n")
        
        # HMM Impact Analysis (only if we have required strategies)
        if has_all_required:
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
        else:
            f.write(f"## HMM Impact Analysis\n\n")
            f.write(f"*Skipped: Requires all strategies (Alpha Only, HMM Only, Alpha + HMM Filter, Alpha + HMM Combine)*\n\n")
        
        # Timing Quality Analysis
        f.write(f"---\n\n")
        f.write(f"## Timing Quality Analysis\n\n")
        f.write(f"This section compares competitive strategies against the Alpha Oracle baseline. ")
        f.write(f"For complete timing analysis of all strategies, see [TIMING_ANALYSIS.md](TIMING_ANALYSIS.md).\n\n")
        
        f.write(f"### Best Entry Timing\n\n")
        f.write("| Model | Strategy | Entry Timing (bars) | Coverage (%) | False Entries |\n")
        f.write("|-------|----------|---------------------|--------------|---------------|\n")
        for _, row in competitive_results.sort_values('Entry Timing (bars)', key=abs).head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['Entry Timing (bars)']:.1f} | {row['Coverage (%)']:.1f} | {row['False Entries']:.0f} |\n")
        f.write("\n")
        f.write("*Negative values = early entry, Positive values = late entry*\n\n")
        
        f.write(f"### Best Exit Timing\n\n")
        f.write("| Model | Strategy | Exit Timing (bars) | False Exits |\n")
        f.write("|-------|----------|-----------------------|-------------|\n")
        for _, row in competitive_results.sort_values('Exit Timing (bars)', key=abs).head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['Exit Timing (bars)']:.1f} | {row['False Exits']:.0f} |\n")
        f.write("\n")
        f.write("*Negative values = early exit, Positive values = late exit*\n\n")
        
        f.write(f"### Best Coverage\n\n")
        f.write("| Model | Strategy | Coverage (%) | Entry Timing (bars) | Total Return (%) |\n")
        f.write("|-------|----------|--------------|---------------------|------------------|\n")
        for _, row in competitive_results.sort_values('Coverage (%)', ascending=False).head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['Coverage (%)']:.1f} | {row['Entry Timing (bars)']:.1f} | {row['Total Return (%)']:.2f} |\n")
        f.write("\n")
        f.write("*Coverage = % of oracle bull periods where strategy was also long*\n\n")
        
        f.write(f"### Fewest False Signals\n\n")
        competitive_results_copy = competitive_results.copy()
        competitive_results_copy['Total False Signals'] = competitive_results_copy['False Entries'] + competitive_results_copy['False Exits']
        f.write("| Model | Strategy | False Entries | False Exits | Total False | Sharpe Ratio |\n")
        f.write("|-------|----------|---------------|-------------|-------------|---------------|\n")
        for _, row in competitive_results_copy.sort_values('Total False Signals').head(10).iterrows():
            f.write(f"| {row['Model']} | {row['Strategy']} | {row['False Entries']:.0f} | {row['False Exits']:.0f} | {row['Total False Signals']:.0f} | {row['Sharpe Ratio']:.2f} |\n")
        f.write("\n")
        f.write("*False Entries = entries during oracle bear periods | False Exits = exits during oracle bull periods*\n\n")
        
        # Files generated
        f.write(f"---\n\n")
        f.write(f"## Generated Files\n\n")
        f.write(f"- **CSV Results:** `comparison_{ticker_filename}_{user_start_date}_{user_end_date}.csv`\n")
        if save_plots:
            f.write(f"- **Summary Plots:** `comparison_plots_{ticker_filename}.png`\n")
            num_plots = len(alpha_models) * 5
            f.write(f"- **Individual Plots:** `individual_plots/` ({num_plots} plots)\n")
        f.write(f"\n")
    
    print(f"✓ Analysis report saved to {md_file}")
    
    # Create separate timing analysis report with ALL strategies
    timing_md_file = os.path.join(output_dir, 'TIMING_ANALYSIS.md')
    with open(timing_md_file, 'w') as f:
        f.write(f"# Complete Timing Quality Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Test Period:** {user_start_date} to {user_end_date}\n\n")
        f.write(f"---\n\n")
        
        f.write(f"## Overview\n\n")
        f.write(f"This report provides complete timing analysis for all tested strategies compared against ")
        f.write(f"the Alpha Oracle baseline. The Alpha Oracle identifies ideal entry/exit points based on ")
        f.write(f"local price extrema using a ZigZag algorithm (5% minimum move).\n\n")
        
        f.write(f"### Timing Metrics Explanation\n\n")
        f.write(f"- **Entry Timing (bars)**: Average bars early (-) or late (+) compared to oracle BUYs\n")
        f.write(f"- **Exit Timing (bars)**: Average bars early (-) or late (+) compared to oracle SELLs\n")
        f.write(f"- **Coverage (%)**: Percentage of oracle bull periods captured by strategy\n")
        f.write(f"- **False Entries**: Number of entries when oracle was in bear (cash) position\n")
        f.write(f"- **False Exits**: Number of exits when oracle was still in bull position\n\n")
        
        f.write(f"---\n\n")
        
        # Oracle baseline reference
        if len(oracle_results) > 0:
            f.write(f"## Oracle Baseline\n\n")
            f.write(f"*The Alpha Oracle represents theoretical perfect timing and is not included in competitive rankings.*\n\n")
            
            oracle_avg = oracle_results.agg({
                'Total Return (%)': 'mean',
                'Num Trades': 'mean',
                'Win Rate (%)': 'mean'
            })
            
            f.write(f"### Average Oracle Performance\n")
            f.write(f"- **Total Return:** {oracle_avg['Total Return (%)']:.2f}%\n")
            f.write(f"- **Avg Trades:** {oracle_avg['Num Trades']:.0f}\n")
            f.write(f"- **Win Rate:** {oracle_avg['Win Rate (%)']:.1f}%\n\n")
            f.write(f"---\n\n")
        
        # All strategies timing analysis
        f.write(f"## Complete Timing Analysis - All Strategies\n\n")
        f.write(f"Total strategies analyzed: {len(competitive_results)}\n\n")
        
        # Sort by multiple criteria
        f.write(f"### Sorted by Entry Timing (Absolute Value)\n\n")
        f.write("| Rank | Model | Strategy | Entry Timing | Coverage (%) | False Entries | Total Return (%) |\n")
        f.write("|------|-------|----------|--------------|--------------|---------------|------------------|\n")
        for rank, (_, row) in enumerate(competitive_results.sort_values('Entry Timing (bars)', key=abs).iterrows(), 1):
            f.write(f"| {rank} | {row['Model']} | {row['Strategy']} | {row['Entry Timing (bars)']:.1f} | ")
            f.write(f"{row['Coverage (%)']:.1f} | {row['False Entries']:.0f} | {row['Total Return (%)']:.2f} |\n")
        f.write("\n")
        
        f.write(f"### Sorted by Exit Timing (Absolute Value)\n\n")
        f.write("| Rank | Model | Strategy | Exit Timing | False Exits | Total Return (%) |\n")
        f.write("|------|-------|----------|-------------|-------------|------------------|\n")
        for rank, (_, row) in enumerate(competitive_results.sort_values('Exit Timing (bars)', key=abs).iterrows(), 1):
            f.write(f"| {rank} | {row['Model']} | {row['Strategy']} | {row['Exit Timing (bars)']:.1f} | ")
            f.write(f"{row['False Exits']:.0f} | {row['Total Return (%)']:.2f} |\n")
        f.write("\n")
        
        f.write(f"### Sorted by Coverage\n\n")
        f.write("| Rank | Model | Strategy | Coverage (%) | Entry Timing | Exit Timing | Total Return (%) |\n")
        f.write("|------|-------|----------|--------------|--------------|-------------|------------------|\n")
        for rank, (_, row) in enumerate(competitive_results.sort_values('Coverage (%)', ascending=False).iterrows(), 1):
            f.write(f"| {rank} | {row['Model']} | {row['Strategy']} | {row['Coverage (%)']:.1f} | ")
            f.write(f"{row['Entry Timing (bars)']:.1f} | {row['Exit Timing (bars)']:.1f} | {row['Total Return (%)']:.2f} |\n")
        f.write("\n")
        
        f.write(f"### Sorted by Total False Signals\n\n")
        competitive_with_false = competitive_results.copy()
        competitive_with_false['Total False'] = competitive_with_false['False Entries'] + competitive_with_false['False Exits']
        f.write("| Rank | Model | Strategy | False Entries | False Exits | Total False | Sharpe Ratio |\n")
        f.write("|------|-------|----------|---------------|-------------|-------------|--------------|\n")
        for rank, (_, row) in enumerate(competitive_with_false.sort_values('Total False').iterrows(), 1):
            f.write(f"| {rank} | {row['Model']} | {row['Strategy']} | {row['False Entries']:.0f} | ")
            f.write(f"{row['False Exits']:.0f} | {row['Total False']:.0f} | {row['Sharpe Ratio']:.2f} |\n")
        f.write("\n")
        
        f.write(f"---\n\n")
        
        # Strategy-by-strategy breakdown
        f.write(f"## Detailed Strategy Breakdown\n\n")
        
        for strategy_name in competitive_results['Strategy'].unique():
            strategy_data = competitive_results[competitive_results['Strategy'] == strategy_name]
            f.write(f"### {strategy_name}\n\n")
            f.write(f"Tested across {len(strategy_data)} alpha model(s)\n\n")
            
            # Calculate averages
            avg_entry = strategy_data['Entry Timing (bars)'].mean()
            avg_exit = strategy_data['Exit Timing (bars)'].mean()
            avg_coverage = strategy_data['Coverage (%)'].mean()
            avg_false_entries = strategy_data['False Entries'].mean()
            avg_false_exits = strategy_data['False Exits'].mean()
            
            f.write(f"**Average Metrics:**\n")
            f.write(f"- Entry Timing: {avg_entry:.1f} bars\n")
            f.write(f"- Exit Timing: {avg_exit:.1f} bars\n")
            f.write(f"- Coverage: {avg_coverage:.1f}%\n")
            f.write(f"- False Entries: {avg_false_entries:.1f}\n")
            f.write(f"- False Exits: {avg_false_exits:.1f}\n\n")
            
            f.write("| Model | Entry Timing | Exit Timing | Coverage (%) | False Entries | False Exits | Total Return (%) |\n")
            f.write("|-------|--------------|-------------|--------------|---------------|-------------|------------------|\n")
            for _, row in strategy_data.iterrows():
                f.write(f"| {row['Model']} | {row['Entry Timing (bars)']:.1f} | {row['Exit Timing (bars)']:.1f} | ")
                f.write(f"{row['Coverage (%)']:.1f} | {row['False Entries']:.0f} | {row['False Exits']:.0f} | ")
                f.write(f"{row['Total Return (%)']:.2f} |\n")
            f.write("\n")
        
        f.write(f"---\n\n")
        f.write(f"## Interpretation Guidelines\n\n")
        f.write(f"### Entry/Exit Timing\n")
        f.write(f"- **Negative values**: Early entries/exits (can be good or risky)\n")
        f.write(f"- **Positive values**: Late entries/exits (missed opportunities)\n")
        f.write(f"- **Target**: -2 to +2 bars (close to optimal)\n\n")
        
        f.write(f"### Coverage\n")
        f.write(f"- **>90%**: Excellent opportunity capture\n")
        f.write(f"- **80-90%**: Good participation\n")
        f.write(f"- **60-80%**: Moderate participation\n")
        f.write(f"- **<60%**: Significant opportunities missed\n\n")
        
        f.write(f"### False Signals\n")
        f.write(f"- **0**: Perfect discipline\n")
        f.write(f"- **1-3**: Excellent signal quality\n")
        f.write(f"- **4-10**: Acceptable noise level\n")
        f.write(f"- **>10**: Poor signal quality or overtrading\n")
    
    print(f"✓ Complete timing analysis saved to {timing_md_file}")
    
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
        plot_file = os.path.join(output_dir, f'comparison_plots_{ticker_filename}.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Summary plots saved to {plot_file}")
        
        # Count individual plots
        num_models = len(alpha_models)
        # 6 strategies + 1 4-state plot (combine) + 2 regime-colored plots (HMM Only + Oracle)
        num_plots = num_models * 9
        print(f"✓ Individual plots saved to {os.path.join(output_dir, 'individual_plots/')}")
        print(f"  Total: {num_plots} plots ({num_models} models × 6 strategies + {num_models} 4-state plots + {num_models * 2} regime-colored plots)")
    
    return results_df, output_dir


if __name__ == '__main__':
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Run comprehensive comparison of AlphaModels with and without HMM filtering.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default multi-asset portfolio (SPY, AGG)
  python run_comparison.py
  
  # Run with single ticker
  python run_comparison.py SPY
  
  # Run with multiple tickers
  python run_comparison.py SPY,AGG
  
  # Run with custom date range (overrides config)
  python run_comparison.py SPY --start-date 2020-01-01 --end-date 2025-12-31
  
  # Run with config file
  python run_comparison.py --config config_optimal.json
  
  # Command-line args override config file
  python run_comparison.py --config config_optimal.json --start-date 2022-01-01
  
  # Run with logging and plots
  python run_comparison.py SPY --enable-logging --save-plots
        """
    )
    
    # Positional argument
    parser.add_argument(
        'ticker',
        nargs='?',
        default=None,
        help='Ticker symbol(s) to test. Use comma-separated for multiple (e.g., SPY,AGG). Default: SPY,AGG'
    )
    
    # Date range arguments
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for backtest (YYYY-MM-DD). Overrides config file if provided.'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for backtest (YYYY-MM-DD). Overrides config file if provided.'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file. Command-line args override config values.'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save results. Default: results/run_<timestamp>'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Generate and save plots'
    )
    
    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help='Alias for --save-plots'
    )
    
    parser.add_argument(
        '--enable-logging', '--log',
        action='store_true',
        help='Enable detailed CSV logging of trading decisions'
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        default=None,
        help='Comma-separated list of strategies to run. Options: alpha_only, hmm_only, oracle, alpha_hmm_filter, alpha_hmm_combine, regime_adaptive_alpha. Default: all strategies'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process ticker argument
    ticker = None
    if args.ticker:
        ticker = args.ticker.split(',') if ',' in args.ticker else [args.ticker]
    
    # Determine final values (command-line args override config)
    # Start with defaults
    final_start_date = '2018-01-01'
    final_end_date = '2024-12-31'
    
    # If config provided, load it first
    if args.config:
        config = ConfigLoader.load_config(args.config)
        final_start_date = config.get('data', {}).get('start_date', final_start_date)
        final_end_date = config.get('data', {}).get('end_date', final_end_date)
    
    # Command-line args override config
    if args.start_date:
        final_start_date = args.start_date
    if args.end_date:
        final_end_date = args.end_date
    
    # Process strategies argument
    strategies_to_run = None
    if args.strategies:
        strategies_to_run = [s.strip() for s in args.strategies.split(',')]
    
    # Run comparison
    results, output_directory = run_comparison(
        ticker=ticker,
        start_date=final_start_date,
        end_date=final_end_date,
        short_window=10,
        long_window=30,
        rebalance_frequency=1,
        transaction_cost=0.001,
        output_dir=args.output_dir,
        save_plots=args.save_plots or args.plot,
        config_path=args.config,
        enable_logging=args.enable_logging,
        strategies=strategies_to_run
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_directory}")
    print("="*80)


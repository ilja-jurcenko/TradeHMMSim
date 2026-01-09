import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# ----------------------------
# Helpers: feature engineering
# ----------------------------
def make_features(close: pd.Series, vol_window: int = 20) -> pd.DataFrame:
    """
    Features for HMM emissions:
      - log return
      - realized volatility (rolling std of returns)
    """
    # Ensure close is a Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    r = np.log(close).diff()
    rv = r.rolling(vol_window).std()
    
    X = pd.DataFrame({
        'ret': r,
        'rv': rv
    }).dropna()
    return X

# --------------------------------------
# Online filtered probs (no lookahead)
# --------------------------------------
def filtered_state_probs(hmm: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute P(S_t = k | x_1:t) for each t (filtered probabilities).
    hmmlearn provides predict_proba which is typically *smoothed* (uses all data).
    So we compute filtered probs by iterating forward with the forward algorithm.

    Returns:
      gamma_filt: shape (T, n_states)
    """
    # Unpack HMM params
    startprob = hmm.startprob_
    transmat = hmm.transmat_

    # For GaussianHMM, we can compute emission likelihoods via score_samples
    # score_samples returns logprob and posterior (smoothed). We only use log likelihood per sample.
    # But hmmlearn doesn't directly expose per-time emission likelihoods.
    #
    # Workaround: use hmm._compute_log_likelihood(X) (private but stable in practice).
    logB = hmm._compute_log_likelihood(X)  # shape (T, n_states)

    T, K = logB.shape
    alpha = np.zeros((T, K))
    scale = np.zeros(T)

    # t=0
    alpha0 = startprob * np.exp(logB[0])
    scale[0] = alpha0.sum()
    alpha[0] = alpha0 / (scale[0] + 1e-300)

    # t>0
    for t in range(1, T):
        alpha_t = (alpha[t-1] @ transmat) * np.exp(logB[t])
        scale[t] = alpha_t.sum()
        alpha[t] = alpha_t / (scale[t] + 1e-300)

    # alpha[t] is proportional to filtered probabilities
    gamma_filt = alpha
    return gamma_filt

# ----------------------------------------------------
# Switch detection rules: threshold + confirmation
# ----------------------------------------------------
def detect_regime_switches(
    probs: pd.DataFrame,
    enter_th: float = 0.70,
    exit_th: float = 0.55,       # hysteresis: must drop below this to exit
    confirm_k: int = 2
) -> pd.Series:
    """
    Given filtered state probabilities (index=time, columns=states),
    produce a "live regime" series with:
      - entry threshold
      - exit threshold (hysteresis)
      - K-step confirmation (debounce)

    Returns:
      regime: pd.Series of active regime (int state id) at each time
    """
    states = probs.columns.to_list()
    idx = probs.index

    active = np.full(len(probs), fill_value=-1, dtype=int)
    current = int(probs.iloc[0].values.argmax())
    active[0] = current

    # confirmation buffer
    candidate = None
    streak = 0

    for t in range(1, len(probs)):
        p = probs.iloc[t]
        top_state = int(p.values.argmax())
        top_prob = float(p.iloc[top_state])

        cur_prob = float(p.iloc[current])

        # If current regime is still "confident enough", keep it
        # (hysteresis exit)
        if cur_prob >= exit_th:
            # consider switching only if another state is strongly dominant
            if top_state != current and top_prob >= enter_th:
                if candidate == top_state:
                    streak += 1
                else:
                    candidate = top_state
                    streak = 1

                if streak >= confirm_k:
                    current = candidate
                    candidate = None
                    streak = 0
            else:
                candidate = None
                streak = 0

        else:
            # current regime lost confidence; allow quicker switch if someone else is strong
            if top_state != current and top_prob >= enter_th:
                if candidate == top_state:
                    streak += 1
                else:
                    candidate = top_state
                    streak = 1
                if streak >= confirm_k:
                    current = candidate
                    candidate = None
                    streak = 0
            else:
                # fallback: pick the top state (but still you may require confirmation)
                current = top_state
                candidate = None
                streak = 0

        active[t] = current

    return pd.Series(active, index=idx, name="regime")

# ----------------------------
# Example end-to-end usage
# ----------------------------
def run_hmm_regime_detection(
    close: pd.Series,
    n_states: int = 3,
    train_window: int = 756,   # ~3 years daily
    vol_window: int = 20,
    refit_every: int = 21      # ~monthly
):
    """
    Walk-forward:
      - build features
      - refit HMM on rolling window every refit_every steps
      - compute filtered probs online
      - detect regime switches
    """
    feats = make_features(close, vol_window=vol_window)
    X_all = feats.values
    
    # Standardize features for better HMM convergence
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # We'll store filtered probabilities for each time step
    prob_list = []
    time_list = []

    hmm = None

    for t in range(train_window, len(feats)):
        # Refit periodically
        if hmm is None or ((t - train_window) % refit_every == 0):
            X_train = X_all[t - train_window : t]

            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",  # Use diagonal covariance for stability
                n_iter=100,
                tol=1e-4,
                random_state=42,
                verbose=False
            )
            hmm.fit(X_train)
            
            if (t - train_window) % (refit_every * 10) == 0:
                print(f"  Progress: {t}/{len(feats)} ({100*t/len(feats):.1f}%)")

        # Compute filtered probs using data up to t (inclusive)
        X_upto = X_all[: t + 1]
        gamma_filt = filtered_state_probs(hmm, X_upto)
        prob_t = gamma_filt[-1]  # only the last time point's filtered probs

        prob_list.append(prob_t)
        time_list.append(feats.index[t])

    probs = pd.DataFrame(prob_list, index=time_list, columns=list(range(n_states)))
    regime = detect_regime_switches(probs, enter_th=0.7, exit_th=0.55, confirm_k=2)

    # Switch events: times when regime changes
    switches = regime[regime.ne(regime.shift(1))].dropna()

    return probs, regime, switches


def download_spy_data(start_date='2005-01-01', end_date='2024-12-31'):
    """Download SPY data from Yahoo Finance."""
    print(f"Downloading SPY data from {start_date} to {end_date}...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    print(f"Downloaded {len(spy)} rows of data")
    return spy

def train_hmm_model(features, n_states=3):
    """Train Gaussian HMM model."""
    print(f"\nTraining HMM model with {n_states} states...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train HMM model
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    
    model.fit(features_scaled)
    
    # Predict hidden states
    hidden_states = model.predict(features_scaled)
    
    print("Model trained successfully")
    return model, hidden_states, scaler

def prepare_features(data):
    """Prepare features for HMM model."""
    # Calculate returns
    data['Returns'] = data['Close'].pct_change()
    
    # Calculate volatility (rolling std of returns)
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    
    # Drop NaN values
    data = data.dropna()
    
    # Create feature matrix
    features = data[['Returns', 'Volatility']].values
    #features = data[['Returns']].values
    return data, features

def calculate_moving_averages(close, short_window=50, long_window=200, ma_type='sma'):
    """Calculate short and long moving averages.
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    short_window : int
        Short window period
    long_window : int
        Long window period
    ma_type : str
        Type of moving average: 'sma' (Simple), 'ema' (Exponential), 'wma' (Weighted),
        'hma' (Hull), 'kama' (Kaufman's Adaptive), 'tema' (Triple Exponential), 
        or 'zlema' (Zero-Lag Exponential)
    
    Returns:
    --------
    ma_short, ma_long : pd.Series
        Short and long moving averages
    """
    if ma_type == 'sma':
        # Simple Moving Average
        ma_short = close.rolling(window=short_window).mean()
        ma_long = close.rolling(window=long_window).mean()
        
    elif ma_type == 'ema':
        # Exponential Moving Average
        ma_short = close.ewm(span=short_window, adjust=False).mean()
        ma_long = close.ewm(span=long_window, adjust=False).mean()
        
    elif ma_type == 'wma':
        # Weighted Moving Average
        def wma(series, window):
            weights = np.arange(1, window + 1)
            return series.rolling(window).apply(
                lambda x: np.dot(x, weights) / weights.sum() if len(x) == window else np.nan,
                raw=True
            )
        ma_short = wma(close, short_window)
        ma_long = wma(close, long_window)
        
    elif ma_type == 'hma':
        # Hull Moving Average
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def hma(series, window):
            half_length = int(window / 2)
            sqrt_length = int(np.sqrt(window))
            
            # WMA function
            def wma_calc(s, w):
                weights = np.arange(1, w + 1)
                return s.rolling(w).apply(
                    lambda x: np.dot(x, weights) / weights.sum() if len(x) == w else np.nan,
                    raw=True
                )
            
            wma_half = wma_calc(series, half_length)
            wma_full = wma_calc(series, window)
            raw_hma = 2 * wma_half - wma_full
            return wma_calc(raw_hma, sqrt_length)
        
        ma_short = hma(close, short_window)
        ma_long = hma(close, long_window)
        
    elif ma_type == 'kama':
        # Kaufman's Adaptive Moving Average
        def kama(series, window, fast_ema=2, slow_ema=30):
            # Efficiency Ratio
            change = abs(series - series.shift(window))
            volatility = series.diff().abs().rolling(window).sum()
            er = change / volatility
            er = er.fillna(0)
            
            # Smoothing constant
            fast_sc = 2 / (fast_ema + 1)
            slow_sc = 2 / (slow_ema + 1)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            # Calculate KAMA
            kama_vals = np.zeros(len(series))
            kama_vals[:window] = np.nan
            kama_vals[window] = series.iloc[window]
            
            for i in range(window + 1, len(series)):
                kama_vals[i] = kama_vals[i-1] + sc.iloc[i] * (series.iloc[i] - kama_vals[i-1])
            
            return pd.Series(kama_vals, index=series.index)
        
        ma_short = kama(close, short_window)
        ma_long = kama(close, long_window)
        
    elif ma_type == 'tema':
        # Triple Exponential Moving Average
        # TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
        def tema(series, window):
            ema1 = series.ewm(span=window, adjust=False).mean()
            ema2 = ema1.ewm(span=window, adjust=False).mean()
            ema3 = ema2.ewm(span=window, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        
        ma_short = tema(close, short_window)
        ma_long = tema(close, long_window)
        
    elif ma_type == 'zlema':
        # Zero-Lag Exponential Moving Average
        # ZLEMA = EMA(data + (data - data[lag]))
        def zlema(series, window):
            lag = int((window - 1) / 2)
            ema_data = series + (series - series.shift(lag))
            return ema_data.ewm(span=window, adjust=False).mean()
        
        ma_short = zlema(close, short_window)
        ma_long = zlema(close, long_window)
        
    else:
        raise ValueError(f"Unknown ma_type: {ma_type}. Must be 'sma', 'ema', 'wma', 'hma', 'kama', 'tema', or 'zlema'")
    
    return ma_short, ma_long


def backtest_regime_filtered_strategy(
    close,
    probs,
    regime,
    short_window=10,
    long_window=30,
    bear_regime_id=None,
    bear_prob_threshold=0.5,
    bull_regime_id=None,
    bull_prob_threshold=0.65,
    strategy_mode='sma_hmm_override',
    ma_type='sma'
):
    """
    Backtest a trend-following strategy filtered by HMM regime probabilities.
    
    Strategy Modes:
    ---------------
    1. 'sma_hmm_override': MA + HMM model - HMM can override MA signals
       Position = (MA_bullish AND NOT bear_regime) OR bull_regime_override
       
    2. 'sma_hmm_filter': HMM filters only incorrect bull MA signals when detecting bear
       Position = MA_bullish AND NOT bear_regime
       (HMM only prevents entries during bear regimes, no bull override)
       
    3. 'hmm_only': HMM only, ignoring MA completely
       Position = bull_regime (based on bull+neutral probability threshold)
       
    4. 'sma_only': Pure MA trend-following without any HMM filtering
       Position = MA_bullish (no regime filtering at all)
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    probs : pd.DataFrame
        Regime probabilities from HMM
    regime : pd.Series
        Active regime at each time point
    short_window : int
        Short SMA window
    long_window : int
        Long SMA window
    bear_regime_id : int or None
        Which regime ID represents bear market (if None, auto-detect as highest volatility)
    bear_prob_threshold : float
        Exit position if bear probability exceeds this
    bull_regime_id : int or None
        Which regime ID represents bull market (if None, auto-detect as lowest volatility)
    bull_prob_threshold : float
        Enter position if bull probability exceeds this (overrides SMA in mode 1, primary signal in mode 3)
    strategy_mode : str
        Strategy mode: 'sma_hmm_override', 'sma_hmm_filter', 'hmm_only', or 'sma_only'
    ma_type : str
        Moving average type: 'sma' (Simple), 'ema' (Exponential), or 'wma' (Weighted)
    
    Returns:
    --------
    results : dict
        Dictionary containing performance metrics and equity curves
    """
    # Align all data to same index
    common_index = regime.index.intersection(close.index)
    close_aligned = close.loc[common_index]
    probs_aligned = probs.loc[common_index]
    regime_aligned = regime.loc[common_index]
    
    # Calculate moving averages
    ma_short, ma_long = calculate_moving_averages(close_aligned, short_window, long_window, ma_type)
    
    # Auto-detect bear regime if not specified (highest volatility regime)
    if bear_regime_id is None:
        # Calculate volatility for each regime
        returns = close_aligned.pct_change()
        regime_vols = {}
        for state in sorted(regime_aligned.unique()):
            mask = regime_aligned == state
            state_returns = returns[mask]
            regime_vols[int(state)] = float(state_returns.std())
        bear_regime_id = max(regime_vols, key=regime_vols.get)
        print(f"Auto-detected bear regime: {bear_regime_id} (volatility: {regime_vols[bear_regime_id]:.4f})")
        print(f"All regime volatilities: {regime_vols}")
    
    # Auto-detect bull regime if not specified (lowest volatility regime)
    if bull_regime_id is None:
        # Calculate volatility for each regime if not already done
        if not 'regime_vols' in locals():
            returns = close_aligned.pct_change()
            regime_vols = {}
            for state in sorted(regime_aligned.unique()):
                mask = regime_aligned == state
                state_returns = returns[mask]
                regime_vols[int(state)] = float(state_returns.std())
        bull_regime_id = min(regime_vols, key=regime_vols.get)
        print(f"Auto-detected bull regime: {bull_regime_id} (volatility: {regime_vols[bull_regime_id]:.4f})")
    
    # Identify neutral regime (medium volatility - neither highest nor lowest)
    if not 'regime_vols' in locals():
        returns = close_aligned.pct_change()
        regime_vols = {}
        for state in sorted(regime_aligned.unique()):
            mask = regime_aligned == state
            state_returns = returns[mask]
            regime_vols[int(state)] = float(state_returns.std())
    
    # Find neutral regime (exclude bear and bull)
    all_regimes = set(regime_vols.keys())
    neutral_regime_id = list(all_regimes - {bear_regime_id, bull_regime_id})[0] if len(all_regimes) == 3 else None
    if neutral_regime_id is not None:
        print(f"Auto-detected neutral regime: {neutral_regime_id} (volatility: {regime_vols[neutral_regime_id]:.4f})")
    
    # Bear regime probability
    bear_prob = probs_aligned[bear_regime_id]
    if isinstance(bear_prob, pd.DataFrame):
        bear_prob = bear_prob.iloc[:, 0]  # Take first column if DataFrame
    
    # Bull regime probability (combine bull + neutral as bullish)
    bull_prob = probs_aligned[bull_regime_id]
    if isinstance(bull_prob, pd.DataFrame):
        bull_prob = bull_prob.iloc[:, 0]  # Take first column if DataFrame
    
    # Add neutral regime probability to bull probability
    if neutral_regime_id is not None:
        neutral_prob = probs_aligned[neutral_regime_id]
        if isinstance(neutral_prob, pd.DataFrame):
            neutral_prob = neutral_prob.iloc[:, 0]
        bull_prob = bull_prob + neutral_prob  # Combined bullish probability
        print(f"Treating neutral regime as bullish (combined bull+neutral probability)")
    
    # Generate signals
    # MA signal: 1 when short > long (bullish), 0 otherwise
    ma_bullish = (ma_short > ma_long).fillna(False)
    if isinstance(ma_bullish, pd.DataFrame):
        ma_bullish = ma_bullish.iloc[:, 0]
    
    # Bear regime filter: 1 when bear prob below threshold, 0 otherwise
    bear_filter = (bear_prob < bear_prob_threshold).fillna(False)
    if isinstance(bear_filter, pd.DataFrame):
        bear_filter = bear_filter.iloc[:, 0]
    
    # Bull regime filter: 1 when bull prob above threshold, 0 otherwise
    bull_filter = (bull_prob > bull_prob_threshold).fillna(False)
    if isinstance(bull_filter, pd.DataFrame):
        bull_filter = bull_filter.iloc[:, 0]
    
    # Find valid indices (where MAs are not NaN)
    valid_mask = ~ma_short.isna() & ~ma_long.isna()
    valid_idx = ma_short[valid_mask].index
    
    # Filter to common index with probs
    valid_idx = valid_idx.intersection(bear_prob.index).intersection(bull_prob.index)
    
    # Use numpy arrays for boolean operations
    ma_bull_vals = np.array(ma_bullish.loc[valid_idx], dtype=bool).ravel()
    bear_filt_vals = np.array(bear_filter.loc[valid_idx], dtype=bool).ravel()
    bull_filt_vals = np.array(bull_filter.loc[valid_idx], dtype=bool).ravel()
    
    # Strategy position logic based on mode
    print(f"\nStrategy Mode: {strategy_mode}")
    
    if strategy_mode == 'sma_hmm_override':
        # Mode 1: MA + HMM with override
        # Position = (MA_bullish AND NOT bear_regime) OR bull_regime_override
        print(f"  Logic: ({ma_type.upper()}_bullish AND NOT bear) OR bull_override")
        strategy_position_vals = ((ma_bull_vals & bear_filt_vals) | bull_filt_vals).astype(int)
        
    elif strategy_mode == 'sma_hmm_filter':
        # Mode 2: HMM only filters incorrect bull MA signals (bear filter only)
        # Position = MA_bullish AND NOT bear_regime
        print(f"  Logic: {ma_type.upper()}_bullish AND NOT bear (no bull override)")
        strategy_position_vals = (ma_bull_vals & bear_filt_vals).astype(int)
        
    elif strategy_mode == 'hmm_only':
        # Mode 3: HMM only, ignore MA completely
        # Position = bull_regime (based on bull probability threshold)
        print(f"  Logic: bull_regime only ({ma_type.upper()} ignored)")
        strategy_position_vals = bull_filt_vals.astype(int)
        
    elif strategy_mode == 'sma_only':
        # Mode 4: Pure MA trend-following, no HMM filtering
        # Position = MA_bullish (no regime considerations)
        print(f"  Logic: {ma_type.upper()}_bullish only (HMM ignored)")
        strategy_position_vals = ma_bull_vals.astype(int)
        
    else:
        raise ValueError(f"Unknown strategy_mode: {strategy_mode}. Must be 'sma_hmm_override', 'sma_hmm_filter', 'hmm_only', or 'sma_only'")
    
    strategy_position = pd.Series(strategy_position_vals, index=valid_idx, dtype=int)
    
    # Reindex to common_index with 0 fill
    strategy_position = strategy_position.reindex(common_index, fill_value=0)
    
    # Debug: print some statistics
    ma_bull_count = int(np.sum(ma_bull_vals))
    bear_filt_count = int(np.sum(bear_filt_vals))
    bull_filt_count = int(np.sum(bull_filt_vals))
    ma_and_bear = int(np.sum(ma_bull_vals & bear_filt_vals))
    strat_pos_count = int(np.sum(strategy_position_vals))
    print(f"{ma_type.upper()} bullish signals: {ma_bull_count} / {len(valid_idx)} ({100*ma_bull_count/len(valid_idx):.1f}%)")
    print(f"Bear filter allows: {bear_filt_count} / {len(valid_idx)} ({100*bear_filt_count/len(valid_idx):.1f}%)")
    print(f"Bull+Neutral filter overrides: {bull_filt_count} / {len(valid_idx)} ({100*bull_filt_count/len(valid_idx):.1f}%)")
    print(f"{ma_type.upper()} + Bear combined: {ma_and_bear} / {len(valid_idx)} ({100*ma_and_bear/len(valid_idx):.1f}%)")
    print(f"Strategy positions: {strat_pos_count} / {len(strategy_position)} ({100*strat_pos_count/len(strategy_position):.1f}%)")
    
    # Calculate returns
    returns = close_aligned.pct_change().fillna(0)
    
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # Align strategy_position with returns index
    strategy_position_aligned = strategy_position.reindex(returns.index, fill_value=0)
    
    # Strategy returns (only when in position)
    strategy_returns = strategy_position_aligned.shift(1).fillna(0) * returns
    
    # Debug: check for issues  
    print(f"Returns type: {type(returns)}, shape: {returns.shape if hasattr(returns, 'shape') else 'N/A'}")
    print(f"Strategy returns type: {type(strategy_returns)}, shape: {strategy_returns.shape if hasattr(strategy_returns, 'shape') else 'N/A'}")
    print(f"Strategy returns sample: {strategy_returns.head()}")
    
    # Buy and hold returns (benchmark)
    benchmark_returns = returns
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    # Calculate performance metrics
    def calculate_metrics(returns_series):
        # Convert to numpy if needed
        if isinstance(returns_series, pd.Series):
            returns_arr = returns_series.values
        else:
            returns_arr = np.array(returns_series)
        
        total_return = float(np.prod(1 + returns_arr) - 1)
        annual_return = float((1 + total_return) ** (252 / len(returns_arr)) - 1)
        volatility = float(np.std(returns_arr) * np.sqrt(252))
        sharpe_ratio = float(annual_return / volatility) if volatility > 0 else 0.0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown))
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    strategy_metrics = calculate_metrics(strategy_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    
    # Calculate additional statistics
    num_trades = int(np.sum(np.abs(np.diff(strategy_position.values))) / 2)
    time_in_market = float(np.sum(strategy_position.values > 0) / len(strategy_position))
    
    results = {
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics,
        'strategy_cumulative': strategy_cumulative,
        'benchmark_cumulative': benchmark_cumulative,
        'strategy_position': strategy_position,
        'bear_prob': bear_prob,
        'bull_prob': bull_prob,
        'ma_short': ma_short,
        'ma_long': ma_long,
        'num_trades': num_trades,
        'time_in_market': time_in_market,
        'bear_regime_id': bear_regime_id,
        'bull_regime_id': bull_regime_id,
        'strategy_mode': strategy_mode,
        'ma_type': ma_type
    }
    
    return results


def print_backtest_results(results):
    """Print formatted backtest results."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"\nStrategy Mode: {results.get('strategy_mode', 'N/A')}")
    print(f"Moving Average Type: {results.get('ma_type', 'sma').upper()}")
    print("-" * 60)
    
    print("\nSTRATEGY PERFORMANCE (Regime-Filtered Trend Following):")
    print("-" * 60)
    strat = results['strategy_metrics']
    print(f"Total Return:      {strat['total_return']*100:>10.2f}%")
    print(f"Annual Return:     {strat['annual_return']*100:>10.2f}%")
    print(f"Volatility:        {strat['volatility']*100:>10.2f}%")
    print(f"Sharpe Ratio:      {strat['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:      {strat['max_drawdown']*100:>10.2f}%")
    
    print("\nBENCHMARK PERFORMANCE (Buy & Hold):")
    print("-" * 60)
    bench = results['benchmark_metrics']
    print(f"Total Return:      {bench['total_return']*100:>10.2f}%")
    print(f"Annual Return:     {bench['annual_return']*100:>10.2f}%")
    print(f"Volatility:        {bench['volatility']*100:>10.2f}%")
    print(f"Sharpe Ratio:      {bench['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:      {bench['max_drawdown']*100:>10.2f}%")
    
    print("\nSTRATEGY STATISTICS:")
    print("-" * 60)
    print(f"Number of Trades:  {results['num_trades']:>10.0f}")
    print(f"Time in Market:    {results['time_in_market']*100:>10.1f}%")
    print(f"Bear Regime ID:    {results['bear_regime_id']:>10d}")
    print(f"Bull Regime ID:    {results['bull_regime_id']:>10d} (includes neutral)")
    
    print("\nOUTPERFORMANCE:")
    print("-" * 60)
    outperformance = strat['total_return'] - bench['total_return']
    sharpe_diff = strat['sharpe_ratio'] - bench['sharpe_ratio']
    print(f"Total Return Diff: {outperformance*100:>10.2f}%")
    print(f"Sharpe Ratio Diff: {sharpe_diff:>10.2f}")


def plot_backtest_results(results, close):
    """Plot backtest results."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Cumulative returns
    ax1 = axes[0]
    strategy_cum = results['strategy_cumulative']
    benchmark_cum = results['benchmark_cumulative']
    
    ax1.plot(strategy_cum.index, strategy_cum.values, label='Regime-Filtered Strategy', 
             color='blue', linewidth=2)
    ax1.plot(benchmark_cum.index, benchmark_cum.values, label='Buy & Hold Benchmark', 
             color='gray', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.set_title('Strategy vs Benchmark Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price with MAs and position
    ax2 = axes[1]
    common_idx = results['ma_short'].index
    close_aligned = close.loc[common_idx]
    ma_type = results.get('ma_type', 'sma').upper()
    
    ax2.plot(common_idx, close_aligned, label='SPY Close', color='black', linewidth=1, alpha=0.7)
    ax2.plot(results['ma_short'].index, results['ma_short'], 
             label=f'{ma_type} Short', color='blue', linewidth=1.5, alpha=0.8)
    ax2.plot(results['ma_long'].index, results['ma_long'], 
             label=f'{ma_type} Long', color='red', linewidth=1.5, alpha=0.8)
    
    # Highlight when in position
    position = results['strategy_position']
    in_position = position > 0
    ax2.fill_between(common_idx, close_aligned.min(), close_aligned.max(), 
                     where=in_position, alpha=0.2, color='green', label='In Position')
    
    ax2.set_ylabel('Price', fontsize=12)
    ax2.set_title('Price with Moving Averages & Position', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bull and Bear regime probabilities
    ax3 = axes[2]
    bear_prob = results['bear_prob']
    bull_prob = results['bull_prob']
    ax3.plot(bear_prob.index, bear_prob.values, color='red', linewidth=1.5, label='Bear Regime Prob', alpha=0.8)
    ax3.plot(bull_prob.index, bull_prob.values, color='green', linewidth=1.5, label='Bull+Neutral Prob', alpha=0.8)
    ax3.axhline(0.65, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Thresholds')
    ax3.fill_between(bear_prob.index, 0, 1, where=in_position, alpha=0.15, color='blue', label='In Position')
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title('Bull+Neutral & Bear Regime Probabilities', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drawdown comparison
    ax4 = axes[3]
    strategy_cum = results['strategy_cumulative']
    benchmark_cum = results['benchmark_cumulative']
    
    strategy_dd = (strategy_cum - strategy_cum.expanding().max()) / strategy_cum.expanding().max()
    benchmark_dd = (benchmark_cum - benchmark_cum.expanding().max()) / benchmark_cum.expanding().max()
    
    ax4.fill_between(strategy_dd.index, strategy_dd.values, 0, alpha=0.5, color='blue', label='Strategy DD')
    ax4.fill_between(benchmark_dd.index, benchmark_dd.values, 0, alpha=0.5, color='gray', label='Benchmark DD')
    ax4.set_ylabel('Drawdown', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def optimize_sma_parameters(
    close,
    probs,
    regime,
    short_range=(5, 30),
    long_range=(20, 100),
    step=1,
    bear_prob_threshold=0.65,
    output_file='sma_optimization_results.csv',
    ma_type='sma'
):
    """
    Optimize MA parameters by searching for best short and long window combination.
    Uses brute force search across all parameter combinations.
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    probs : pd.DataFrame
        Regime probabilities from HMM
    regime : pd.Series
        Active regime at each time point
    short_range : tuple
        (min, max) for short MA window
    long_range : tuple
        (min, max) for long MA window
    step : int
        Step size for parameter search (default 1 for brute force)
    bear_prob_threshold : float
        Bear regime probability threshold
    output_file : str
        CSV filename to save results
    ma_type : str
        Moving average type: 'sma' (Simple), 'ema' (Exponential), or 'wma' (Weighted)
    
    Returns:
    --------
    best_params : dict
        Dictionary with best parameters and results
    results_df : pd.DataFrame
        DataFrame with all optimization results
    """
    print("\n" + "="*60)
    print(f"BRUTE FORCE {ma_type.upper()} PARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Moving Average Type: {ma_type.upper()}")
    print(f"Short window range: {short_range[0]} to {short_range[1]} (step {step})")
    print(f"Long window range: {long_range[0]} to {long_range[1]} (step {step})")
    
    results_list = []
    best_return = -np.inf
    best_params = None
    
    # Generate parameter combinations
    short_windows = range(short_range[0], short_range[1] + 1, step)
    long_windows = range(long_range[0], long_range[1] + 1, step)
    
    total_combinations = len(list(short_windows)) * len(list(long_windows))
    current = 0
    
    print(f"\nTesting {total_combinations} parameter combinations...")
    
    for short_window in range(short_range[0], short_range[1] + 1, step):
        for long_window in range(long_range[0], long_range[1] + 1, step):
            current += 1
            
            # Skip invalid combinations (short must be < long)
            if short_window >= long_window:
                continue
            
            try:
                # Run backtest with current parameters (suppress debug output)
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                results = backtest_regime_filtered_strategy(
                    close=close,
                    probs=probs,
                    regime=regime,
                    short_window=short_window,
                    long_window=long_window,
                    bear_regime_id=None,
                    bear_prob_threshold=bear_prob_threshold
                )
                
                sys.stdout = old_stdout
                
                # Extract metrics
                total_return = results['strategy_metrics']['total_return']
                sharpe_ratio = results['strategy_metrics']['sharpe_ratio']
                max_drawdown = results['strategy_metrics']['max_drawdown']
                volatility = results['strategy_metrics']['volatility']
                num_trades = results['num_trades']
                time_in_market = results['time_in_market']
                
                # Store results
                results_list.append({
                    'short_window': short_window,
                    'long_window': long_window,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'num_trades': num_trades,
                    'time_in_market': time_in_market
                })
                
                # Track best result
                if total_return > best_return:
                    best_return = total_return
                    best_params = {
                        'short_window': short_window,
                        'long_window': long_window,
                        'total_return': total_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'volatility': volatility,
                        'num_trades': num_trades,
                        'time_in_market': time_in_market
                    }
                
                # Progress update every 10%
                if current % max(1, total_combinations // 10) == 0:
                    progress = 100 * current / total_combinations
                    print(f"  Progress: {current}/{total_combinations} ({progress:.1f}%)")
                    
            except Exception as e:
                sys.stdout = old_stdout
                print(f"  Error with short={short_window}, long={long_window}: {e}")
                continue
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('total_return', ascending=False)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print top 10 results
    print("\n" + "="*60)
    print("TOP 10 PARAMETER COMBINATIONS (by Total Return)")
    print("="*60)
    print(f"{'Rank':<6}{'Short':<8}{'Long':<8}{'Total Return':<15}{'Sharpe':<10}{'Max DD':<12}{'Trades':<10}")
    print("-" * 60)
    
    for idx, row in results_df.head(10).iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"{rank:<6}{row['short_window']:<8.0f}{row['long_window']:<8.0f}"
              f"{row['total_return']*100:<14.2f}%{row['sharpe_ratio']:<10.2f}"
              f"{row['max_drawdown']*100:<11.2f}%{row['num_trades']:<10.0f}")
    
    # Print best result
    print("\n" + "="*60)
    print("BEST PARAMETERS (Highest Total Return)")
    print("="*60)
    print(f"Short Window:       {best_params['short_window']}")
    print(f"Long Window:        {best_params['long_window']}")
    print(f"Total Return:       {best_params['total_return']*100:.2f}%")
    print(f"Sharpe Ratio:       {best_params['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {best_params['max_drawdown']*100:.2f}%")
    print(f"Volatility:         {best_params['volatility']*100:.2f}%")
    print(f"Number of Trades:   {best_params['num_trades']:.0f}")
    print(f"Time in Market:     {best_params['time_in_market']*100:.1f}%")
    
    return best_params, results_df


def main(show_plots=True, optimize=False, strategy_mode='sma_hmm_override', ma_type='sma'):
    """Main execution function.
    
    Parameters:
    -----------
    show_plots : bool
        Whether to display plots
    optimize : bool
        Whether to run parameter optimization
    strategy_mode : str
        Strategy mode: 'sma_hmm_override' (default), 'sma_hmm_filter', 'hmm_only', or 'sma_only'
    ma_type : str
        Moving average type: 'sma' (Simple), 'ema' (Exponential), or 'wma' (Weighted)
    """
    # Download data
    spy_data = download_spy_data(start_date='2018-01-01', end_date='2025-12-26')
    
    # Extract close prices as Series
    close = spy_data['Close']
    
    print("\n" + "="*60)
    print("Running HMM Regime Detection with Switch Probability")
    print("="*60)
    
    # Run regime detection with switch calculation
    probs, regime, switches = run_hmm_regime_detection(
        close=close,
        n_states=3,
        train_window=504,    # ~2 years
        vol_window=20,
        refit_every=42       # ~bi-monthly retraining for speed
    )
    
    # Display results
    print(f"\nTotal time periods analyzed: {len(regime)}")
    print(f"Number of regime switches detected: {len(switches)}")
    
    print("\n" + "="*60)
    print("REGIME SWITCHES")
    print("="*60)
    print("\nFirst 20 regime switches:")
    print(switches.head(20))
    
    print("\n" + "="*60)
    print("REGIME DISTRIBUTION")
    print("="*60)
    regime_counts = regime.value_counts().sort_index()
    for state, count in regime_counts.items():
        percentage = (count / len(regime)) * 100
        print(f"Regime {state}: {count} days ({percentage:.2f}%)")
    
    # Calculate switch statistics
    print("\n" + "="*60)
    print("SWITCH STATISTICS")
    print("="*60)
    
    # Time between switches
    switch_dates = switches.index
    if len(switch_dates) > 1:
        time_diffs = [(switch_dates[i+1] - switch_dates[i]).days 
                      for i in range(len(switch_dates)-1)]
        print(f"Average days between switches: {np.mean(time_diffs):.1f}")
        print(f"Min days between switches: {np.min(time_diffs)}")
        print(f"Max days between switches: {np.max(time_diffs)}")
    
    # Transition matrix
    print("\n" + "="*60)
    print("TRANSITION PATTERNS")
    print("="*60)
    transitions = pd.DataFrame({
        'From': regime[:-1].values,
        'To': regime[1:].values
    })
    trans_counts = transitions.groupby(['From', 'To']).size().reset_index(name='Count')
    # Only show actual switches (where From != To)
    switches_only = trans_counts[trans_counts['From'] != trans_counts['To']]
    print("\nSwitch transitions (From -> To):")
    print(switches_only.to_string(index=False))
    
    # Plot results
    if show_plots:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATION")
        print("="*60)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: SPY close price with regime coloring
        ax1 = axes[0]
        colors = ['green', 'yellow', 'red']
        
        # Align close prices with regime index
        close_aligned = close.loc[regime.index]
        
        for state in sorted(regime.unique()):
            mask = regime == state
            state_dates = regime[mask].index
            state_prices = close_aligned[mask]
            ax1.scatter(state_dates, state_prices, 
                       c=colors[state], alpha=0.6, s=10, label=f'Regime {state}')
        
        # Mark switch points
        for switch_date in switches.index:
            ax1.axvline(switch_date, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
        
        ax1.set_ylabel('SPY Close Price', fontsize=12)
        ax1.set_title('SPY Price with Regime Coloring & Switch Points', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regime probabilities over time
        ax2 = axes[1]
        for state in range(probs.shape[1]):
            ax2.plot(probs.index, probs[state], label=f'P(Regime {state})', alpha=0.7)
        
        ax2.axhline(0.7, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Enter Threshold')
        ax2.axhline(0.55, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Exit Threshold')
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Filtered State Probabilities', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Active regime over time
        ax3 = axes[2]
        ax3.plot(regime.index, regime.values, drawstyle='steps-post', color='blue', linewidth=1.5)
        ax3.set_ylabel('Active Regime', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title('Detected Regime Over Time', fontsize=14, fontweight='bold')
        ax3.set_yticks(sorted(regime.unique()))
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        
        print("\nVisualization complete!")
    
    # Run optimization if requested
    if optimize:
        best_params, results_df = optimize_sma_parameters(
            close=close,
            probs=probs,
            regime=regime,
            short_range=(10, 200),
            long_range=(30, 200),
            step=1,  # Brute force: test every value
            bear_prob_threshold=0.65,
            output_file='sma_optimization_results_bruteforce_extended.csv',
            ma_type=ma_type
        )
        
        # Use best parameters for final backtest
        short_window = best_params['short_window']
        long_window = best_params['long_window']
        print(f"\n✓ Using optimized parameters: short={short_window}, long={long_window}")
    else:
        short_window = 10
        long_window = 30
    
    # Run backtest
    print("\n" + "="*60)
    print("RUNNING BACKTEST")
    print("="*60)
    
    backtest_results = backtest_regime_filtered_strategy(
        close=close,
        probs=probs,
        regime=regime,
        short_window=short_window,
        long_window=long_window,
        bear_regime_id=None,  # Auto-detect
        bear_prob_threshold=0.65,  # Less restrictive threshold
        strategy_mode=strategy_mode,
        ma_type=ma_type
    )
    
    print_backtest_results(backtest_results)
    
    if show_plots:
        print("\n" + "="*60)
        print("GENERATING BACKTEST VISUALIZATION")
        print("="*60)
        
        plot_backtest_results(backtest_results, close)
    
    print("\nBacktest complete!")
    
    if show_plots:
        plt.show()  # Keep all plots open at the end


if __name__ == "__main__":
    import sys
    
    # Parse command-line arguments
    optimize_mode = '--optimize' in sys.argv or '-o' in sys.argv
    no_plots = '--no-plots' in sys.argv
    
    # Parse moving average type
    ma_type = 'sma'  # Default
    if '--ma-type' in sys.argv:
        idx = sys.argv.index('--ma-type')
        if idx + 1 < len(sys.argv):
            ma_arg = sys.argv[idx + 1].lower()
            if ma_arg in ['sma', 'ema', 'wma', 'hma', 'kama', 'tema', 'zlema']:
                ma_type = ma_arg
            else:
                print(f"Warning: Unknown MA type '{ma_arg}'. Using 'sma' (Simple Moving Average)")
                print("Valid options: sma, ema, wma, hma, kama, tema, zlema")
    
    # Parse strategy mode
    strategy_mode = 'sma_hmm_override'  # Default
    if '--mode' in sys.argv:
        mode_idx = sys.argv.index('--mode')
        if mode_idx + 1 < len(sys.argv):
            mode_arg = sys.argv[mode_idx + 1]
            if mode_arg in ['1', 'sma_hmm_override']:
                strategy_mode = 'sma_hmm_override'
            elif mode_arg in ['2', 'sma_hmm_filter']:
                strategy_mode = 'sma_hmm_filter'
            elif mode_arg in ['3', 'hmm_only']:
                strategy_mode = 'hmm_only'
            elif mode_arg in ['4', 'sma_only']:
                strategy_mode = 'sma_only'
            else:
                print(f"Warning: Unknown mode '{mode_arg}', using default 'sma_hmm_override'")
    
    # Print banner with mode information
    print("\n" + "="*60)
    print("STRATEGY CONFIGURATION")
    print("="*60)
    ma_names = {
        'sma': 'Simple', 
        'ema': 'Exponential', 
        'wma': 'Weighted',
        'hma': 'Hull',
        'kama': 'Kaufman Adaptive',
        'tema': 'Triple Exponential',
        'zlema': 'Zero-Lag Exponential'
    }
    print(f"Moving Average: {ma_type.upper()} ({ma_names[ma_type]})")
    if optimize_mode:
        print("Mode: OPTIMIZATION")
    print(f"Strategy: {strategy_mode}")
    if strategy_mode == 'sma_hmm_override':
        print("  Description: SMA + HMM - HMM can override SMA signals")
    elif strategy_mode == 'sma_hmm_filter':
        print("  Description: HMM filters incorrect bull SMA signals (bear only)")
    elif strategy_mode == 'hmm_only':
        print("  Description: HMM only - SMA ignored")
    elif strategy_mode == 'sma_only':
        print("  Description: Pure SMA trend-following - HMM ignored")
    print("="*60 + "\n")
    
    main(show_plots=not no_plots, optimize=optimize_mode, strategy_mode=strategy_mode, ma_type=ma_type)

"""
HMM-based signal filter for regime detection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM, GMMHMM, PoissonHMM
from typing import Optional, Tuple, List


def zigzag_pivots(close: pd.Series, threshold_pct: float = 0.02):
    """
    Identify zigzag pivot points and regime segments.
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    threshold_pct : float
        Minimum percentage move to confirm a pivot (e.g., 0.02 = 2%)
    
    Returns:
    --------
    Tuple[List[Tuple], pd.Series]
        - pivots: list of tuples (idx, price, type) where type in {"LOW","HIGH"}
        - seg: pd.Series with values in {"UP","DOWN"} for each bar (NaN before first full segment)
    """
    c = close.values
    idxs = close.index

    pivots = []

    # Start: assume first point is a LOW pivot candidate
    last_pivot_i = 0
    last_pivot_p = c[0]
    direction = None  # None, "UP", "DOWN"

    # Track extremes since last pivot
    run_high_i, run_high_p = 0, c[0]
    run_low_i, run_low_p = 0, c[0]

    for i in range(1, len(c)):
        price = c[i]

        # Update running extremes
        if price > run_high_p:
            run_high_p, run_high_i = price, i
        if price < run_low_p:
            run_low_p, run_low_i = price, i

        if direction is None:
            # Wait until we move enough from the starting point to define direction
            if price >= last_pivot_p * (1 + threshold_pct):
                direction = "UP"
                run_high_p, run_high_i = price, i
            elif price <= last_pivot_p * (1 - threshold_pct):
                direction = "DOWN"
                run_low_p, run_low_i = price, i

        elif direction == "UP":
            # Confirm a HIGH pivot if we drop enough from the running high
            if price <= run_high_p * (1 - threshold_pct):
                pivots.append((idxs[run_high_i], run_high_p, "HIGH"))
                last_pivot_i, last_pivot_p = run_high_i, run_high_p
                direction = "DOWN"
                # reset running low from here
                run_low_i, run_low_p = i, price
                run_high_i, run_high_p = i, price  # reset too for safety

        elif direction == "DOWN":
            # Confirm a LOW pivot if we rise enough from the running low
            if price >= run_low_p * (1 + threshold_pct):
                pivots.append((idxs[run_low_i], run_low_p, "LOW"))
                last_pivot_i, last_pivot_p = run_low_i, run_low_p
                direction = "UP"
                # reset running high from here
                run_high_i, run_high_p = i, price
                run_low_i, run_low_p = i, price  # reset too for safety

    # Build segment labels
    seg = pd.Series(index=idxs, dtype="object")
    if len(pivots) >= 2:
        for (t1, p1, _), (t2, p2, _) in zip(pivots[:-1], pivots[1:]):
            label = "UP" if p2 > p1 else "DOWN"
            seg.loc[t1:t2] = label
        
        # Label the segment after the last pivot to the end
        last_pivot_date, last_pivot_price, last_pivot_type = pivots[-1]
        if last_pivot_type == "LOW":
            # After a LOW pivot, we're in an UP segment
            seg.loc[last_pivot_date:] = "UP"
        else:  # HIGH
            # After a HIGH pivot, we're in a DOWN segment
            seg.loc[last_pivot_date:] = "DOWN"
    elif len(pivots) == 1:
        # Only one pivot - label from it to the end based on its type
        pivot_date, pivot_price, pivot_type = pivots[0]
        if pivot_type == "LOW":
            seg.loc[pivot_date:] = "UP"
        else:  # HIGH
            seg.loc[pivot_date:] = "DOWN"

    return pivots, seg


class HMMRegimeFilter:
    """
    Hidden Markov Model for market regime detection and signal filtering.
    """
    
    def __init__(self, n_states: int = 3, covariance_type: str = "diag", 
                 n_iter: int = 100, tol: float = 1e-3, random_state: int = 42,
                 short_vol_window: int = 10, long_vol_window: int = 30,
                 short_ma_window: int = 10, long_ma_window: int = 30):
        """
        Initialize HMM Regime Filter.
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states (regimes)
        covariance_type : str
            Type of covariance matrix ('diag', 'full', 'spherical', 'tied')
        n_iter : int
            Maximum number of iterations for EM algorithm
        tol : float
            Convergence threshold for EM algorithm
        random_state : int
            Random seed for reproducibility
        short_vol_window : int
            Window for short-term volatility calculation
        long_vol_window : int
            Window for long-term volatility calculation
        short_ma_window : int
            Window for short-term moving average calculation
        long_ma_window : int
            Window for long-term moving average calculation
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.short_vol_window = short_vol_window
        self.long_vol_window = long_vol_window
        self.short_ma_window = short_ma_window
        self.long_ma_window = long_ma_window
        self.model: Optional[GaussianHMM] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def make_features(self, close: pd.Series, short_vol_window: Optional[int] = None, 
                     long_vol_window: Optional[int] = None,
                     short_ma_window: Optional[int] = None,
                     long_ma_window: Optional[int] = None,
                     volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Create features for HMM from close prices.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        short_vol_window : int, optional
            Window for short-term volatility calculation (uses self.short_vol_window if None)
        long_vol_window : int, optional
            Window for long-term volatility calculation (uses self.long_vol_window if None)
        short_ma_window : int, optional
            Window for short-term moving average calculation (uses self.short_ma_window if None)
        long_ma_window : int, optional
            Window for long-term moving average calculation (uses self.long_ma_window if None)
        volume : pd.Series, optional
            Volume data for volume-based features
            
        Returns:
        --------
        pd.DataFrame
            Enhanced feature matrix with reduced redundancy (removed highly correlated features):
            - ret: 1-period log return (r1)
            - ret_5: 5-period log return (r5) for short-term momentum
            - ret_20: 20-period log return (r20) for medium-term momentum
            - rv_short: short-term realized volatility
            - rv_long: long-term realized volatility
            - sma_long: long-term simple moving average (trend baseline)
            - ma_spread: (sma_short - sma_long) / close (trend strength)
            - slope_slow: (sma_long - sma_long[t-5]) / close (trend momentum)
            - rsi: Relative Strength Index (14-period, overbought/oversold)
            - macd: MACD line (12/26 EMA difference, trend following)
            - drawdown: Distance from all-time high (bear market indicator)
            - vol_regime: Volatility percentile rank (126-day rolling, regime stability)
            - volume_ratio: Volume / 20-day average (if volume provided)
            - volume_volatility: Volume volatility / mean ratio (if volume provided)
            
            Removed features based on correlation analysis:
            - vol_ratio: 0.911 correlation with vol_regime (kept vol_regime as more stable)
            - sma_short: 0.973 correlation with sma_long (kept ma_spread which captures difference)
            - price_momentum_5d: 1.000 correlation with ret_5 (redundant)
            - price_momentum_20d: 1.000 correlation with ret_20 (redundant)
        """
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Use instance defaults if not provided
        short_vol_win = short_vol_window if short_vol_window is not None else self.short_vol_window
        long_vol_win = long_vol_window if long_vol_window is not None else self.long_vol_window
        short_ma_win = short_ma_window if short_ma_window is not None else self.short_ma_window
        long_ma_win = long_ma_window if long_ma_window is not None else self.long_ma_window
        
        # Log returns at multiple horizons (multi-scale turning points)
        log_close = np.log(close)
        r1 = log_close.diff()  # r1 = log(P_t / P_{t-1})
        r5 = log_close.diff(5)  # r5 = log(P_t / P_{t-5})
        r20 = log_close.diff(20)  # r20 = log(P_t / P_{t-20})
        
        # Short-term realized volatility
        rv_short = r1.rolling(short_vol_win).std()
        
        # Long-term realized volatility
        rv_long = r1.rolling(long_vol_win).std()
        
        # Volatility ratio for regime detection (captures volatility regime transitions)
        vol_ratio = rv_short / rv_long
        
        # Short-term simple moving average
        sma_short = log_close.rolling(short_ma_win).mean()
        
        # Long-term simple moving average
        sma_long = log_close.rolling(long_ma_win).mean()
        
        # Trend strength features
        # MA spread normalized by price (captures trend strength)
        ma_spread = (sma_short - sma_long) / log_close
        
        # Slope of slow MA (captures trend weakening/strengthening)
        # Use k=5 periods to measure slope
        slope_slow = (sma_long - sma_long.shift(5)) / log_close
        
        # Create base feature dataframe
        feats = pd.DataFrame({
            'ret': r1,
            'ret_5': r5,
            'ret_20': r20,
            'rv_short': rv_short,
            'rv_long': rv_long,
            # Removed vol_ratio - highly correlated with vol_regime (0.911)
            # Keep sma_long for long-term trend, ma_spread captures short-long difference
            'sma_long': sma_long,
            'ma_spread': ma_spread,
            'slope_slow': slope_slow
        })
        
        # Add RSI indicator
        feats['rsi'] = self._compute_rsi(close, window=14)
        
        # Note: MACD is 0.858 correlated with ret_20, but provides different signal
        # (exponential weighting vs simple returns), so keep it
        feats['macd'] = self._compute_macd(close)
        
        # Removed price_momentum_5d - perfectly correlated with ret_5 (1.000)
        # Removed price_momentum_20d - perfectly correlated with ret_20 (1.000)
        # Log returns already capture this information
        
        # Add drawdown feature
        rolling_max = close.expanding().max()
        feats['drawdown'] = (close - rolling_max) / rolling_max
        
        # Add volume features if available
        if volume is not None:
            feats['volume_ratio'] = volume / volume.rolling(20).mean()
            feats['volume_volatility'] = volume.rolling(20).std() / volume.rolling(20).mean()
        
        # Add volatility regime feature (high/low vol indicator)
        # This is percentile rank which is more stable than raw vol_ratio
        vol_percentile = feats['rv_short'].rolling(126).rank(pct=True)
        feats['vol_regime'] = vol_percentile
        
        return feats.dropna()
    
    def fit(self, features: pd.DataFrame) -> None:
        """
        Fit HMM model to features.
        
        Parameters:
        -----------
        features : pd.DataFrame or np.ndarray
            Feature matrix (typically returns and volatility)
        """
        # Handle both DataFrame and numpy array inputs
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=False
        )
        
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def predict_states(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict most likely states.
        
        Parameters:
        -----------
        features : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Predicted states
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle both DataFrame and numpy array inputs
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict state probabilities (smoothed).
        
        Parameters:
        -----------
        features : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            State probabilities (T, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle both DataFrame and numpy array inputs
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def filtered_state_probs(self, features: pd.DataFrame) -> np.ndarray:
        """
        Compute filtered probabilities P(S_t = k | x_1:t) for each t.
        Uses forward algorithm for online filtering (no lookahead).
        
        Parameters:
        -----------
        features : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Filtered state probabilities (T, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle both DataFrame and numpy array inputs
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        X_scaled = self.scaler.transform(X)
        
        # Get model parameters
        startprob = self.model.startprob_
        transmat = self.model.transmat_
        
        # Compute log-likelihood for each observation
        logB = self.model._compute_log_likelihood(X_scaled)
        
        T, K = logB.shape
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        
        # Initialize at t=0
        alpha0 = startprob * np.exp(logB[0])
        scale[0] = alpha0.sum()
        alpha[0] = alpha0 / (scale[0] + 1e-300)
        
        # Forward pass
        for t in range(1, T):
            alpha_t = (alpha[t-1] @ transmat) * np.exp(logB[t])
            scale[t] = alpha_t.sum()
            alpha[t] = alpha_t / (scale[t] + 1e-300)
        
        return alpha
    
    def compute_switch_probability(self, probs: pd.DataFrame) -> pd.Series:
        """
        Compute probability of switching states at next step.
        
        Formula: P_switch(t) = 1 - Σ_i p_t(i) * A_ii
        
        This calculates the probability of transitioning to a different state
        in the next time step, based on current state probabilities and the
        transition matrix persistence (diagonal elements).
        
        Parameters:
        -----------
        probs : pd.DataFrame
            Filtered state probabilities (index=time, columns=states)
            
        Returns:
        --------
        pd.Series
            Probability of switching at each time point
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing switch probability")
        
        # Get diagonal of transition matrix (persistence probabilities)
        A_diag = np.diag(self.model.transmat_)  # Shape: (n_states,)
        
        # Compute weighted sum of persistence probabilities
        # probs: (T, n_states), A_diag: (n_states,)
        persistence_prob = (probs.values * A_diag).sum(axis=1)
        
        # Switch probability = 1 - persistence probability
        switch_prob = 1.0 - persistence_prob
        
        return pd.Series(switch_prob, index=probs.index, name='switch_prob')
    
    def compute_state_entropy(self, probs: pd.DataFrame, normalize: bool = True) -> pd.Series:
        """
        Compute state uncertainty (entropy) at each time point.
        
        Formula: H(t) = -Σ_i p_t(i) * log(p_t(i))
        Normalized: H_norm(t) = H(t) / log(K) where K = number of states
        
        High entropy indicates uncertainty about the current state, which often
        occurs near regime boundaries. Normalized entropy is in [0, 1] range.
        
        Parameters:
        -----------
        probs : pd.DataFrame
            State probabilities (index=time, columns=states)
        normalize : bool
            If True, normalize entropy to [0, 1] by dividing by log(K)
            
        Returns:
        --------
        pd.Series
            State entropy at each time point
        """
        # Compute entropy H(t) = -Σ p_t(i) * log(p_t(i))
        # Add small epsilon to avoid log(0)
        probs_safe = np.maximum(probs.values, 1e-10)
        
        # Element-wise: p * log(p)
        entropy_terms = probs_safe * np.log(probs_safe)
        
        # Sum across states and negate
        entropy = -entropy_terms.sum(axis=1)
        
        # Normalize to [0, 1] if requested
        if normalize:
            max_entropy = np.log(self.n_states)
            entropy = entropy / max_entropy
        
        return pd.Series(entropy, index=probs.index, name='entropy')
    
    def compute_turn_alert_score(self, switch_prob: pd.Series, entropy: pd.Series,
                                 switch_prob_delta: Optional[pd.Series] = None,
                                 weight_switch: float = 0.5, weight_entropy: float = 0.2,
                                 weight_delta: float = 0.3) -> pd.Series:
        """
        Compute turn alert score combining switch probability, entropy, and switch probability change.
        
        Formula: score(t) = w_s × P_switch(t) + w_e × H_norm(t) + w_d × max(0, ΔP_switch(t))
        
        Default weights: w_s = 0.5 (switch probability), w_e = 0.2 (entropy), w_d = 0.3 (delta)
        
        The delta component (ΔP_switch) captures rising switch probability, which indicates
        increasing likelihood of regime change. Only positive changes are included since
        we're interested in growing switch probability. Research shows positive correlation
        of 0.111 between switch probability rises and labeled turning points.
        
        Parameters:
        -----------
        switch_prob : pd.Series
            Switch probability at each time point
        entropy : pd.Series
            Normalized entropy at each time point
        switch_prob_delta : pd.Series, optional
            Change in switch probability (normalized positive changes)
        weight_switch : float, default=0.5
            Weight for switch probability component
        weight_entropy : float, default=0.2
            Weight for entropy component
        weight_delta : float, default=0.3
            Weight for switch probability change component
            
        Returns:
        --------
        pd.Series
            Turn alert score at each time point
        """
        # Align indices
        common_idx = switch_prob.index.intersection(entropy.index)
        sp_aligned = switch_prob.loc[common_idx]
        ent_aligned = entropy.loc[common_idx]
        
        # Compute weighted sum with base components
        alert_score = weight_switch * sp_aligned + weight_entropy * ent_aligned
        
        # Add delta component if available
        if switch_prob_delta is not None:
            common_idx_delta = common_idx.intersection(switch_prob_delta.index)
            if len(common_idx_delta) > 0:
                delta_aligned = switch_prob_delta.loc[common_idx_delta]
                # Align alert_score to common delta indices
                alert_score_aligned = alert_score.loc[common_idx_delta]
                alert_score = alert_score_aligned + weight_delta * delta_aligned
                common_idx = common_idx_delta
        
        return pd.Series(alert_score.values, index=common_idx, name='alert_score')
    
    def detect_regime_switches(self, probs: pd.DataFrame, 
                               enter_th: float = 0.70,
                               exit_th: float = 0.55,
                               confirm_k: int = 2) -> pd.Series:
        """
        Detect regime switches with hysteresis and confirmation.
        
        Parameters:
        -----------
        probs : pd.DataFrame
            State probabilities (index=time, columns=states)
        enter_th : float
            Probability threshold to enter a regime
        exit_th : float
            Probability threshold to exit a regime (hysteresis)
        confirm_k : int
            Number of consecutive periods to confirm switch
            
        Returns:
        --------
        pd.Series
            Active regime at each time point
        """
        idx = probs.index
        active = np.full(len(probs), fill_value=-1, dtype=int)
        current = int(probs.iloc[0].values.argmax())
        active[0] = current
        
        candidate = None
        streak = 0
        
        for t in range(1, len(probs)):
            p = probs.iloc[t]
            top_state = int(p.values.argmax())
            top_prob = float(p.iloc[top_state])
            cur_prob = float(p.iloc[current])
            
            # Hysteresis: keep current regime if still confident enough
            if cur_prob >= exit_th:
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
                # Current regime lost confidence
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
                    current = top_state
                    candidate = None
                    streak = 0
            
            active[t] = current
        
        return pd.Series(active, index=idx, name="regime")
    
    def label_turning_points(self, close: pd.Series, 
                            horizon: int = 20, 
                            threshold: float = 0.05,
                            epsilon: float = 0.02) -> pd.DataFrame:
        """
        Label turning points using forward-looking reversal logic for supervised evaluation.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        horizon : int
            Forward-looking window (H bars)
        threshold : float
            Minimum price change to qualify as reversal (τ), e.g., 0.05 = 5%
        epsilon : float
            Tolerance for identifying local extrema (ε), e.g., 0.02 = within 2% of local max/min
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns:
            - 'is_top': 1 if turning point top, 0 otherwise
            - 'is_bottom': 1 if turning point bottom, 0 otherwise
            - 'label': 'top', 'bottom', or 'none'
        """
        labels = pd.DataFrame(index=close.index)
        labels['is_top'] = 0
        labels['is_bottom'] = 0
        labels['label'] = 'none'
        
        for i in range(len(close) - horizon):
            current_price = close.iloc[i]
            
            # Look forward H bars
            future_window = close.iloc[i:i+horizon+1]
            
            # Check for TOP: price falls more than threshold within horizon
            # and current price is within epsilon of local max
            local_max = future_window.max()
            if current_price >= local_max * (1 - epsilon):
                # Current price is near local max
                future_min = future_window[1:].min()  # Exclude current price
                if (current_price - future_min) / current_price >= threshold:
                    # Price falls more than threshold
                    labels.iloc[i, labels.columns.get_loc('is_top')] = 1
                    labels.iloc[i, labels.columns.get_loc('label')] = 'top'
            
            # Check for BOTTOM: price rises more than threshold within horizon
            # and current price is within epsilon of local min
            local_min = future_window.min()
            if current_price <= local_min * (1 + epsilon):
                # Current price is near local min
                future_max = future_window[1:].max()  # Exclude current price
                if (future_max - current_price) / current_price >= threshold:
                    # Price rises more than threshold
                    labels.iloc[i, labels.columns.get_loc('is_bottom')] = 1
                    labels.iloc[i, labels.columns.get_loc('label')] = 'bottom'
        
        return labels
    
    def label_zigzag_pivots(self, close: pd.Series, 
                           threshold_pct: float = 0.02) -> pd.DataFrame:
        """
        Label turning points and regimes using zigzag algorithm.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        threshold_pct : float
            Minimum percentage move to confirm a pivot (e.g., 0.02 = 2%)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns:
            - 'is_top': 1 if turning point top (HIGH pivot), 0 otherwise
            - 'is_bottom': 1 if turning point bottom (LOW pivot), 0 otherwise
            - 'is_labeled_turn': 1 if any turning point, 0 otherwise
            - 'regime_label': 'bullish' (UP segment) or 'bearish' (DOWN segment)
            - 'label': 'top', 'bottom', or 'none'
        """
        # Get pivots and segments from zigzag algorithm
        pivots, segments = zigzag_pivots(close, threshold_pct=threshold_pct)
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=close.index)
        result['is_top'] = 0
        result['is_bottom'] = 0
        result['is_labeled_turn'] = 0
        result['label'] = 'none'
        
        # Mark pivot points
        for pivot_date, pivot_price, pivot_type in pivots:
            if pivot_type == "HIGH":
                result.loc[pivot_date, 'is_top'] = 1
                result.loc[pivot_date, 'is_labeled_turn'] = 1
                result.loc[pivot_date, 'label'] = 'top'
            elif pivot_type == "LOW":
                result.loc[pivot_date, 'is_bottom'] = 1
                result.loc[pivot_date, 'is_labeled_turn'] = 1
                result.loc[pivot_date, 'label'] = 'bottom'
        
        # Map segment labels to regime labels
        # UP -> bullish, DOWN -> bearish
        regime_map = {'UP': 'bullish', 'DOWN': 'bearish'}
        result['regime_label'] = segments.map(regime_map)
        
        # Fill NaN regime labels (before first segment) with 'neutral'
        result['regime_label'] = result['regime_label'].fillna('neutral')
        
        return result
    
    def identify_regimes(self, close: pd.Series, regime: pd.Series) -> dict:
        """
        Identify regime characteristics based on returns and volatility.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        regime : pd.Series
            Regime series
            
        Returns:
        --------
        dict
            Dictionary with regime IDs and statistics
        """
        returns = close.pct_change()
        regime_stats = {}
        
        # Align regime and returns indices
        common_idx = regime.index.intersection(returns.index)
        regime_aligned = regime.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        
        # Calculate mean return and volatility for each regime
        for state in sorted(regime_aligned.unique()):
            mask = regime_aligned == state
            state_returns = returns_aligned[mask]
            regime_stats[int(state)] = {
                'mean_return': float(state_returns.mean()),
                'volatility': float(state_returns.std())
            }
        
        # Sort regimes by mean return to identify bull/bear/neutral
        sorted_by_return = sorted(regime_stats.keys(), 
                                 key=lambda k: regime_stats[k]['mean_return'])
        
        if len(sorted_by_return) == 2:
            # 2-state model: lowest return = bear, highest return = bull
            bear_regime = sorted_by_return[0]
            bull_regime = sorted_by_return[1]
            neutral_regime = None
        else:
            # 3-state model: lowest = bear, middle = neutral, highest = bull
            bear_regime = sorted_by_return[0]
            neutral_regime = sorted_by_return[1]
            bull_regime = sorted_by_return[2]
        
        # Extract volatilities for backward compatibility
        regime_vols = {k: v['volatility'] for k, v in regime_stats.items()}
        
        return {
            'regime_volatilities': regime_vols,
            'regime_returns': {k: v['mean_return'] for k, v in regime_stats.items()},
            'bear_regime': bear_regime,
            'bull_regime': bull_regime,
            'neutral_regime': neutral_regime
        }
    
    def initialize_hmm_from_data(self, X_train_scaled: np.ndarray) -> GaussianHMM:
        """
        Initialize HMM with regime-aware parameters based on training data.
        
        Uses returns-based clustering to identify initial bear/bull/neutral regimes,
        then computes means, covariances, and transition probabilities from the
        clustered data. This approach is more meaningful than random K-means.
        
        Parameters:
        -----------
        X_train_scaled : np.ndarray
            Scaled training features (T, n_features)
            
        Returns:
        --------
        GaussianHMM
            Initialized HMM model ready for fitting
        """
        n_samples, n_features = X_train_scaled.shape
        
        # Regime-aware initialization based on returns
        # Assume first feature is daily return (ret)
        print(f"  Performing regime-aware initialization...")
        returns = X_train_scaled[:, 0]  # First feature is 'ret'
        
        # Define regime thresholds based on return quantiles
        if self.n_states == 2:
            # Bear/Bull split at median
            median_ret = np.median(returns)
            labels = (returns > median_ret).astype(int)
            print(f"    2-state split at median return: {median_ret:.4f}")
        elif self.n_states == 3:
            # Bear/Neutral/Bull at 33rd/67th percentiles
            q33 = np.percentile(returns, 33)
            q67 = np.percentile(returns, 67)
            labels = np.zeros(len(returns), dtype=int)
            labels[returns > q33] = 1
            labels[returns > q67] = 2
            print(f"    3-state split at 33rd/67th percentiles: {q33:.4f} / {q67:.4f}")
        else:
            # For other n_states, fall back to quantile-based splitting
            quantiles = np.linspace(0, 100, self.n_states + 1)[1:-1]
            thresholds = np.percentile(returns, quantiles)
            labels = np.digitize(returns, thresholds)
            print(f"    {self.n_states}-state split at quantiles: {thresholds}")
        
        # Compute statistics for each regime
        cluster_stats = {}
        for k in range(self.n_states):
            mask = labels == k
            cluster_data = X_train_scaled[mask]
            
            # Ensure we have enough samples
            if len(cluster_data) < 2:
                print(f"  WARNING: Regime {k} has only {len(cluster_data)} samples, using default covariance")
                cluster_stats[k] = {
                    'count': mask.sum(),
                    'mean': cluster_data.mean(axis=0) if len(cluster_data) > 0 else np.zeros(n_features),
                    'cov': np.eye(n_features) * 1.0
                }
            else:
                # Compute covariance with proper handling
                cov_matrix = np.cov(cluster_data.T)
                
                # Check for NaN or Inf
                if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                    print(f"  WARNING: Regime {k} has invalid covariance, using identity matrix")
                    cov_matrix = np.eye(n_features) * 1.0
                
                # Ensure positive definiteness by adding regularization
                cov_matrix += np.eye(n_features) * 0.1
                
                cluster_stats[k] = {
                    'count': mask.sum(),
                    'mean': cluster_data.mean(axis=0),
                    'cov': cov_matrix
                }
        
        # Sort clusters by mean return to identify bear/bull/neutral
        # Returns are in features 0-2 (ret, ret_5, ret_20)
        avg_returns = {k: stats['mean'][:3].mean() for k, stats in cluster_stats.items()}
        sorted_clusters = sorted(avg_returns.keys(), key=lambda k: avg_returns[k])
        
        print(f"    Cluster sizes: {[cluster_stats[k]['count'] for k in sorted_clusters]}")
        print(f"    Mean returns by state: {[avg_returns[k] for k in sorted_clusters]}")
        
        # Estimate transition probabilities from label sequence
        trans_count = np.zeros((self.n_states, self.n_states))
        for i in range(len(labels) - 1):
            trans_count[labels[i], labels[i+1]] += 1
        
        # Normalize and add smoothing
        transmat = trans_count + 0.1  # Laplace smoothing
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        
        # Enhance persistence (increase diagonal elements)
        persistence_boost = 0.15
        for i in range(self.n_states):
            off_diag_sum = transmat[i, :].sum() - transmat[i, i]
            transmat[i, i] += persistence_boost
            transmat[i, :] /= transmat[i, :].sum()  # Renormalize
        
        # Compute starting probabilities from cluster frequencies
        startprob = np.array([cluster_stats[k]['count'] for k in range(self.n_states)])
        startprob = startprob / startprob.sum()
        
        # Blend with regime priors
        if self.n_states == 2:
            # Bear (34%), Bull (66%)
            regime_prior = np.array([0.34, 0.66])
            startprob = 0.7 * startprob + 0.3 * regime_prior
        elif self.n_states == 3:
            # Bear (20%), Neutral (30%), Bull (50%)
            regime_prior = np.array([0.20, 0.30, 0.50])
            startprob = 0.7 * startprob + 0.3 * regime_prior
        
        # Create model and set parameters
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False
        )
        
        model.startprob_ = startprob
        model.transmat_ = transmat
        
        # Set means from cluster centers
        means = np.zeros((self.n_states, n_features))
        for k in range(self.n_states):
            means[k] = cluster_stats[k]['mean']
        model.means_ = means
        
        # Set covariances based on covariance_type
        if self.covariance_type == "diag":
            # Diagonal: extract diagonal elements from cluster covariances
            covars = np.zeros((self.n_states, n_features))
            for k in range(self.n_states):
                cov_matrix = cluster_stats[k]['cov']
                covars[k] = np.diag(cov_matrix)
            # Add minimum variance to avoid singularity and ensure positive values
            covars = np.maximum(covars, 0.1)
            # Check for NaN/Inf
            if np.any(np.isnan(covars)) or np.any(np.isinf(covars)):
                print(f"  WARNING: Invalid diagonal covariances detected, using default values")
                covars = np.ones((self.n_states, n_features)) * 0.5
            model.covars_ = covars
            
        elif self.covariance_type == "full":
            # Full covariance matrices
            covars = np.zeros((self.n_states, n_features, n_features))
            for k in range(self.n_states):
                cov_matrix = cluster_stats[k]['cov']
                # Additional regularization to ensure positive definiteness
                cov_matrix = cov_matrix + np.eye(n_features) * 0.1
                
                # Check for NaN/Inf
                if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                    print(f"  WARNING: Cluster {k} has invalid full covariance, using identity matrix")
                    cov_matrix = np.eye(n_features) * 1.0
                
                covars[k] = cov_matrix
            model.covars_ = covars
            
        elif self.covariance_type == "spherical":
            # Single variance per state
            covars = np.zeros(self.n_states)
            for k in range(self.n_states):
                cov_matrix = cluster_stats[k]['cov']
                covars[k] = np.diag(cov_matrix).mean()
            covars = np.maximum(covars, 0.1)
            # Check for NaN/Inf
            if np.any(np.isnan(covars)) or np.any(np.isinf(covars)):
                print(f"  WARNING: Invalid spherical covariances detected, using default values")
                covars = np.ones(self.n_states) * 0.5
            model.covars_ = covars
            
        elif self.covariance_type == "tied":
            # Shared covariance matrix across states
            cov_tied = np.zeros((n_features, n_features))
            for k in range(self.n_states):
                weight = cluster_stats[k]['count'] / n_samples
                cov_tied += weight * cluster_stats[k]['cov']
            cov_tied += np.eye(n_features) * 0.1
            
            # Check for NaN/Inf
            if np.any(np.isnan(cov_tied)) or np.any(np.isinf(cov_tied)):
                print(f"  WARNING: Invalid tied covariance detected, using identity matrix")
                cov_tied = np.eye(n_features) * 1.0
            
            model.covars_ = cov_tied
        
        # Print initialization summary
        print(f"  Initialized {self.n_states}-state model from training data:")
        print(f"    Starting probabilities: {startprob}")
        print(f"    Transition persistence (diagonal): {np.diag(transmat)}")
        print(f"    Cluster sizes: {[cluster_stats[k]['count'] for k in range(self.n_states)]}")
        print(f"    Mean returns by state: {[means[k, :4].mean() for k in range(self.n_states)]}")
        
        return model
    
    def _compute_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Compute MACD line."""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def walkforward_filter(self, close: pd.Series, 
                          train_window: int = 504,
                          short_vol_window: Optional[int] = None,
                          long_vol_window: Optional[int] = None,
                          short_ma_window: Optional[int] = None,
                          long_ma_window: Optional[int] = None,
                          refit_every: int = 21,
                          return_switch_prob: bool = False,
                          return_entropy: bool = False,
                          return_switch_prob_delta: bool = False,
                          return_alert_score: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Walk-forward regime detection with periodic refitting.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        train_window : int
            Initial training window size
        short_vol_window : int, optional
            Window for short-term volatility calculation (uses instance default if None)
        long_vol_window : int, optional
            Window for long-term volatility calculation (uses instance default if None)
        short_ma_window : int, optional
            Window for short-term moving average calculation (uses instance default if None)
        long_ma_window : int, optional
            Window for long-term moving average calculation (uses instance default if None)
        refit_every : int
            Refit model every N periods
        return_switch_prob : bool
            If True, return switch probability. For backward compatibility, defaults to False.
        return_entropy : bool
            If True, return state entropy (uncertainty). Defaults to False.
        return_switch_prob_delta : bool
            If True, return normalized positive change in switch probability. Defaults to False.
        return_alert_score : bool
            If True, return turn alert score combining switch_prob, entropy, and delta. Defaults to False.
            Note: Best results when return_switch_prob=True, return_entropy=True, and return_switch_prob_delta=True.
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]
            (probabilities, regime, switches, switch_probability, entropy, switch_prob_delta, alert_score)
            If all return flags are False, returns (probabilities, regime, switches)
        """
        # Create features
        feats = self.make_features(close, short_vol_window=short_vol_window, 
                                  long_vol_window=long_vol_window,
                                  short_ma_window=short_ma_window,
                                  long_ma_window=long_ma_window)
        
        # Check if we have enough data
        if len(feats) < train_window:
            print(f"  Warning: Not enough data ({len(feats)} < {train_window}). Using all data for initial training.")
            train_window = max(60, len(feats) // 2)  # Use at least 60 days or half the data
            print(f"  Adjusted training window: {train_window}")
        
        prob_list = []
        time_list = []
        switch_prob_list = []
        entropy_list = []
        switch_prob_delta_list = []
        
        temp_scaler = StandardScaler()
        
        print(f"Running walk-forward HMM regime detection...")
        print(f"  Training window: {train_window}, Refit every: {refit_every}")
        print(f"  Initializing HMM from training data...")
        
        # Initialize first model using data-driven initialization
        X_init = feats.iloc[:train_window].values
        X_init_scaled = temp_scaler.fit_transform(X_init)
        
        # Initialize model with training data statistics
        temp_model = self.initialize_hmm_from_data(X_init_scaled)
        
        # Fit the model with initialized parameters
        temp_model.init_params = ""  # Don't reinitialize - use our data-driven parameters
        temp_model.fit(X_init_scaled)
        
        print(f"  Initial training complete")
        
        for t in range(train_window, len(feats)):
            # Refit periodically
            if ((t - train_window) % refit_every == 0):
                X_train = feats.iloc[t - train_window : t].values
                X_train_scaled = temp_scaler.fit_transform(X_train)
                
                # Reinitialize model with data-driven parameters for robustness
                #temp_model = self.initialize_hmm_from_data(X_train_scaled)
                temp_model.init_params = "stmc"  # Don't reinitialize during fit
                
                try:
                    temp_model.fit(X_train_scaled)
                except (ValueError, np.linalg.LinAlgError) as e:
                    print(f"    WARNING: Refit failed at t={t}: {str(e)}")
                    print(f"    Keeping previous model parameters")
                    # Continue with previous model
                
                if (t - train_window) % (refit_every * 10) == 0:
                    print(f"    Progress: {t}/{len(feats)} ({100*t/len(feats):.1f}%)")
            
            # Compute filtered probs using data up to t (inclusive)
            X_upto = feats.iloc[: t + 1].values
            X_upto_scaled = temp_scaler.transform(X_upto)
            
            # Use library's predict_proba for robust probability calculation
            probs_upto = temp_model.predict_proba(X_upto_scaled)
            
            # Take only the probability for time t (last row)
            prob_t = probs_upto[-1]
            prob_list.append(prob_t)
            time_list.append(feats.index[t])
            
            # Compute switch probability if requested: P_switch(t) = 1 - Σ_i p_t(i) * A_ii
            if return_switch_prob or return_switch_prob_delta:
                A_diag = np.diag(temp_model.transmat_)
                persistence_prob = (prob_t * A_diag).sum()
                switch_prob_t = 1.0 - persistence_prob
                
                # Compute change in switch probability if requested (before appending)
                if return_switch_prob_delta:
                    if return_switch_prob and len(switch_prob_list) > 0:
                        # Delta = current - previous
                        delta = switch_prob_t - switch_prob_list[-1]
                        # Only keep positive changes (rising switch probability)
                        delta_norm = max(0.0, delta)
                        switch_prob_delta_list.append(delta_norm)
                    elif not return_switch_prob and len(switch_prob_delta_list) > 0:
                        # If not tracking switch_prob separately, need a temporary variable
                        # Delta from last computed value (stored separately)
                        # Store the previous value
                        if not hasattr(self, '_last_switch_prob'):
                            self._last_switch_prob = switch_prob_t
                            switch_prob_delta_list.append(0.0)
                        else:
                            delta = switch_prob_t - self._last_switch_prob
                            delta_norm = max(0.0, delta)
                            switch_prob_delta_list.append(delta_norm)
                            self._last_switch_prob = switch_prob_t
                    else:
                        # First point, no delta available
                        switch_prob_delta_list.append(0.0)
                        if not return_switch_prob:
                            self._last_switch_prob = switch_prob_t
                
                # Append to switch_prob list after delta computation
                if return_switch_prob:
                    switch_prob_list.append(switch_prob_t)
            
            # Compute entropy if requested: H(t) = -Σ p_t(i) * log(p_t(i))
            if return_entropy:
                prob_t_safe = np.maximum(prob_t, 1e-10)
                entropy_t = -(prob_t_safe * np.log(prob_t_safe)).sum()
                # Normalize to [0, 1]
                max_entropy = np.log(self.n_states)
                entropy_t_norm = entropy_t / max_entropy
                entropy_list.append(entropy_t_norm)
        
        probs = pd.DataFrame(prob_list, index=time_list, columns=list(range(self.n_states)))
        regime = self.detect_regime_switches(probs, enter_th=0.99, exit_th=0.99, confirm_k=100)
        switches = regime[regime.ne(regime.shift(1))].dropna()
        
        print(f"  Complete! Detected {len(switches)} regime switches")
        
        # Store the last model for potential later use
        self.model = temp_model
        self.scaler = temp_scaler
        self.is_fitted = True
        
        # Build return tuple based on requested outputs
        if return_switch_prob or return_entropy or return_switch_prob_delta or return_alert_score:
            results = [probs, regime, switches]
            
            switch_prob = None
            entropy = None
            switch_prob_delta = None
            
            if return_switch_prob:
                switch_prob = pd.Series(switch_prob_list, index=time_list, name='switch_prob')
                results.append(switch_prob)
            else:
                results.append(None)
            
            if return_entropy:
                entropy = pd.Series(entropy_list, index=time_list, name='entropy')
                results.append(entropy)
            else:
                results.append(None)
            
            if return_switch_prob_delta:
                switch_prob_delta = pd.Series(switch_prob_delta_list, index=time_list, name='switch_prob_delta')
                results.append(switch_prob_delta)
            else:
                results.append(None)
            
            # Compute alert score if requested
            if return_alert_score:
                if switch_prob is not None and entropy is not None:
                    # Weighted sum: 0.5 * switch_prob + 0.2 * entropy + 0.3 * delta (if available)
                    alert_score = 0.5 * switch_prob + 0.2 * entropy
                    
                    # Add delta component if available
                    if switch_prob_delta is not None:
                        alert_score = alert_score + 0.3 * switch_prob_delta
                    
                    alert_score.name = 'alert_score'
                    results.append(alert_score)
                else:
                    # If alert score requested but dependencies not available, return None
                    results.append(None)
            else:
                results.append(None)
            
            return tuple(results)
        else:
            return probs, regime, switches
    
    def walkforward_filter_predict_proba(self, close: pd.Series, 
                                        train_window: int = 504,
                                        short_vol_window: Optional[int] = None,
                                        long_vol_window: Optional[int] = None,
                                        short_ma_window: Optional[int] = None,
                                        long_ma_window: Optional[int] = None,
                                        refit_every: int = 21) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Walk-forward regime detection using predict_proba (simpler but slower).
        
        This version uses the library's predict_proba() function instead of manual
        forward algorithm. It's simpler but less efficient due to redundant computation.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        train_window : int
            Initial training window size
        short_vol_window : int, optional
            Window for short-term volatility calculation (uses instance default if None)
        long_vol_window : int, optional
            Window for long-term volatility calculation (uses instance default if None)
        short_ma_window : int, optional
            Window for short-term moving average calculation (uses instance default if None)
        long_ma_window : int, optional
            Window for long-term moving average calculation (uses instance default if None)
        refit_every : int
            Refit model every N periods
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, pd.Series]
            (probabilities, regime, switches)
        """
        # Create features
        feats = self.make_features(close, short_vol_window=short_vol_window,
                                  long_vol_window=long_vol_window,
                                  short_ma_window=short_ma_window,
                                  long_ma_window=long_ma_window)
        
        # Check if we have enough data
        if len(feats) < train_window:
            print(f"  Warning: Not enough data ({len(feats)} < {train_window}). Using all data for initial training.")
            train_window = max(60, len(feats) // 2)
            print(f"  Adjusted training window: {train_window}")
        
        prob_list = []
        time_list = []
        
        temp_model = None
        temp_scaler = StandardScaler()
        
        print(f"Running walk-forward HMM regime detection (using predict_proba)...")
        print(f"  Training window: {train_window}, Refit every: {refit_every}")
        
        for t in range(train_window, len(feats)):
            # Refit periodically
            if temp_model is None or ((t - train_window) % refit_every == 0):
                X_train = feats.iloc[t - train_window : t].values
                X_train_scaled = temp_scaler.fit_transform(X_train)
                
                temp_model = GMMHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                    verbose=False
                )
                temp_model.fit(X_train_scaled)
                
                if (t - train_window) % (refit_every * 10) == 0:
                    print(f"    Progress: {t}/{len(feats)} ({100*t/len(feats):.1f}%)")
            
            # Use predict_proba on data up to t (inclusive)
            X_upto = feats.iloc[: t + 1].values
            X_upto_scaled = temp_scaler.transform(X_upto)
            
            # Get probabilities for all timesteps up to t
            probs_upto = temp_model.predict_proba(X_upto_scaled)
            
            # Take only the probability for time t (last row)
            prob_t = probs_upto[-1]
            prob_list.append(prob_t)
            time_list.append(feats.index[t])
        
        probs = pd.DataFrame(prob_list, index=time_list, columns=list(range(self.n_states)))
        regime = self.detect_regime_switches(probs, enter_th=0.77, exit_th=0.55, confirm_k=2)
        switches = regime[regime.ne(regime.shift(1))].dropna()
        
        print(f"  Complete! Detected {len(switches)} regime switches")
        
        return probs, regime, switches

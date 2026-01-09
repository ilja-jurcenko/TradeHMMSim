"""
HMM-based signal filter for regime detection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from typing import Optional, Tuple


class HMMRegimeFilter:
    """
    Hidden Markov Model for market regime detection and signal filtering.
    """
    
    def __init__(self, n_states: int = 3, covariance_type: str = "diag", 
                 n_iter: int = 100, random_state: int = 42):
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
        random_state : int
            Random seed for reproducibility
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def make_features(self, close: pd.Series, vol_window: int = 20) -> pd.DataFrame:
        """
        Create features for HMM from close prices.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        vol_window : int
            Window for volatility calculation
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix with returns and volatility
        """
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Log returns
        r = np.log(close).diff()
        
        # Realized volatility (rolling std of returns)
        rv = r.rolling(vol_window).std()
        
        X = pd.DataFrame({
            'ret': r,
            'rv': rv
        }).dropna()
        
        return X
    
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
    
    def identify_regimes(self, close: pd.Series, regime: pd.Series) -> dict:
        """
        Identify regime characteristics (volatility-based).
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        regime : pd.Series
            Regime series
            
        Returns:
        --------
        dict
            Dictionary with regime IDs and volatilities
        """
        returns = close.pct_change()
        regime_vols = {}
        
        # Align regime and returns indices
        common_idx = regime.index.intersection(returns.index)
        regime_aligned = regime.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        
        for state in sorted(regime_aligned.unique()):
            mask = regime_aligned == state
            state_returns = returns_aligned[mask]
            regime_vols[int(state)] = float(state_returns.std())
        
        # Identify bear (highest vol), bull (lowest vol), neutral (medium vol)
        bear_regime = max(regime_vols, key=regime_vols.get)
        bull_regime = min(regime_vols, key=regime_vols.get)
        
        all_regimes = set(regime_vols.keys())
        neutral_regime = None
        if len(all_regimes) == 3:
            neutral_regime = list(all_regimes - {bear_regime, bull_regime})[0]
        
        return {
            'regime_volatilities': regime_vols,
            'bear_regime': bear_regime,
            'bull_regime': bull_regime,
            'neutral_regime': neutral_regime
        }
    
    def walkforward_filter(self, close: pd.Series, 
                          train_window: int = 504,
                          vol_window: int = 20,
                          refit_every: int = 21) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Walk-forward regime detection with periodic refitting.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        train_window : int
            Initial training window size
        vol_window : int
            Window for volatility calculation
        refit_every : int
            Refit model every N periods
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, pd.Series]
            (probabilities, regime, switches)
        """
        # Create features
        feats = self.make_features(close, vol_window=vol_window)
        
        # Check if we have enough data
        if len(feats) < train_window:
            print(f"  Warning: Not enough data ({len(feats)} < {train_window}). Using all data for initial training.")
            train_window = max(60, len(feats) // 2)  # Use at least 60 days or half the data
            print(f"  Adjusted training window: {train_window}")
        
        prob_list = []
        time_list = []
        
        temp_model = None
        temp_scaler = StandardScaler()
        
        print(f"Running walk-forward HMM regime detection...")
        print(f"  Training window: {train_window}, Refit every: {refit_every}")
        
        for t in range(train_window, len(feats)):
            # Refit periodically
            if temp_model is None or ((t - train_window) % refit_every == 0):
                X_train = feats.iloc[t - train_window : t].values
                X_train_scaled = temp_scaler.fit_transform(X_train)
                
                temp_model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                    verbose=False
                )
                temp_model.fit(X_train_scaled)
                
                if (t - train_window) % (refit_every * 10) == 0:
                    print(f"    Progress: {t}/{len(feats)} ({100*t/len(feats):.1f}%)")
            
            # Compute filtered probs using data up to t (inclusive)
            X_upto = feats.iloc[: t + 1].values
            X_upto_scaled = temp_scaler.transform(X_upto)
            
            # Forward algorithm for filtered probs
            startprob = temp_model.startprob_
            transmat = temp_model.transmat_
            logB = temp_model._compute_log_likelihood(X_upto_scaled)
            
            T, K = logB.shape
            alpha = np.zeros((T, K))
            scale = np.zeros(T)
            
            alpha0 = startprob * np.exp(logB[0])
            scale[0] = alpha0.sum()
            alpha[0] = alpha0 / (scale[0] + 1e-300)
            
            for i in range(1, T):
                alpha_t = (alpha[i-1] @ transmat) * np.exp(logB[i])
                scale[i] = alpha_t.sum()
                alpha[i] = alpha_t / (scale[i] + 1e-300)
            
            prob_t = alpha[-1]
            prob_list.append(prob_t)
            time_list.append(feats.index[t])
        
        probs = pd.DataFrame(prob_list, index=time_list, columns=list(range(self.n_states)))
        regime = self.detect_regime_switches(probs, enter_th=0.7, exit_th=0.55, confirm_k=2)
        switches = regime[regime.ne(regime.shift(1))].dropna()
        
        print(f"  Complete! Detected {len(switches)} regime switches")
        
        return probs, regime, switches

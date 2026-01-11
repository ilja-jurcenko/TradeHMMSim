"""
Unit tests for HMM Signal Filter.
"""

import unittest
import pandas as pd
import numpy as np
from signal_filter import HMMRegimeFilter


class TestHMMFilter(unittest.TestCase):
    """Test cases for HMM regime filter."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic price data with regime changes
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Bull market (low vol, uptrend)
        bull = 100 + np.cumsum(np.random.randn(200) * 0.5 + 0.2)
        
        # Bear market (high vol, downtrend)
        bear = bull[-1] + np.cumsum(np.random.randn(150) * 2.0 - 0.3)
        
        # Recovery (medium vol, uptrend)
        recovery = bear[-1] + np.cumsum(np.random.randn(150) * 1.0 + 0.15)
        
        prices = np.concatenate([bull, bear, recovery])
        self.prices = pd.Series(prices, index=dates, name='close')
        
    def test_hmm_initialization(self):
        """Test HMM filter initialization."""
        hmm = HMMRegimeFilter(n_states=3)
        
        self.assertEqual(hmm.n_states, 3)
        self.assertFalse(hmm.is_fitted)
        self.assertIsNone(hmm.model)
        
    def test_feature_creation(self):
        """Test feature creation from prices."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices, vol_window=20)
        
        # Check feature shape
        self.assertEqual(features.shape[1], 2)
        self.assertIn('ret', features.columns)
        self.assertIn('rv', features.columns)
        
        # Check that features are created
        self.assertGreater(len(features), 0)
        self.assertLess(len(features), len(self.prices))  # Some NaN dropped
        
    def test_hmm_fitting(self):
        """Test HMM model fitting."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        
        # Fit model
        hmm.fit(features)
        
        self.assertTrue(hmm.is_fitted)
        self.assertIsNotNone(hmm.model)
        
    def test_state_prediction(self):
        """Test state prediction."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        hmm.fit(features)
        
        states = hmm.predict_states(features)
        
        # Check output shape
        self.assertEqual(len(states), len(features))
        
        # Check states are valid
        self.assertTrue(all(state in [0, 1, 2] for state in states))
        
    def test_probability_prediction(self):
        """Test probability prediction."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        hmm.fit(features)
        
        probs = hmm.predict_proba(features)
        
        # Check shape
        self.assertEqual(probs.shape, (len(features), 3))
        
        # Check probabilities sum to 1
        prob_sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(features)), decimal=5)
        
        # Check probabilities are in [0, 1]
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))
        
    def test_filtered_probabilities(self):
        """Test filtered probability calculation."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        hmm.fit(features)
        
        filtered_probs = hmm.filtered_state_probs(features)
        
        # Check shape
        self.assertEqual(filtered_probs.shape, (len(features), 3))
        
        # Check probabilities are valid
        self.assertTrue(np.all(filtered_probs >= 0))
        self.assertTrue(np.all(filtered_probs <= 1.01))  # Allow small numerical error
        
    def test_regime_identification(self):
        """Test regime identification."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        hmm.fit(features)
        
        states = hmm.predict_states(features)
        regime_series = pd.Series(states, index=features.index)
        
        regime_info = hmm.identify_regimes(self.prices, regime_series)
        
        # Check structure
        self.assertIn('regime_volatilities', regime_info)
        self.assertIn('bear_regime', regime_info)
        self.assertIn('bull_regime', regime_info)
        
        # Check that bear has higher vol than bull
        bear_vol = regime_info['regime_volatilities'][regime_info['bear_regime']]
        bull_vol = regime_info['regime_volatilities'][regime_info['bull_regime']]
        self.assertGreater(bear_vol, bull_vol)
        
    def test_regime_switch_detection(self):
        """Test regime switch detection."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        hmm.fit(features)
        
        probs = hmm.filtered_state_probs(features)
        probs_df = pd.DataFrame(probs, index=features.index, columns=[0, 1, 2])
        
        regime = hmm.detect_regime_switches(probs_df, enter_th=0.7, exit_th=0.55, confirm_k=2)
        
        # Check output
        self.assertIsInstance(regime, pd.Series)
        self.assertEqual(len(regime), len(features))
        
        # Check that regimes are valid
        self.assertTrue(all(state in [0, 1, 2] for state in regime.unique()))
        
        # Check that switches are detected
        switches = regime[regime.ne(regime.shift(1))].dropna()
        self.assertGreater(len(switches), 0)
        
    def test_walkforward_filter(self):
        """Test walk-forward filtering."""
        hmm = HMMRegimeFilter(n_states=3)
        
        # Use shorter windows for faster test
        probs, regime, switches = hmm.walkforward_filter(
            self.prices,
            train_window=100,
            refit_every=20
        )
        
        # Check outputs
        self.assertIsInstance(probs, pd.DataFrame)
        self.assertIsInstance(regime, pd.Series)
        self.assertIsInstance(switches, pd.Series)
        
        # Check that we got results
        self.assertGreater(len(probs), 0)
        self.assertGreater(len(regime), 0)
        
    def test_hysteresis_in_switches(self):
        """Test that hysteresis prevents excessive switching."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        hmm.fit(features)
        
        probs = hmm.filtered_state_probs(features)
        probs_df = pd.DataFrame(probs, index=features.index, columns=[0, 1, 2])
        
        # Test with high hysteresis (fewer switches)
        regime_high = hmm.detect_regime_switches(probs_df, enter_th=0.8, exit_th=0.4, confirm_k=3)
        switches_high = regime_high[regime_high.ne(regime_high.shift(1))].dropna()
        
        # Test with low hysteresis (more switches)
        regime_low = hmm.detect_regime_switches(probs_df, enter_th=0.6, exit_th=0.55, confirm_k=1)
        switches_low = regime_low[regime_low.ne(regime_low.shift(1))].dropna()
        
        # High hysteresis should have fewer switches
        self.assertLessEqual(len(switches_high), len(switches_low))
        
    def test_model_persistence(self):
        """Test that fitted model can be reused."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        
        hmm.fit(features)
        
        # First prediction
        states1 = hmm.predict_states(features)
        
        # Second prediction (should be identical)
        states2 = hmm.predict_states(features)
        
        np.testing.assert_array_equal(states1, states2)
        
    def test_error_handling(self):
        """Test error handling for unfitted model."""
        hmm = HMMRegimeFilter(n_states=3)
        features = hmm.make_features(self.prices)
        
        # Should raise error when predicting without fitting
        with self.assertRaises(ValueError):
            hmm.predict_states(features)
        
        with self.assertRaises(ValueError):
            hmm.predict_proba(features)
        
        with self.assertRaises(ValueError):
            hmm.filtered_state_probs(features)


if __name__ == '__main__':
    unittest.main()

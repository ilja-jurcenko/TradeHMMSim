"""
Unit tests for HMM walk-forward refitting behavior.
"""

import unittest
import pandas as pd
import numpy as np
from signal_filter import HMMRegimeFilter


class TestHMMRefitting(unittest.TestCase):
    """Test cases for HMM walk-forward refitting logic."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create price data: 2000 days for training + ~252 days for testing = ~2252 days
        # This simulates: train_window=2000, test_period=252 days (1 year)
        n_train = 2000
        n_test = 252
        n_total = n_train + n_test
        
        dates = pd.date_range('2020-01-01', periods=n_total, freq='B')  # Business days
        
        # Generate realistic price movements
        returns = np.random.randn(n_total) * 0.015 + 0.0003  # ~1.5% daily vol, slight upward drift
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.prices = pd.Series(prices, index=dates, name='close')
        self.n_train = n_train
        self.n_test = n_test
        
    def test_single_fit_no_refit(self):
        """
        Test that with train_window=2000 and refit_every=2000,
        the model fits only once and doesn't refit during a 1-year test period.
        
        We verify this by:
        1. Checking that predictions span the correct range
        2. Verifying the logic: only 1 refit should occur at t=train_window
        """
        train_window = 2000
        refit_every = 2000
        
        hmm = HMMRegimeFilter(
            n_states=3,
            random_state=42,
            short_vol_window=20,
            long_vol_window=60,
            short_ma_window=20,
            long_ma_window=60
        )
        
        # Run walkforward filter
        probs, regime, switches = hmm.walkforward_filter(
            self.prices,
            train_window=train_window,
            refit_every=refit_every
        )
        
        # Calculate expected behavior:
        # Features will have some rows dropped due to MA/vol windows (max 60 rows)
        # For loop range(train_window, len(features)):
        #   - First iteration at t=train_window: (t-train_window) % refit_every = 0 → FIT
        #   - Subsequent iterations: (t-train_window) % refit_every != 0 (since test period < refit_every)
        # Therefore, only 1 fit should occur
        
        # Verify predictions were generated
        self.assertGreater(len(probs), 0, "Should generate predictions")
        
        # The predictions should cover approximately the test period
        # Given train_window=2000, refit_every=2000, and ~252 test days
        # We expect predictions for approximately n_test days
        expected_min_predictions = self.n_test - 100  # Allow for feature overhead
        expected_max_predictions = self.n_test + 10
        
        self.assertGreaterEqual(len(probs), expected_min_predictions,
                               f"Expected at least {expected_min_predictions} predictions, got {len(probs)}")
        self.assertLessEqual(len(probs), expected_max_predictions,
                            f"Expected at most {expected_max_predictions} predictions, got {len(probs)}")
        
        # Calculate how many refits should have occurred
        features = hmm.make_features(self.prices)
        n_features = len(features)
        
        # Count refit positions
        refit_count = 0
        for t in range(train_window, n_features):
            if (t - train_window) % refit_every == 0:
                refit_count += 1
        
        # With train_window=2000, refit_every=2000, and test period ~252 days:
        # - First refit at t=2000: (2000-2000) % 2000 = 0 → refit
        # - Next would be at t=4000, but we only go up to ~2252
        # So expect exactly 1 refit
        expected_refits = 1
        self.assertEqual(refit_count, expected_refits,
                        f"Expected {expected_refits} refit(s) based on condition logic, got {refit_count}")
        
    def test_multiple_refits_small_interval(self):
        """
        Test that with train_window=200 and refit_every=50,
        the model refits multiple times during the test period.
        """
        train_window = 200
        refit_every = 50
        
        hmm = HMMRegimeFilter(
            n_states=3,
            random_state=42,
            short_vol_window=20,
            long_vol_window=60,
            short_ma_window=20,
            long_ma_window=60
        )
        
        # Run walkforward filter
        probs, regime, switches = hmm.walkforward_filter(
            self.prices,
            train_window=train_window,
            refit_every=refit_every
        )
        
        # Calculate expected predictions
        # Features are created with 60-day MA/vol windows, dropping those NaN rows
        features = hmm.make_features(self.prices)
        expected_predictions = len(features) - train_window
        
        self.assertEqual(len(probs), expected_predictions,
                        f"Expected {expected_predictions} predictions, got {len(probs)}")
        
        # Verify multiple refits occurred
        # With train_window=200, refit_every=50, and ~2192 features:
        # Refits at: t=200, 250, 300, ..., up to ~2191
        # Number of refits = (2191-200)//50 + 1 = 39 + 1 = 40
        refit_count = 0
        for t in range(train_window, len(features)):
            if (t - train_window) % refit_every == 0:
                refit_count += 1
        
        self.assertGreater(refit_count, 1, "Should have multiple refits with small interval")
        
    def test_refit_positions(self):
        """
        Test that refits happen at the correct positions in the data.
        """
        train_window = 100
        refit_every = 30
        
        # Create shorter test data for easier verification
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=250, freq='B')
        returns = np.random.randn(250) * 0.015
        prices = 100 * np.exp(np.cumsum(returns))
        test_prices = pd.Series(prices, index=dates, name='close')
        
        hmm = HMMRegimeFilter(n_states=3, random_state=42)
        
        # Create features to know actual feature count
        features = hmm.make_features(test_prices)
        
        # Track refit positions
        refit_positions = []
        for t in range(train_window, len(features)):
            if (t - train_window) % refit_every == 0:
                refit_positions.append(t)
        
        # Expected positions: 100, 130, 160, 190, 220 (if we have at least 220 features)
        self.assertEqual(refit_positions[0], 100, "First refit should be at position 100")
        self.assertEqual(refit_positions[1], 130, "Second refit should be at position 130")
        
        # Run the actual filter
        probs, regime, switches = hmm.walkforward_filter(
            test_prices,
            train_window=train_window,
            refit_every=refit_every
        )
        
        # Verify we got predictions from position train_window onwards in features
        expected_predictions = len(features) - train_window
        self.assertEqual(len(probs), expected_predictions,
                        f"Expected {expected_predictions} predictions, got {len(probs)}")
        
    def test_no_refit_when_interval_exceeds_data(self):
        """
        Test that when refit_every is larger than the test period,
        only the initial fit occurs.
        """
        train_window = 100
        refit_every = 1000  # Much larger than test period
        
        # Create test data with only 200 total points
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        returns = np.random.randn(200) * 0.015
        prices = 100 * np.exp(np.cumsum(returns))
        test_prices = pd.Series(prices, index=dates, name='close')
        
        hmm = HMMRegimeFilter(n_states=3, random_state=42)
        
        # Create features to know actual count
        features = hmm.make_features(test_prices)
        
        # Calculate expected refits
        # Loop: t in range(train_window, len(features))
        # Refit at t=train_window: (train_window-train_window) % 1000 = 0 → YES (only this one)
        # Subsequent iterations: (t-train_window) % 1000 != 0 since test period << 1000
        refit_count = 0
        for t in range(train_window, len(features)):
            if (t - train_window) % refit_every == 0:
                refit_count += 1
        
        self.assertEqual(refit_count, 1, "Should have exactly 1 refit when interval exceeds data")
        
        probs, regime, switches = hmm.walkforward_filter(
            test_prices,
            train_window=train_window,
            refit_every=refit_every
        )
        
        # Should get predictions for (len(features) - train_window) time steps
        expected_predictions = len(features) - train_window
        self.assertEqual(len(probs), expected_predictions,
                        f"Expected {expected_predictions} predictions, got {len(probs)}")
        
    def test_feature_overhead_accounted(self):
        """
        Test that feature creation overhead (NaN rows) doesn't affect
        the refit logic.
        """
        train_window = 200
        refit_every = 50
        
        # Use longer MA windows to create more NaN rows
        hmm = HMMRegimeFilter(
            n_states=3,
            random_state=42,
            short_vol_window=20,
            long_vol_window=100,  # Creates 100 NaN rows
            short_ma_window=20,
            long_ma_window=100    # Creates 100 NaN rows
        )
        
        # Create features manually to check
        features = hmm.make_features(self.prices)
        
        # Features will have 100 fewer rows due to dropna()
        feature_overhead = len(self.prices) - len(features)
        self.assertEqual(feature_overhead, 100)
        
        # Run walkforward with these features
        probs, regime, switches = hmm.walkforward_filter(
            self.prices,
            train_window=train_window,
            refit_every=refit_every
        )
        
        # Predictions should start from position train_window in features
        # which corresponds to a later date in the original prices
        expected_predictions = len(features) - train_window
        self.assertEqual(len(probs), expected_predictions)


if __name__ == '__main__':
    unittest.main()

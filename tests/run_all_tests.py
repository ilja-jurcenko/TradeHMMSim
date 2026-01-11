#!/usr/bin/env python3
"""
Run all tests (unit + integration) for TradeHMMSim.

This runs both unit and integration test suites sequentially.
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Discover and run all tests (unit + integration)."""
    # Start from the tests directory
    test_dir = Path(__file__).parent
    
    # Discover all tests
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern='test_*.py',
        top_level_dir=str(project_root)
    )
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on results
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    print("=" * 70)
    print("Running All Tests (Unit + Integration)")
    print("=" * 70)
    print()
    
    exit_code = run_all_tests()
    sys.exit(exit_code)

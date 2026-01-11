#!/usr/bin/env python3
"""
Run all unit tests for TradeHMMSim.

Unit tests are fast, use mocked dependencies, and don't require network access.
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_unit_tests():
    """Discover and run all unit tests."""
    # Start from the unit test directory
    test_dir = Path(__file__).parent / 'unit'
    
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
    exit_code = run_unit_tests()
    sys.exit(exit_code)

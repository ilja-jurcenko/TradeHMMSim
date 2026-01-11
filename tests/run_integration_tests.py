#!/usr/bin/env python3
"""
Run all integration tests for TradeHMMSim.

Integration tests are slower, use real data sources, and require network access.
These tests verify that components work correctly with actual data.
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_integration_tests():
    """Discover and run all integration tests."""
    # Start from the integration test directory
    test_dir = Path(__file__).parent / 'integration'
    
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
    print("Running Integration Tests")
    print("Note: These tests require network access and may take longer to run")
    print("=" * 70)
    print()
    
    exit_code = run_integration_tests()
    sys.exit(exit_code)

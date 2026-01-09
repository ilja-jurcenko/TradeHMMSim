"""Quick test to verify HMM combine logic is using bull+neutral"""
from portfolio import Portfolio
from alpha_models import EMA
from backtest import BacktestEngine

# Setup
portfolio = Portfolio(['SPY'], '2020-01-01', '2025-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

alpha_model = EMA(short_window=10, long_window=30)
engine = BacktestEngine(close, alpha_model, initial_capital=100000)

# Run combine strategy
print("Running EMA Alpha + HMM Combine...")
results = engine.run(
    strategy_mode='alpha_hmm_combine',
    walk_forward=True,
    train_window=504,
    refit_every=21,
    bull_prob_threshold=0.65,
    bear_prob_threshold=0.65
)

print(f"\nTotal Return: {results['metrics']['total_return']:.2f}%")
print(f"Time in Market: {results['time_in_market']*100:.1f}%")
print(f"Num Trades: {results['num_trades']}")

# Check regime info
if hasattr(engine, 'regime_info'):
    regime_info = engine.regime_info
    print(f"\nRegime Info:")
    print(f"  Bear: {regime_info['bear_regime']}")
    print(f"  Bull: {regime_info['bull_regime']}")
    print(f"  Neutral: {regime_info['neutral_regime']}")
    
    # Check probs
    probs = engine.regime_probs
    common_idx = results['positions'].index.intersection(probs.index)
    
    bear_prob = probs[regime_info['bear_regime']].loc[common_idx]
    bull_prob = probs[regime_info['bull_regime']].loc[common_idx]
    neutral_prob = probs[regime_info['neutral_regime']].loc[common_idx] if regime_info['neutral_regime'] is not None else None
    
    print(f"\nProbability Stats:")
    print(f"  Bear: mean={bear_prob.mean():.3f}, max={bear_prob.max():.3f}")
    print(f"  Bull: mean={bull_prob.mean():.3f}, max={bull_prob.max():.3f}")
    if neutral_prob is not None:
        print(f"  Neutral: mean={neutral_prob.mean():.3f}, max={neutral_prob.max():.3f}")
        bull_combined = bull_prob + neutral_prob
        print(f"  Bull+Neutral: mean={bull_combined.mean():.3f}, max={bull_combined.max():.3f}")
        print(f"  Bull+Neutral > 0.65: {(bull_combined > 0.65).sum()} / {len(bull_combined)} = {(bull_combined > 0.65).sum()/len(bull_combined)*100:.1f}%")

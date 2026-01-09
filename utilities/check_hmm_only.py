"""Check HMM Only strategy behavior"""
from portfolio import Portfolio
from alpha_models import EMA, SMA, KAMA
from backtest import BacktestEngine
from signal_filter import HMMRegimeFilter

# Load data
portfolio = Portfolio(['SPY'], '2020-01-01', '2025-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Test with different alpha models but same HMM
hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)

models = [
    ('EMA', EMA(short_window=10, long_window=30)),
    ('SMA', SMA(short_window=10, long_window=30)),
    ('KAMA', KAMA(short_window=10, long_window=30))
]

print("="*80)
print("HMM ONLY STRATEGY TEST")
print("="*80)
print("\nTesting if HMM Only returns same result regardless of alpha model...")
print("(It should, since HMM Only ignores the alpha model)\n")

results = []
for name, model in models:
    engine = BacktestEngine(close, model, hmm_filter=HMMRegimeFilter(n_states=3, random_state=42))
    result = engine.run(
        strategy_mode='hmm_only',
        walk_forward=True,
        train_window=504,
        refit_every=21,
        bull_prob_threshold=0.65,
        bear_prob_threshold=0.65
    )
    
    total_return = result['metrics']['total_return']
    time_in_market = result['time_in_market'] * 100
    num_trades = result['num_trades']
    
    results.append({
        'model': name,
        'return': total_return,
        'time_in_market': time_in_market,
        'num_trades': num_trades
    })
    
    print(f"{name:10s}: Return={total_return:6.2f}%, Time={time_in_market:5.1f}%, Trades={num_trades:3d}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Check if all returns are identical
returns = [r['return'] for r in results]
if len(set(returns)) == 1:
    print("✓ EXPECTED: All models return the same (HMM doesn't use alpha model)")
    print(f"  Common return: {returns[0]:.2f}%")
else:
    print("✗ UNEXPECTED: Returns differ (HMM might be using alpha model somehow)")
    for r in results:
        print(f"  {r['model']}: {r['return']:.2f}%")

# Check regime distribution
print(f"\n{'='*80}")
print("HMM REGIME ANALYSIS")
print(f"{'='*80}")

# Run one more time to get regime info
engine = BacktestEngine(close, EMA(short_window=10, long_window=30), 
                       hmm_filter=HMMRegimeFilter(n_states=3, random_state=42))
result = engine.run(
    strategy_mode='hmm_only',
    walk_forward=True,
    train_window=504,
    refit_every=21,
    bull_prob_threshold=0.65,
    bear_prob_threshold=0.65
)

if hasattr(engine, 'regime_info'):
    regime_info = engine.regime_info
    print(f"\nRegime Identification:")
    print(f"  Bear regime: {regime_info['bear_regime']}")
    print(f"  Bull regime: {regime_info['bull_regime']}")
    print(f"  Neutral regime: {regime_info['neutral_regime']}")
    
    probs = engine.regime_probs
    positions = result['positions']
    common_idx = positions.index.intersection(probs.index)
    
    bear_prob = probs[regime_info['bear_regime']].loc[common_idx]
    bull_prob = probs[regime_info['bull_regime']].loc[common_idx]
    neutral_prob = probs[regime_info['neutral_regime']].loc[common_idx] if regime_info['neutral_regime'] is not None else None
    
    bull_prob_combined = bull_prob.copy()
    if neutral_prob is not None:
        bull_prob_combined = bull_prob + neutral_prob
    
    print(f"\nProbability Statistics:")
    print(f"  Bear prob: mean={bear_prob.mean():.3f}, max={bear_prob.max():.3f}")
    print(f"  Bull prob: mean={bull_prob.mean():.3f}, max={bull_prob.max():.3f}")
    if neutral_prob is not None:
        print(f"  Neutral prob: mean={neutral_prob.mean():.3f}, max={neutral_prob.max():.3f}")
        print(f"  Bull+Neutral: mean={bull_prob_combined.mean():.3f}, max={bull_prob_combined.max():.3f}")
    
    print(f"\nHMM Signal (bull+neutral > 0.65):")
    hmm_signal = (bull_prob_combined > 0.65)
    print(f"  Days with signal: {hmm_signal.sum()} / {len(hmm_signal)} ({hmm_signal.sum()/len(hmm_signal)*100:.1f}%)")
    
    print(f"\nActual positions:")
    print(f"  Days in market: {positions.sum()} / {len(positions)} ({positions.sum()/len(positions)*100:.1f}%)")
    
    # Check if they match
    positions_bool = positions.astype(bool).loc[common_idx]
    if (positions_bool == hmm_signal).all():
        print(f"  ✓ Positions match HMM signal exactly")
    else:
        print(f"  ✗ Positions don't match HMM signal")
        print(f"    Difference: {(positions_bool != hmm_signal).sum()} days")

"""Verify HMM Only with transaction costs"""
from portfolio import Portfolio
from alpha_models import EMA
from backtest import BacktestEngine
from signal_filter import HMMRegimeFilter

portfolio = Portfolio(['SPY'], '2020-01-01', '2025-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
model = EMA(short_window=10, long_window=30)

print("="*80)
print("HMM ONLY - TRANSACTION COST IMPACT")
print("="*80)

for tx_cost in [0.0, 0.001, 0.002]:
    engine = BacktestEngine(close, model, hmm_filter=HMMRegimeFilter(n_states=3, random_state=42))
    result = engine.run(
        strategy_mode='hmm_only',
        walk_forward=True,
        train_window=504,
        refit_every=21,
        bull_prob_threshold=0.65,
        bear_prob_threshold=0.65,
        transaction_cost=tx_cost
    )
    
    print(f"\nTransaction cost: {tx_cost*100:.1f}%")
    print(f"  Return: {result['metrics']['total_return']:.2f}%")
    print(f"  Trades: {result['num_trades']}")
    print(f"  Time in market: {result['time_in_market']*100:.1f}%")
    
    if tx_cost > 0:
        # Each trade = entry + exit, so 2 transactions
        total_tx_cost = result['num_trades'] * 2 * tx_cost * 100
        print(f"  Total TX cost: ~{total_tx_cost:.2f}%")

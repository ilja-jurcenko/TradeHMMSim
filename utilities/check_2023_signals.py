"""
Quick check of alpha signals in 2023-2024 period
"""
import pandas as pd
from portfolio import Portfolio
from alpha_models import EMA

# Load data
portfolio = Portfolio(['SPY'], '2020-01-01', '2025-12-31')
portfolio.load_data()
close = portfolio.get_close_prices('SPY')

# Setup alpha model
alpha_model = EMA(short_window=10, long_window=30)
alpha_signals = alpha_model.generate_signals(close)

# Filter to 2023-2024
mask = (alpha_signals.index >= '2023-01-01') & (alpha_signals.index <= '2024-12-31')
signals_2023_2024 = alpha_signals[mask]

# Find signal changes
signal_changes = signals_2023_2024.diff()
entries = signal_changes[signal_changes == 1].index
exits = signal_changes[signal_changes == -1].index

print(f"Period: 2023-01-01 to 2024-12-31")
print(f"Total days: {len(signals_2023_2024)}")
print(f"Days in position: {signals_2023_2024.sum()} ({signals_2023_2024.sum()/len(signals_2023_2024)*100:.1f}%)")
print(f"\nBuy signals (entries): {len(entries)}")
if len(entries) > 0:
    for date in entries:
        print(f"  {date.strftime('%Y-%m-%d')}: Price ${close.loc[date]:.2f}")

print(f"\nSell signals (exits): {len(exits)}")
if len(exits) > 0:
    for date in exits:
        print(f"  {date.strftime('%Y-%m-%d')}: Price ${close.loc[date]:.2f}")

# Check what position we're in at key dates
print(f"\nPosition status at key dates:")
key_dates = ['2023-01-03', '2023-06-30', '2023-12-29', '2024-06-28', '2024-12-31']
for date_str in key_dates:
    if date_str in signals_2023_2024.index.strftime('%Y-%m-%d'):
        date = pd.Timestamp(date_str)
        pos = "IN" if signals_2023_2024.loc[date] == 1 else "OUT"
        print(f"  {date_str}: {pos} position (Price: ${close.loc[date]:.2f})")

"""
TESTE DO BUG: Divisao incorreta dos retornos diarios.

O bug: quando alguma acao do portfolio nao tem retorno numa data,
o codigo divide por 'valid' (menor) em vez de 'top_n' (fixo em 5).

Exemplo: portfolio tem 5 acoes, mas 2 nao tem dados nesse dia.
  - Codigo atual: daily_ret / 3  (inflado!)
  - Correto:      daily_ret / 5  (capital dividido entre 5 acoes)

Isso infla o retorno em fator = top_n / valid.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

TICKER = 'AAPL'
top_n = 5

# Download data
prices = yf.download(TICKER, period="10y", progress=False, auto_adjust=True)['Close'].squeeze()
stock_ret = prices.pct_change()

raw = yf.download([TICKER, "^GSPC"], period="10y", progress=False, auto_adjust=True)
data2 = pd.DataFrame()
data2['Close'] = raw['Close'][TICKER]
data2 = data2.dropna()

test_start = int(len(data2) * 0.8)
test_dates = list(data2.index[test_start:])

print(f"Test dates: {len(test_dates)}")
print(f"AAPL dates in stock_ret: {len(stock_ret)}")
print()

# Simulate model with 5 identical copies of AAPL (to isolate the bug)
# If valid < 5, the buggy code divides by valid instead of 5
model_buggy = 10000
model_fixed = 10000
days_with_bug = 0
total_bug_factor = 0

for d in test_dates:
    if d in stock_ret.index:
        r = stock_ret.loc[d]
        if not np.isnan(r):
            # Simulate: model holds 5 stocks, but only 3 have data on this date
            # (AAPL has data, but 2 "phantom" stocks don't)
            valid = 3  # Only 3 out of 5 stocks have data
            daily_ret = r * valid  # Sum of returns (3 stocks each with return r)
            
            # Buggy: divide by valid (3)
            buggy_avg = daily_ret / valid
            model_buggy *= (1 + buggy_avg)
            
            # Fixed: divide by top_n (5) 
            fixed_avg = daily_ret / top_n
            model_fixed *= (1 + fixed_avg)
            
            if valid < top_n:
                days_with_bug += 1
                total_bug_factor += top_n / valid

print("=== SIMULATION: 5 stocks, but only 3 have data each day ===")
print(f"Days where valid < top_n: {days_with_bug} / {len(test_dates)}")
print(f"Average bug factor: {total_bug_factor / max(days_with_bug, 1):.3f}")
print(f"Buggy return:   {(model_buggy/10000-1)*100:+.1f}%")
print(f"Fixed return:   {(model_fixed/10000-1)*100:+.1f}%")
print(f"Ratio:          {model_buggy/model_fixed:.2f}x")
print()

# Now test with REALISTIC scenario: 
# Some stocks genuinely missing on some dates
print("=== REALISTIC TEST: 5 real stocks with varying data availability ===")
tickers_5 = ['AAPL', 'MSFT', 'NVDA', 'DKNG', 'COIN']
all_ret = {}
for t in tickers_5:
    try:
        p = yf.download(t, period="10y", progress=False, auto_adjust=True)['Close'].squeeze()
        all_ret[t] = p.pct_change()
    except:
        pass

print(f"Got data for: {list(all_ret.keys())}")

# Check how many dates each stock is missing
for t in all_ret:
    missing = [d for d in test_dates if d not in all_ret[t].index]
    total_valid = sum(1 for d in test_dates if d in all_ret[t].index and not np.isnan(all_ret[t].get(d, np.nan)))
    print(f"  {t}: {total_valid}/{len(test_dates)} dates have returns ({len(missing)} missing)")

# Now simulate the actual bug with real data
model_buggy_real = 10000
model_fixed_real = 10000
bug_days = 0
bug_factors = []

for d in test_dates:
    # Get returns for all 5 stocks on this date
    rets = []
    for t in all_ret:
        if d in all_ret[t].index:
            r = all_ret[t].loc[d]
            if not np.isnan(r):
                rets.append(r)
    
    valid = len(rets)
    if valid > 0:
        total_ret = sum(rets)
        
        # Buggy: divide by valid
        buggy_avg = total_ret / valid
        model_buggy_real *= (1 + buggy_avg)
        
        # Fixed: divide by top_n (5)
        fixed_avg = total_ret / top_n
        model_fixed_real *= (1 + fixed_avg)
        
        if valid < top_n:
            bug_days += 1
            bug_factors.append(top_n / valid)

print(f"\nDays where valid < {top_n}: {bug_days} / {len(test_dates)}")
if bug_factors:
    print(f"Avg bug factor on those days: {np.mean(bug_factors):.3f}")
    print(f"Max bug factor: {max(bug_factors):.3f}")

print(f"\nBuggy return:   {(model_buggy_real/10000-1)*100:+.1f}%")
print(f"Fixed return:   {(model_fixed_real/10000-1)*100:+.1f}%")
print(f"Ratio:          {model_buggy_real/model_fixed_real:.2f}x")

# KEY TEST: What happens when COIN/DKNG are missing (early period)?
print("\n=== EARLY PERIOD IMPACT (before COIN/DKNG IPO) ===")
early_dates = [d for d in test_dates if d < pd.Timestamp('2021-04-14')]  # Before COIN IPO
print(f"Early dates (before COIN IPO): {len(early_dates)}")

early_bug_days = 0
for d in early_dates:
    rets = []
    for t in all_ret:
        if d in all_ret[t].index:
            r = all_ret[t].loc[d]
            if not np.isnan(r):
                rets.append(r)
    valid = len(rets)
    if valid < top_n and valid > 0:
        early_bug_days += 1

print(f"Days with valid < {top_n} in early period: {early_bug_days} / {len(early_dates)}")
print(f"This means {early_bug_days}/{len(early_dates)} = {early_bug_days/max(len(early_dates),1)*100:.0f}% of early days have inflated returns!")

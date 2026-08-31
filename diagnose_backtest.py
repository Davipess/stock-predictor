"""
CRITICAL diagnostic: Trace the exact backtest loop to find the bug.
Runs the actual backtest logic with step-by-step logging.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier

# Simplified prepare_data without sentiment
def prepare_quick(ticker, years=10):
    raw = yf.download([ticker, "^GSPC"], period=f"{years}y", progress=False, auto_adjust=True)
    d = pd.DataFrame()
    try:
        d['Close'] = raw['Close'][ticker]
        d['High'] = raw['High'][ticker]
        d['Low'] = raw['Low'][ticker]
        d['Volume'] = raw['Volume'][ticker]
    except:
        return None, None
    
    d['Ret_1d'] = d['Close'].pct_change(1)
    max_move = d['Ret_1d'].abs().max()
    if max_move > 0.50:
        return None, None
    
    d.ta.sma(length=20, append=True)
    d.ta.sma(length=50, append=True)
    d.ta.rsi(length=14, append=True)
    d.ta.macd(fast=12, slow=26, signal=9, append=True)
    d.ta.atr(length=14, append=True)
    
    bollinger = ta.bbands(d['Close'], length=20, std=2)
    d['BB_Width'] = (bollinger['BBU_20_2.0_2.0'] - bollinger['BBL_20_2.0_2.0']) / d['Close']
    d['Dist_52w_High_%'] = (d['Close'] - d['High'].rolling(252, min_periods=1).max()) / d['High'].rolling(252, min_periods=1).max()
    d['Volatility_20d'] = d['Ret_1d'].rolling(20).std()
    d['Rel_Volume'] = d['Volume'] / d['Volume'].rolling(20).mean()
    
    sp500 = raw['Close']['^GSPC'].pct_change()
    d['Corr_SP500_20d'] = d['Ret_1d'].rolling(20).corr(sp500)
    
    d['Price_vs_SMA20'] = (d['Close'] - d['SMA_20']) / d['SMA_20']
    d['Price_vs_SMA50'] = (d['Close'] - d['SMA_50']) / d['SMA_50']
    
    d['Close_Future'] = d['Close'].shift(-10)
    d = d.dropna()
    d['Target_10d'] = (d['Close_Future'] > d['Close']).astype(int)
    d = d.drop(columns=['Close_Future'])
    
    for col in ['Close', 'High', 'Low', 'SMA_20', 'SMA_50']:
        if col in d.columns:
            d = d.drop(columns=[col])
    
    return d, 'Target_10d'

# Build universe
universe = ['AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'GOOGL',
            'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'JPM', 'V',
            'INTC', 'BA', 'T', 'VZ', 'PFE', 'CVX', 'XOM',
            'PYPL', 'DKNG', 'RBLX', 'COIN', 'ABNB', 'UBER',
            'PTON']

print("Building universe...")
all_stock_data = {}
for t in universe:
    data, tc = prepare_quick(t)
    if data is not None:
        all_stock_data[t] = data
        print(f"  {t}: {len(data)} rows")
    else:
        print(f"  {t}: SKIPPED")

# Price data (SEPARATE download)
price_data = {}
for t in all_stock_data:
    p = yf.download(t, period="10y", progress=False, auto_adjust=True)['Close'].squeeze()
    price_data[t] = p

# Test dates from first ticker
min_len = min(len(d) for d in all_stock_data.values())
test_start_idx = int(min_len * 0.8)
first_ticker = list(all_stock_data.keys())[0]
test_dates = list(all_stock_data[first_ticker].index[test_start_idx:])

print(f"\nShortest stock: {min_len} rows")
print(f"Test starts at index: {test_start_idx}")
print(f"Test dates: {len(test_dates)} ({test_dates[0].date()} to {test_dates[-1].date()})")

# Stock returns from SEPARATE download
stock_returns = {}
for t in all_stock_data:
    if t in price_data:
        stock_returns[t] = price_data[t].pct_change()

# KEY TEST: Check alignment for first 10 test dates
print(f"\n=== ALIGNMENT TEST ===")
for d in test_dates[:10]:
    found = 0
    missing = []
    for t in all_stock_data:
        if t in stock_returns and d in stock_returns[t].index:
            found += 1
        else:
            missing.append(t)
    print(f"  {d.date()}: {found}/{len(all_stock_data)} stocks have returns | Missing: {missing[:3]}")

# Run abbreviated backtest with logging
print(f"\n=== BACKTEST DIAGNOSTIC ===")
initial_capital = 10000
model_value = initial_capital
ew_value = initial_capital
model_holdings = {}
rebalance_counter = 0
rebalance_days = 10
top_n = 5

# Log key metrics
daily_model_rets = []
daily_ew_rets = []
rebalance_count = 0

for i, date in enumerate(test_dates):
    # Track model daily return
    if model_holdings:
        daily_ret = 0
        valid = 0
        for t in model_holdings:
            if t in stock_returns and date in stock_returns[t].index:
                r = stock_returns[t].loc[date]
                if not np.isnan(r):
                    daily_ret += r
                    valid += 1
        if valid > 0:
            avg_ret = daily_ret / valid
            model_value *= (1 + avg_ret)
            daily_model_rets.append(avg_ret)
    
    # Track EW daily return
    ew_ret = 0
    ew_valid = 0
    for t in stock_returns:
        if date in stock_returns[t].index:
            r = stock_returns[t].loc[date]
            if not np.isnan(r):
                ew_ret += r
                ew_valid += 1
    if ew_valid > 0:
        avg_ew = ew_ret / ew_valid
        ew_value *= (1 + avg_ew)
        daily_ew_rets.append(avg_ew)
    
    # Rebalance
    rebalance_counter += 1
    if rebalance_counter >= rebalance_days:
        rebalance_counter = 0
        rebalance_count += 1
        
        # Retrain XGBoost for each stock
        scores = {}
        for t in all_stock_data:
            try:
                fd = all_stock_data[t]
                date_idx = fd.index.get_loc(date)
                start_idx = max(0, date_idx - 5 * 252)
                if date_idx < 100:
                    continue
                window = fd.iloc[start_idx:date_idx]
                if len(window) < 100:
                    continue
                tc = [c for c in window.columns if c.startswith('Target_')][0]
                X = window.drop(columns=[tc])
                y = window[tc]
                if y.nunique() < 2:
                    continue
                n_up = (y == 1).sum()
                n_down = (y == 0).sum()
                model = XGBClassifier(
                    eval_metric='logloss', random_state=42,
                    scale_pos_weight=n_down/n_up if n_up > 0 else 1,
                    n_estimators=80, max_depth=4, learning_rate=0.05
                )
                model.fit(X, y)
                today = fd.iloc[date_idx:date_idx+1].drop(columns=[tc], errors='ignore')
                if len(today) > 0 and list(today.columns) == list(X.columns):
                    prob = model.predict_proba(today)[:, 1][0]
                    scores[t] = prob
            except:
                continue
        
        if scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            new_top = set(t for t, _ in ranked[:top_n])
            
            if rebalance_count <= 3 or rebalance_count % 10 == 0:
                print(f"\n  Rebalance #{rebalance_count} at {date.date()}:")
                print(f"    Top 5: {[(t, f'{s:.3f}') for t, s in ranked[:5]]}")
                print(f"    Model value: ${model_value:,.0f} | EW value: ${ew_value:,.0f}")
                
                # Check what these stocks actually return over next 10 days
                next_dates = test_dates[i:i+10] if i+10 < len(test_dates) else test_dates[i:]
                for t, _ in ranked[:3]:
                    if t in stock_returns:
                        period_rets = [stock_returns[t].get(d, 0) for d in next_dates if d in stock_returns[t].index]
                        if period_rets:
                            total_ret = np.prod([1+r for r in period_rets if not np.isnan(r)]) - 1
                            print(f"    {t} next 10d: {total_ret*100:+.1f}%")
            
            # Apply transaction costs
            old = set(model_holdings.keys())
            trades = len(old.symmetric_difference(new_top))
            cost = trades * 0.001
            model_value *= (1 - cost)
            model_holdings = {t: 1 for t in new_top}

# Final summary
print(f"\n{'='*60}")
print(f"DIAGNOSTIC RESULTS")
print(f"{'='*60}")
print(f"Rebalances: {rebalance_count}")
print(f"Model final: ${model_value:,.0f} ({(model_value/initial_capital-1)*100:+.1f}%)")
print(f"EW final: ${ew_value:,.0f} ({(ew_value/initial_capital-1)*100:+.1f}%)")
print(f"\nModel avg daily return: {np.mean(daily_model_rets)*100:.4f}%")
print(f"EW avg daily return: {np.mean(daily_ew_rets)*100:.4f}%")
print(f"Ratio: {np.mean(daily_model_rets)/np.mean(daily_ew_rets):.1f}x")

# Compounding sanity
print(f"\nIf avg daily ret of {np.mean(daily_model_rets)*100:.4f}% compounds for 2000 days:")
print(f"  = {(1+np.mean(daily_model_rets))**2000:.1f}x = {((1+np.mean(daily_model_rets))**2000 - 1)*100:.0f}%")

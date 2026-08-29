import yfinance as yf
import pandas_ta as ta
import pandas as pd

def prepare_data(ticker="AAPL", target_days=5):
    # Fetch the data on a stock (2 years of history)
    stock = yf.Ticker(ticker)
    data = stock.history(period="2y") 

    """ Calculate Returns 
        Essentially we look at how much a specific stock raised or declined in a percentage.
        This is useful because we can then compare the momentums, it might raise by 5% in 1 day 
        but in 20 days it had fell down 20% so it could still be undervalued, depending on other factors of course.
    """
    data['Ret_1d'] = data['Close'].pct_change(periods=1)
    data['Ret_5d'] = data['Close'].pct_change(periods=5)
    data['Ret_20d'] = data['Close'].pct_change(periods=20)

    # Add Trend & Momentum Indicators such as RSI SMA and MACD
    data.ta.sma(length=20, append=True) # 20-day Simple Moving Average (creates 'SMA_20')
    data.ta.rsi(length=14, append=True) # 14-day RSI (creates 'RSI_14')
    data.ta.macd(fast=12, slow=26, signal=9, append=True) # MACD // this is what dictates if there has been optimism lately

    # Calculate Distance to 52-week high (252 trading days)
    rolling_52w_high = data['High'].rolling(window=252).max()
    data['Dist_52w_High_%'] = (data['Close'] - rolling_52w_high) / rolling_52w_high

    # Clean up rows with missing values
    data = data.dropna()

    # Create the column for the future price (dynamic days ahead)
    data['Close_Future'] = data['Close'].shift(-target_days)

    # Clean the last 'target_days' because of the value wipe on the shift
    data = data.dropna()

    # Create the target, 1 if it rises, 0 if it declines
    target_col_name = f'Target_{target_days}d'
    data[target_col_name] = (data['Close_Future'] > data['Close']).astype(int)

    # Clear the Close_Future column so the model doesn't become biased
    data = data.drop(columns=['Close_Future'])

    return data, target_col_name

# Test the function for a specific number of days (ex, 3 days)
dataset, target_col = prepare_data(ticker="AAPL", target_days=3)

# Results printed out
print(f"Features created successfully. Target is set to {target_col}. Latest metrics:")
print(dataset[['Close', 'Ret_1d', 'SMA_20', 'RSI_14', target_col]].tail())
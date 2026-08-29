import yfinance as yf

apple = yf.Ticker("AAPL")

data_apple = apple.history(period = "1y")

print(data_apple.head())
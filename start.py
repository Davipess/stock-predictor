import yfinance as yf
import pandas_ta as ta
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

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

    # Create the column for the future price for X days ahead
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

def train_baseline_model(dataset, target_col, n_splits=3):
    """  
        Splits the data chronologically to prevent lookahead bias and trains an XGBoost model.
    """
    # Separate Features/Numbers (X) from the Target (y)

    X = dataset.drop(columns=[target_col]) # Everything
    y = dataset[target_col] # Everything except the predictions

    """ Time Series Cross-Validation (Project Rule: NEVER use random train_test_split)
        this is because if we feed it a date ahead and ask for one in the past, he will get it right
        but then he will be biased and won't produce meaningfull predictions in the future
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize the XGBoost Classifier
    model = XGBClassifier(eval_metric='logloss', random_state=42)

    print(f"\nTraining XGBoost Model to predict {target_col}...")

    fold = 1
    for train_index, test_index in tscv.split(X):

        # Split the data into chronologically ordered train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model on the past
        model.fit(X_train, y_train)
        
        # Make predictions on the "future", practice on the test set
        predictions = model.predict(X_test)
        
        # Evaluate how many directions (Up/Down) it guessed correctly
        accuracy = accuracy_score(y_test, predictions)
        print(f"Fold {fold} Accuracy: {accuracy * 100:.2f}%")
        
        fold += 1
        
    return model

def main():
    print("Starting Stock Predictor Pipeline...")
    dataset, target_col = prepare_data(ticker="AAPL", target_days=3)
    
    trained_model = train_baseline_model(dataset, target_col, n_splits=3)
    
    print("\nPipeline execution finished successfully!")


if __name__ == "__main__":
    main()
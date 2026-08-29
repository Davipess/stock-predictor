import yfinance as yf
import pandas_ta as ta
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

def prepare_data(ticker="AAPL", target_days=5):

    """ Fetch 5 years of history instead of 2.
        More data = more patterns for the model to learn from.
        With 2 years we only had ~250 usable rows after rolling windows.
        
        yf.download() fetches multiple tickers at once and aligns their dates.
        This avoids the NaN problem where different tickers had different date formats.
    """
    tickers = [ticker, "^GSPC", "^VIX"]
    raw_data = yf.download(tickers, period="5y", progress=False)
    
    """ yf.download() returns a DataFrame with MultiIndex columns like (Close, AAPL).
        We extract each ticker's Close price and build our main DataFrame from the stock.
    """
    data = pd.DataFrame()
    data['Close'] = raw_data['Close'][ticker]
    data['High'] = raw_data['High'][ticker]
    data['Low'] = raw_data['Low'][ticker]
    data['Open'] = raw_data['Open'][ticker]
    data['Volume'] = raw_data['Volume'][ticker]
    data['SP500_Close'] = raw_data['Close']['^GSPC']
    data['VIX'] = raw_data['Close']['^VIX']

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

    """ Bollinger Bands: Upper and lower bands around the SMA based on volatility.
        When price touches the upper band, it might be overbought (expensive).
        When it touches the lower band, it might be oversold (cheap).
        We store the width of the band (upper - lower) as a measure of volatility.
    """
    bollinger = ta.bbands(data['Close'], length=20, std=2)
    data['BB_Upper'] = bollinger['BBU_20_2.0_2.0']
    data['BB_Lower'] = bollinger['BBL_20_2.0_2.0']
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['Close']  # Normalized width

    """ ATR (Average True Range): Measures the average size of price movements.
        Unlike volatility (which uses returns), ATR uses actual price ranges (High - Low).
        High ATR = big daily swings = more volatile market.
    """
    data.ta.atr(length=14, append=True)  # Creates 'ATRr_14'

    """ Stochastic Oscillator: Compares the closing price to the price range over a period.
        %K = where is the price relative to the high/low range (0-100)
        %D = smoothed version of %K (signal line)
        Above80 = overbought, below20 = oversold.
        Useful for spotting reversals.
    """
    stoch = ta.stoch(data['High'], data['Low'], data['Close'])
    data['Stoch_K'] = stoch['STOCHk_14_3_3']
    data['Stoch_D'] = stoch['STOCHd_14_3_3']

    # Calculate Distance to 52-week high (252 trading days)
    rolling_52w_high = data['High'].rolling(window=252).max()
    data['Dist_52w_High_%'] = (data['Close'] - rolling_52w_high) / rolling_52w_high

    """ Volatility: How "unstable" the price is.
        Calculated as the standard deviation of daily returns over 20 days.
        High volatility = big swings = more risk but also more opportunity.
    """
    data['Volatility_20d'] = data['Ret_1d'].rolling(window=20).std()

    """ Relative Volume: Today's volume compared to the 20-day average.
        A spike in volume often means something important is happening.
        This is a key signal for FOMO detection later.
    """
    avg_volume = data['Volume'].rolling(window=20).mean()
    data['Rel_Volume'] = data['Volume'] / avg_volume

    """ Market Context: S&P 500 returns and VIX correlation.
        If the whole market is up, a stock going up might not mean anything special.
        We calculate correlation to separate stock-specific moves from market moves.
    """
    data['SP500_Ret'] = data['SP500_Close'].pct_change(periods=1)
    data['Corr_SP500_20d'] = data['Ret_1d'].rolling(window=20).corr(data['SP500_Ret'])

    print(f"DEBUG: Raw data rows: {len(data)}")

    # Clean up rows with missing values
    data = data.dropna()
    print(f"DEBUG: After dropna (indicators): {len(data)}")

    # Create the column for the future price for X days ahead
    data['Close_Future'] = data['Close'].shift(-target_days)

    # Clean the last 'target_days' because of the value wipe on the shift 
    data = data.dropna()
    print(f"DEBUG: After dropna (target): {len(data)}")

    # Create the target, 1 if it rises, 0 if it declines
    target_col_name = f'Target_{target_days}d'
    data[target_col_name] = (data['Close_Future'] > data['Close']).astype(int)

    # Clear the Close_Future column so the model doesn't become biased
    data = data.drop(columns=['Close_Future'], errors='ignore')

    # Drop columns we used only for calculations earlier
    data = data.drop(columns=['SP500_Close', 'SP500_Ret'], errors='ignore')

    return data, target_col_name

def check_class_balance(dataset, target_col):
    """ 
        Check if the dataset is balanced between Up (1) and Down (0).
        If 70% of days are up, the model can just predict "Up" always and get 70% accuracy
        without learning anything useful.
    """
    total = len(dataset)
    
    if total == 0:
        print("\nERROR: Dataset is empty! No data to train on.")
        return
    
    counts = dataset[target_col].value_counts()
    
    down_count = counts.get(0, 0)
    up_count = counts.get(1, 0)
    
    print(f"\nClass Distribution:")
    print(f"  Down (0): {down_count} days ({down_count/total*100:.1f}%)")
    print(f"  Up   (1): {up_count} days ({up_count/total*100:.1f}%)")
    
    # A ratio close to50/50 is ideal
    imbalance = abs(down_count - up_count) / total
    if imbalance > 0.1:
        print(f"  WARNING: Dataset is imbalanced ({imbalance*100:.1f}% difference)")
    else:
        print(f"  Dataset is well balanced.")

def train_baseline_model(dataset, target_col, n_splits=3):
    """  
        Splits the data chronologically to prevent lookahead bias and trains an XGBoost model.
        Now includes Precision, Recall, and F1-Score for better evaluation.
    """
    # Separate Features/Numbers (X) from the Target (y)

    X = dataset.drop(columns=[target_col]) # Everything except the predictions
    y = dataset[target_col] # Just the predictions

    """ Time Series Cross-Validation (Project Rule: NEVER use random train_test_split)
        this is because if we feed it a date ahead and ask for one in the past, he will get it right
        but then he will be biased and won't produce meaningfull predictions in the future
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize the XGBoost Classifier
    model = XGBClassifier(eval_metric='logloss', random_state=42)

    print(f"\nTraining XGBoost Model to predict {target_col}...")

    fold = 1
    all_accuracies = []
    all_f1s = []
    
    for train_index, test_index in tscv.split(X):

        # Split the data into chronologically ordered train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model on the past
        model.fit(X_train, y_train)
        
        # Make predictions on the "future", practice on the test set
        predictions = model.predict(X_test)
        
        """ Evaluate multiple metrics, not just accuracy.
            - Precision: Of all "Up" predictions, how many were correct?
            - Recall: Of all actual "Up" days, how many did we catch?
            - F1-Score: Balance between Precision and Recall
        """
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        print(f"Fold {fold}: Accuracy={accuracy*100:.2f}% | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}")
        
        all_accuracies.append(accuracy)
        all_f1s.append(f1)
        fold += 1
        
    # Print summary
    print(f"\n--- Summary ---")
    print(f"Average Accuracy: {sum(all_accuracies)/len(all_accuracies)*100:.2f}%")
    print(f"Average F1-Score: {sum(all_f1s)/len(all_f1s):.3f}")
    
    return model

def main():
    print("Starting Stock Predictor Pipeline...")
    dataset, target_col = prepare_data(ticker="AAPL", target_days=5)
    
    check_class_balance(dataset, target_col)
    
    print(f"\nDataset shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(f"Features: {list(dataset.columns)}")
    
    trained_model = train_baseline_model(dataset, target_col, n_splits=5)
    
    print("\nPipeline execution finished successfully!")


if __name__ == "__main__":
    main()
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

CONFIG = {
    'ticker': 'AAPL',           # Stock ticker to predict
    'target_days': 10,           # Days ahead to predict (5 = 1 week)
    'data_years': 10,            # Years of history to fetch
    'n_splits_cv': 5,           # Folds for cross-validation
    'feature_threshold': 0.02,  # Min importance to keep a feature (0.02 = 2%)
    'n_iter_tuning': 100,        # Hyperparameter combinations to test
    'random_ticker_columns': [  # Raw price columns to remove (data leakage)
        'Close', 'High', 'Low', 'Open',
        'SMA_20', 'SMA_50'
    ],
}

def prepare_data(ticker=None, target_days=None):

    """ Fetch 5 years of history instead of 2.
        More data = more patterns for the model to learn from.
        With 2 years we only had ~250 usable rows after rolling windows.
        
        yf.download() fetches multiple tickers at once and aligns their dates.
        This avoids the NaN problem where different tickers had different date formats.
    """
    if ticker is None:
        ticker = CONFIG['ticker']
    if target_days is None:
        target_days = CONFIG['target_days']
    
    years = CONFIG['data_years']
    period = f"{years}y"
    tickers = [ticker, "^GSPC", "^VIX"]
    raw_data = yf.download(tickers, period=period, progress=False)
    
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

    """ Different rolling windows for volatility and volume.
        Short windows (10d) = react fast to changes
        Long windows (50d) = smoother, less noise
    """
    data['Volatility_10d'] = data['Ret_1d'].rolling(window=10).std()
    data['Volatility_50d'] = data['Ret_1d'].rolling(window=50).std()
    
    data['Rel_Volume_10d'] = data['Volume'] / data['Volume'].rolling(window=10).mean()
    data['Rel_Volume_50d'] = data['Volume'] / data['Volume'].rolling(window=50).mean()

    """ Different rolling windows for SMA.
        Comparing SMA_20 vs SMA_50 tells us about trend strength.
    """
    data.ta.sma(length=50, append=True)  # Creates 'SMA_50'

    """ RSI rate of change: Is RSI rising or falling?
        RSI going from30 to40 (rising) is bullish momentum.
        RSI going from70 to60 (falling) is bearish momentum.
    """
    data['RSI_Change'] = data['RSI_14'] - data['RSI_14'].shift(1)

    """ Stochastic rate of change: Same logic as RSI.
    """
    data['Stoch_K_Change'] = data['Stoch_K'] - data['Stoch_K'].shift(1)

    """ Feature interactions: Combine features to reveal hidden patterns.
        Example: RSI high + Volume high = strong overbought with conviction
    """
    data['RSI_x_Volume'] = data['RSI_14'] * data['Rel_Volume']
    data['MACD_x_Volume'] = data['MACD_12_26_9'] * data['Rel_Volume']

    """ Price relative to moving averages.
        How far is the price from its average? Far above = potentially overextended.
    """
    data['Price_vs_SMA20'] = (data['Close'] - data['SMA_20']) / data['SMA_20']
    data['Price_vs_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']

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

def tune_hyperparameters(dataset, target_col, n_splits=3):
    """  
        RandomizedSearchCV tests random combinations of hyperparameters.
        Instead of trying every combination (GridSearch), it samples random ones.
        This is faster and usually finds good enough parameters.
        
        Key parameters we're tuning:
        - n_estimators: How many trees to build (more = more complex, risk of overfitting)
        - max_depth: How deep each tree can go (deeper = more complex patterns)
        - learning_rate: How fast the model learns (slower = more careful, needs more trees)
        - subsample: Fraction of data used for each tree (prevents overfitting)
        - colsample_bytree: Fraction of features used for each tree (prevents overfitting)
    """
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]

    # Calculate class weights for imbalance handling
    n_down = (y == 0).sum()
    n_up = (y == 1).sum()
    scale_pos_weight = n_down / n_up

    tscv = TimeSeriesSplit(n_splits=n_splits)

    """ Define the parameter distributions to sample from.
        Each parameter has a range of values to try.
    """
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'scale_pos_weight': uniform(0.5, 1.5),  # Tune class weight
    }

    # Initialize base model with class weight handling
    base_model = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)

    """ RandomizedSearchCV: Tests n_iter random combinations.
        cv=tscv uses time series split instead of random split.
        scoring='f1' optimizes for F1-score (balance of precision and recall).
        n_jobs=-1 uses all CPU cores (faster).
    """
    n_iter = CONFIG['n_iter_tuning']
    print(f"\nTuning hyperparameters (testing {n_iter} random combinations)...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,            # Test N random combinations
        cv=tscv,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X, y)

    # Print the best parameters found
    print(f"\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best F1-Score: {random_search.best_score_:.3f}")

    return random_search.best_estimator_

def train_baseline_model(dataset, target_col, model, n_splits=3):
    """  
        Splits the data chronologically to prevent lookahead bias and trains the model.
        Now includes Precision, Recall, and F1-Score for evaluation.
        
        The model is passed as parameter (from tune_hyperparameters) instead of being
        created inside this function. This ensures we train with the best parameters found.
    """
    # Separate Features/Numbers (X) from the Target (y)

    X = dataset.drop(columns=[target_col]) # Everything except the predictions
    y = dataset[target_col] # Just the predictions

    """ Time Series Cross-Validation (Project Rule: NEVER use random train_test_split)
        this is because if we feed it a date ahead and ask for one in the past, he will get it right
        but then he will be biased and won't produce meaningfull predictions in the future
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\nTraining Model to predict {target_col}...")

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

def select_features(dataset, target_col, threshold=None):
    """  
        Uses feature_importances_ from XGBoost to rank features.
        Features with importance below the threshold are removed.
        Also removes raw price columns (data leakage).
        
        Trains XGBoost on the data, each feature gets an importance score (how much it helps predictions)
        Features with very low importance are noise, they confuse the model
        We remove them and retrain with only the useful features
    """
    if threshold is None:
        threshold = CONFIG['feature_threshold']
    
    # Remove raw price columns (data leakage)
    columns_to_remove = CONFIG['random_ticker_columns']
    existing_to_remove = [col for col in columns_to_remove if col in dataset.columns]
    dataset = dataset.drop(columns=existing_to_remove, errors='ignore')
    print(f"\nRemoved raw price columns: {existing_to_remove}")

    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]

    # Train a model to get feature importances
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X, y)

    """ Get importance scores and sort them from highest to lowest.
        Importance = how much each feature contributes to the model's decisions.
    """
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print(f"\n--- Feature Importance Ranking ---")
    for feature, importance in importances.items():
        status = "KEEP" if importance >= threshold else "REMOVE"
        print(f"  {feature}: {importance:.4f} [{status}]")

    """ Select only features above the threshold.
        These are the features that actually help the model predict.
    """
    important_features = importances[importances >= threshold].index.tolist()
    
    print(f"\nKeeping {len(important_features)} out of {len(X.columns)} features")
    print(f"Removed {len(X.columns) - len(important_features)} noisy features")

    # Return dataset with only important features
    filtered_dataset = dataset[important_features + [target_col]]
    return filtered_dataset, important_features

def backtest(model, dataset, target_col, ticker=None):
    """
        Simulates three trading strategies and compares their performance:
        
        -Model Strategy: Buy when model predicts Up (1), sell when predicts Down (0)
        -S&P 500 Buy & Hold: Buy on day 1, hold forever (benchmark)
        -Random Strategy: Randomly buy/sell (to show that any strategy beats randomness)
        
        The chart shows cumulative returns over time for all three strategies.
    """
    if ticker is None:
        ticker = CONFIG['ticker']
    
    # Fetch S&P 500 data for the same period
    # .squeeze() converts single-column DataFrame to Series
    sp500_data = yf.download("^GSPC", period=f"{CONFIG['data_years']}y", progress=False)['Close'].squeeze()
    
    # Get the stock's close prices aligned with our dataset
    stock_data = yf.download(ticker, period=f"{CONFIG['data_years']}y", progress=False)['Close'].squeeze()
    
    # Prepare features and make predictions
    X = dataset.drop(columns=[target_col])
    predictions = model.predict(X)
    
    """ Add predictions and dates to the stock price data.
        We need to align the predictions with the actual dates and prices.
    """
    results = pd.DataFrame({
        'Price': stock_data.loc[dataset.index].values,
        'Prediction': predictions,
        'Actual': dataset[target_col].values
    }, index=dataset.index)
    
    """ Calculate daily returns for each strategy:
        
        Model Strategy:
        - If prediction is1 (Up): stay invested, earn the return
        - If prediction is0 (Down): cash out, earn0
        
        S&P 500:
        - Always invested, earn the market return
        
        Random:
        - Randomly decide to be invested or cash (50/50 chance)
    """
    import numpy as np
    np.random.seed(42)  # For reproducibility
    
    results['Stock_Return'] = results['Price'].pct_change()
    
    # Model strategy: invest only when predicting Up
    results['Model_Return'] = results['Stock_Return'] * results['Prediction']
    
    # S&P 500: get returns for the same dates
    results['SP500_Return'] = sp500_data.loc[results.index].pct_change()
    
    # Random strategy: randomly invest or stay cash
    results['Random_Decision'] = np.random.randint(0, 2, size=len(results))
    results['Random_Return'] = results['Stock_Return'] * results['Random_Decision']
    
    """ Calculate cumulative returns (growth of $1 invested).
        Cumulative return = (1 + r1) * (1 + r2) * ... * (1 + rn)
        This shows how $1 would have grown over time.
    """
    results['Model_Cumulative'] = (1 + results['Model_Return']).cumprod()
    results['SP500_Cumulative'] = (1 + results['SP500_Return']).cumprod()
    results['Random_Cumulative'] = (1 + results['Random_Return']).cumprod()
    
    """ Print performance summary.
        Total return = final value - initial value (as percentage)
    """
    model_total = (results['Model_Cumulative'].iloc[-1] - 1) * 100
    sp500_total = (results['SP500_Cumulative'].iloc[-1] - 1) * 100
    random_total = (results['Random_Cumulative'].iloc[-1] - 1) * 100
    
    print(f"\n{'='*50}")
    print(f"BACKTEST RESULTS ({len(results)} trading days)")
    print(f"{'='*50}")
    print(f"  Model Strategy:     {model_total:+.2f}%")
    print(f"  S&P 500 Buy&Hold:   {sp500_total:+.2f}%")
    print(f"  Random Strategy:    {random_total:+.2f}%")
    print(f"{'='*50}")
    
    # Calculate how many days the model was invested vs cash
    invested_days = (results['Prediction'] == 1).sum()
    cash_days = (results['Prediction'] == 0).sum()
    print(f"  Days Invested:      {invested_days} ({invested_days/len(results)*100:.1f}%)")
    print(f"  Days in Cash:       {cash_days} ({cash_days/len(results)*100:.1f}%)")
    
    """ Plot the three strategies on the same chart.
        This visually shows which strategy performed best over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results.index, results['Model_Cumulative'], label='Model Strategy', linewidth=2)
    ax.plot(results.index, results['SP500_Cumulative'], label='S&P 500 Buy & Hold', linewidth=2, linestyle='--')
    ax.plot(results.index, results['Random_Cumulative'], label='Random Strategy', linewidth=1, alpha=0.7, linestyle=':')
    
    ax.set_title(f'Backtest: Model vs S&P 500 vs Random ({ticker})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    plt.show()
    
    print(f"\nChart saved as 'backtest_results.png'")
    
    return results

def main():
    print("Starting Stock Predictor Pipeline...")
    print(f"Ticker: {CONFIG['ticker']} | Target: {CONFIG['target_days']} days | Data: {CONFIG['data_years']} years")
    
    dataset, target_col = prepare_data()
    
    check_class_balance(dataset, target_col)
    
    print(f"\nDataset shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(f"Features: {list(dataset.columns)}")
    
    # Remove raw price columns and noisy features
    filtered_dataset, kept_features = select_features(dataset, target_col)
    
    # Find the best hyperparameters on filtered data
    best_model = tune_hyperparameters(filtered_dataset, target_col, n_splits=CONFIG['n_splits_cv'])
    
    # Train the final model with the best parameters found by tuning
    print("\nTraining final model with best parameters...")
    trained_model = train_baseline_model(filtered_dataset, target_col, model=best_model, n_splits=CONFIG['n_splits_cv'])
    
    # Run backtest to compare model vs S&P 500 vs Random
    backtest_results = backtest(best_model, filtered_dataset, target_col)
    
    print("\nPipeline execution finished successfully!")


if __name__ == "__main__":
    main()
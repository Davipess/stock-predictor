import yfinance as yf
import pandas_ta as ta
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

CONFIG = {
    # MIXED universe: winners, losers, and average performers
    # This avoids survivorship bias
    'universe': [
        # WINNERS (large cap tech/growth)
        'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'GOOGL',
        # AVERAGE performers (steady but not spectacular)
        'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'JPM', 'V',
        # UNDERPERFORMERS (stocks that struggled)
        'INTC', 'BA', 'T', 'VZ', 'PFE', 'CVX', 'XOM',
        # SMALL/MID CAP (more volatile, harder to predict)
        'RIVN', 'PLTR', 'SOFI', 'ROKU', 'SNAP', 'PYPL', 'SQ',
        'DKNG', 'HOOD', 'RBLX', 'COIN', 'ABNB', 'UBER',
        'ZM', 'PTON', 'BYND', 'NIO', 'LCID', 'RIVN',
    ],
    'target_days': 10,           # Days ahead to predict
    'data_years': 10,            # Years of history to fetch
    'n_splits_cv': 5,           # Folds for cross-validation
    'feature_threshold': 0.02,  # Min importance to keep a feature
    'n_iter_tuning': 100,        # Hyperparameter combinations to test
    'test_size': 0.2,           # % of data held out for backtest
    'portfolio_top_n': 5,       # Buy top N stocks each rebalance
    'rebalance_days': 10,       # Rebalance every N trading days
    'initial_capital': 10000,   # Starting capital in $
    'transaction_cost': 0.001,  # 0.1% cost per trade (buy or sell)
    'random_ticker_columns': [  # Raw price columns to remove
        'Close', 'High', 'Low', 'Open',
        'SMA_20', 'SMA_50'
    ],
}

def prepare_data(ticker=None, target_days=None):

    """ Fetch data for a single stock.
        yf.download() fetches multiple tickers at once and aligns their dates.
        This avoids the NaN problem where different tickers had different date formats.
    """
    if ticker is None:
        ticker = CONFIG['tickers'][0]  # Default to first ticker
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

def backtest():
    """
        REALISTIC PORTFOLIO MANAGER BOT:
        
        1. Mixed universe (winners + losers + small caps) — no survivorship bias
        2. Transaction costs on every trade (0.1% per buy/sell)
        3. Fair benchmarks: S&P 500 AND Equal-Weight portfolio of same stocks
        4. Tracks total costs paid
        
        If the model can't beat a simple equal-weight portfolio of the same stocks
        after costs, it has no real value.
    """
    import numpy as np
    np.random.seed(42)
    
    universe = CONFIG['universe']
    test_size = CONFIG['test_size']
    top_n = CONFIG['portfolio_top_n']
    rebalance_days = CONFIG['rebalance_days']
    initial_capital = CONFIG['initial_capital']
    tx_cost = CONFIG['transaction_cost']  # 0.1% per trade
    
    print(f"\n{'='*60}")
    print(f"REALISTIC PORTFOLIO MANAGER BOT")
    print(f"Universe: {len(universe)} stocks | Top {top_n} | Rebalance: {rebalance_days}d")
    print(f"Transaction cost: {tx_cost*100:.1f}% per trade")
    print(f"{'='*60}")
    
    """ Train models and get predictions + probabilities for all stocks.
    """
    all_data = {}
    
    for ticker in universe:
        try:
            dataset, target_col = prepare_data(ticker=ticker)
            
            cols_to_remove = [c for c in CONFIG['random_ticker_columns'] if c in dataset.columns]
            dataset = dataset.drop(columns=cols_to_remove, errors='ignore')
            
            split_idx = int(len(dataset) * (1 - test_size))
            train_data = dataset.iloc[:split_idx]
            test_data = dataset.iloc[split_idx:]
            
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col]
            
            from xgboost import XGBClassifier
            n_down = (y_train == 0).sum()
            n_up = (y_train == 1).sum()
            model = XGBClassifier(
                eval_metric='logloss', random_state=42,
                scale_pos_weight=n_down/n_up if n_up > 0 else 1,
                n_estimators=100, max_depth=4, learning_rate=0.05
            )
            model.fit(X_train, y_train)
            
            X_test = test_data.drop(columns=[target_col])
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            
            stock_data = yf.download(ticker, period=f"{CONFIG['data_years']}y", progress=False)['Close'].squeeze()
            test_prices = stock_data.loc[test_data.index]
            
            all_data[ticker] = {
                'predictions': pd.Series(predictions, index=test_data.index),
                'probabilities': pd.Series(probabilities, index=test_data.index),
                'prices': test_prices,
                'returns': test_prices.pct_change()
            }
            
            print(f"  {ticker}: OK")
            
        except Exception as e:
            print(f"  {ticker}: SKIPPED ({e})")
            continue
    
    if not all_data:
        print("ERROR: No stocks processed.")
        return None
    
    # Common test dates
    all_dates = set()
    for ticker in all_data:
        all_dates.update(all_data[ticker]['probabilities'].index)
    common_dates = sorted(all_dates)
    
    print(f"\nTest period: {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} days)")
    print(f"Stocks with data: {len(all_data)}")
    
    """ FOUR strategies to compare:
        
        1. Model: Top N stocks by confidence, rebalance, with costs
        2. Equal-Weight: Hold ALL stocks equally, rebalance, with costs (FAIR benchmark)
        3. S&P 500: Buy and hold (no costs)
        4. Random: Pick N stocks randomly, rebalance, with costs
    """
    # Portfolios
    model_value = initial_capital
    ew_value = initial_capital      # Equal-weight benchmark
    sp500_value = initial_capital
    random_value = initial_capital
    
    model_history = []
    ew_history = []
    sp500_history = []
    random_history = []
    
    # Track costs
    model_total_costs = 0
    ew_total_costs = 0
    random_total_costs = 0
    
    # S&P 500
    sp500_data = yf.download("^GSPC", period=f"{CONFIG['data_years']}y", progress=False)['Close'].squeeze()
    sp500_returns = sp500_data.pct_change().reindex(common_dates, method='ffill').fillna(0)
    
    # Holdings
    model_holdings = {}
    ew_holdings = {}
    random_holdings = {}
    rebalance_counter = 0
    
    # Calculate equal-weight returns (buy and hold all stocks)
    all_returns = pd.DataFrame({t: all_data[t]['returns'] for t in all_data})
    ew_daily_returns = all_returns.mean(axis=1).reindex(common_dates, fill_value=0)
    
    for i, date in enumerate(common_dates):
        sp500_ret = sp500_returns.loc[date]
        
        """ Model portfolio: update value from holdings.
        """
        if model_holdings:
            daily_ret = 0
            valid = 0
            for ticker in model_holdings:
                if date in all_data[ticker]['returns'].index:
                    r = all_data[ticker]['returns'].loc[date]
                    if not np.isnan(r):
                        daily_ret += r
                        valid += 1
            if valid > 0:
                model_value *= (1 + daily_ret / valid)
        
        model_history.append(model_value)
        
        """ Equal-weight: always invested in all stocks.
        """
        ew_ret = ew_daily_returns.loc[date] if date in ew_daily_returns.index else 0
        if not np.isnan(ew_ret):
            ew_value *= (1 + ew_ret)
        ew_history.append(ew_value)
        
        """ S&P 500: buy and hold.
        """
        sp500_value *= (1 + sp500_ret)
        sp500_history.append(sp500_value)
        
        """ Random portfolio.
        """
        if random_holdings:
            rand_ret = 0
            rand_valid = 0
            for ticker in random_holdings:
                if date in all_data[ticker]['returns'].index:
                    r = all_data[ticker]['returns'].loc[date]
                    if not np.isnan(r):
                        rand_ret += r
                        rand_valid += 1
            if rand_valid > 0:
                random_value *= (1 + rand_ret / rand_valid)
        random_history.append(random_value)
        
        """ REBALANCE every N days.
            Compare old holdings to new, charge costs for changes.
        """
        rebalance_counter += 1
        if rebalance_counter >= rebalance_days:
            rebalance_counter = 0
            
            # Score all stocks
            scores = {}
            for ticker in all_data:
                if date in all_data[ticker]['probabilities'].index:
                    scores[ticker] = all_data[ticker]['probabilities'].loc[date]
            
            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                new_top = set(ticker for ticker, _ in ranked[:top_n])
                
                # MODEL: Sell old, buy new — count trades
                old_model = set(model_holdings.keys())
                sells = old_model - new_top
                buys = new_top - old_model
                model_trades = len(sells) + len(buys)
                model_cost = model_trades * tx_cost
                model_value *= (1 - model_cost)
                model_total_costs += model_cost * model_value
                
                # Also sell stocks predicted Down (not just top N)
                for ticker in list(model_holdings.keys()):
                    if ticker in all_data and date in all_data[ticker]['predictions'].index:
                        if all_data[ticker]['predictions'].loc[date] == 0:
                            model_value *= (1 - tx_cost)  # Cost to sell
                            model_total_costs += tx_cost * model_value
                
                model_holdings = {t: 1 for t in new_top}
                
                # EQUAL-WEIGHT: Rebalance to equal weight (costs apply)
                available = [t for t in all_data if date in all_data[t]['returns'].index]
                old_ew = set(ew_holdings.keys())
                new_ew = set(available)
                ew_trades = len(old_ew.symmetric_difference(new_ew))
                ew_cost = ew_trades * tx_cost * 0.5  # Only rebalance drift
                ew_value *= (1 - ew_cost)
                ew_total_costs += ew_cost * ew_value
                ew_holdings = {t: 1 for t in available}
                
                # RANDOM: Pick random stocks
                if available:
                    n_rand = min(top_n, len(available))
                    old_rand = set(random_holdings.keys())
                    new_rand = set(np.random.choice(available, n_rand, replace=False))
                    rand_trades = len(old_rand.symmetric_difference(new_rand))
                    rand_cost = rand_trades * tx_cost
                    random_value *= (1 - rand_cost)
                    random_total_costs += rand_cost * random_value
                    random_holdings = {t: 1 for t in new_rand}
                
                if i % (rebalance_days * 10) == 0:
                    print(f"  {date.date()}: Rebalanced | Model trades: {model_trades} | Top: {[t for t,_ in ranked[:3]]}")
    
    """ Final results.
    """
    model_return = (model_value / initial_capital - 1) * 100
    ew_return = (ew_value / initial_capital - 1) * 100
    sp500_return = (sp500_value / initial_capital - 1) * 100
    random_return = (random_value / initial_capital - 1) * 100
    
    print(f"\n{'='*60}")
    print(f"RESULTS (test set only, with transaction costs)")
    print(f"{'='*60}")
    print(f"  {'Strategy':<25} {'Final Value':>12} {'Return':>10} {'Costs Paid':>12}")
    print(f"  {'-'*59}")
    print(f"  {'Model (Top N)':<25} ${model_value:>10,.0f} {model_return:>+9.1f}% ${model_total_costs:>10,.0f}")
    print(f"  {'Equal-Weight (All)':<25} ${ew_value:>10,.0f} {ew_return:>+9.1f}% ${ew_total_costs:>10,.0f}")
    print(f"  {'S&P 500':<25} ${sp500_value:>10,.0f} {sp500_return:>+9.1f}% ${'0':>10}")
    print(f"  {'Random (Top N)':<25} ${random_value:>10,.0f} {random_return:>+9.1f}% ${random_total_costs:>10,.0f}")
    print(f"  {'='*60}")
    
    # Did the model beat equal-weight?
    alpha = model_return - ew_return
    print(f"\n  Model vs Equal-Weight (TRUE alpha): {alpha:+.1f}%")
    if alpha > 0:
        print(f"  Model ADDS value over passive holding!")
    else:
        print(f"  Model does NOT beat just buying everything equally.")
    
    """ Plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1.plot(common_dates, model_history, label=f'Model ({model_return:+.1f}%)', linewidth=2.5)
    ax1.plot(common_dates, ew_history, label=f'Equal-Weight ({ew_return:+.1f}%)', linewidth=2, linestyle='-.')
    ax1.plot(common_dates, sp500_history, label=f'S&P 500 ({sp500_return:+.1f}%)', linewidth=2, linestyle='--')
    ax1.plot(common_dates, random_history, label=f'Random ({random_return:+.1f}%)', linewidth=1.5, alpha=0.7, linestyle=':')
    
    ax1.set_title(f'Realistic Backtest: Transaction Costs + Mixed Universe\n({len(all_data)} stocks, Top {top_n}, rebalance every {rebalance_days}d, {tx_cost*100:.1f}% cost/trade)', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Alpha over time (Model vs Equal-Weight)
    alpha_over_time = [(m/ew - 1) * 100 for m, ew in zip(model_history, ew_history)]
    ax2.fill_between(common_dates, alpha_over_time, 0, alpha=0.3, color='green')
    ax2.plot(common_dates, alpha_over_time, label='Alpha vs Equal-Weight', linewidth=2, color='green')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f'True Alpha (Model minus Equal-Weight Benchmark)')
    ax2.set_ylabel('Alpha (%)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_backtest.png', dpi=150)
    plt.show()
    
    print(f"\nChart saved as 'realistic_backtest.png'")
    
    return {
        'model_return': model_return,
        'ew_return': ew_return,
        'sp500_return': sp500_return,
        'random_return': random_return,
        'alpha': alpha,
        'total_costs': model_total_costs
    }

def main():
    print("Starting Portfolio Manager Bot...")
    print(f"Universe: {len(CONFIG['universe'])} stocks | Target: {CONFIG['target_days']}d | Rebalance: {CONFIG['rebalance_days']}d")
    
    # Run autonomous portfolio manager backtest
    results = backtest()
    
    print("\nPipeline execution finished successfully!")


if __name__ == "__main__":
    main()
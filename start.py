import yfinance as yf
import pandas_ta as ta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from lstm_model import StockLSTM, create_sequences, train_lstm, predict_lstm

# Load environment variables (.env file)
# Look for .env in the same directory as this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, '.env'))

CONFIG = {
    # Mixed universe: winners, losers, and average performers
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
        'ZM', 'PTON', 'RBLX', 'NIO', 'LCID', 'RIVN',
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
    'retrain_years': 5,         # Years of data used for retraining
    'random_ticker_columns': [  # Raw price columns to remove (data leakage)
        'Close', 'High', 'Low', 'Open'
    ],
    # News Sentiment (Phase 2)
    'use_news_sentiment': True,   # Enable/disable news features
    'news_lookback_days': 30,     # Days of news to fetch per request
    'sentiment_window': 5,        # Rolling window for sentiment features
    # LSTM Ensemble (Phase 3)
    'use_lstm': False,            # LSTM experimental (too slow for full backtest)
    'lstm_weight': 0.2,           # Weight for LSTM (1 - weight = XGBoost weight)
    'lstm_seq_length': 20,        # Days of history for LSTM sequences
    'lstm_epochs': 30,            # Training epochs for LSTM
    'lstm_hidden': 32,            # LSTM hidden units
}

# ============================================
# NEWS SENTIMENT MODULE (Phase 2)
# ============================================

# Global variables for FinBERT model (loaded once)
_finbert_model = None
_finbert_tokenizer = None

def load_finbert():
    """Load FinBERT model and tokenizer (lazy loading - only when needed)."""
    global _finbert_model, _finbert_tokenizer
    
    if _finbert_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        print("Loading FinBERT model...")
        model_name = "ProsusAI/finbert"
        _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("FinBERT loaded successfully.")
    
    return _finbert_model, _finbert_tokenizer

def analyze_sentiment(text):
    """Analyze sentiment of a single text using FinBERT.
    Returns: 'positive', 'negative', or 'neutral'
    """
    import torch
    
    model, tokenizer = load_finbert()
    labels = ["positive", "negative", "neutral"]
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    return labels[probs.argmax()]

def fetch_news_finnhub(ticker, days=None):
    """Fetch news articles from Finnhub for a given ticker.
    Returns list of dicts with: headline, summary, datetime, source
    """
    import finnhub
    
    if days is None:
        days = CONFIG['news_lookback_days']
    
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        print(f"  WARNING: No FINNHUB_API_KEY found in .env. Skipping news for {ticker}.")
        return []
    
    client = finnhub.Client(api_key=api_key)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        news = client.company_news(
            ticker,
            _from=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        )
        
        # Process and clean news
        processed = []
        for article in news:
            if article.get('headline') and article.get('datetime'):
                processed.append({
                    'headline': article['headline'],
                    'summary': article.get('summary', ''),
                    'datetime': datetime.fromtimestamp(article['datetime']),
                    'source': article.get('source', 'Unknown')
                })
        
        return processed
        
    except Exception as e:
        print(f"  ERROR fetching news for {ticker}: {e}")
        return []

def get_daily_sentiment(ticker, stock_dates):
    """Get daily sentiment features for a stock.
    
    For each trading day, calculates:
    - News_Sentiment: average sentiment of news that day (+1=pos, -1=neg, 0=neutral)
    - News_Volume: number of news articles that day
    - News_Positive_Ratio: % of positive articles that day
    
    Args:
        ticker: stock ticker
        stock_dates: DatetimeIndex of trading dates to align with
    
    Returns:
        DataFrame with sentiment features indexed by date
    """
    # Fetch news
    news_articles = fetch_news_finnhub(ticker)
    
    if not news_articles:
        # No news available - return zeros with all expected columns
        return pd.DataFrame({
            'News_Sentiment': 0.0,
            'News_Volume': 0,
            'News_Positive_Ratio': 0.0,
            'Sentiment_MA': 0.0,
            'Sentiment_Std': 0.0
        }, index=stock_dates)
    
    # Analyze sentiment for each article
    print(f"  Analyzing sentiment for {len(news_articles)} articles...")
    for article in news_articles:
        article['sentiment'] = analyze_sentiment(article['headline'])
    
    # Convert to DataFrame
    news_df = pd.DataFrame(news_articles)
    news_df['date'] = news_df['datetime'].dt.date
    
    # Aggregate by date
    sentiment_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
    
    daily_sentiment = []
    for date in stock_dates:
        day_news = news_df[news_df['date'] == date.date()]
        
        if len(day_news) > 0:
            sentiments = day_news['sentiment'].map(sentiment_map)
            avg_sentiment = sentiments.mean()
            news_count = len(day_news)
            positive_ratio = (day_news['sentiment'] == 'positive').sum() / news_count
        else:
            avg_sentiment = 0.0
            news_count = 0
            positive_ratio = 0.0
        
        daily_sentiment.append({
            'date': date,
            'News_Sentiment': avg_sentiment,
            'News_Volume': news_count,
            'News_Positive_Ratio': positive_ratio
        })
    
    result = pd.DataFrame(daily_sentiment).set_index('date')
    
    # Add rolling sentiment features
    window = CONFIG['sentiment_window']
    result['Sentiment_MA'] = result['News_Sentiment'].rolling(window, min_periods=1).mean()
    result['Sentiment_Std'] = result['News_Sentiment'].rolling(window, min_periods=1).std().fillna(0)
    
    return result

def prepare_data(ticker=None, target_days=None):

    """ Fetch data for a single stock.
        yf.download() fetches multiple tickers at once and aligns their dates.
        This avoids the NaN problem where different tickers had different date formats.
    """
    if ticker is None:
        ticker = CONFIG['universe'][0]  # Default to first ticker
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

    """ Data Quality Filter: Detect reverse splits or data errors.
        A single-day move >50% is almost certainly a split, not a real market move.
        These corrupt the model's ability to learn real patterns.
    """
    max_daily_move = data['Ret_1d'].abs().max()
    if max_daily_move > 0.50:
        print(f"  WARNING: {ticker} has a {max_daily_move*100:.0f}% single-day move (likely split). Skipping.")
        return None, None

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
    # min_periods=1 allows calculation even with less than 1 year of data
    rolling_52w_high = data['High'].rolling(window=252, min_periods=1).max()
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

    """ ============================================
        ADDITIONAL INDICATORS (from research paper)
        Paper found these are top predictors for S&P 500.
        ============================================

        HLC3: Average of High, Low, Close — smoother price proxy.
    """
    data['HLC3'] = (data['High'] + data['Low'] + data['Close']) / 3

    """ TEMA: Triple Exponential Moving Average — reacts faster than SMA/EMA.
        Less lag = better at catching trend changes early.
    """
    data.ta.tema(length=20, append=True)  # Creates 'TEMA_20'

    """ FWMA: Forecaster Weighted Moving Average — uses Fibonacci weights.
        Recent prices get Fibonacci-weighted importance.
    """
    data.ta.fwma(length=20, append=True)  # Creates 'FWMA_20'

    """ OBV: On-Balance Volume — accumulates volume on up days, subtracts on down.
        Rising OBV with rising price = strong trend confirmation.
        Divergence between OBV and price = potential reversal.
    """
    data.ta.obv(append=True)  # Creates 'OBV'

    """ RVI: Relative Vigor Index — measures conviction of a move.
        Uses Open/Close relationship to gauge bullish vs bearish energy.
    """
    data.ta.rvi(length=14, append=True)  # Creates 'RVI_14'

    """ KAMA: Kaufman Adaptive Moving Average — adjusts to market noise.
        Flat in sideways markets (avoids whipsaws), fast in trends.
    """
    data.ta.kama(length=10, append=True)  # Creates 'KAMA_10'

    """ SWMA: Sine Weighted Moving Average — uses sine function for weights.
        Middle of the window gets highest weight, smooths edges.
    """
    data.ta.swma(length=20, append=True)  # Creates 'SWMA_20'

    """ MFI: Money Flow Index — RSI weighted by volume.
        >80 = overbought with volume confirmation, <20 = oversold.
        More reliable than RSI alone because it confirms with volume.
    """
    data.ta.mfi(length=14, append=True)  # Creates 'MFI_14'

    """ PVT: Price Volume Trend — cumulative volume weighted by price change.
        Similar to OBV but proportional to price change magnitude.
    """
    data.ta.pvt(append=True)  # Creates 'PVT'

    """ EMA: Exponential Moving Average — exponential decay weighting.
        More responsive than SMA, gives more weight to recent prices.
    """
    data.ta.ema(length=20, append=True)  # Creates 'EMA_20'
    data.ta.ema(length=50, append=True)  # Creates 'EMA_50'

    """ HMA: Hull Moving Average — near-zero lag with smooth curve.
        Uses weighted moving average of WMA differences.
    """
    data.ta.hma(length=20, append=True)  # Creates 'HMA_20'

    """ HWMA: Holt-Winter Moving Average — exponential smoothing with trend.
        Captures both level and trend components of price.
    """
    data.ta.hwma(length=20, append=True)  # Creates 'HWMA_20'

    """ Ichimoku Cloud — Japanese trend system.
        Tenkan/Kijun cross = signal, Cloud = support/resistance zone.
    """
    nine_high = data['High'].rolling(window=9).max()
    nine_low = data['Low'].rolling(window=9).min()
    data['Ichimoku_Tenkan'] = (nine_high + nine_low) / 2

    twenty_six_high = data['High'].rolling(window=26).max()
    twenty_six_low = data['Low'].rolling(window=26).min()
    data['Ichimoku_Kijun'] = (twenty_six_high + twenty_six_low) / 2

    data['Ichimoku_SpanA'] = ((data['Ichimoku_Tenkan'] + data['Ichimoku_Kijun']) / 2).shift(26)
    fifty_two_high = data['High'].rolling(window=52).max()
    fifty_two_low = data['Low'].rolling(window=52).min()
    data['Ichimoku_SpanB'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)
    data['Ichimoku_Cloud_Width'] = (data['Ichimoku_SpanA'] - data['Ichimoku_SpanB']) / data['Close']

    """ EMA crossover signals: EMA_20 vs EMA_50.
        Golden cross (EMA20 > EMA50) = bullish, Death cross = bearish.
    """
    data['EMA_Cross'] = data['EMA_20'] - data['EMA_50']

    """ Price relative to KAMA: Is price above/below adaptive average?
    """
    data['Price_vs_KAMA'] = (data['Close'] - data['KAMA_10_2_30']) / data['KAMA_10_2_30']

    """ 
        Fetch news from Finnhub and analyze sentiment with FinBERT.
        These features capture market sentiment that isn't in price data.
    """
    if CONFIG['use_news_sentiment']:
        print(f"\nFetching news sentiment for {ticker}...")
        sentiment_features = get_daily_sentiment(ticker, data.index)
        
        # Merge sentiment features with price data
        data = data.join(sentiment_features, how='left')
        
        # Fill missing sentiment values using forward fill
        # (no news for a day = same sentiment as last day with news)
        # Then fill remaining NaN at the start with0 (neutral)
        for col in ['News_Sentiment', 'News_Volume', 'News_Positive_Ratio',
                     'Sentiment_MA', 'Sentiment_Std']:
            data[col] = data[col].ffill().fillna(0)
        
        print(f"  Added5 sentiment features")
    else:
        print(f"\nNews sentiment disabled in CONFIG")

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
        ADAPTIVE PORTFOLIO MANAGER BOT:
        
        Retrains the model every rebalance period using the most recent data.
        This allows the model to adapt to changing market conditions.
        
        1. Mixed universe (winners + losers + small caps) — no survivorship bias
        2. Transaction costs on every trade (0.1% per buy/sell)
        3. Fair benchmarks: S&P 500 AND Equal-Weight portfolio of same stocks
        4. Adaptive retraining every N days with sliding window
    """
    import numpy as np
    from xgboost import XGBClassifier
    np.random.seed(42)
    
    universe = CONFIG['universe']
    top_n = CONFIG['portfolio_top_n']
    rebalance_days = CONFIG['rebalance_days']
    initial_capital = CONFIG['initial_capital']
    tx_cost = CONFIG['transaction_cost']  # 0.1% per trade
    retrain_years = CONFIG.get('retrain_years', 5)  # Years of data for retraining
    
    print(f"\n{'='*60}")
    print(f"ADAPTIVE PORTFOLIO MANAGER BOT")
    print(f"Universe: {len(universe)} stocks | Top {top_n} | Rebalance: {rebalance_days}d")
    print(f"Transaction cost: {tx_cost*100:.1f}% per trade")
    print(f"Retrain every {rebalance_days}d with last {retrain_years}y of data")
    if CONFIG.get('use_lstm', False):
        print(f"LSTM Ensemble: weight={CONFIG['lstm_weight']:.0%}, seq={CONFIG['lstm_seq_length']}d, hidden={CONFIG['lstm_hidden']}")
    print(f"{'='*60}")
    
    """ Pre-fetch all stock data once to avoid repeated downloads.
    """
    print(f"\nPre-fetching stock data...")
    all_stock_data = {}
    for ticker in universe:
        try:
            full_data, target_col = prepare_data(ticker=ticker)
            if full_data is None:
                print(f"  {ticker}: SKIPPED (bad data)")
                continue
            cols_to_remove = [c for c in CONFIG['random_ticker_columns'] if c in full_data.columns]
            full_data = full_data.drop(columns=cols_to_remove, errors='ignore')
            all_stock_data[ticker] = full_data
            print(f"  {ticker}: {len(full_data)} rows")
        except Exception as e:
            print(f"  {ticker}: SKIPPED ({e})")
    
    if not all_stock_data:
        print("ERROR: No stocks processed.")
        return None
    
    # Get price data for all stocks
    print(f"\nFetching price data...")
    price_data = {}
    for ticker in all_stock_data:
        try:
            prices = yf.download(ticker, period=f"{CONFIG['data_years']}y", progress=False)['Close'].squeeze()
            price_data[ticker] = prices
        except:
            pass
    
    # S&P 500
    sp500_data = yf.download("^GSPC", period=f"{CONFIG['data_years']}y", progress=False)['Close'].squeeze()
    
    # Find the test start date (80% of the earliest stock data)
    min_len = min(len(d) for d in all_stock_data.values())
    test_start_idx = int(min_len * 0.8)
    first_ticker = list(all_stock_data.keys())[0]
    test_dates = list(all_stock_data[first_ticker].index[test_start_idx:])
    
    print(f"\nTest period: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
    print(f"Stocks with data: {len(all_stock_data)}")
    
    # Pre-compute all returns for each stock to avoid repeated calculations
    print(f"\nPre-computing returns...")
    stock_returns = {}
    for ticker in all_stock_data:
        if ticker in price_data:
            prices = price_data[ticker]
            returns = prices.pct_change()
            stock_returns[ticker] = returns
    
    # Pre-compute S&P 500 returns
    sp500_all_returns = sp500_data.pct_change()
    
    """ FOUR strategies to compare.
    """
    model_value = initial_capital
    ew_value = initial_capital
    sp500_value = initial_capital
    random_value = initial_capital
    
    model_history = []
    ew_history = []
    sp500_history = []
    random_history = []
    
    model_total_costs = 0
    ew_total_costs = 0
    random_total_costs = 0
    
    model_holdings = {}
    ew_holdings = {}
    random_holdings = {}
    rebalance_counter = 0
    models_trained = 0
    
    sp500_returns = sp500_all_returns.reindex(test_dates, method='ffill').fillna(0)
    
    for i, date in enumerate(test_dates):
        sp500_ret = sp500_returns.loc[date] if date in sp500_returns.index else 0
        
        """ Model portfolio: update value from holdings.
        """
        if model_holdings:
            daily_ret = 0
            valid = 0
            for ticker in model_holdings:
                if ticker in stock_returns and date in stock_returns[ticker].index:
                    r = stock_returns[ticker].loc[date]
                    if not np.isnan(r):
                        daily_ret += r
                        valid += 1
            if valid > 0:
                model_value *= (1 + daily_ret / valid)
        
        model_history.append(model_value)
        
        """ Equal-weight: always invested in all stocks.
        """
        ew_ret = 0
        ew_valid = 0
        for ticker in stock_returns:
            if date in stock_returns[ticker].index:
                r = stock_returns[ticker].loc[date]
                if not np.isnan(r):
                    ew_ret += r
                    ew_valid += 1
        if ew_valid > 0:
            ew_value *= (1 + ew_ret / ew_valid)
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
                if ticker in stock_returns and date in stock_returns[ticker].index:
                    r = stock_returns[ticker].loc[date]
                    if not np.isnan(r):
                        rand_ret += r
                        rand_valid += 1
            if rand_valid > 0:
                random_value *= (1 + rand_ret / rand_valid)
        random_history.append(random_value)
        
        """ REBALANCE + RETRAIN every N days.
        """
        rebalance_counter += 1
        if rebalance_counter >= rebalance_days:
            rebalance_counter = 0
            
            """ ADAPTIVE RETRAINING: Train new models with recent data only.
                Uses ensemble: XGBoost + LSTM (if enabled).
            """
            xgb_scores = {}
            lstm_scores = {}
            use_lstm = CONFIG.get('use_lstm', False)
            lstm_weight = CONFIG.get('lstm_weight', 0.2)
            seq_length = CONFIG.get('lstm_seq_length', 20)
            
            import torch
            from sklearn.preprocessing import StandardScaler
            
            for ticker in all_stock_data:
                try:
                    full_data = all_stock_data[ticker]
                    
                    date_idx = full_data.index.get_loc(date)
                    start_idx = max(0, date_idx - retrain_years * 252)
                    
                    if date_idx < 100:
                        continue
                    
                    window_data = full_data.iloc[start_idx:date_idx]
                    
                    if len(window_data) < 100:
                        continue
                    
                    target_col_name = [c for c in window_data.columns if c.startswith('Target_')][0]
                    X = window_data.drop(columns=[target_col_name])
                    y = window_data[target_col_name]
                    
                    if y.nunique() < 2:
                        continue
                    
                    # === XGBoost ===
                    n_down = (y == 0).sum()
                    n_up = (y == 1).sum()
                    xgb_model = XGBClassifier(
                        eval_metric='logloss', random_state=42,
                        scale_pos_weight=n_down/n_up if n_up > 0 else 1,
                        n_estimators=50, max_depth=2, learning_rate=0.05,
                        min_child_weight=30, gamma=0.5,
                        subsample=0.6, colsample_bytree=0.5
                    )
                    xgb_model.fit(X, y)
                    
                    today_row = full_data.iloc[date_idx:date_idx + 1]
                    today_row = today_row.drop(columns=[target_col_name], errors='ignore')
                    if len(today_row) > 0 and list(today_row.columns) == list(X.columns):
                        xgb_prob = xgb_model.predict_proba(today_row)[:, 1][0]
                        xgb_scores[ticker] = xgb_prob
                    
                    # === LSTM (if enabled) ===
                    if use_lstm and len(window_data) >= seq_length + 50:
                        try:
                            feature_cols = list(X.columns)
                            X_raw = X.values
                            y_raw = y.values
                            
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_raw)
                            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
                            
                            X_seq, y_seq = create_sequences(X_scaled, y_raw, seq_length)
                            
                            if len(X_seq) >= 30:
                                # Select top 10 features by variance for LSTM
                                variances = np.var(X_scaled, axis=0)
                                top_idx = np.argsort(variances)[-10:]
                                X_seq_small = X_seq[:, :, top_idx]
                                
                                input_size = X_seq_small.shape[2]
                                lstm_model = StockLSTM(
                                    input_size=input_size,
                                    hidden_size=CONFIG.get('lstm_hidden', 32),
                                    num_layers=1,
                                    dropout=0.3
                                )
                                lstm_model, _ = train_lstm(
                                    lstm_model, X_seq_small, y_seq,
                                    epochs=CONFIG.get('lstm_epochs', 50),
                                    lr=0.001, batch_size=64
                                )
                                
                                # Predict on today
                                today_full = full_data.iloc[max(0, date_idx - seq_length + 1):date_idx + 1]
                                today_features = today_full[feature_cols].values
                                today_scaled = scaler.transform(today_features)
                                today_scaled = np.nan_to_num(today_scaled, nan=0.0)
                                today_small = today_scaled[:, top_idx]
                                
                                if len(today_small) == seq_length:
                                    X_today = today_small.reshape(1, seq_length, -1)
                                    lstm_prob = predict_lstm(lstm_model, X_today)[0]
                                    lstm_scores[ticker] = lstm_prob
                        except Exception:
                            pass
                    
                    models_trained += 1
                except Exception as e:
                    continue
            
            # === COMBINE SCORES (Ensemble) ===
            # LSTM only for top 10 XGBoost candidates (not all stocks — too slow)
            current_scores = {}
            xgb_ranked = sorted(xgb_scores.items(), key=lambda x: x[1], reverse=True)
            top_candidates = [t for t, _ in xgb_ranked[:10]]
            
            for ticker in top_candidates:
                xgb_p = xgb_scores[ticker]
                if use_lstm and ticker in lstm_scores:
                    lstm_p = lstm_scores[ticker]
                    current_scores[ticker] = (1 - lstm_weight) * xgb_p + lstm_weight * lstm_p
                else:
                    current_scores[ticker] = xgb_p
            
            # Add remaining stocks with XGBoost only
            for ticker in xgb_scores:
                if ticker not in current_scores:
                    current_scores[ticker] = xgb_scores[ticker]
            
            if current_scores:
                ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
                new_top = set(ticker for ticker, _ in ranked[:top_n])
                
                # DEBUG: Show what the model is picking
                if models_trained % 30 == 0:
                    print(f"\n  DEBUG - Date: {date.date()}")
                    print(f"  Top picks: {[(t, f'{s:.3f}') for t, s in ranked[:5]]}")
                    # Check returns of picked stocks over next 10 days
                    next_10_dates = test_dates[i:i+10] if i+10 < len(test_dates) else test_dates[i:]
                    for t, _ in ranked[:3]:
                        if t in stock_returns:
                            period_ret = sum(stock_returns[t].get(d, 0) for d in next_10_dates if d in stock_returns[t].index)
                            print(f"    {t} next 10d return: {period_ret*100:+.1f}%")
                    print(f"  Model value: ${model_value:,.0f}")
                
                # MODEL: Sell old, buy new — count trades
                old_model = set(model_holdings.keys())
                sells = old_model - new_top
                buys = new_top - old_model
                model_trades = len(sells) + len(buys)
                model_cost = model_trades * tx_cost
                model_value *= (1 - model_cost)
                model_total_costs += model_cost * model_value
                
                model_holdings = {t: 1 for t in new_top}
                
                # EQUAL-WEIGHT: Rebalance (costs apply)
                available = [t for t in all_stock_data if t in price_data and date in price_data[t].index]
                old_ew = set(ew_holdings.keys())
                new_ew = set(available)
                ew_trades = len(old_ew.symmetric_difference(new_ew))
                ew_cost = ew_trades * tx_cost * 0.5
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
                
                if i % (rebalance_days * 5) == 0:
                    top3 = [t for t, _ in ranked[:3]]
                    print(f"  {date.date()}: Retrained {models_trained} models | Top: {top3}")
    
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
    
    alpha = model_return - ew_return
    print(f"\n  Model vs Equal-Weight (TRUE alpha): {alpha:+.1f}%")
    if alpha > 0:
        print(f"  Model ADDS value over passive holding!")
    else:
        print(f"  Model does NOT beat just buying everything equally.")
    
    """ Plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1.plot(test_dates, model_history, label=f'Model ({model_return:+.1f}%)', linewidth=2.5)
    ax1.plot(test_dates, ew_history, label=f'Equal-Weight ({ew_return:+.1f}%)', linewidth=2, linestyle='-.')
    ax1.plot(test_dates, sp500_history, label=f'S&P 500 ({sp500_return:+.1f}%)', linewidth=2, linestyle='--')
    ax1.plot(test_dates, random_history, label=f'Random ({random_return:+.1f}%)', linewidth=1.5, alpha=0.7, linestyle=':')
    
    ax1.set_title(f'Adaptive Backtest: Retrains every {rebalance_days}d ({retrain_years}y window)\n({len(all_stock_data)} stocks, Top {top_n}, {tx_cost*100:.1f}% cost/trade)', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    alpha_over_time = [(m/ew - 1) * 100 for m, ew in zip(model_history, ew_history)]
    ax2.fill_between(test_dates, alpha_over_time, 0, alpha=0.3, color='green')
    ax2.plot(test_dates, alpha_over_time, label='Alpha vs Equal-Weight', linewidth=2, color='green')
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
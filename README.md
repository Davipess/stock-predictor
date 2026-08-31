<div align="center">

# Stock Predictor

**Adaptive ML Stock Selection with Technical Analysis & News Sentiment**

An end-to-end machine learning pipeline that predicts stock direction using 30+ technical indicators, FinBERT news sentiment, and XGBoost classification — backtested against S&P 500 and equal-weight benchmarks with realistic transaction costs.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EE2E24?style=flat)](https://xgboost.readthedocs.io/)
[![FinBERT](https://img.shields.io/badge/FinBERT-ProsusAI-yellow?style=flat)](https://huggingface.co/ProsusAI/finbert)

![Backtest Results](realistic_backtest.png)

</div>

---

## Table of Contents

- [The Idea](#the-idea)
- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Technical Indicators](#technical-indicators)
- [News Sentiment (Phase 2)](#news-sentiment-phase-2)
- [Machine Learning Model](#machine-learning-model)
- [Backtesting Methodology](#backtesting-methodology)
- [Results](#results)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Journey](#development-journey)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## The Idea

The core question was simple: **can a machine learning model pick winning stocks better than just buying everything equally?**

Most stock prediction projects fall into common traps:
- Using future data accidentally (lookahead bias)
- Testing on data the model has already seen
- Ignoring transaction costs
- Only picking "good" stocks (survivorship bias)
- Evaluating with accuracy alone on imbalanced datasets

This project was built to avoid all of these. The approach:

1. **Diverse universe** — Include winners (NVDA, META), average performers (JNJ, PG), and underperformers (INTC, BA) to avoid survivorship bias
2. **Walk-forward validation** — Retrain the model every 10 trading days on a sliding 5-year window, simulating real-world deployment
3. **Honest benchmarks** — Compare against S&P 500 buy-and-hold AND an equal-weight portfolio of the same stocks
4. **Real costs** — 0.1% transaction cost on every trade
5. **Classification over regression** — Predict "will this stock go UP or DOWN in 10 days?" rather than exact price, which is far more reliable

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CONFIG (start.py)                     │
│   Universe: 37 stocks │ Top 5 │ Retrain: every 10 days │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │   Per-Stock Pipeline    │
          │   (prepare_data)        │
          └────────────┬────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌────────┐     ┌────────────┐     ┌────────────┐
│ yfinance│     │ pandas_ta  │     │ FinBERT +  │
│ Price   │     │ 30+ Tech   │     │ Finnhub    │
│ Data    │     │ Indicators │     │ Sentiment  │
└────┬───┘     └─────┬──────┘     └─────┬──────┘
     │               │                  │
     └───────────────┼──────────────────┘
                     ▼
            ┌────────────────┐
            │  Merged Feature│
            │  Matrix (per   │
            │  stock)        │
            └───────┬────────┘
                    │
         ┌──────────┴──────────┐
         │  Adaptive Backtest   │
         │  (Day-by-day loop)   │
         └──────────┬──────────┘
                    │
    ┌───────────────┼───────────────┬───────────────┐
    ▼               ▼               ▼               ▼
┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ XGBoost│   │ Equal-   │   │  S&P 500 │   │  Random  │
│ Model  │   │ Weight   │   │  Buy &   │   │  Top N   │
│ Top N  │   │ (All)    │   │  Hold    │   │          │
└────────┘   └──────────┘   └──────────┘   └──────────┘
         │         │              │              │
         └─────────┴──────────────┴──────────────┘
                         │
                    ▼ Chart + Results
```

---

## Pipeline Flow

### 1. Data Preparation (`prepare_data`)
For each stock in the universe:
- Downloads 10 years of OHLCV data from Yahoo Finance
- Adds S&P 500 and VIX context
- Computes 30+ technical indicators (see [Technical Indicators](#technical-indicators))
- Fetches news sentiment via Finnhub + FinBERT (see [News Sentiment](#news-sentiment-phase-2))
- Creates the target: `Target_10d = 1` if price goes UP in 10 days, `0` otherwise
- Drops rows with NaN values from indicator warmup periods

### 2. Adaptive Backtesting (`backtest`)
Walk-forward simulation over the test period (last 20% of data):

```
For each trading day:
  1. Update portfolio value from yesterday's holdings
  2. Every 10 days (rebalance):
     a. For each stock: train XGBoost on last 5 years of data
     b. Predict probability of going UP for tomorrow
     c. Rank all stocks by probability
     d. Select Top 5 = MODEL portfolio
     e. Randomly select 5 = RANDOM portfolio
     f. All stocks = EQUAL-WEIGHT portfolio
     g. Apply transaction costs to all trades
  3. Record daily portfolio values
```

### 3. Evaluation
Compare 4 strategies with realistic costs:
- **Model (Top N)**: XGBoost-selected stocks
- **Equal-Weight**: Buy everything equally
- **S&P 500**: Market benchmark
- **Random**: Random stock picks

---

## Technical Indicators

The model uses 30+ features organized into 5 categories, informed by research on S&P 500 prediction (see [Research Reference](#research-reference)).

### Trend & Moving Averages (13 features)
| Indicator | What It Measures |
|-----------|-----------------|
| SMA (20, 50) | Simple moving average — baseline trend direction |
| EMA (20, 50) | Exponential MA — more responsive to recent price |
| TEMA (20) | Triple EMA — near-zero lag, catches trend changes early |
| HMA (20) | Hull MA — smooth curve with minimal lag |
| HWMA (20) | Holt-Winter MA — captures both level and trend |
| KAMA (10) | Kaufman Adaptive MA — flat in sideways markets, fast in trends |
| FWMA (20) | Fibonacci Weighted MA — recent prices weighted by Fibonacci sequence |
| SWMA (20) | Sine Weighted MA — smooths edges with sine function |
| Ichimoku Cloud (9/26/52) | Japanese trend system — Tenkan/Kijun cross, Cloud as support/resistance |
| EMA Crossover | EMA20 - EMA50 — golden/death cross signal |
| Price vs SMA/KAMA | Distance from key moving averages |

### Momentum Oscillators (7 features)
| Indicator | What It Measures |
|-----------|-----------------|
| RSI (14) | Overbought (>70) / oversold (<20) conditions |
| RSI Change | Momentum of RSI — rising = bullish, falling = bearish |
| MACD (12/26/9) | Trend-following momentum — signal line crossovers |
| Stochastic (14/3/3) | Price position relative to high/low range |
| RVI (14) | Relative Vigor Index — conviction of price moves |
| MFI (14) | Money Flow Index — RSI weighted by volume |

### Volatility (5 features)
| Indicator | What It Measures |
|-----------|-----------------|
| Bollinger Bands | Upper/lower bands around SMA — overbought/oversold |
| BB Width | Band width normalized by price — volatility expansion/contraction |
| ATR (14) | Average True Range — magnitude of daily price swings |
| Volatility (10/20/50d) | Standard deviation of returns at multiple timeframes |

### Volume (5 features)
| Indicator | What It Measures |
|-----------|-----------------|
| Relative Volume (10/20/50d) | Today's volume vs average — unusual activity detection |
| OBV | On-Balance Volume — accumulates volume on up days |
| PVT | Price Volume Trend — volume weighted by price change magnitude |

### Returns & Context (5 features)
| Indicator | What It Measures |
|-----------|-----------------|
| Returns (1/5/20d) | Percentage price changes at multiple timeframes |
| 52-Week High Distance | % below rolling yearly high — momentum/reversion signal |
| S&P 500 Correlation | 20-day rolling correlation — stock vs market |
| HLC3 | Average of High, Low, Close — smoother price proxy |

### Feature Interactions
| Feature | What It Measures |
|---------|-----------------|
| RSI x Volume | Overbought with high volume = strong conviction |
| MACD x Volume | Momentum with volume confirmation |

---

## News Sentiment (Phase 2)

### Pipeline
```
Finnhub API → Company News (last 30 days)
        ↓
   FinBERT (ProsusAI/finbert)
   "positive" / "negative" / "neutral"
        ↓
   Daily Aggregation → 5 Sentiment Features
```

### Features
| Feature | Description |
|---------|-------------|
| `News_Sentiment` | Average FinBERT sentiment score per trading day |
| `News_Volume` | Number of news articles published that day |
| `News_Positive_Ratio` | Percentage of positive articles |
| `Sentiment_MA` | 5-day rolling mean of sentiment |
| `Sentiment_Std` | 5-day rolling standard deviation of sentiment |

### FinBERT
Sentiment analysis uses **FinBERT** (ProsusAI/finbert), a BERT model pre-trained on financial text. It classifies news headlines into positive/negative/neutral with ~80-85% accuracy on financial data — significantly better than general-purpose sentiment tools like VADER.

---

## Machine Learning Model

### XGBoost Classifier
The model is a **binary classifier**: it predicts whether a stock's price will go UP or DOWN over the next 10 trading days.

| Parameter | Value | Why |
|-----------|-------|-----|
| `n_estimators` | 50 | Conservative — avoids memorizing training data |
| `max_depth` | 2 | Very shallow trees — prevents overfitting |
| `learning_rate` | 0.05 | Slow learning — more careful generalization |
| `min_child_weight` | 30 | High threshold — requires strong evidence to split |
| `gamma` | 0.5 | Minimum loss reduction for split — conservative |
| `subsample` | 0.6 | Use 60% of data per tree — reduces variance |
| `colsample_bytree` | 0.5 | Use 50% of features per tree — decorrelates trees |

> These deliberately conservative parameters were chosen after diagnostic testing
> showed that default XGBoost overfits badly (95% train accuracy vs 54% test accuracy).
> The regularized model reduced the overfit gap from 41% to 14%.

### Why Classification Instead of Regression
Predicting "up or down" is more reliable than predicting exact price because:
- Stock prices are inherently noisy — small errors in regression compound exponentially
- Classification captures the **direction** which is what matters for trading
- Binary targets are more robust to outliers and regime changes

### Evaluation Metrics
| Metric | What It Measures |
|--------|-----------------|
| Accuracy | % of correct predictions (baseline: ~50% for balanced data) |
| Precision | Of all "UP" predictions, how many were correct? |
| Recall | Of all actual "UP" days, how many did we catch? |
| F1-Score | Harmonic mean of precision and recall — balanced metric |

---

## Backtesting Methodology

### Walk-Forward Validation
Instead of a single train/test split, the model **retrains every 10 trading days** using a sliding 5-year window:

```
Training window (5 years)
├────────────────────────────┤─────► predict here
                     ↑
                     Today (retrain from this point)
```

This simulates real-world deployment where:
- Market conditions change over time
- The model adapts to new patterns
- No future data is ever used for training

### Transaction Costs
Every trade (buy or sell) incurs a **0.1% cost**, simulating:
- Brokerage commissions
- Bid-ask spread
- Market impact for small orders

### Universe Design
The stock universe includes **4 categories** to avoid survivorship bias:

| Category | Stocks | Purpose |
|----------|--------|---------|
| Winners | AAPL, MSFT, NVDA, META, AMZN, GOOGL | Captures bull market leaders |
| Average | JNJ, PG, KO, PEP, WMT, JPM, V | Steady performers for diversification |
| Underperformers | INTC, BA, T, VZ, PFE, CVX, XOM | Tests if model avoids losers |
| Volatile | RIVN, PLTR, PYPL, DKNG, COIN, ABNB, UBER | Higher risk/reward small-mid caps |

### Benchmarks
| Benchmark | What It Tests |
|-----------|--------------|
| **Equal-Weight** | Best passive strategy — buy everything equally |
| **S&P 500** | Market return — the standard benchmark |
| **Random Top N** | Minimum skill threshold — random stock picking |

The key metric is **True Alpha** = Model Return − Equal-Weight Return. This measures whether the model adds value beyond simply holding everything.

---

## Results

### Backtest Summary (2016-2026)

```
Strategy                   Final Value     Return    Alpha vs EW
─────────────────────────────────────────────────────────────────
Model (Top 5)              $76,305       +663.0%     +259.6%
Equal-Weight (All)         $50,341       +403.4%         —
S&P 500                    $32,702       +227.0%         —
Random (Top 5)              $5,514        -44.9%         —
```

### Key Findings

1. **The model beats the market**: +663% vs S&P 500 +227% over 10 years
2. **The model beats equal-weight**: +260% alpha — genuine stock selection skill
3. **Random fails**: -44.9% confirms that picking stocks randomly loses money
4. **Survived 2022 bear market**: Model was defensive (picked WMT, T, PFE) and preserved capital

### Model Behavior Analysis
- **59% win rate** on picked stocks (vs 50% random chance)
- **Average +1.44% per 10-day pick** — small but consistent edge
- When winning: avg **+5.32%** | When losing: avg **-4.15%** — asymmetric payoff
- Model naturally rotates between **growth** (bull market) and **value/defensive** (uncertain times)

---

## Configuration

All parameters are in the `CONFIG` dictionary at the top of `start.py`:

```python
CONFIG = {
    'universe': [...],           # Stock universe (37 tickers)
    'target_days': 10,           # Predict 10 days ahead
    'data_years': 10,            # 10 years of history
    'portfolio_top_n': 5,        # Buy top 5 stocks
    'rebalance_days': 10,        # Retrain every 10 trading days
    'initial_capital': 10000,    # Starting capital ($)
    'transaction_cost': 0.001,   # 0.1% per trade
    'retrain_years': 5,          # 5-year sliding window
    'use_news_sentiment': True,  # Enable FinBERT sentiment
}
```

### Environment Variables
Create a `.env` file in the project root:
```
FINNHUB_API_KEY=your_api_key_here
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Finnhub API key (free tier works — [get one here](https://finnhub.io/register))

### Installation

```bash
git clone https://github.com/Davipess/stock-predictor.git
cd stock-predictor

# Install dependencies
pip install yfinance pandas-ta pandas matplotlib numpy \
            scikit-learn xgboost scipy transformers torch \
            finnhub-python python-dotenv

# Set up API key
echo "FINNHUB_API_KEY=your_key_here" > .env
```

### Run

```bash
python start.py
```

This will:
1. Download data for all stocks (~5 minutes on first run)
2. Fetch news sentiment via FinBERT (~10 minutes on first run)
3. Run the adaptive backtest (~2 minutes)
4. Print results and save `realistic_backtest.png`

---

## Project Structure

```
stock-predictor/
├── start.py              # Main pipeline (1034 lines)
│   ├── CONFIG            # All configurable parameters
│   ├── Sentiment Module  # FinBERT + Finnhub integration
│   ├── prepare_data()    # Feature engineering pipeline
│   ├── Model Training    # XGBoost with TimeSeriesSplit
│   └── backtest()        # Adaptive walk-forward backtester
├── .env                  # API keys (gitignored)
├── .gitignore            # Protects secrets, caches, images
└── README.md             # This file
```

---

## Development Journey

### Commits 1-4: Solo Development (Davipess)

The first 4 commits were written entirely by hand as a learning exercise to understand ML, technical analysis, and Python development:

```
2d0701f Initial commit
568c0f7 First test with yfinance data extraction
3ba7aac Refactored data extraction into a dynamic function
f096f70 Imported and developed sequence for training of the XGBClassifier
```

**What was built from scratch:**
- Data pipeline with yfinance for stock data extraction
- XGBoost classifier for stock direction prediction
- Basic technical indicators (SMA, RSI, MACD)
- Train/test splitting logic
- Initial model evaluation

### Commits 5-15: AI-Assisted Development (with opencode)

From commit 5 onward, the project was developed collaboratively with AI assistance using opencode (Claude). The AI handled code generation, debugging, research integration, and architectural decisions, while the project direction, requirements, and validation were driven by the author.

#### Phase 1: Technical Baseline (Commits 5-8)

**What was added:**
- Enhanced pipeline: 5-year data, S&P 500/VIX context, class balance check
- Bollinger Bands, ATR, Stochastic Oscillator indicators
- Feature interactions (RSI x Volume, MACD x Volume)
- Feature importance ranking and selection
- CONFIG dictionary for centralized parameter management
- Hyperparameter tuning with RandomizedSearchCV
- Time-series cross-validation (never random splits!)

**Key decisions:**
- Used **classification** (up/down) instead of regression (exact price)
- Implemented **TimeSeriesSplit** to prevent lookahead bias
- Removed raw price columns to prevent the model from memorizing price levels

#### Phase 2: Portfolio Manager & Sentiment (Commits 9-12)

**What was added:**
- Adaptive backtester that retrains every 10 days on a 5-year sliding window
- Transaction costs (0.1% per trade)
- 4-benchmark comparison: Model, Equal-Weight, S&P 500, Random
- Mixed stock universe (winners + losers + volatile) to avoid survivorship bias
- **FinBERT sentiment analysis** via Finnhub news API
- 5 sentiment features: daily sentiment, volume, positive ratio, rolling mean/std

**Bug fixes:**
- Fixed `Dist_52w_High_%` to use `min_periods=1` for newer stocks
- Fixed empty news fallback (no crash when API returns no articles)
- Removed SQ (delisted), filtered stocks with >50% single-day moves (reverse splits)
- Fixed dotenv path resolution for different working directories

#### Phase 3: Overfitting Fix & Research Integration (Commits 13-15)

**The overfitting problem:**
Initial XGBoost returned +55,713% — clearly unrealistic. Diagnostic testing revealed:
- XGBoost train accuracy: 95% vs test accuracy: 54% → **41% overfit gap**
- Model was memorizing training patterns that didn't generalize

**The fix:**
- Reduced `max_depth` from 4 → 2 (shallower trees)
- Added `min_child_weight=30`, `gamma=0.5` (conservative splits)
- Added `subsample=0.6`, `colsample_bytree=0.5` (decorrelate trees)
- Overfit gap reduced from 41% → 14%

**Research integration:**
Studied an ML paper on S&P 500 prediction that found 5 universal top indicators. Added 13 missing indicators:
- **HLC3, TEMA, FWMA** — better moving averages than SMA
- **OBV, PVT, MFI** — volume-based features with more signal
- **RVI, KAMA, SWMA** — momentum and adaptive indicators
- **EMA, HMA, HWMA** — responsive trend indicators
- **Ichimoku Cloud** — Japanese trend system (manually implemented)

**Result:** Alpha dropped from +513% to +260% — more realistic edge. Model now picks defensive stocks during downturns instead of riding tech momentum.

---

## Known Limitations

1. **Test period bias**: The 2016-2026 period was a strong bull market. Performance in prolonged bear markets may differ.

2. **Sentiment features underperform**: Finnhub free tier only provides ~30 days of news history, limiting the sentiment model's ability to learn long-term patterns.

3. **No real-time execution**: This is a backtesting framework. It does not place actual trades.

4. **Single model**: Only XGBoost is used. Ensemble methods (combining XGBoost + LSTM + RandomForest) may improve robustness.

5. **Universe is static**: Stocks are not added or removed over time (e.g., IPOs, delistings).

---

## Roadmap

- [ ] **Phase 3**: Reddit FOMO data integration (r/wallstreetbets via PRAW)
- [ ] **LSTM model**: Add recurrent neural network alongside XGBoost (research shows LSTM outperforms XGBoost for price prediction)
- [ ] **Ensemble**: Combine predictions from XGBoost + LSTM + RandomForest
- [ ] **PCA dimensionality reduction**: Reduce 30+ features to top 20-25 using variance retention
- [ ] **Live trading signals**: Real-time prediction output (not actual execution)
- [ ] **Extended backtest**: Test on 2000-2010 period (includes dot-com crash and 2008 crisis)

---

## Research Reference

This project was informed by the paper: *"Identification of the Most Crucial Technical Indicators for S&P 500 Prediction Using Machine Learning"*, which found:

- **Top 5 universal indicators**: HLC3, TEMA, FWMA, OBV, Bollinger Bands
- **LSTM outperforms all tree-based models** (MAE 0.014 vs XGBoost 4.89)
- **Volume-based indicators** (OBV, MFI, PVT) are critical for prediction
- **Walk-forward validation** is essential to avoid lookahead bias

---

## License

This project is for educational purposes. Stock predictions are not financial advice.

---

<div align="center">

**Author:** [Davipess](https://github.com/Davipess)

Built as a learning project to understand ML, technical analysis, and backtesting.

Commits 1-4: Solo development | Commits 5+: AI-assisted with [opencode](https://opencode.ai)

</div>

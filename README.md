# stock-predictor

Quantitative machine learning pipeline for short-term stock direction prediction using technical indicators and walk-forward validation.

## Status

Personal research project, not in production.

## What It Does

- Predicts 10-day forward return direction for a 37-stock universe using engineered technical features.
- Implements rolling walk-forward validation: trains on a rolling 5-year window and retrains every 10 trading days to avoid lookahead bias.
- Simulates portfolio performance with a 0.1% transaction cost applied to turnover on every rebalance.
- Benchmarks predictions against passive S&P 500 and equal-weight buy-and-hold baselines.
- Extracts financial headline sentiment using ProsusAI/finbert via Finnhub API for live analysis.

## How I Built It

I directed AI-assisted development of this project to explore quantitative finance and machine learning. I selected the feature definitions, defined the walk-forward rebalancing protocol, and analyzed model overfitting. Initial unregularized models showed severe overfitting (95% training accuracy vs 54% test accuracy, a 41% gap). By constraining tree depth (max_depth=2), enforcing minimum sample weights, and adding subsampling, I brought the train/test gap down from 41% to 14%.

## Tech Stack

- Python 3.10+
- LightGBM and XGBoost
- scikit-learn
- pandas and numpy
- yfinance
- PyTorch (for the LSTM benchmark in lstm_model.py)
- Hugging Face Transformers (ProsusAI/finbert)

## How to Run

1. Clone repository and set up a virtual environment:
```bash
git clone https://github.com/Davipess/stock-predictor.git
cd stock-predictor
python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

2. Configure optional API key (only needed for live news sentiment):
```bash
cp .env.example .env
# Optional: Add FINNHUB_API_KEY
```

3. Run the backtest pipeline:
```bash
python start.py
```

The script downloads historical data, computes indicators, runs the walk-forward backtest, and saves a performance chart to `assets/realistic_backtest.png`.

## Tests

No automated unit test suite. The pipeline is validated end-to-end by running `start.py`, which executes data verification, feature generation, model retraining loops, and backtest reporting.

## Known Limitations and Next Steps

- Hindsight bias in stock universe: the 37 tickers were chosen retrospectively (including modern large-cap winners like NVDA and MSFT), which introduces survivorship and selection bias into historical simulations.
- Market friction assumptions: the 0.1% cost model does not account for slippage during high-volatility sessions or borrow fees for short positions.
- Not investment advice: this codebase is for academic study and quantitative research only.
- Next steps include testing survivorship-free S&P 500 historical constituents and building an automated daily paper-trading runner.

## License

MIT License.

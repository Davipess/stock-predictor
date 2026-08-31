"""
    TEST: Fetch news from Finnhub for a single stock
    This script tests the Finnhub API connection and shows the structure of news data.
"""
import finnhub
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load API key from .env file
load_dotenv()
api_key = os.getenv('FINNHUB_API_KEY')

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=api_key)

# Test: Get news for AAPL from the last30 days
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print(f"Fetching news for {ticker}...")
print(f"Period: {start_date.date()} to {end_date.date()}")

# Fetch news
news = finnhub_client.company_news(ticker, _from=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))

print(f"\nFound {len(news)} articles")

# Show first3 articles as example
print("\n--- Sample Articles ---")
for i, article in enumerate(news[:3]):
    print(f"\nArticle {i+1}:")
    print(f"  Headline: {article.get('headline', 'N/A')[:100]}")
    print(f"  Source: {article.get('source', 'N/A')}")
    print(f"  Published: {datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')}")
    summary = article.get('summary', 'N/A')[:150]
    # Remove non-ASCII characters for console output
    summary = summary.encode('ascii', 'ignore').decode('ascii')
    print(f"  Summary: {summary}...")

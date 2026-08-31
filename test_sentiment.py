"""
    TEST: Analyze sentiment of news headlines using FinBERT
    This script tests if FinBERT can classify financial text correctly.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model (will download ~400MB on first run)
print("Loading FinBERT model (first time download ~400MB)...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Labels: positive, negative, negative (FinBERT has 3 labels)
labels = ["positive", "negative", "neutral"]

def analyze_sentiment(text):
    """Analyze sentiment of a single text using FinBERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.numpy()[0]
    
    # Get the label with highest probability
    max_idx = probs.argmax()
    
    return {
        'label': labels[max_idx],
        'positive': float(probs[0]),
        'negative': float(probs[1]),
        'neutral': float(probs[2])
    }

# Test with financial headlines
test_headlines = [
    "Apple reports record quarterly revenue, beats expectations",
    "Tesla stock crashes after disappointing delivery numbers",
    "Federal Reserve maintains interest rates unchanged",
    "Microsoft announces massive layoffs affecting 10,000 employees",
    "NVIDIA stock surges on strong AI chip demand",
    "Bank of America warns of potential recession",
]

print("\n--- Sentiment Analysis Results ---\n")

for headline in test_headlines:
    result = analyze_sentiment(headline)
    print(f"Headline: {headline[:60]}...")
    print(f"  Sentiment: {result['label'].upper()}")
    print(f"  Scores: +{result['positive']:.2%} / -{result['negative']:.2%} / ={result['neutral']:.2%}")
    print()

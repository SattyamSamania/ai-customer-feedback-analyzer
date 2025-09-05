from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

def get_sentiment(texts):
    results = sentiment_model(texts)
    return [res['label'] for res in results]

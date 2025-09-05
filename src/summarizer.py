from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_len=100):
    return summarizer(text, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']

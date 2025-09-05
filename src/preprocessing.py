import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove links
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # remove numbers, punctuation
    text = re.sub(r"\s+", ' ', text).strip()
    return text

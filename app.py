import streamlit as st
import pandas as pd
import io
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------- FILE LOADER ----------
def load_reviews(uploaded_file):
    try:
        # Try to load as a normal CSV
        df = pd.read_csv(uploaded_file, quotechar='"', escapechar="\\")
        if 'review_text' in df.columns:
            return df
    except Exception:
        pass

    # Reset pointer & handle raw Kaggle __label__1/2 format
    uploaded_file.seek(0)
    cleaned_data = []
    for line in io.TextIOWrapper(uploaded_file, encoding="utf-8"):
        line = line.strip()
        if line.startswith("__label__2"):
            label = "POSITIVE"
            text = line.replace("__label__2", "", 1).strip()
        elif line.startswith("__label__1"):
            label = "NEGATIVE"
            text = line.replace("__label__1", "", 1).strip()
        else:
            continue
        cleaned_data.append([text, label])
    return pd.DataFrame(cleaned_data, columns=["review_text", "label"])


# ---------- CLEANING ----------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


# ---------- MODELS ----------
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return sentiment_model, embedder

sentiment_model, embedder = load_models()


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI Customer Feedback Analyzer", layout="wide")
st.title("📊 AI Customer Feedback Analyzer")

uploaded_file = st.file_uploader("Upload a CSV or raw Kaggle reviews file", type=["csv", "txt"])

# ---------- Fallback Sample Data ----------
if not uploaded_file:
    st.warning("⚠️ No file uploaded. Using sample reviews for demo.")
    sample_data = {
        "review_text": [
            "The delivery was late and the package was damaged.",
            "Amazing product! I love the new design and user experience.",
            "Customer support was unhelpful, I waited 3 days for a reply.",
            "The app keeps crashing after the new update.",
            "Fast delivery and great quality packaging.",
            "Payment failed twice before going through.",
            "The new update fixed many issues, very happy with the app now.",
            "Horrible experience, refund took too long to process.",
            "The website is easy to navigate and I found what I wanted quickly.",
            "Product quality is terrible, broke after 2 days of use."
        ],
        "label": ["NEGATIVE","POSITIVE","NEGATIVE","NEGATIVE","POSITIVE","NEGATIVE","POSITIVE","NEGATIVE","POSITIVE","NEGATIVE"]
    }
    df = pd.DataFrame(sample_data)
else:
    df = load_reviews(uploaded_file)

# ---------- Processing ----------
df['cleaned'] = df['review_text'].apply(clean_text)

# Sentiment Analysis
df['sentiment'] = [res['label'] for res in sentiment_model(df['cleaned'].tolist())]

# Clustering
embeddings = embedder.encode(df['cleaned'].tolist())
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
df['cluster'] = clusterer.fit_predict(embeddings)

# ---------- 1. Sentiment Chart ----------
st.subheader("🔹 Sentiment Distribution")
st.bar_chart(df['sentiment'].value_counts())

# ---------- 2. Complaint Clusters ----------
st.subheader("🔹 Complaint Clusters")
st.dataframe(df[['review_text', 'sentiment', 'cluster']])

# ---------- 3. Executive Insights ----------
st.subheader("🔹 Executive Insights")
total = len(df)
pos = (df['sentiment'] == 'POSITIVE').sum()
neg = (df['sentiment'] == 'NEGATIVE').sum()
st.info(f"Out of {total} reviews, {pos} are positive and {neg} are negative.")

# ---------- 4. Top Keywords per Cluster ----------
st.subheader("🔹 Top Keywords per Cluster")
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df['cleaned'])
terms = vectorizer.get_feature_names_out()

cluster_keywords = {}
for c in set(df['cluster']):
    if c == -1:
        continue
    mask = (df['cluster'] == c).values  # FIXED
    cluster_docs = X[mask].toarray().sum(axis=0)
    top_indices = cluster_docs.argsort()[-5:][::-1]
    cluster_keywords[c] = [terms[i] for i in top_indices]

for c, words in cluster_keywords.items():
    st.write(f"**Cluster {c}** → {', '.join(words)}")

# ---------- 5. Review Explorer ----------
st.subheader("🔹 Explore Reviews")
sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "POSITIVE", "NEGATIVE"])
cluster_filter = st.selectbox("Filter by Cluster", ["All"] + list(map(str, set(df['cluster']))))

filtered = df.copy()
if sentiment_filter != "All":
    filtered = filtered[filtered['sentiment'] == sentiment_filter]
if cluster_filter != "All":
    filtered = filtered[filtered['cluster'] == int(cluster_filter)]

st.dataframe(filtered[['review_text', 'sentiment', 'cluster']])

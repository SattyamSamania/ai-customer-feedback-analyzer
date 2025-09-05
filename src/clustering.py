from sentence_transformers import SentenceTransformer
import hdbscan

model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_reviews(reviews):
    embeddings = model.encode(reviews)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    return labels

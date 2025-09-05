# AI-Customer-Feedback-Analyzer

The AI Customer Feedback Analyzer is an NLP-powered system that helps companies analyze large volumes of customer reviews, support tickets, and survey responses. Instead of manually reading thousands of comments, this tool extracts sentiment, groups similar complaints, and summarizes customer pain points in a clear dashboard.

---

## **Project Overview**

Many companies receive hundreds or thousands of customer reviews daily. Manually analyzing them is time-consuming and prone to errors. This project uses **Natural Language Processing (NLP)** and **unsupervised clustering** to automatically summarize feedback and provide actionable insights to improve customer satisfaction.

**Key Features:**

- Upload reviews in CSV or Kaggle raw format.
- Clean and preprocess text for accurate analysis.
- Sentiment analysis (POSITIVE / NEGATIVE) using a transformer model.
- Cluster similar complaints using embeddings and HDBSCAN.
- Extract top keywords per cluster using TF-IDF.
- Interactive Streamlit dashboard for:
  - Sentiment visualization
  - Cluster exploration
  - Executive metrics and insights
  - Highlighting actionable items for business

---

## **Tech Stack**

- Python 3.9+
- Streamlit
- Hugging Face Transformers
- Sentence Transformers
- HDBSCAN
- Scikit-learn
- Pandas

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/SattyamSamania/ai-customer-feedback-analyzer.git
cd ai-customer-feedback-analyzer
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run app.py
```

## **Usage**

- Upload a CSV file or raw Kaggle review file.

- Explore sentiment distribution in the dashboard.

- View clusters and top keywords to understand common complaints.

- Filter reviews by sentiment or cluster.

- Use the Business Impact Insight section to identify actionable items.

## **Sample Data**

- If no file is uploaded, the app uses a built-in sample dataset of customer reviews.

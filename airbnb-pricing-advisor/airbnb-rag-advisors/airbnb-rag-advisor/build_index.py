import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False
    print("sentence-transformers unavailable (import error) — falling back to TF-IDF.")


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def build_listing_chunks(listings_df):
    fields = [
        "id", "name", "description", "neighborhood_overview",
        "neighbourhood_cleansed", "property_type", "room_type",
        "accommodates", "bedrooms", "beds", "bathrooms_text",
        "amenities", "price", "review_scores_rating",
    ]
    documents = []
    for _, row in listings_df.iterrows():
        listing_id = row.get("id", "")
        parts = []
        for field in fields:
            if field in listings_df.columns:
                value = clean_text(row.get(field, ""))
                if value:
                    parts.append(f"{field}: {value}")
        text = " | ".join(parts)
        if text:
            documents.append({"source": "listing", "listing_id": str(listing_id), "text": text})
    return documents


def build_review_chunks(reviews_df):
    documents = []
    if "listing_id" not in reviews_df.columns or "comments" not in reviews_df.columns:
        return documents
    for listing_id, group in reviews_df.groupby("listing_id"):
        comments = [clean_text(c) for c in group["comments"].head(5) if clean_text(c)]
        if comments:
            documents.append({
                "source": "review",
                "listing_id": str(listing_id),
                "text": " | ".join(comments),
            })
    return documents


def main():
    print("Loading CSV files...")
    listings_df = pd.read_csv("data/listings.csv.gz")
    reviews_df = pd.read_csv("data/reviews.csv.gz")

    print("Building listing chunks...")
    listing_docs = build_listing_chunks(listings_df)

    print("Building review chunks...")
    review_docs = build_review_chunks(reviews_df)

    documents = listing_docs + review_docs
    print(f"Total documents: {len(documents)}")

    texts = [doc["text"] for doc in documents]

    if SBERT_AVAILABLE:
        print("Encoding with sentence-transformers (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        index_data = {
            "documents": documents,
            "embeddings": embeddings,        # shape (N, 384)
            "vectorizer": None,
            "tfidf_matrix": None,
        }
        print(f"Embeddings shape: {embeddings.shape}")
    else:
        print("Vectorizing with TF-IDF...")
        vectorizer = TfidfVectorizer(
            strip_accents="unicode", lowercase=True,
            stop_words="english", max_features=20000,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        index_data = {
            "documents": documents,
            "embeddings": None,
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
        }

    with open("index.pkl", "wb") as f:
        pickle.dump(index_data, f)

    method = "sentence-transformers" if SBERT_AVAILABLE else "TF-IDF"
    print(f"Saved {len(documents)} chunks to index.pkl using {method}")


if __name__ == "__main__":
    main()

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

# ── Load pricing data ──────────────────────────────────────────────────────────
df = pd.read_csv('pricing_recommendations.csv')
listings = pd.read_csv('listings.csv', usecols=[
    'id', 'name', 'neighbourhood_cleansed', 'room_type',
    'price', 'bedrooms', 'accommodates', 'review_scores_rating'
])
listings['price'] = listings['price'].replace(r'[\$,]', '', regex=True).astype(float)
merged = df.merge(listings, on='id', how='left')

# ── Load RAG index ─────────────────────────────────────────────────────────────
try:
    with open('index.pkl', 'rb') as f:
        index_data = pickle.load(f)
    rag_docs        = index_data['documents']
    rag_embeddings  = index_data.get('embeddings')   # sentence-transformers embeddings
    rag_tfidf       = index_data.get('tfidf_matrix')
    rag_vectorizer  = index_data.get('vectorizer')
    RAG_AVAILABLE   = True
    method = "sentence-transformers" if rag_embeddings is not None else "TF-IDF"
    print(f"RAG index loaded — {len(rag_docs)} docs, {method} retrieval.")
except Exception as e:
    RAG_AVAILABLE = False
    print(f"RAG index not available ({e}). Run build_index.py first.")

# ── Claude client ──────────────────────────────────────────────────────────────
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Return top-k most semantically similar documents."""
    if not RAG_AVAILABLE:
        return []

    if rag_embeddings is not None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            q_vec = _model.encode([query], convert_to_numpy=True)
            scores = cosine_similarity(rag_embeddings, q_vec).ravel()
        except Exception:
            return []
    else:
        q_vec = rag_vectorizer.transform([query])
        if q_vec.nnz == 0:
            return []
        scores = cosine_similarity(rag_tfidf, q_vec).ravel()

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [rag_docs[i] for i in top_idx if scores[i] > 0]


def build_data_context(query: str) -> str:
    """Extract relevant statistics from the pricing DataFrame as plain text for Claude."""
    q = query.lower()
    parts = []

    # Always: summary stats
    rc  = len(merged[merged['action'] == 'RAISE'])
    dc  = len(merged[merged['action'] == 'DISCOUNT'])
    hc  = len(merged[merged['action'] == 'HOLD'])
    avg = merged['recommended_price'].mean()
    avg_rating = merged['review_scores_rating'].mean() if 'review_scores_rating' in merged.columns else None
    rating_str = f", avg rating {avg_rating:.2f}" if avg_rating and not pd.isna(avg_rating) else ""
    parts.append(
        f"Dataset: {len(merged)} listings — RAISE: {rc:,}, DISCOUNT: {dc:,}, HOLD: {hc:,}. "
        f"Avg recommended price: ${avg:.0f}/night{rating_str}."
    )

    # Raise / underpriced
    if any(x in q for x in ['raise', 'increase', 'underpriced', 'bump']):
        top = merged[merged['action'] == 'RAISE'][
            ['name', 'neighbourhood_cleansed', 'baseline_price', 'recommended_price', 'nearest_event']
        ].head(8)
        parts.append("Top listings to RAISE prices:\n" + top.to_string(index=False))

    # Discount / overpriced
    if any(x in q for x in ['discount', 'lower', 'decrease', 'overpriced', 'reduce']):
        top = merged[merged['action'] == 'DISCOUNT'][
            ['name', 'neighbourhood_cleansed', 'baseline_price', 'recommended_price', 'nearest_event']
        ].head(8)
        parts.append("Top listings to DISCOUNT:\n" + top.to_string(index=False))

    # Revenue / profit
    if any(x in q for x in ['revenue', 'money', 'earning', 'profit', 'opportunity', 'roi']):
        raise_df = merged[merged['action'] == 'RAISE']
        avg_uplift = (raise_df['recommended_price'] - raise_df['baseline_price']).mean()
        total = avg_uplift * 240 * len(raise_df)
        parts.append(
            f"Revenue opportunity: {len(raise_df):,} underpriced listings, "
            f"avg nightly uplift +${avg_uplift:.0f}, "
            f"estimated annual opportunity ${total:,.0f}."
        )

    # Events
    if 'event' in q:
        if 'nearest_event' in merged.columns:
            ev = merged.groupby('nearest_event').agg(
                count=('id', 'count'),
                avg_uplift=('price_uplift', 'mean')
            ).sort_values('avg_uplift', ascending=False).head(8)
            parts.append("Top events by price uplift:\n" + ev.to_string())

        if 'distance_to_event_km' in merged.columns and any(
            x in q for x in ['distance', 'proximity', 'near', 'close', 'km']
        ):
            bins   = [0, 2, 5, 10, 20, 999]
            labels = ['<2 km', '2-5 km', '5-10 km', '10-20 km', '>20 km']
            tmp = merged.copy()
            tmp['dist_band'] = pd.cut(tmp['distance_to_event_km'], bins=bins, labels=labels)
            by_dist = tmp.groupby('dist_band', observed=True).agg(
                avg_price=('recommended_price', 'mean'),
                avg_uplift=('price_uplift', 'mean'),
                count=('id', 'count')
            )
            parts.append("Price impact by distance from events:\n" + by_dist.to_string())

    # Neighbourhoods
    if any(x in q for x in ['neighbourhood', 'neighborhood', 'area', 'district', 'location']):
        if 'neighbourhood_cleansed' in merged.columns:
            by_hood = merged.groupby('neighbourhood_cleansed').agg(
                avg_price=('recommended_price', 'mean'),
                avg_uplift=('price_uplift', 'mean'),
                count=('id', 'count'),
                raise_pct=('action', lambda x: round((x == 'RAISE').mean() * 100, 1))
            ).sort_values('avg_price', ascending=False).head(10)
            parts.append("Top neighbourhoods by avg price:\n" + by_hood.to_string())

    # Room types
    if any(x in q for x in ['room type', 'entire home', 'private room', 'shared room', 'room comparison']):
        if 'room_type' in merged.columns:
            by_type = merged.groupby('room_type').agg(
                avg_price=('recommended_price', 'mean'),
                avg_uplift=('price_uplift', 'mean'),
                count=('id', 'count')
            ).sort_values('avg_price', ascending=False)
            parts.append("Pricing by room type:\n" + by_type.to_string())

    # Bedrooms
    if any(x in q for x in ['bedroom', 'studio', 'br ', '1 bed', '2 bed', '3 bed']):
        if 'bedrooms' in merged.columns:
            by_beds = merged.groupby('bedrooms').agg(
                avg_price=('recommended_price', 'mean'),
                count=('id', 'count')
            ).sort_values('bedrooms')
            parts.append("Pricing by bedroom count:\n" + by_beds.to_string())

    # Average price
    if any(x in q for x in ['average', 'avg', 'mean', 'typical']):
        baseline = merged['baseline_price'].mean()
        uplift   = avg - baseline
        parts.append(f"Price stats: avg baseline ${baseline:.0f}, avg recommended ${avg:.0f}, avg uplift +${uplift:.0f}.")

    # Most profitable
    if any(x in q for x in ['most profitable', 'best listing', 'highest uplift', 'top opportunity', 'biggest gap']):
        top = merged.nlargest(8, 'price_uplift')[
            ['name', 'neighbourhood_cleansed', 'baseline_price', 'recommended_price', 'price_uplift']
        ]
        parts.append("Highest uplift listings:\n" + top.to_string(index=False))

    # Ratings
    if any(x in q for x in ['rating', 'review', 'score', 'stars', 'quality']):
        if 'review_scores_rating' in merged.columns:
            top_rated = merged.dropna(subset=['review_scores_rating']).nlargest(8, 'review_scores_rating')[
                ['name', 'neighbourhood_cleansed', 'review_scores_rating', 'recommended_price', 'action']
            ]
            parts.append("Top rated listings:\n" + top_rated.to_string(index=False))

    return "\n\n".join(parts)


def answer(query: str) -> str:
    # 1. Build structured data context
    data_ctx = build_data_context(query)

    # 2. Retrieve relevant listing/review docs
    docs = retrieve(query, top_k=5)
    doc_text = ""
    if docs:
        excerpts = [f"- [{d['source']} #{d['listing_id']}]: {d['text'][:250]}" for d in docs]
        doc_text = "Relevant listing/review excerpts:\n" + "\n".join(excerpts)

    # 3. Call Claude
    system = (
        "You are an expert Airbnb pricing advisor for Washington DC. "
        "Answer questions concisely using the provided pricing data. "
        "Be specific with numbers. Format responses clearly with bullet points or short paragraphs."
    )
    user_msg = f"Pricing data:\n{data_ctx}"
    if doc_text:
        user_msg += f"\n\n{doc_text}"
    user_msg += f"\n\nQuestion: {query}"

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


print("Airbnb Pricing Advisor  (RAG + Claude)")
print("Ask anything about pricing, listings, events, neighbourhoods (type 'exit' to quit)\n")

while True:
    query = input("> ").strip()
    if not query:
        continue
    if query.lower() in {"exit", "quit"}:
        break
    print("\n" + answer(query) + "\n")

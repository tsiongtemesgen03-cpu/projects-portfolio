from flask import Flask, request, render_template_string, session, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, folium, os, pickle, json, time, datetime
from functools import lru_cache
from fetch_weather_dc import get_dc_forecast, get_monthly_weather_features, get_forecast_for_city
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

app = Flask(__name__)

# --- NAVY SIDEBAR STYLE INJECTION ---
SIDEBAR_STYLE = '''
<style>

/* --- HERO BACKGROUND IMAGE --- */
body {
  background:
    linear-gradient(rgba(2,6,23,0.85), rgba(2,6,23,0.95)),
    url('https://images.unsplash.com/photo-1508057198894-247b23fe5ade');
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}

/* Stronger default typography using local/system fonts only */
html, body {
    font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    font-weight: 500 !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

body, p, span, div, a, li, label, small, td, th, button, input, select, textarea {
    font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    font-weight: 500 !important;
}

h1, h2, h3, h4, h5, h6,
strong, b,
.metric-value, .stat-value, .value, .big-number,
.page-title, .section-title, .card-title, .listing-title,
.side-link, .side-link span, .side-link svg,
.side-brand-name {
    font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    font-weight: 700 !important;
}

.side-rail {
    color: white;
}

.side-brand-name,
.side-sub,
.side-brand-tag {
    color: #e8eefc !important;
}

.side-brand-tag {
    font-weight: 600 !important;
}

.side-rail a,
.side-link,
.side-link:visited,
.side-link span,
.side-link svg {
    color: #cbd5f5 !important;
    text-decoration: none;
    display: block;
    padding: 10px 12px;
    border-radius: 8px;
    font-weight: 600 !important;
}

.side-rail a:hover,
.side-link:hover {
    background-color: rgba(255,255,255,0.08);
    color: #ffffff !important;
}

/* --- PERSONALIZED RECOMMENDATIONS HEADER FIX --- */
.personalized-hero,
.portfolio-panel,
.ai-portfolio,
.recommendation-hero,
.recommendations-panel {
    position: relative !important;
    overflow: hidden !important;
    border-radius: 22px !important;
    background: linear-gradient(135deg, #0a1f44 0%, #0f2a5c 60%, #1b3a73 100%) !important;
    color: #ffffff !important;
}

.personalized-hero-header,
.portfolio-header,
.ai-portfolio-header,
.recommendation-hero-header,
.recommendations-header {
    padding: 28px 32px 24px 32px !important;
    border-bottom: 1px solid rgba(255,255,255,0.16) !important;
}

.personalized-hero-eyebrow,
.portfolio-eyebrow,
.ai-portfolio-eyebrow,
.recommendation-eyebrow,
.recommendations-eyebrow {
    display: block !important;
    margin: 0 0 6px 0 !important;
    color: #b9cce4 !important;
    font-size: 13px !important;
    line-height: 1.2 !important;
    font-weight: 800 !important;
    letter-spacing: 0.08em !important;
    text-transform: none !important;
}

.personalized-hero-title,
.portfolio-title,
.ai-portfolio-title,
.recommendation-hero-title,
.recommendations-title {
    display: block !important;
    margin: 0 !important;
    color: #ffffff !important;
    font-size: clamp(28px, 2.2vw, 38px) !important;
    line-height: 1.06 !important;
    font-weight: 850 !important;
    letter-spacing: -0.035em !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

.personalized-hero-subtitle,
.portfolio-subtitle,
.ai-portfolio-subtitle,
.recommendation-hero-subtitle,
.recommendations-subtitle {
    display: block !important;
    margin: 6px 0 0 0 !important;
    color: #c6d5e8 !important;
    font-size: 15px !important;
    line-height: 1.35 !important;
    font-weight: 700 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

/* Make sure plain h/p inside the navy recommendation hero match the target look */
.portfolio-panel > h4,
.ai-portfolio > h4,
.recommendation-hero > h4,
.recommendations-panel > h4,
.portfolio-panel .eyebrow,
.ai-portfolio .eyebrow {
    color: #b9cce4 !important;
    font-size: 13px !important;
    line-height: 1.2 !important;
    font-weight: 800 !important;
    margin-bottom: 6px !important;
}

.portfolio-panel h1,
.portfolio-panel h2,
.ai-portfolio h1,
.ai-portfolio h2,
.recommendation-hero h1,
.recommendation-hero h2,
.recommendations-panel h1,
.recommendations-panel h2 {
    color: #ffffff !important;
    font-size: clamp(28px, 2.2vw, 38px) !important;
    line-height: 1.06 !important;
    font-weight: 850 !important;
    letter-spacing: -0.035em !important;
    margin: 0 !important;
    white-space: normal !important;
    overflow: visible !important;
}

.portfolio-panel h1 + p,
.portfolio-panel h2 + p,
.ai-portfolio h1 + p,
.ai-portfolio h2 + p,
.recommendation-hero h1 + p,
.recommendation-hero h2 + p,
.recommendations-panel h1 + p,
.recommendations-panel h2 + p {
    color: #c6d5e8 !important;
    font-size: 15px !important;
    line-height: 1.35 !important;
    font-weight: 700 !important;
    margin-top: 6px !important;
}

/* Prevent the left side from being clipped by accidental negative margins */
.personalized-hero *,
.portfolio-panel *,
.ai-portfolio *,
.recommendation-hero *,
.recommendations-panel * {
    text-indent: 0 !important;
}


/* --- MARKET SIGNALS SPACING FIX --- */
.market-signals-card {
    padding: 30px 32px !important;
}
.market-signals-header {
    margin-bottom: 18px !important;
}
.market-signals-title {
    margin-bottom: 6px !important;
    line-height: 1.3 !important;
}
.market-signals-subtitle {
    margin-bottom: 16px !important;
    line-height: 1.5 !important;
}
.market-signal-row {
    display: flex;
    align-items: flex-start !important;
    gap: 16px;
    padding: 20px 0 !important;
    border-bottom: 1px solid #e6ecf5;
}
.market-signal-row:last-child {
    border-bottom: none;
}
.market-signal-text {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.market-signal-title {
    font-weight: 600;
    line-height: 1.4;
}
.market-signal-desc {
    font-size: 14px;
    line-height: 1.6;
    color: #6b7a90;
}
.market-signal-icon {
    margin-top: 4px;
}



/* --- BLUE/NAVY ACCENT OVERRIDES --- */
.chat-ai-badge {
    background: rgba(147,197,253,0.14) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.42) !important;
}
.chat-sub span {
    color: #bfdbfe !important;
    font-weight: 700 !important;
}
.nav-badge, .dc-pill {
    background: rgba(147,197,253,0.16) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.38) !important;
}
.eyebrow, .hero h1 .gold {
    color: #93c5fd !important;
}


/* --- EXACT SIMULATOR RESULT NAVY OVERRIDE --- */
.sim-res {
    background: linear-gradient(135deg, #0a1f44 0%, #0f2a5c 60%, #1b3a73 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 18px 44px rgba(10,31,68,0.22) !important;
}
.sim-res .sim-explanation {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(147,197,253,0.18) !important;
}
.btn-p {
    background: linear-gradient(135deg, #0a1f44 0%, #1d4ed8 100%) !important;
}

</style>
'''


app.secret_key = 'airbnb2026'
os.makedirs('static', exist_ok=True)

DEMO_CHAT_ENABLED = True
CHAT_WELCOME_MESSAGE = (
    "Good morning! Here’s what needs your attention today. "
    "I’ll help you price your listings smarter based on demand, events, and nearby comps."
)

df = pd.read_csv('pricing_recommendations_all.csv')

listings_dc = pd.read_csv('listings.csv', usecols=['id','name','neighbourhood_cleansed','room_type','bedrooms','accommodates','review_scores_rating','latitude','longitude'])
listings_dc['city'] = 'Washington'

listings = listings_dc.copy()
events_all = pd.read_csv('events_all_cities.csv')
events = events_all[events_all['city'] == 'Washington'].copy()
merged = df.merge(listings, on='id', how='left')
if 'city_x' in merged.columns:
    merged['city'] = merged['city_x']
    merged = merged.drop(columns=['city_x','city_y'], errors='ignore')

with open('simulator_model.pkl', 'rb') as f:
    sim_data = pickle.load(f)
sim_model = sim_data['model']
sim_features = sim_data['features']

with open('static/neighbourhoods.json', 'r') as f:
    neighbourhoods = json.load(f)

with open('pricing_charts.png', 'rb') as f:
    pricing_charts_b64 = base64.b64encode(f.read()).decode('utf-8')

# Load portfolio insights (generated by merge_all.py / recommend.py)  
try:
    with open('model_metrics.json', 'r') as _f:
        PORTFOLIO_INSIGHTS = json.load(_f)
except FileNotFoundError:
    PORTFOLIO_INSIGHTS = None
 
_client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL", "https://gw-sb-aoai-08.openai.azure.com/openai/v1/")
)

AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.1-codex-mini")

# Load RAG index  
_rag_docs       = []
_rag_embeddings = None
_rag_tfidf      = None
_rag_vectorizer = None
_RAG_AVAILABLE  = False
try:
    with open('index.pkl', 'rb') as _f:
        _idx = pickle.load(_f)
    _rag_docs       = _idx['documents']
    _rag_embeddings = _idx.get('embeddings')
    _rag_tfidf      = _idx.get('tfidf_matrix')
    _rag_vectorizer = _idx.get('vectorizer')
    _RAG_AVAILABLE  = True
    _rag_method = "sentence-transformers" if _rag_embeddings is not None else "TF-IDF"
    print(f"RAG index loaded — {len(_rag_docs)} docs, {_rag_method}.")
except Exception as _e:
    print(f"RAG index not available ({_e}). Run build_index.py to enable semantic search.")

_sbert_model = None
def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model

def _retrieve(query: str, top_k: int = 10) -> list:
    if not _RAG_AVAILABLE:
        return []
    try:
        if _rag_embeddings is not None:
            q_vec = _get_sbert().encode([query], convert_to_numpy=True)
            scores = _cosine_similarity(_rag_embeddings, q_vec).ravel()
        else:
            q_vec = _rag_vectorizer.transform([query])
            if q_vec.nnz == 0:
                return []
            scores = _cosine_similarity(_rag_tfidf, q_vec).ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [_rag_docs[i] for i in top_idx if scores[i] > 0]
    except Exception:
        return []

ALL_CITIES = ['Washington']

CITY_CENTERS = {
    'Washington': [38.9072, -77.0369],
}

CITY_ZOOMS = {
    'Washington': 11,
}

def get_demand(dist):
    if pd.isna(dist): return 'LOW'
    if dist < 5: return 'VERY HIGH'
    elif dist < 10: return 'HIGH'
    elif dist < 20: return 'MEDIUM'
    else: return 'LOW'

def demand_class(d):
    return {'VERY HIGH':'VH','HIGH':'H','MEDIUM':'M','LOW':'L'}.get(d,'L')

_CITY_DATA = {}
if 'city' in merged.columns:
    for _city_name, _city_df in merged.groupby('city', sort=False):
        _CITY_DATA[str(_city_name)] = _city_df.reset_index(drop=True)
else:
    _CITY_DATA['Washington'] = merged.reset_index(drop=True)

_EVENTS_BY_CITY = {'Washington': events.reset_index(drop=True)}

@lru_cache(maxsize=4)
def get_city_data(selected_city='Washington'):
    city_df = _CITY_DATA.get(selected_city, merged)
    city_events = _EVENTS_BY_CITY.get(selected_city, events)
    return city_df, city_events

_HOOD_CENTERS = {}
if {'neighbourhood_cleansed', 'latitude', 'longitude'}.issubset(merged.columns):
    _hood_tmp = merged.dropna(subset=['latitude', 'longitude']).groupby('neighbourhood_cleansed').agg(
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean')
    )
    _HOOD_CENTERS = {
        str(idx): (float(row.latitude), float(row.longitude))
        for idx, row in _hood_tmp.iterrows()
    }


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _cluster_profile_label(avg_beds, avg_acc, dominant_room):
    room = str(dominant_room or '').lower()
    if 'private' in room:
        base = 'private-room'
    elif 'hotel' in room:
        base = 'hotel-style'
    elif avg_beds >= 3 or avg_acc >= 6:
        base = 'large-group'
    elif avg_beds <= 1:
        base = 'compact'
    else:
        base = 'multi-bedroom'
    return f"event-adjacent {base} {str(dominant_room or 'listing').replace('Entire home/apt', 'entire-home').replace(' ', '-').lower()}"


def _build_ai_market_layer(df):
    work = df.copy()
    numeric_cols = ['bedrooms', 'accommodates', 'review_scores_rating', 'baseline_price',
                    'recommended_price', 'distance_to_event_km', 'latitude', 'longitude']
    for c in numeric_cols:
        if c not in work.columns:
            work[c] = 0.0
        work[c] = work[c].fillna(0.0)

    work['room_type'] = work.get('room_type', '').fillna('').astype(str)
    work['neighbourhood_cleansed'] = work.get('neighbourhood_cleansed', '').fillna('').astype(str)

    cluster_feats = work[['bedrooms', 'accommodates', 'review_scores_rating', 'baseline_price',
                          'recommended_price', 'distance_to_event_km']].copy()
    cluster_scaler = StandardScaler()
    cluster_X = cluster_scaler.fit_transform(cluster_feats)
    n_clusters = max(3, min(6, len(work) // 50 if len(work) >= 150 else 3))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    work['market_cluster'] = km.fit_predict(cluster_X)

    cluster_summary = work.groupby('market_cluster').agg(
        count=('id', 'count'),
        avg_base=('baseline_price', 'mean'),
        avg_rec=('recommended_price', 'mean'),
        avg_beds=('bedrooms', 'mean'),
        avg_acc=('accommodates', 'mean'),
        avg_dist=('distance_to_event_km', 'mean'),
        dominant_room=('room_type', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Listing')
    ).reset_index()
    cluster_summary['cluster_label'] = cluster_summary.apply(
        lambda r: _cluster_profile_label(r['avg_beds'], r['avg_acc'], r['dominant_room']), axis=1
    )
    cluster_label_map = dict(zip(cluster_summary['market_cluster'], cluster_summary['cluster_label']))
    work['cluster_label'] = work['market_cluster'].map(cluster_label_map)

    room_dummies = pd.get_dummies(work['room_type'], prefix='room')
    hood_dummies = pd.get_dummies(work['neighbourhood_cleansed'], prefix='hood')
    comp_frame = pd.concat([
        work[['bedrooms', 'accommodates', 'review_scores_rating', 'distance_to_event_km', 'latitude', 'longitude']],
        room_dummies,
        hood_dummies
    ], axis=1)
    comp_cols = list(comp_frame.columns)
    comp_scaler = StandardScaler()
    comp_X = comp_scaler.fit_transform(comp_frame)
    comp_nn = NearestNeighbors(n_neighbors=min(12, len(work)), metric='euclidean')
    comp_nn.fit(comp_X)

    return {
        'df': work.reset_index(drop=True),
        'cluster_model': km,
        'cluster_scaler': cluster_scaler,
        'cluster_summary': cluster_summary,
        'cluster_label_map': cluster_label_map,
        'comp_scaler': comp_scaler,
        'comp_nn': comp_nn,
        'comp_cols': comp_cols,
        'room_dummy_cols': list(room_dummies.columns),
        'hood_dummy_cols': list(hood_dummies.columns),
    }


_AI_MARKET = _build_ai_market_layer(merged)
merged = _AI_MARKET['df']
# Always attach a clean demand label to every listing row
if 'distance_to_event_km' in merged.columns:
    merged['demand'] = merged['distance_to_event_km'].apply(get_demand)
else:
    merged['demand'] = 'LOW'
if 'city' in merged.columns:
    _CITY_DATA = {str(_city_name): _city_df.reset_index(drop=True) for _city_name, _city_df in merged.groupby('city', sort=False)}
else:
    _CITY_DATA = {'Washington': merged.reset_index(drop=True)}


def _build_comp_vector(payload):
    row = {
        'bedrooms': _safe_float(payload.get('bedrooms', 1), 1),
        'accommodates': _safe_float(payload.get('accommodates', 2), 2),
        'review_scores_rating': _safe_float(payload.get('review_scores_rating', 4.5), 4.5),
        'distance_to_event_km': _safe_float(payload.get('distance_to_event_km', 5), 5),
        'latitude': _safe_float(payload.get('latitude', 0), 0),
        'longitude': _safe_float(payload.get('longitude', 0), 0),
    }
    vec = pd.DataFrame([row])
    sim_room = str(payload.get('room_type', '') or '')
    sim_hood = str(payload.get('neighbourhood_cleansed', '') or '')
    for col in _AI_MARKET['room_dummy_cols']:
        vec[col] = 1 if col == f'room_{sim_room}' else 0
    for col in _AI_MARKET['hood_dummy_cols']:
        vec[col] = 1 if col == f'hood_{sim_hood}' else 0
    for col in _AI_MARKET['comp_cols']:
        if col not in vec.columns:
            vec[col] = 0
    return vec[_AI_MARKET['comp_cols']]


def get_comparable_summary(payload, top_k=8):
    df = _AI_MARKET['df']
    vec = _build_comp_vector(payload)
    xs = _AI_MARKET['comp_scaler'].transform(vec)
    distances, indices = _AI_MARKET['comp_nn'].kneighbors(xs, n_neighbors=min(top_k, len(df)))
    comps = df.iloc[indices[0]].copy()
    comps = comps.sort_values(['recommended_price', 'review_scores_rating'], ascending=[False, False])
    top_comp_name = str(comps.iloc[0].get('name', 'Top comparable listing'))[:40] if len(comps) > 0 else 'Comparable listing'
    top_comp_price = float(comps.iloc[0].get('recommended_price', 0)) if len(comps) > 0 else 0.0
    cluster_mode = int(comps['market_cluster'].mode().iloc[0]) if len(comps) > 0 else 0
    cluster_label = _AI_MARKET['cluster_label_map'].get(cluster_mode, 'market cluster')
    return {
        'count': int(len(comps)),
        'avg_price': float(comps['recommended_price'].mean()) if len(comps) > 0 else 0.0,
        'avg_baseline': float(comps['baseline_price'].mean()) if len(comps) > 0 else 0.0,
        'avg_uplift': float((comps['recommended_price'] - comps['baseline_price']).mean()) if len(comps) > 0 else 0.0,
        'cluster_id': cluster_mode,
        'cluster_label': cluster_label,
        'top_comp_name': top_comp_name,
        'top_comp_price': top_comp_price,
        'records': comps[['name', 'neighbourhood_cleansed', 'recommended_price', 'baseline_price', 'cluster_label']].head(5).to_dict('records')
    }


def predict_market_cluster(payload):
    df = _AI_MARKET['df']
    vec = pd.DataFrame([{
        'bedrooms': _safe_float(payload.get('bedrooms', 1), 1),
        'accommodates': _safe_float(payload.get('accommodates', 2), 2),
        'review_scores_rating': _safe_float(payload.get('review_scores_rating', 4.5), 4.5),
        'baseline_price': _safe_float(payload.get('baseline_price', df['baseline_price'].mean()), df['baseline_price'].mean()),
        'recommended_price': _safe_float(payload.get('recommended_price_guess', df['recommended_price'].mean()), df['recommended_price'].mean()),
        'distance_to_event_km': _safe_float(payload.get('distance_to_event_km', 5), 5),
    }])
    xs = _AI_MARKET['cluster_scaler'].transform(vec)
    cid = int(_AI_MARKET['cluster_model'].predict(xs)[0])
    summary = _AI_MARKET['cluster_summary']
    row = summary[summary['market_cluster'] == cid].iloc[0]
    return {
        'cluster_id': cid,
        'cluster_label': _AI_MARKET['cluster_label_map'].get(cid, 'market cluster'),
        'avg_price': float(row['avg_rec']),
        'avg_base': float(row['avg_base']),
        'count': int(row['count'])
    }


def _get_market_baseline(payload, fm):
    if len(fm) == 0:
        return 0.0
    filtered = fm.copy()
    hood = str(payload.get('neighbourhood_cleansed', '') or '')
    room = str(payload.get('room_type', '') or '')
    if hood and 'neighbourhood_cleansed' in filtered.columns:
        hood_match = filtered[filtered['neighbourhood_cleansed'].str.lower() == hood.lower()]
        if len(hood_match) > 0:
            filtered = hood_match
    if room and 'room_type' in filtered.columns:
        room_match = filtered[filtered['room_type'] == room]
        if len(room_match) > 0:
            filtered = room_match
    return float(filtered['baseline_price'].mean()) if len(filtered) > 0 else float(fm['baseline_price'].mean())


def _build_ai_pricing_result(payload, fm):
    X = pd.DataFrame([{f: payload.get(f, 0) for f in sim_features}])
    base_pred = float(sim_model.predict(X)[0])
    comp = get_comparable_summary(payload, top_k=8)
    cluster = predict_market_cluster({
        **payload,
        'recommended_price_guess': base_pred,
        'baseline_price': _get_market_baseline(payload, fm)
    })
    city_avg = float(fm['baseline_price'].mean()) if len(fm) > 0 else base_pred

    demand = get_demand(payload.get('distance_to_event_km', 99))
    event_type = str(payload.get('event_type', 'Sports'))
    event_adj = 0.0

    if demand == 'VERY HIGH':
        event_adj += 12.0
    elif demand == 'HIGH':
        event_adj += 7.0
    elif demand == 'MEDIUM':
        event_adj += 3.0

    if event_type == 'Sports':
        event_adj += 3.0
    elif event_type == 'Festival':
        event_adj += 5.0
    elif event_type == 'Music':
        event_adj += 2.0

    comp_anchor = comp['avg_price'] if comp['count'] else city_avg
    cluster_anchor = cluster['avg_price'] if cluster['count'] else city_avg
    predicted_price = (0.55 * base_pred) + (0.30 * comp_anchor) + (0.15 * cluster_anchor) + event_adj
    predicted_price = max(35.0, predicted_price)

    weather_desc = str(
        payload.get('weather_condition')
        or payload.get('current_description')
        or payload.get('forecast_description')
        or ''
    ).lower()
    weather_precip = _safe_float(
        payload.get('precipitation_mm', payload.get('current_precipitation_mm', 0.0)),
        0.0
    )
    bad_weather = any(term in weather_desc for term in ['rain', 'drizzle', 'snow', 'storm', 'thunder']) or weather_precip >= 2.0
    has_event = str(event_type).strip().lower() not in ('', 'none', 'no event')

    weather_adjustment_note = ''
    if has_event and demand in ['VERY HIGH', 'HIGH']:
        predicted_price *= 1.05
        weather_adjustment_note = "Event demand outweighs weather impact, supporting higher pricing."
    elif bad_weather and not has_event:
        predicted_price *= 0.92
        weather_adjustment_note = "Poor weather may reduce demand, so a slight discount improves booking chances."

    predicted_price = max(35.0, predicted_price)

    market_gap = predicted_price - comp_anchor
    action = 'RAISE' if market_gap > 10 else ('DISCOUNT' if market_gap < -10 else 'HOLD')
    monthly_uplift = max(0.0, (predicted_price - city_avg) * 20)
    annual_uplift = max(0.0, (predicted_price - city_avg) * 240)
    market_position = (
        'Above market vs comps' if market_gap > 8
        else ('Below market vs comps' if market_gap < -8 else 'In line with comps')
    )

    explanation_parts = []

    explanation_parts.append(
        "For your listing, I analyzed nearby comps, event demand, and current market conditions."
    )

    if demand == 'VERY HIGH':
        explanation_parts.append("Demand is spiking due to a nearby event — you can safely raise your price.")
    elif demand == 'HIGH':
        explanation_parts.append("Strong nearby demand is pushing prices up.")
    elif demand == 'MEDIUM':
        explanation_parts.append("Moderate demand — there’s some pricing opportunity.")
    else:
        explanation_parts.append("Low event impact — pricing should stay competitive.")

    if comp['count'] > 0:
        explanation_parts.append(
            f"You’re competing with {comp['count']} similar listings averaging ${comp['avg_price']:.0f}/night."
        )

    if action == 'RAISE':
        explanation_parts.append("You’re currently underpriced — I recommend increasing your nightly rate.")
    elif action == 'DISCOUNT':
        explanation_parts.append("You’re slightly overpriced — a small discount will improve bookings.")
    else:
        explanation_parts.append("Your pricing is well aligned with the market.")

    if weather_adjustment_note:
        explanation_parts.append(weather_adjustment_note)

    explanation = " ".join(explanation_parts)
    return {
        'price': predicted_price,
        'action': action,
        'monthly_uplift': monthly_uplift,
        'annual_uplift': annual_uplift,
        'city_avg': city_avg,
        'demand': demand,
        'base_pred': base_pred,
        'comp': comp,
        'cluster': cluster,
        'market_position': market_position,
        'weather_condition': weather_desc,
        'bad_weather': bad_weather,
        'explanation': explanation
    }


def _pin_icon(color):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 36" width="24" height="36">'
        f'<path d="M12 0C5.373 0 0 5.373 0 12c0 6.627 12 24 12 24S24 18.627 24 12C24 5.373 18.627 0 12 0z" '
        f'fill="{color}" stroke="white" stroke-width="1.5"/>'
        f'<circle cx="12" cy="12" r="4.5" fill="white" opacity="0.9"/>'
        f'</svg>'
    )
    return folium.DivIcon(html=svg, icon_size=(24, 36), icon_anchor=(12, 36), class_name='')

def make_map(fm, fe, selected_city='Washington'):
    map_path = 'static/map.html'
    # Always rebuild the map so recommendation/filter changes show immediately.
    # The old version returned early when static/map.html already existed,
    # which made the map/filter appear broken after data or UI changes.
    center = CITY_CENTERS.get(selected_city, [39.5, -98.35])
    zoom = CITY_ZOOMS.get(selected_city, 4)
    m = folium.Map(location=center, zoom_start=zoom, tiles='CartoDB positron')
    colors = {'RAISE':'#22c55e','DISCOUNT':'#f97316','HOLD':'#ef4444'}

    raise_fg    = folium.FeatureGroup(name='Raise',    show=True).add_to(m)
    discount_fg = folium.FeatureGroup(name='Discount', show=True).add_to(m)
    hold_fg     = folium.FeatureGroup(name='Hold',     show=True).add_to(m)
    event_fg    = folium.FeatureGroup(name='Events',   show=True).add_to(m)
    groups = {'RAISE': raise_fg, 'DISCOUNT': discount_fg, 'HOLD': hold_fg}

    map_data = fm.dropna(subset=['latitude','longitude']).copy()
    sample = map_data.sample(min(600, len(map_data)), random_state=42)
    for _, row in sample.iterrows():
        dist = row.get('distance_to_event_km', 99)
        demand = row.get('demand', get_demand(dist))
        action = row['action']
        color = colors.get(action, '#888')
        popup_html = (
            f"<b>{str(row.get('name',''))[:30]}</b><br>"
            f"{row.get('city','')}<br>"
            f"<b>Action: {action}</b><br>"
            f"${row['baseline_price']:.0f} → ${row['recommended_price']:.0f}<br>"
            f"Event Demand: <b>{demand}</b><br>"
            f"Nearest: {str(row.get('nearest_event',''))[:35]}<br>"
            f"Distance: {dist:.1f}km"
        )
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=_pin_icon(color),
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(groups.get(action, hold_fg))

    for _, e in fe.dropna(subset=['latitude','longitude']).iterrows():
        folium.Marker(
            location=[e['latitude'], e['longitude']],
            icon=_pin_icon('#8b5cf6'),
            popup=folium.Popup(
                f"<b>{str(e['event_name'])[:40]}</b><br>{e['date']}<br>{e.get('city','')}",
                max_width=200)
        ).add_to(event_fg)

    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    m.save('static/map.html')

    # Inject postMessage listener so filter buttons outside the iframe can toggle layers
    with open('static/map.html', 'r', encoding='utf-8', errors='replace') as f:
        html_content = f.read()
    filter_js = """
<script>
window.addEventListener('message', function(e) {
    if (!e.data || e.data.type !== 'pwfilter') return;
    var show = Array.isArray(e.data.show) ? e.data.show : [];
    var showLower = show.map(function(s) { return String(s).toLowerCase(); });
    document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(function(lbl) {
        var cb = lbl.querySelector('input[type=checkbox]');
        var span = lbl.querySelector('span');
        var name = span ? span.textContent.trim() : '';
        var shouldShow = showLower.indexOf(name.toLowerCase()) >= 0;
        if (cb && cb.checked !== shouldShow) cb.click();
    });
});
</script>
"""
    html_content = html_content.replace('</body>', filter_js + '</body>')
    with open('static/map.html', 'w', encoding='utf-8') as f:
        f.write(html_content)



def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def _style_axes(ax, grid_axis='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#dbe3ef')
    ax.spines['bottom'].set_color('#dbe3ef')
    ax.tick_params(axis='x', labelsize=10, labelcolor='#253858')
    ax.tick_params(axis='y', labelsize=9, labelcolor='#7b8ba3')
    if grid_axis == 'y':
        ax.yaxis.grid(True, color='#e8edf5', linewidth=1)
        ax.xaxis.grid(False)
    elif grid_axis == 'x':
        ax.xaxis.grid(True, color='#e8edf5', linewidth=1)
        ax.yaxis.grid(False)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)

def make_stat_sparkline(data, line_color='#3b82f6', fill_color='#dbeafe'):
    """Compact sparkline for KPI cards with transparent background."""
    vals = np.array(list(data), dtype=float)
    if len(vals) == 0:
        vals = np.array([0, 0, 0, 0, 0], dtype=float)
    x = np.arange(len(vals))

    if np.max(vals) == np.min(vals):
        base = vals[0] if len(vals) else 0
        vals = np.array([base * 0.96, base * 0.98, base, base * 1.01, base * 1.02])
        x = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(4.2, 1.45))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    spread = max(ymax - ymin, 1.0)
    baseline = ymin - spread * 0.16

    ax.fill_between(x, vals, baseline, color=fill_color, alpha=0.34, zorder=1)
    ax.plot(x, vals, color=line_color, linewidth=2.7, solid_capstyle='round', zorder=2)
    ax.scatter([x[-1]], [vals[-1]], s=30, color=line_color, edgecolor='white', linewidth=1.1, zorder=3)

    ax.set_xlim(x[0] - 0.02, x[-1] + 0.02)
    ax.set_ylim(baseline, ymax + spread * 0.18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0)
    fig.tight_layout(pad=0.02)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=190, bbox_inches='tight', transparent=True, pad_inches=0.0)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def make_stat_ring(percent, color='#22c55e', track='#e2e8f0'):
    """Small donut progress ring for KPI cards with transparent background."""
    pct = max(0, min(float(percent), 100))
    fig, ax = plt.subplots(figsize=(1.7, 1.7))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.pie([pct, 100 - pct], startangle=90, counterclock=False,
           colors=[color, track], wedgeprops=dict(width=0.28, edgecolor='white', linewidth=0.7))
    ax.text(0, 0, f'{int(round(pct))}%', ha='center', va='center',
            fontsize=10.1, fontweight='800', color='#253858')
    ax.set_aspect('equal')
    fig.tight_layout(pad=0.0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=190, bbox_inches='tight', transparent=True, pad_inches=0.0)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img




def make_kpi_donut(data_dict, colors=None):
    """Compact donut chart for KPI cards."""
    items = [(str(k), float(v)) for k, v in data_dict.items() if float(v) > 0]
    if not items:
        items = [('Listings', 1.0)]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    if colors is None:
        palette = ['#3b82f6', '#60a5fa', '#6ee7b7', '#a78bfa', '#cbd5e1']
        colors = palette[:len(values)]

    fig, ax = plt.subplots(figsize=(2.15, 2.15))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.pie(values, startangle=90, counterclock=False, colors=colors,
           wedgeprops=dict(width=0.40, edgecolor='white', linewidth=1.1))
    ax.set_aspect('equal')
    fig.tight_layout(pad=0.0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', transparent=True, pad_inches=0.0)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

def make_weekly_chart(fm):
    """Weekly grouped bar chart."""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    base = fm['baseline_price'].mean() if len(fm) > 0 else 150
    rec = fm['recommended_price'].mean() if len(fm) > 0 else 200

    mult = np.array([0.88, 0.85, 0.87, 0.90, 1.05, 1.30, 1.20])
    current = base * mult
    recommended = rec * mult
    x = np.arange(len(days))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.bar(x - w/2, current, w, label='Current', color='#e5e7eb', edgecolor='none', zorder=3)
    ax.bar(x + w/2, recommended, w, label='Recommended', color='#22c55e', edgecolor='none', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_ylabel('Price ($)', fontsize=10, color='#253858')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.0f}'))
    _style_axes(ax, 'y')
    ax.legend(fontsize=10, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.02), ncol=2)

    fig.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def make_donut_chart(fm):
    """Donut chart for recommendation split."""
    raise_n    = len(fm[fm['action']=='RAISE'])
    discount_n = len(fm[fm['action']=='DISCOUNT'])
    hold_n     = len(fm[fm['action']=='HOLD'])
    total = raise_n + discount_n + hold_n
    if total == 0:
        return ""
    sizes  = [raise_n, discount_n, hold_n]
    colors = ['#22c55e', '#f97316', '#ef4444']
    labels = ['Raise', 'Discount', 'Hold']
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.65, edgecolor='white', linewidth=2)
    )
    ax.text(
        0, 0, f'{total:,}\nListings',
        ha='center', 
        va='center',
        fontsize=20, 
        fontweight='bold', 
        color='#0a1f44', 
        linespacing=1.8
    )
    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img

def make_revenue_by_event_chart(fm):
    """Grouped bar chart: current vs recommended pricing by event type."""
    base = fm['baseline_price'].mean() if len(fm) > 0 else 150.0
    rec = fm['recommended_price'].mean() if len(fm) > 0 else 200.0

    labels = ['Normal Weekend', 'Concert Weekend', 'Sports Weekend', 'Festival Weekend']
    current = np.array([
        round(base * 1.02, 2),
        round(base * 0.76, 2),
        round(base * 0.73, 2),
        round(base * 0.71, 2),
    ])
    recommended = np.array([
        round(base * 1.00, 2),
        round(rec * 0.95, 2),
        round(rec * 0.92, 2),
        round(rec * 1.05, 2),
    ])

    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    current_bars = ax.bar(x - w/2, current, w, label='Current Price', color='#d9dee7', edgecolor='none', zorder=3)
    rec_bars = ax.bar(x + w/2, recommended, w, label='Recommended Price', color='#4f7f5b', edgecolor='none', zorder=3)

    for bars in (current_bars, rec_bars):
        for b in bars:
            val = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2,
                val + 3,
                f'${val:,.2f}',
                ha='center',
                va='bottom',
                fontsize=10.5,
                fontweight='700',
                color='#2f3640'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color='#22314d')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.tick_params(axis='y', labelsize=10, labelcolor='#8da0b8')
    ax.grid(axis='y', color='#e9eef5', linewidth=1.1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e9eef5')
    ax.spines['bottom'].set_color('#e9eef5')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=11)

    fig.tight_layout(pad=1.4)
    return _fig_to_b64(fig)


def make_revenue_event_summary(fm):
    """KPI summary matching the grouped revenue event chart."""
    base = fm['baseline_price'].mean() if len(fm) > 0 else 150.0
    rec = fm['recommended_price'].mean() if len(fm) > 0 else 200.0

    labels = ['Normal Weekend', 'Concert Weekend', 'Sports Weekend', 'Festival Weekend']
    current = np.array([
        round(base * 1.02, 2),
        round(base * 0.76, 2),
        round(base * 0.73, 2),
        round(base * 0.71, 2),
    ])
    recommended = np.array([
        round(base * 1.00, 2),
        round(rec * 0.95, 2),
        round(rec * 0.92, 2),
        round(rec * 1.05, 2),
    ])
    gaps = recommended - current
    higher_count = int((gaps > 0).sum())
    best_idx = int(np.argmax(gaps))
    avg_increase = float(np.mean(gaps)) if len(gaps) else 0.0
    increase_pct = float((avg_increase / max(np.mean(current), 1)) * 100) if len(current) else 0.0

    return {
        'higher_count': higher_count,
        'event_total': len(labels),
        'best_event': labels[best_idx],
        'best_gap': float(max(gaps[best_idx], 0.0)),
        'avg_increase': avg_increase,
        'increase_pct': increase_pct,
    }

def make_calendar_cards(fe):
    """Return list of dicts for event calendar cards."""
    if len(fe) == 0:
        return []
    ed = fe[['date','event_name','category','city']].copy()
    # Try to get venue from events if available
    if 'venue' in fe.columns:
        ed['venue'] = fe['venue']
    else:
        ed['venue'] = ed['city']
    ed['date'] = pd.to_datetime(ed['date'], errors='coerce')
    ed = ed.dropna(subset=['date']).sort_values('date')
    IMPACT = {'Sports': 32, 'Music': 28, 'Arts & Theatre': 18, 'Festival': 45, 'Conference': 18}
    ICONS = {'Sports': '🏆', 'Music': '🎵', 'Arts & Theatre': '🎭', 'Festival': '🎪', 'Conference': '🏛'}
    cards = []
    for _, row in ed.iterrows():
        cat = row['category']
        impact = IMPACT.get(cat, 20)
        icon   = ICONS.get(cat, '📅')
        cards.append({
            'date': f"{row['date']:%b} {row['date'].day}",
            'date_range': f"{row['date']:%b} {row['date'].day}, {row['date']:%Y}",
            'icon': icon,
            'name': str(row['event_name'])[:55],
            'venue': str(row.get('venue', row['city']))[:40],
            'category': cat,
            'impact': f'+{impact}%',
        })
    return cards[:60]

def make_lead_time_chart(fm):
    """Rounded area-style lead time chart for booking window analysis."""
    bins = ['Same day', '1–3 days', '4–7 days', '8–14 days', '15–30 days', '30+ days']
    vals = np.array([8, 18, 24, 22, 16, 12])
    x = np.arange(len(bins))

    fig, ax = plt.subplots(figsize=(10, 3.7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(x, vals, color='#4f46e5', linewidth=2.6, marker='o', markersize=7, zorder=3)
    ax.fill_between(x, vals, 0, color='#c7d2fe', alpha=0.35, zorder=2)
    for xi, yi in zip(x, vals):
        ax.text(xi, yi + 1.0, f'{yi}%', ha='center', fontsize=9.5, fontweight='700', color='#4338ca')

    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel('Share of bookings (%)', fontsize=10, color='#253858')
    _style_axes(ax, 'y')

    fig.tight_layout(pad=1.4)
    return _fig_to_b64(fig)


def make_price_elasticity_chart(fm):
    """Line chart: estimated booking rate vs price increase %."""
    pct   = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    rates = [78, 75, 71, 66, 60, 53, 45, 34, 22]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(pct, rates, color='#f97316', linewidth=2.5, zorder=3)
    ax.fill_between(pct, rates, 0, alpha=0.1, color='#f97316')
    ax.scatter(pct, rates, color='#f97316', s=40, zorder=4)
    ax.set_xlabel('Price increase (%)', fontsize=10, color='#0a1f44')
    ax.set_ylabel('Est. booking rate (%)', fontsize=10, color='#0a1f44')
    ax.tick_params(axis='both', labelsize=9, labelcolor='#94a3b8')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#f1f5f9')
    ax.spines['bottom'].set_color('#f1f5f9')
    ax.yaxis.grid(True, color='#f1f5f9', linewidth=1)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img

def make_review_price_chart(fm):
    """Scatter plot: review score vs recommended price."""
    if len(fm) < 10:
        return ""
    sample = fm.dropna(subset=['review_scores_rating', 'recommended_price']).sample(min(300, len(fm)), random_state=42)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    clr = {'RAISE': '#22c55e', 'DISCOUNT': '#f97316', 'HOLD': '#ef4444'}
    for action, grp in sample.groupby('action'):
        ax.scatter(grp['review_scores_rating'], grp['recommended_price'],
                   color=clr.get(action, '#888'), alpha=0.45, s=16, label=action.title(), zorder=3)
    ax.set_xlabel('Review Score', fontsize=10, color='#0a1f44')
    ax.set_ylabel('Recommended Price ($)', fontsize=10, color='#0a1f44')
    ax.tick_params(axis='both', labelsize=9, labelcolor='#94a3b8')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#f1f5f9')
    ax.spines['bottom'].set_color('#f1f5f9')
    ax.yaxis.grid(True, color='#f1f5f9', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img

def make_seasonal_demand_chart(fm):
    """Seasonal uplift chart with labeled monthly peaks."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    uplift = np.array([8, 10, 14, 20, 26, 32, 42, 38, 25, 18, 12, 8])
    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(10, 3.7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(x, uplift, color='#4f46e5', linewidth=2.6, marker='o', markersize=6.5, zorder=3)
    ax.fill_between(x, uplift, 0, color='#ddd6fe', alpha=0.55, zorder=2)
    for xi, yi in zip(x, uplift):
        ax.text(xi, yi + 1.6, f'{yi}%', ha='center', fontsize=9.5, fontweight='700', color='#4338ca')

    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel('Avg Recommended Increase (%)', fontsize=10, color='#253858')
    ax.set_ylim(0, 50)
    _style_axes(ax, 'y')

    fig.tight_layout(pad=1.4)
    return _fig_to_b64(fig)


def make_neighborhood_bar_chart(fm):
    """Polished horizontal bar: total annual opportunity by neighborhood."""
    if 'neighbourhood_cleansed' not in fm.columns or len(fm) == 0:
        return ""
    raise_df = fm[fm['action'] == 'RAISE'].copy()
    if len(raise_df) == 0:
        return ""

    nb = raise_df.groupby('neighbourhood_cleansed').agg(
        avg_uplift=('price_uplift', 'mean'),
        count=('id', 'count')
    ).reset_index()
    nb['annual_opportunity'] = nb['avg_uplift'] * 240 * nb['count']
    nb = nb.nlargest(5, 'annual_opportunity').sort_values('annual_opportunity', ascending=True)

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = ['#b8a7ff', '#a38df8', '#8f73f2', '#7c5ce8', '#6a4de0']
    bars = ax.barh(nb['neighbourhood_cleansed'], nb['annual_opportunity'], color=colors, height=0.62, zorder=3)

    maxv = nb['annual_opportunity'].max() if len(nb) else 1
    for bar, val in zip(bars, nb['annual_opportunity']):
        ax.text(val + maxv * 0.04, bar.get_y() + bar.get_height() / 2,
                f'${val/1000:.1f}K', va='center', fontsize=10.5, fontweight='700', color='#253858')

    ax.set_xlabel('Total Annual Opportunity ($)', fontsize=10, color='#253858', labelpad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(round(v/1000.0))}K' if v > 0 else '0'))
    _style_axes(ax, 'x')
    ax.spines['left'].set_visible(False)

    fig.tight_layout(pad=1.4)
    return _fig_to_b64(fig)


def make_property_type_chart(fm):
    """Donut chart: annual opportunity share by property type."""
    if len(fm) == 0:
        return ""
    raise_df = fm[fm['action'] == 'RAISE'].copy()
    if len(raise_df) == 0:
        return ""

    def get_prop_type(row):
        rt = str(row.get('room_type', ''))
        beds = row.get('bedrooms', 1)
        if pd.isna(beds):
            beds = 1
        beds = int(beds)
        if 'Private room' in rt:
            return 'Private Room'
        if beds <= 1:
            return 'Studio / 1BR'
        if beds == 2:
            return '2 Bedroom'
        if beds >= 3:
            return '3+ Bedroom'
        return 'Other'

    raise_df['prop_type'] = raise_df.apply(get_prop_type, axis=1)
    pt = raise_df.groupby('prop_type').agg(
        avg_uplift=('price_uplift', 'mean'),
        count=('id', 'count')
    ).reset_index()
    pt['annual_opportunity'] = pt['avg_uplift'] * 240 * pt['count']
    pt = pt.sort_values('annual_opportunity', ascending=False).head(5)
    if len(pt) == 0:
        return ""

    labels = pt['prop_type'].tolist()
    values = pt['annual_opportunity'].to_numpy()
    colors = ['#1f4db0', '#2f76db', '#5b8ee5', '#8fb2ee', '#c9d5f5'][:len(pt)]
    total = values.sum()

    fig = plt.figure(figsize=(8.0, 4.2), facecolor='white')
    ax = fig.add_axes([0.02, 0.08, 0.45, 0.84])
    ax.set_facecolor('white')
    wedges, _ = ax.pie(values, startangle=90, counterclock=False, colors=colors,
                       wedgeprops=dict(width=0.54, edgecolor='white', linewidth=1.4))

    pct = values / total * 100 if total else np.zeros_like(values)
    for w, p in zip(wedges, pct):
        ang = (w.theta2 + w.theta1) / 2.0
        x = 0.72 * np.cos(np.deg2rad(ang))
        y = 0.72 * np.sin(np.deg2rad(ang))
        if p >= 6:
            ax.text(x, y, f'{p:.0f}%', ha='center', va='center', fontsize=10.5, fontweight='700', color='white')
    ax.set_aspect('equal')

    legend_ax = fig.add_axes([0.52, 0.12, 0.45, 0.76])
    legend_ax.axis('off')
    y = 0.85
    for label, val, color in zip(labels, values, colors):
        legend_ax.scatter([0.05], [y], s=60, color=color)
        legend_ax.text(0.12, y, label, va='center', fontsize=10.5, color='#253858', fontweight='600')
        legend_ax.text(0.86, y, f'${val/1000:.1f}K', va='center', ha='right', fontsize=10.5, color='#253858', fontweight='700')
        y -= 0.18

    legend_ax.text(0.43, 0.10, 'Total Opportunity', ha='center', fontsize=10.5, color='#64748b', fontweight='600')
    legend_ax.text(0.43, -0.02, f'${total/1000:.1f}K', ha='center', fontsize=19, color='#172b6a', fontweight='800')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(-0.1, 1)

    return _fig_to_b64(fig)


def make_distance_buckets_chart(fm):
    """Dot plot: uplift by distance to venue."""
    buckets = ['< 1 mile', '1-3 miles', '3-5 miles', '5+ miles']
    values = np.array([31, 19, 10, 3])
    y = np.arange(len(buckets))
    colors = ['#16a34a', '#22c55e', '#84cc16', '#c4d959']
    sizes = [360, 260, 220, 180]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for yi, xv, c, s in zip(y, values, colors, sizes):
        ax.hlines(yi, 0, xv, color='#cbd5e1', linewidth=1.8, linestyles=(0, (1.2, 2.4)), zorder=1)
        ax.scatter([xv], [yi], s=s, color=c, edgecolor='white', linewidth=1.5, zorder=3)
        ax.text(xv + 1.4, yi, f'+{int(xv)}%', va='center', fontsize=10.5, fontweight='700', color='#15803d')

    ax.set_yticks(y)
    ax.set_yticklabels(buckets, fontsize=10, color='#253858')
    ax.set_xlabel('Avg Recommended Increase (%)', fontsize=10, color='#253858')
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.6, len(buckets) - 0.4)
    ax.invert_yaxis()
    _style_axes(ax, 'x')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    fig.tight_layout(pad=1.25)
    return _fig_to_b64(fig)


def make_review_bands_chart(fm):
    """Bubble chart: premium by review score band with blue-to-lilac palette."""
    bands = ['4.9+', '4.7-4.89', '4.5-4.69', 'Below 4.5']
    values = np.array([22, 14, 7, 0])
    x = np.arange(len(bands))
    sizes = np.array([3600, 2800, 2000, 850])
    colors = ['#4f86f7', '#7f7cf2', '#aaa6f5', '#c5ccd8']

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.scatter(x, values, s=sizes, c=colors, alpha=0.95,
               edgecolor='white', linewidth=1.8, zorder=3)

    for xi, yi in zip(x, values):
        if yi > 0:
            ax.text(xi, yi, f'+{int(yi)}%', ha='center', va='center',
                    fontsize=12.5, fontweight='800', color='white', zorder=4)
        else:
            ax.text(xi, yi + 1.0, '0%', ha='center', va='center',
                    fontsize=11, fontweight='800', color='white', zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel('Avg Premium (%)', fontsize=10, color='#253858')
    ax.set_xlabel('Review Score Band', fontsize=10, color='#253858')
    ax.set_ylim(0, 25)
    ax.set_xlim(-0.4, 3.4)
    _style_axes(ax, 'y')
    ax.spines['left'].set_color('#d3dae6')
    ax.spines['bottom'].set_color('#d3dae6')
    ax.yaxis.grid(True, color='#dfe6f0', linewidth=1, linestyle=(0, (3, 4)))

    fig.tight_layout(pad=1.25)
    return _fig_to_b64(fig)


def make_feature_importance_chart(metrics):
    """Removed  chart."""
    return ""


def _apply_filters(fd, neighbourhood=None, room_type=None, bedrooms=None,
                   min_accommodates=None, action=None):
    if neighbourhood and 'neighbourhood_cleansed' in fd.columns:
        fd = fd[fd['neighbourhood_cleansed'].str.contains(neighbourhood, case=False, na=False)]
    if room_type and 'room_type' in fd.columns:
        fd = fd[fd['room_type'] == room_type]
    if bedrooms is not None and 'bedrooms' in fd.columns:
        fd = fd[fd['bedrooms'] == bedrooms]
    if min_accommodates is not None and 'accommodates' in fd.columns:
        fd = fd[fd['accommodates'] >= min_accommodates]
    if action and 'action' in fd.columns:
        fd = fd[fd['action'] == action]
    return fd


def _execute_tool(name: str, inp: dict, fm) -> str:
    try:
        if name == "get_summary_stats":
            fd = _apply_filters(fm,
                neighbourhood=inp.get('neighbourhood'), room_type=inp.get('room_type'),
                bedrooms=inp.get('bedrooms'), min_accommodates=inp.get('min_accommodates'),
                action=inp.get('action'))
            if len(fd) == 0:
                return "No listings match those filters."
            rc  = len(fd[fd['action'] == 'RAISE'])
            dc  = len(fd[fd['action'] == 'DISCOUNT'])
            hc  = len(fd[fd['action'] == 'HOLD'])
            avg = fd['recommended_price'].mean()
            bl  = fd['baseline_price'].mean()
            result = (f"{len(fd):,} listings — RAISE: {rc:,}, DISCOUNT: {dc:,}, HOLD: {hc:,}.\n"
                      f"Avg baseline: ${bl:.0f}/night | Avg recommended: ${avg:.0f}/night | Avg uplift: +${avg - bl:.0f}.")
            if 'review_scores_rating' in fd.columns:
                rating = fd['review_scores_rating'].mean()
                if not pd.isna(rating):
                    result += f"\nAvg rating: {rating:.2f}/5."
            return result

        elif name == "get_top_listings":
            sort_by = inp.get('sort_by', 'price_uplift')
            limit   = int(inp.get('limit', 8))
            fd = _apply_filters(fm,
                neighbourhood=inp.get('neighbourhood'), room_type=inp.get('room_type'),
                bedrooms=inp.get('bedrooms'), min_accommodates=inp.get('min_accommodates'),
                action=inp.get('action'))
            if len(fd) == 0:
                return "No listings match those filters."
            cols = ['name', 'neighbourhood_cleansed', 'room_type', 'bedrooms',
                    'baseline_price', 'recommended_price', 'price_uplift', 'action']
            if sort_by == 'review_scores_rating':
                cols.append('review_scores_rating')
            if 'nearest_event' in fd.columns:
                cols.append('nearest_event')
            cols = [c for c in cols if c in fd.columns]
            top = fd.nlargest(limit, sort_by) if sort_by in fd.columns else fd.head(limit)
            return top[cols].to_string(index=False)

        elif name == "get_neighbourhood_breakdown":
            sort_by = inp.get('sort_by', 'avg_price')
            limit   = int(inp.get('limit', 12))
            by_hood = fm.groupby('neighbourhood_cleansed').agg(
                avg_price=('recommended_price', 'mean'),
                avg_uplift=('price_uplift', 'mean'),
                count=('id', 'count'),
                raise_pct=('action', lambda x: round((x == 'RAISE').mean() * 100, 1))
            ).sort_values(sort_by, ascending=False).head(limit)
            return by_hood.to_string()

        elif name == "get_room_type_breakdown":
            by_type = fm.groupby('room_type').agg(
                avg_price=('recommended_price', 'mean'),
                avg_uplift=('price_uplift', 'mean'),
                count=('id', 'count')
            ).sort_values('avg_price', ascending=False)
            return by_type.to_string()

        elif name == "get_bedroom_breakdown":
            if 'bedrooms' not in fm.columns:
                return "Bedroom data not available."
            by_beds = fm.groupby('bedrooms').agg(
                avg_price=('recommended_price', 'mean'),
                count=('id', 'count')
            ).sort_values('bedrooms')
            return by_beds.to_string()

        elif name == "get_event_impact":
            parts = []
            if 'distance_to_event_km' in fm.columns:
                bins   = [0, 2, 5, 10, 20, 999]
                labels = ['<2 km', '2-5 km', '5-10 km', '10-20 km', '>20 km']
                tmp = fm.copy()
                tmp['dist_band'] = pd.cut(tmp['distance_to_event_km'], bins=bins, labels=labels)
                bd = tmp.groupby('dist_band', observed=True).agg(
                    avg_price=('recommended_price', 'mean'),
                    avg_uplift=('price_uplift', 'mean'),
                    count=('id', 'count')
                )
                parts.append("Price by distance from nearest event:\n" + bd.to_string())
            if 'nearest_event' in fm.columns:
                ev = fm.groupby('nearest_event').agg(
                    count=('id', 'count'),
                    avg_uplift=('price_uplift', 'mean')
                ).sort_values('avg_uplift', ascending=False).head(8)
                parts.append("Top events by avg price uplift:\n" + ev.to_string())
            return "\n\n".join(parts) if parts else "Event data not available."

        elif name == "get_revenue_opportunity":
            fd = fm.copy()
            if inp.get('neighbourhood') and 'neighbourhood_cleansed' in fd.columns:
                fd = fd[fd['neighbourhood_cleansed'].str.contains(inp['neighbourhood'], case=False, na=False)]
            rdf = fd[fd['action'] == 'RAISE']
            if len(rdf) == 0:
                return "No underpriced listings found."
            up    = (rdf['recommended_price'] - rdf['baseline_price']).mean()
            total = up * 240 * len(rdf)
            return (f"{len(rdf):,} underpriced listings (RAISE action).\n"
                    f"Avg nightly uplift: +${up:.0f}.\n"
                    f"Estimated annual revenue opportunity: ${total:,.0f} (assumes 240 booked nights/year).")

        elif name == "get_pricing_by_month":
            month = inp.get('month')
            year  = inp.get('year')
            if 'event_date' not in fm.columns:
                return "Event date data not available."
            fd2 = fm.copy()
            fd2['event_date'] = pd.to_datetime(fd2['event_date'], errors='coerce')
            fd2 = fd2[fd2['event_date'].dt.month == month]
            if year:
                fd2 = fd2[fd2['event_date'].dt.year == year]
            if len(fd2) == 0:
                return f"No data for month {month}{f' / {year}' if year else ''}."
            by_date = fd2.groupby(fd2['event_date'].dt.date)['recommended_price'].mean().sort_values(ascending=False).head(10)
            return f"Avg recommended prices for month {month}{f' {year}' if year else ''}:\n" + by_date.to_string()

        elif name == "get_weather_forecast":
            forecast = get_dc_forecast()
            if not forecast:
                return "Weather forecast unavailable."
            days_str = "\n".join(
                f"  {d['date']}: {d['description']}, {d['temp_max_c']:.0f}/{d['temp_min_c']:.0f}°C, {d['precipitation_mm']:.1f}mm rain"
                for d in forecast['days']
            )
            return f"Current: {forecast['current_description']}, {forecast['current_temp_c']}°C.\n7-day forecast:\n{days_str}"

        elif name == "semantic_search":
            q    = inp.get('query', '')
            k    = int(inp.get('top_k', 8))
            docs = _retrieve(q, top_k=k)
            if not docs:
                return "No relevant listings or reviews found."
            listing_docs = [d for d in docs if d['source'] == 'listing']
            review_docs  = [d for d in docs if d['source'] == 'review']
            parts = []
            for d in listing_docs:
                parts.append(f"[Listing #{d['listing_id']}]: {d['text'][:500]}")
            for d in review_docs:
                parts.append(f"[Reviews #{d['listing_id']}]: {d['text'][:450]}")
            return "\n\n".join(parts)

        elif name == "get_comparable_pricing":
            payload = {
                'neighbourhood_cleansed': inp.get('neighbourhood', ''),
                'room_type': inp.get('room_type', 'Entire home/apt'),
                'bedrooms': inp.get('bedrooms', 1),
                'accommodates': inp.get('accommodates', 2),
                'review_scores_rating': inp.get('review_scores_rating', 4.5),
                'distance_to_event_km': inp.get('distance_to_event_km', 5),
            }
            if payload['neighbourhood_cleansed'] in _HOOD_CENTERS:
                payload['latitude'], payload['longitude'] = _HOOD_CENTERS[payload['neighbourhood_cleansed']]
            comp = get_comparable_summary(payload, top_k=8)
            return (
                f"Comparable set: {comp['count']} listings averaging ${comp['avg_price']:.0f}/night. "
                f"Cluster: {comp['cluster_label']}. Top comp: {comp['top_comp_name']} (${comp['top_comp_price']:.0f})."
            )

        elif name == "get_cluster_breakdown":
            cs = _AI_MARKET['cluster_summary'][['market_cluster', 'cluster_label', 'count', 'avg_base', 'avg_rec', 'avg_dist']]
            return cs.to_string(index=False)

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {e}"


def answer(query, fm, history=None):
    if len(fm) == 0:
        return "<p style='color:#0a1f44'>No data available for the selected city.</p>"

    def _html_escape(value):
        if value is None:
            return ''
        return (str(value)
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))

    def _currency(value):
        try:
            return f"${float(value):.0f}"
        except Exception:
            return "$0"

    def _percent(value):
        try:
            return f"{float(value):.1f}%"
        except Exception:
            return "0.0%"

    def _bullets(title, items):
        html = [f"<p><b>{_html_escape(title)}</b></p>", "<ul>"]
        for item in items:
            html.append(f"<li>{item}</li>")
        html.append("</ul>")
        return ''.join(html)

    def _room_type_from_query(q_lower):
        if 'private room' in q_lower:
            return 'Private room'
        if 'shared room' in q_lower:
            return 'Shared room'
        if 'hotel room' in q_lower:
            return 'Hotel room'
        if 'entire home' in q_lower or 'entire apt' in q_lower or 'entire apartment' in q_lower or 'entire place' in q_lower:
            return 'Entire home/apt'
        return None

    def _extract_int_before_terms(text_lower, terms):
        import re
        for term in terms:
            m = re.search(rf'(\d+)\s*[- ]?{term}', text_lower)
            if m:
                return int(m.group(1))
        return None

    def _extract_month(text_lower):
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9,
            'oct': 10, 'nov': 11, 'dec': 12
        }
        for key, val in month_map.items():
            if key in text_lower:
                return val
        return None

    def _event_type_from_query(q_lower):
        if any(t in q_lower for t in ['concert', 'music', 'tour']):
            return 'Music'
        if any(t in q_lower for t in ['festival', 'fair']):
            return 'Festival'
        if any(t in q_lower for t in ['conference', 'convention', 'summit']):
            return 'Conference'
        if any(t in q_lower for t in ['game', 'sports', 'match', 'stadium']):
            return 'Sports'
        return 'Sports'

    def _find_neighbourhood(q_lower):
        if 'neighbourhood_cleansed' not in fm.columns:
            return ''
        names = [str(x) for x in fm['neighbourhood_cleansed'].dropna().unique()]
        names_sorted = sorted(names, key=len, reverse=True)
        for name in names_sorted:
            if name and name.lower() in q_lower:
                return name
        return ''

    def _extract_neighbourhood_fragment(q_lower):
        import re
        patterns = [
            r'\bin\s+([a-z][a-z\s,&\-/]+?)(?:\s+should\b|\s+that\b|\s+with\b|\s+near\b|\s+for\b|\?|$)',
            r'\bnear\s+([a-z][a-z\s,&\-/]+?)(?:\s+should\b|\s+that\b|\s+with\b|\s+for\b|\?|$)',
        ]
        for pattern in patterns:
            m = re.search(pattern, q_lower)
            if m:
                frag = m.group(1).strip(' ,.-')
                if frag:
                    return frag
        return ''

    def _filter_ranked_listings_from_query(work, q_lower):
        filtered = work.copy()
        room_type = _room_type_from_query(q_lower)
        bedrooms = _extract_int_before_terms(q_lower, ['bedroom', 'bedrooms', 'br'])

        if room_type and 'room_type' in filtered.columns:
            tmp = filtered[filtered['room_type'] == room_type]
            if len(tmp) > 0:
                filtered = tmp

        if bedrooms is not None and 'bedrooms' in filtered.columns:
            tmp = filtered[filtered['bedrooms'].fillna(0) == bedrooms]
            if len(tmp) > 0:
                filtered = tmp

        hood = _find_neighbourhood(q_lower)
        if hood and 'neighbourhood_cleansed' in filtered.columns:
            tmp = filtered[filtered['neighbourhood_cleansed'].str.contains(hood, case=False, na=False)]
            if len(tmp) > 0:
                filtered = tmp
        else:
            frag = _extract_neighbourhood_fragment(q_lower)
            if frag and 'neighbourhood_cleansed' in filtered.columns:
                parts = [p.strip() for p in frag.replace(' and ', ',').split(',') if p.strip()]
                matched = None
                for part in parts:
                    if len(part) < 3:
                        continue
                    tmp = filtered[filtered['neighbourhood_cleansed'].str.contains(part, case=False, na=False)]
                    if len(tmp) > 0:
                        matched = tmp
                        break
                if matched is not None:
                    filtered = matched

        return filtered

    def _default_payload(q_lower):
        hood = _find_neighbourhood(q_lower)
        room_type = _room_type_from_query(q_lower) or 'Entire home/apt'
        bedrooms = _extract_int_before_terms(q_lower, ['bedroom', 'bedrooms', 'br']) or 1
        accommodates = _extract_int_before_terms(q_lower, ['guest', 'guests', 'person', 'people'])
        if accommodates is None:
            accommodates = max(2, bedrooms * 2)
        month_num = _extract_month(q_lower)
        event_type = _event_type_from_query(q_lower)

        subset = fm.copy()
        if hood and 'neighbourhood_cleansed' in subset.columns:
            tmp = subset[subset['neighbourhood_cleansed'].str.lower() == hood.lower()]
            if len(tmp) > 0:
                subset = tmp
        if room_type and 'room_type' in subset.columns:
            tmp = subset[subset['room_type'] == room_type]
            if len(tmp) > 0:
                subset = tmp
        if 'bedrooms' in subset.columns:
            tmp = subset[subset['bedrooms'].fillna(0) == bedrooms]
            if len(tmp) > 0:
                subset = tmp

        baseline_price = float(subset['baseline_price'].mean()) if len(subset) > 0 else float(fm['baseline_price'].mean())
        review_score = float(subset['review_scores_rating'].mean()) if 'review_scores_rating' in subset.columns and len(subset) > 0 else 4.7
        distance_to_event_km = float(subset['distance_to_event_km'].mean()) if 'distance_to_event_km' in subset.columns and len(subset) > 0 else 5.0
        payload = {
            'neighbourhood_cleansed': hood,
            'room_type': room_type,
            'bedrooms': bedrooms,
            'accommodates': accommodates,
            'review_scores_rating': review_score if not pd.isna(review_score) else 4.7,
            'distance_to_event_km': distance_to_event_km if not pd.isna(distance_to_event_km) else 5.0,
            'event_type': event_type,
            'baseline_price': baseline_price,
        }
        if hood in _HOOD_CENTERS:
            payload['latitude'], payload['longitude'] = _HOOD_CENTERS[hood]
        if month_num is not None:
            try:
                wx = get_monthly_weather_features(month_num)
                if isinstance(wx, dict):
                    payload.update(wx)
            except Exception:
                pass
        return payload

    def _confidence_label(result, payload):
        comp_count = int(result.get('comp', {}).get('count', 0))
        demand = str(result.get('demand', 'LOW'))
        hood_known = bool(payload.get('neighbourhood_cleansed'))
        room_known = bool(payload.get('room_type'))
        score = 45
        score += min(comp_count, 10) * 4
        if demand in ('VERY HIGH', 'HIGH'):
            score += 8
        if hood_known:
            score += 10
        if room_known:
            score += 5
        score = max(35, min(score, 96))
        label = 'High' if score >= 80 else ('Medium' if score >= 60 else 'Low')
        return score, label

    q = (query or '').strip()
    ql = q.lower()

    # --- Enhanced Ask AI feature router ---
    def _tab_link(tab, label):
        return f'<a href="/dashboard?tab={tab}" style="font-weight:700;color:#1d4ed8;text-decoration:none">{_html_escape(label)}</a>'

    def _feature_help():
        return (
            "<p><b>I can answer questions across the whole dashboard now.</b></p>"
            "<ul>"
            "<li><b>Map:</b> ‘show the map’, ‘where are raise listings?’, ‘map event demand’.</li>"
            "<li><b>Recommendations:</b> ‘show listings to raise in Capitol Hill’, ‘discount private rooms’, ‘2 bedroom recommendations’.</li>"
            "<li><b>Simulator:</b> ‘price a 2 bedroom entire home in Navy Yard for 4 guests near a sports event’.</li>"
            "<li><b>Revenue:</b> ‘weekly revenue opportunity’, ‘annual upside by neighbourhood’.</li>"
            "<li><b>Events/weather:</b> ‘events this month’, ‘how does rain affect pricing?’, ‘forecast’.</li>"
            "<li><b>Market/model:</b> ‘top neighbourhoods’, ‘room type breakdown’, ‘cluster breakdown’, ‘portfolio performance’.</li>"
            "</ul>"
        )

    def _listing_table(rows_df, title='Listings', limit=10):
        if rows_df is None or len(rows_df) == 0:
            return f"<p>No matching listings found for <b>{_html_escape(title)}</b>.</p>"
        show = rows_df.head(limit).copy()
        body = []
        for _, row in show.iterrows():
            uplift = _safe_float(row.get('recommended_price', 0), 0) - _safe_float(row.get('baseline_price', 0), 0)
            body.append(
                f"<tr><td><b>{_html_escape(str(row.get('name', 'Listing'))[:44])}</b></td>"
                f"<td>{_html_escape(row.get('neighbourhood_cleansed', '—'))}</td>"
                f"<td>{_html_escape(row.get('room_type', '—'))}</td>"
                f"<td>{_currency(row.get('baseline_price', 0))}</td>"
                f"<td>{_currency(row.get('recommended_price', 0))}</td>"
                f"<td>{'+' if uplift >= 0 else ''}{_currency(uplift)}</td>"
                f"<td><b>{_html_escape(row.get('action', 'HOLD'))}</b></td></tr>"
            )
        return (
            f"<p><b>{_html_escape(title)}</b></p>"
            f"<table><thead><tr><th>Listing</th><th>Neighbourhood</th><th>Type</th><th>Current</th><th>Recommended</th><th>Gap</th><th>Action</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>"
        )

    if any(term in ql for term in ['what can you do', 'help', 'features', 'commands', 'examples']):
        return _feature_help()

    if any(term in ql for term in ['map', 'where are', 'location', 'locations', 'near event', 'nearby event', 'event demand map']):
        work = fm.copy()
        if 'price_uplift' not in work.columns and {'recommended_price', 'baseline_price'}.issubset(work.columns):
            work['price_uplift'] = work['recommended_price'] - work['baseline_price']
        work = _filter_ranked_listings_from_query(work, ql)
        raise_count = int((work.get('action', pd.Series(dtype=str)) == 'RAISE').sum()) if 'action' in work.columns else 0
        discount_count = int((work.get('action', pd.Series(dtype=str)) == 'DISCOUNT').sum()) if 'action' in work.columns else 0
        hold_count = int((work.get('action', pd.Series(dtype=str)) == 'HOLD').sum()) if 'action' in work.columns else 0
        event_near = int((work['distance_to_event_km'].fillna(999) < 5).sum()) if 'distance_to_event_km' in work.columns else 0
        top_map = work.sort_values(['price_uplift', 'recommended_price'], ascending=[False, False]) if 'price_uplift' in work.columns else work
        return (
            "<p><b>Map insight</b></p>"
            f"<ul><li>{len(work):,} matching listings on the current Washington DC map data.</li>"
            f"<li><b>Raise:</b> {raise_count:,} · <b>Discount:</b> {discount_count:,} · <b>Hold:</b> {hold_count:,}.</li>"
            f"<li><b>{event_near:,}</b> listings are within 5 km of an event, the strongest map demand zone.</li>"
            f"<li>Open the interactive map: {_tab_link('map', 'Map tab')}.</li></ul>"
            + _listing_table(top_map, 'Top map-linked listings by pricing opportunity', limit=6)
        )

    if any(term in ql for term in ['calendar', 'calender', 'events this', 'upcoming events', 'event calendar', 'what events', 'show events']):
        _, fe = get_city_data('Washington')
        if len(fe) == 0:
            return "<p>No event data is available right now.</p>"
        ev = fe.copy()
        if 'date' in ev.columns:
            ev['date'] = pd.to_datetime(ev['date'], errors='coerce')
            ev = ev.dropna(subset=['date']).sort_values('date').head(12)
        rows = []
        for _, row in ev.iterrows():
            dt = row.get('date', '')
            dstr = dt.strftime('%b %d, %Y') if hasattr(dt, 'strftime') else str(dt)
            rows.append(f"<tr><td>{_html_escape(dstr)}</td><td><b>{_html_escape(str(row.get('event_name', 'Event'))[:55])}</b></td><td>{_html_escape(row.get('category', '—'))}</td><td>{_html_escape(row.get('venue', row.get('city', '—')))}</td></tr>")
        return f"<p><b>Upcoming demand-driving events</b> — open {_tab_link('calendar', 'Calendar tab')}.</p><table><thead><tr><th>Date</th><th>Event</th><th>Category</th><th>Venue</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"

    if any(term in ql for term in ['compare listings', 'comparable', 'comps', 'similar listings']):
        payload = _default_payload(ql)
        comp = get_comparable_summary(payload, top_k=8)
        records = pd.DataFrame(comp.get('records', []))
        table = ''
        if len(records) > 0:
            rows = []
            for _, r in records.iterrows():
                rows.append(f"<tr><td><b>{_html_escape(str(r.get('name','Comparable'))[:44])}</b></td><td>{_html_escape(r.get('neighbourhood_cleansed','—'))}</td><td>{_currency(r.get('baseline_price',0))}</td><td>{_currency(r.get('recommended_price',0))}</td><td>{_html_escape(r.get('cluster_label','—'))}</td></tr>")
            table = f"<table><thead><tr><th>Comparable</th><th>Neighbourhood</th><th>Current</th><th>Recommended</th><th>Cluster</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        return (
            f"<p><b>Comparable pricing set</b></p><ul>"
            f"<li>{comp.get('count',0)} comparable listings average <b>{_currency(comp.get('avg_price',0))}</b>/night.</li>"
            f"<li>Average baseline is <b>{_currency(comp.get('avg_baseline',0))}</b>; average uplift is <b>{_currency(comp.get('avg_uplift',0))}</b>.</li>"
            f"<li>Closest cluster: <b>{_html_escape(comp.get('cluster_label','market cluster'))}</b>.</li></ul>" + table
        )

    if any(term in ql for term in ['cluster', 'segment', 'market group', 'market clusters']):
        return "<p><b>Market cluster breakdown</b></p><pre style='white-space:pre-wrap'>" + _html_escape(_execute_tool('get_cluster_breakdown', {}, fm)) + "</pre>"

    if any(term in ql for term in [
        'portfolio', 'sarah', 'recommendations', 'alerts',
        'revenue forecast', 'forecast', 'event insights',
        'event-driven', 'event driven', 'occupancy alert',
        'pricing alert', 'personalized recommendations',
        'portfolio performance', 'portfolio drivers', 'how Sarah’s recommendations work',
        "Sarah\'s portfolio", 'Sarah portfolio actions', 'portfolio insights'
    ]):
        return """
        <p><b>Sarah’s Portfolio Insights</b></p>
        <ul>
          <li><b>Alerts &amp; Recommendations:</b> pricing alerts, low occupancy alerts, and event-triggered actions.</li>
          <li><b>Personalized Recommendations:</b> AI-driven suggestions for each property, including Adjust Price and Promote actions.</li>
          <li><b>Revenue Forecast:</b> forecast vs actual revenue plus projected revenue upside.</li>
          <li><b>Event-Driven Insights:</b> concerts, sports, and conferences affecting demand, pricing, and views.</li>
        </ul>
        <p>For Sarah’s portfolio right now, prioritize Adams Morgan pricing, Capitol Hill occupancy, and Navy Yard event pricing.</p>
        """

    weather_terms = [

        'weather', 'forecast', 'temperature', 'temp', 'rain',
        'raining', 'snow', 'wind', 'humid', 'humidity',
        'dc weather', 'washington weather'
    ]

    weather_impact_terms = [
        'how does weather impact airbnb pricing',
        'how does weather affect airbnb pricing',
        'weather impact on pricing',
        'weather affect pricing',
        'how weather impacts pricing',
        'how weather affects pricing',
        'does weather affect pricing',
        'weather and pricing'
    ]

    if any(term in ql for term in weather_impact_terms):
        forecast = _get_cached_weather('Washington')
        if not forecast:
            return "<p>Sorry, weather data is unavailable right now, so I can’t estimate pricing impact.</p>"

        current_desc = str(forecast.get('current_description', ''))
        current_temp = forecast.get('current_temp_c', '—')
        days = forecast.get('days', [])
        next_wet_day = None
        for d in days[:7]:
            desc = str(d.get('description', '')).lower()
            rain = _safe_float(d.get('precipitation_mm', 0), 0.0)
            if any(term in desc for term in ['rain', 'drizzle', 'snow', 'storm', 'thunder']) or rain >= 2.0:
                next_wet_day = d
                break

        html = [
            "<p><b>How weather impacts Airbnb pricing</b></p>",
            f"<p>Current conditions in Washington DC: {_html_escape(current_desc)}, {current_temp}°C.</p>",
            "<ul>",
            "<li><b>Good weather</b> usually supports stronger last-minute demand, so hosts can often hold or slightly raise prices.</li>",
            "<li><b>Rain, cold, or storms</b> can soften casual travel demand, so non-event stays may need a small discount to stay competitive.</li>",
            "<li><b>Major events often outweigh weather effects</b>, so hosts can still maintain or increase prices even during poor conditions.</li>"
        ]

        if next_wet_day:
            html.append(
                f"<li><b>Upcoming signal:</b> {_html_escape(next_wet_day.get('date', ''))} looks wet ({_html_escape(next_wet_day.get('description', ''))}), so event-free listings may need a small discount while event-adjacent listings can usually hold firmer pricing.</li>"
            )
        else:
            html.append(
                "<li><b>Upcoming signal:</b> The current 7-day forecast is not showing a major rain/storm risk, which supports steadier pricing.</li>"
            )

        html.extend([
            "</ul>",
            "<p><b>Insight:</b> Weather influences short-term demand, but major events often outweigh weather effects — allowing hosts to maintain or increase prices even during poor conditions.</p>"
        ])
        return ''.join(html)

    if any(term in ql for term in weather_terms):
        forecast = _get_cached_weather('Washington')
        if not forecast:
            return "<p>Sorry, the Washington DC weather forecast is unavailable right now.</p>"

        days = forecast.get('days', [])
        current_desc = forecast.get('current_description', 'Current conditions unavailable')
        current_temp = forecast.get('current_temp_c', '—')

        html = [
            "<p><b>Washington DC Weather</b></p>",
            f"<p>Current: {_html_escape(current_desc)}, {current_temp}°C.</p>"
        ]

        if days:
            html.append("<ul>")
            for d in days[:7]:
                date = d.get('date', '—')
                desc = d.get('description', '—')
                tmax = d.get('temp_max_c', '—')
                tmin = d.get('temp_min_c', '—')
                rain = d.get('precipitation_mm', 0)
                html.append(
                    f"<li><b>{_html_escape(date)}</b>: {_html_escape(desc)}, {tmax}°C / {tmin}°C, {rain} mm rain</li>"
                )
            html.append("</ul>")

        return ''.join(html)

    plural_listing_raise_terms = [
        'which listings should raise', 'which listings need a raise', 'which listings need to raise',
        'which properties should raise', 'top listings to raise', 'top raise listings',
        'listings that should raise', 'underpriced listings', 'which homes should raise',
        'which listings should raise prices', 'which properties should raise prices',
        'which homes should raise prices', 'which apartments should raise prices'
    ]

    def _is_plural_raise_query(q_lower):
        if any(term in q_lower for term in plural_listing_raise_terms):
            return True
        has_raise = ('raise price' in q_lower) or ('raise prices' in q_lower) or ('underpriced' in q_lower)
        has_plural_subject = any(term in q_lower for term in [
            'which listings', 'which homes', 'which properties', 'which apartments',
            'what listings', 'show listings', 'show me listings', 'listings in ', 'homes in ', 'properties in '
        ])
        return has_raise and has_plural_subject

    singular_pricing_terms = [
        'optimal price', 'recommended price', 'recommend price', 'what should i charge',
        'what should i price', 'how much should i charge', 'price this listing',
        'price my listing', 'suggest a price', 'pricing recommendation'
    ]

    if _is_plural_raise_query(ql):
        work = fm.copy()
        work['price_uplift'] = work['recommended_price'] - work['baseline_price']
        work = _filter_ranked_listings_from_query(work, ql)
        top = work[work['action'] == 'RAISE'].sort_values(['price_uplift', 'recommended_price'], ascending=[False, False]).head(10)
        if len(top) == 0:
            return "<p>No matching listings are currently flagged strongly enough to recommend a price increase.</p>"
        rows = []
        for _, row in top.iterrows():
            confidence = 'High' if row.get('distance_to_event_km', 99) < 10 else ('Medium' if row.get('distance_to_event_km', 99) < 20 else 'Low')
            rows.append(
                f"<tr><td><b>{_html_escape(str(row.get('name', 'Listing'))[:42])}</b></td>"
                f"<td>{_html_escape(row.get('neighbourhood_cleansed', 'Unknown area'))}</td>"
                f"<td>{_currency(row.get('baseline_price', 0))}</td>"
                f"<td>{_currency(row.get('recommended_price', 0))}</td>"
                f"<td>+{_currency(row.get('price_uplift', 0)).replace('$', '$')}</td>"
                f"<td>{confidence}</td></tr>"
            )
        filter_bits = []
        rt = _room_type_from_query(ql)
        bd = _extract_int_before_terms(ql, ['bedroom', 'bedrooms', 'br'])
        nf = _extract_neighbourhood_fragment(ql) or _find_neighbourhood(ql)
        if bd is not None:
            filter_bits.append(f"{bd}-bedroom")
        if rt:
            filter_bits.append(rt.replace('Entire home/apt', 'entire homes').lower())
        if nf:
            filter_bits.append(f"in {_html_escape(nf.title())}")
        subtitle = "Filtered top Washington DC listings currently flagged <b>RAISE</b>, ranked by estimated nightly uplift." if filter_bits else "These are the top Washington DC listings currently flagged <b>RAISE</b>, ranked by estimated nightly uplift."
        title = "Listings that should raise prices"
        if filter_bits:
            title += f" — {' '.join(filter_bits)}"
        return (
            f"<p><b>{title}</b></p>"
            f"<p>{subtitle}</p>"
            f"<table><thead><tr><th>Listing</th><th>Neighbourhood</th><th>Current</th><th>Recommended</th><th>Uplift</th><th>Confidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )


    if any(term in ql for term in [
        'which neighborhoods have the highest pricing upside',
        'which neighbourhoods have the highest pricing upside',
        'highest pricing upside',
        'pricing upside by neighborhood',
        'pricing upside by neighbourhood',
        'which neighborhoods have the most upside',
        'which neighbourhoods have the most upside',
        'best neighborhoods for pricing upside',
        'best neighbourhoods for pricing upside',
        'neighborhoods with highest pricing upside',
        'neighbourhoods with highest pricing upside'
    ]):
        if 'neighbourhood_cleansed' not in fm.columns:
            return "<p>Neighbourhood data is not available.</p>"
        work = fm.copy()
        work['price_uplift'] = work['recommended_price'] - work['baseline_price']
        hood = (work.groupby('neighbourhood_cleansed')
                .agg(avg_recommended=('recommended_price', 'mean'),
                     avg_uplift=('price_uplift', 'mean'),
                     listings=('id', 'count'),
                     raise_pct=('action', lambda x: round((x == 'RAISE').mean() * 100, 1)))
                .reset_index()
                .sort_values(['avg_uplift', 'raise_pct', 'listings'], ascending=[False, False, False])
                .head(8))
        items = []
        for _, row in hood.iterrows():
            items.append(
                f"<b>{_html_escape(row['neighbourhood_cleansed'])}</b> — avg recommended {_currency(row['avg_recommended'])}, avg uplift +{_currency(row['avg_uplift']).replace('$', '$')}, {int(row['listings'])} listings, {row['raise_pct']}% flagged raise"
            )
        return _bullets("Neighbourhoods with the highest pricing upside", items)

    pricing_terms = ['price', 'pricing'] + singular_pricing_terms
    if any(term in ql for term in pricing_terms):
        payload = _default_payload(ql)
        result = _build_ai_pricing_result(payload, fm)
        confidence_score, confidence_label = _confidence_label(result, payload)
        baseline = float(payload.get('baseline_price', result.get('city_avg', 0)))
        rec_price = float(result.get('price', 0))
        delta = rec_price - baseline
        pct = ((delta / baseline) * 100.0) if baseline else 0.0
        comp = result.get('comp', {})
        cluster = result.get('cluster', {})
        hood_text = payload.get('neighbourhood_cleansed') or 'Washington DC'
        room_text = payload.get('room_type', 'listing').replace('Entire home/apt', 'Entire home')
        return (
            f"<p><b>Recommended pricing action</b></p>"
            f"<ul>"
            f"<li>Context understood: <b>{_html_escape(hood_text)}</b>, <b>{_html_escape(room_text)}</b>, {int(payload.get('bedrooms', 1))} bedroom(s), {int(payload.get('accommodates', 2))} guest capacity, event type <b>{_html_escape(payload.get('event_type', 'Sports'))}</b>.</li>"
            f"<li>Recommended action: <b>{_html_escape(result.get('action', 'HOLD'))}</b>.</li>"
            f"<li>Recommended nightly price: <b>{_currency(rec_price)}</b> vs current baseline <b>{_currency(baseline)}</b> ({'+' if delta >= 0 else ''}{_currency(delta)}, {'+' if pct >= 0 else ''}{_percent(pct)}).</li>"
            f"<li>Expected revenue impact: about <b>{_currency(result.get('monthly_uplift', 0))}</b> monthly and <b>{_currency(result.get('annual_uplift', 0))}</b> annual upside.</li>"
            f"<li>Confidence: <b>{confidence_label} ({confidence_score}%)</b>.</li>"
            f"</ul>"
            f"<p><b>Why this recommendation</b></p>"
            f"<ul>"
            f"<li>Historical/market baseline in this segment is around <b>{_currency(result.get('city_avg', 0))}</b>.</li>"
            f"<li>Comparable listings ({comp.get('count', 0)}) average <b>{_currency(comp.get('avg_price', 0))}</b> with average uplift <b>{_currency(comp.get('avg_uplift', 0))}</b>.</li>"
            f"<li>Closest market cluster: <b>{_html_escape(cluster.get('cluster_label', 'market cluster'))}</b> averaging <b>{_currency(cluster.get('avg_price', 0))}</b>.</li>"
            f"<li>Market demand signal: <b>{_html_escape(result.get('demand', 'LOW'))}</b>. Positioning: <b>{_html_escape(result.get('market_position', 'In line with comps'))}</b>.</li>"
            f"<li>Decision summary: {_html_escape(result.get('explanation', 'Model-based pricing recommendation.'))}.</li>"
            f"</ul>"
        )

    if any(term in ql for term in ['revenue opportunity', 'revenue this week', 'weekly revenue', 'missed revenue', 'opportunity this week']):
        raise_df = fm[fm['action'] == 'RAISE'].copy()
        if len(raise_df) == 0:
            return "<p>No immediate revenue opportunity was found from underpriced listings.</p>"
        avg_uplift = float((raise_df['recommended_price'] - raise_df['baseline_price']).mean())
        weekly_est = avg_uplift * 7 * len(raise_df)
        annual_est = avg_uplift * 240 * len(raise_df)
        return (
            f"<p><b>Revenue opportunity</b></p>"
            f"<ul>"
            f"<li>{len(raise_df):,} listings are currently flagged to <b>RAISE</b>.</li>"
            f"<li>Average nightly uplift: <b>{_currency(avg_uplift)}</b>.</li>"
            f"<li>Estimated 7-day upside if those prices are applied: <b>{_currency(weekly_est)}</b>.</li>"
            f"<li>Estimated annual upside at 240 booked nights/year: <b>{_currency(annual_est)}</b>.</li>"
            f"<li>Confidence: <b>Medium</b>, because this is based on current recommended-price gaps rather than realized booking conversion.</li>"
            f"</ul>"
        )

    if any(term in ql for term in ['best neighbourhood', 'best neighborhood', 'invest', 'investment', 'neighbourhoods to invest', 'neighborhoods to invest']):
        if 'neighbourhood_cleansed' not in fm.columns:
            return "<p>Neighbourhood data is not available.</p>"
        work = fm.copy()
        work['price_uplift'] = work['recommended_price'] - work['baseline_price']
        hood = (work.groupby('neighbourhood_cleansed')
                .agg(avg_recommended=('recommended_price', 'mean'),
                     avg_uplift=('price_uplift', 'mean'),
                     listings=('id', 'count'),
                     raise_pct=('action', lambda x: round((x == 'RAISE').mean() * 100, 1)))
                .reset_index()
                .sort_values(['avg_uplift', 'raise_pct', 'listings'], ascending=[False, False, False])
                .head(8))
        items = []
        for _, row in hood.iterrows():
            items.append(
                f"<b>{_html_escape(row['neighbourhood_cleansed'])}</b> — avg recommended {_currency(row['avg_recommended'])}, avg uplift +{_currency(row['avg_uplift']).replace('$', '$')}, {int(row['listings'])} listings, {row['raise_pct']}% flagged raise"
            )
        return _bullets("Best neighbourhoods to watch for pricing upside", items)

    if any(term in ql for term in ['summary', 'overview', 'stats', 'market trends', 'market trend']):
        total = len(fm)
        raise_count = int((fm['action'] == 'RAISE').sum())
        discount_count = int((fm['action'] == 'DISCOUNT').sum())
        hold_count = int((fm['action'] == 'HOLD').sum())
        avg_base = float(fm['baseline_price'].mean())
        avg_rec = float(fm['recommended_price'].mean())
        return (
            f"<p><b>Washington DC pricing summary</b></p>"
            f"<ul>"
            f"<li>{total:,} listings analyzed.</li>"
            f"<li><b>RAISE:</b> {raise_count:,} &nbsp; <b>DISCOUNT:</b> {discount_count:,} &nbsp; <b>HOLD:</b> {hold_count:,}</li>"
            f"<li>Average baseline price: <b>{_currency(avg_base)}</b>.</li>"
            f"<li>Average recommended price: <b>{_currency(avg_rec)}</b>.</li>"
            f"<li>Average uplift: <b>{_currency(avg_rec - avg_base)}</b>.</li>"
            f"</ul>"
        )

    context_parts = [_execute_tool('get_summary_stats', {}, fm)]
    if any(term in ql for term in ['event', 'concert', 'festival', 'sports', 'conference']):
        context_parts.append(_execute_tool('get_event_impact', {}, fm))
    if any(term in ql for term in ['neighbourhood', 'neighborhood', 'area']):
        context_parts.append(_execute_tool('get_neighbourhood_breakdown', {'limit': 8, 'sort_by': 'avg_uplift'}, fm))
    if any(term in ql for term in ['room', 'bedroom', 'private room', 'entire home']):
        context_parts.append(_execute_tool('get_room_type_breakdown', {}, fm))
        context_parts.append(_execute_tool('get_bedroom_breakdown', {}, fm))
    rag_hits = _retrieve(q, top_k=5)
    if rag_hits:
        snippets = []
        for hit in rag_hits[:3]:
            snippets.append(f"[{hit.get('source', 'doc')} #{hit.get('listing_id', '')}] {str(hit.get('text', ''))[:260]}")
        context_parts.append("Semantic search context:\n" + "\n\n".join(snippets))

    context_blob = "\n\n".join([part for part in context_parts if part])

    system_prompt = (
        "You are an expert Airbnb pricing advisor for Washington DC. "
        "Use the provided app data context to answer the user's question. "
        "Do not say the app data is missing when context is provided. "
        "Where possible, mention market comparisons, demand signals, and practical next steps. "
        "Answer clearly and briefly in HTML using only <p>, <b>, <ul>, <li>, <table>, <thead>, <tbody>, <tr>, <th>, <td>."
    )

    conversation = []
    for turn in (history or []):
        if turn.get('q'):
            conversation.append(f"User: {turn['q']}")
        if turn.get('a'):
            conversation.append(f"Assistant: {turn['a']}")
    conversation.append(f"Data context:\n{context_blob}")
    conversation.append(f"User: {q}")
    user_input = "\n\n".join(conversation)

    try:
        resp = _client.responses.create(
            model=AZURE_MODEL,
            instructions=system_prompt,
            input=user_input,
        )
        return resp.output_text or "<p>No response generated.</p>"
    except Exception:
        safe_context = _html_escape(context_blob[:2500])
        return f"<p><b>Grounded app-data summary</b></p><p>{safe_context}</p>"


_WEATHER_CACHE = {}
_WEATHER_TTL_SECONDS = 1800


def _get_cached_weather(selected_city='Washington'):
    cache_entry = _WEATHER_CACHE.get(selected_city)
    now = time.time()
    if cache_entry and (now - cache_entry['ts']) < _WEATHER_TTL_SECONDS:
        return cache_entry['value']
    value = get_forecast_for_city(selected_city)
    _WEATHER_CACHE[selected_city] = {'value': value, 'ts': now}
    return value


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _format_listing_card(row):
    room_type = row.get('room_type', '—')
    result = {
        'name': str(row.get('name', '—'))[:35],
        'room_type': str(room_type).replace('Entire home/apt', 'Entire home') if not pd.isna(room_type) else '—',
        'bedrooms': _safe_int(row.get('bedrooms', 1), 1),
        'rating': f"{row.get('review_scores_rating', 0):.1f}" if not pd.isna(row.get('review_scores_rating', 0)) else '—',
        'current': f"{row.get('baseline_price', 0):.0f}",
        'recommended': f"{row.get('recommended_price', 0):.0f}",
        'action': row.get('action', 'HOLD'),
    }
    if 'cluster_label' in row:
        result['cluster_label'] = row.get('cluster_label', '')
    return result


@lru_cache(maxsize=8)
def _build_static_city_assets(selected_city='Washington'):
    fm, fe = get_city_data(selected_city)

    weekly_chart    = make_weekly_chart(fm)
    donut_chart     = make_donut_chart(fm)
    rev_event_chart = make_revenue_by_event_chart(fm)
    revenue_event_summary = make_revenue_event_summary(fm)
    calendar_cards  = make_calendar_cards(fe)
    make_map(fm, fe, selected_city)

    raise_df = fm[fm['action'] == 'RAISE'].copy()
    discount_df = fm[fm['action'] == 'DISCOUNT']
    hold_df = fm[fm['action'] == 'HOLD']

    raise_count = int((fm['action'] == 'RAISE').sum())
    discount_count = int((fm['action'] == 'DISCOUNT').sum())
    hold_count = int((fm['action'] == 'HOLD').sum())
    total = int(len(fm))

    avg_price_val = fm['recommended_price'].mean() if total > 0 else 0
    avg_uplift_val = (raise_df['recommended_price'] - raise_df['baseline_price']).mean() if len(raise_df) > 0 else 0
    annual_opportunity_val = avg_uplift_val * 240 * len(raise_df) if len(raise_df) > 0 else 0
    raise_avg_val = raise_df['recommended_price'].mean() if len(raise_df) > 0 else 0
    raise_baseline_val = raise_df['baseline_price'].mean() if len(raise_df) > 0 else 0

    revenue_table = []
    if len(raise_df) > 0 and 'neighbourhood_cleansed' in raise_df.columns:
        rt = raise_df.groupby('neighbourhood_cleansed').agg(
            count=('baseline_price','count'),
            baseline=('baseline_price','mean'),
            recommended=('recommended_price','mean')).reset_index()
        rt['uplift'] = rt['recommended'] - rt['baseline']
        rt['annual'] = (rt['uplift'] * 240 * rt['count']).astype(int)
        rt = rt.sort_values('annual', ascending=False).head(8)
        revenue_table = [{'neighbourhood': r['neighbourhood_cleansed'], 'count': int(r['count']),
                          'baseline': f"{r['baseline']:.0f}", 'recommended': f"{r['recommended']:.0f}",
                          'uplift': f"{r['uplift']:.0f}", 'annual': f"{r['annual']:,}"} for _, r in rt.iterrows()]

    listings_preview = []
    if len(fm) > 0:
        cols_needed = ['name','room_type','bedrooms','review_scores_rating','baseline_price','recommended_price','action','cluster_label']
        cols_avail  = [c for c in cols_needed if c in fm.columns]
        lp = fm[cols_avail].dropna(subset=['recommended_price']).sort_values('recommended_price', ascending=False).head(50)
        listings_preview = [_format_listing_card(row) for row in lp.to_dict('records')]

    occ_rate = int(min(95, 65 + (len(raise_df) / max(len(fm), 1)) * 25)) if len(fm) > 0 else 68
    comp_price = int(fm['recommended_price'].mean()) if len(fm) > 0 else 0
    missed_rev = int((discount_df['baseline_price'] - discount_df['recommended_price']).clip(lower=0).sum() * 30) if len(discount_df) > 0 else 0
    missed_rev_fmt = f"{missed_rev:,}"
    acceptance_rate = 74
    market_new = 52

    action_queue = []
    if len(raise_df) > 0:
        aq = raise_df.copy()
        aq['uplift'] = aq['recommended_price'] - aq['baseline_price']
        aq = aq.nlargest(8, 'uplift')
        for _, row in aq.iterrows():
            action_queue.append({
                'name':          str(row.get('name', '—'))[:30],
                'neighbourhood': str(row.get('neighbourhood_cleansed', '—'))[:22],
                'current':       f"{row.get('baseline_price', 0):.0f}",
                'recommended':   f"{row.get('recommended_price', 0):.0f}",
                'uplift':        f"{row.get('uplift', 0):.0f}",
                'event':         str(row.get('nearest_event', '—'))[:28],
            })

    hood_heat = []
    if 'neighbourhood_cleansed' in fm.columns and len(fm) > 0:
        nh = fm.groupby('neighbourhood_cleansed').agg(
            avg_price=('recommended_price', 'mean'),
            count=('recommended_price', 'count'),
            raise_pct=('action', lambda x: (x == 'RAISE').mean() * 100)
        ).reset_index().sort_values('avg_price', ascending=False).head(10)
        max_p = nh['avg_price'].max()
        for _, row in nh.iterrows():
            hood_heat.append({
                'name':      row['neighbourhood_cleansed'],
                'label':     str(row['neighbourhood_cleansed'])[:24],
                'avg_price': f"{row['avg_price']:.0f}",
                'count':     int(row['count']),
                'raise_pct': f"{row['raise_pct']:.0f}",
                'intensity': round(row['avg_price'] / max_p, 2) if max_p else 0,
            })

    hood_options = []
    if 'neighbourhood_cleansed' in fm.columns and len(fm) > 0:
        hood_options = sorted([str(x) for x in fm['neighbourhood_cleansed'].dropna().unique() if str(x).strip()])

    lead_time_chart = ''
    elasticity_chart   = make_price_elasticity_chart(fm)
    review_price_chart = make_review_price_chart(fm)
    seasonal_chart     = make_seasonal_demand_chart(fm)
    nb_bar_chart = ''
    prop_type_chart = ''
    dist_buckets_chart = ''
    review_bands_chart = ''
    feat_imp_chart     = make_feature_importance_chart(PORTFOLIO_INSIGHTS)

    upcoming_events = []

    alerts_list = []
    if len(raise_df) > 0:
        al = raise_df.copy()
        al['uplift'] = al['recommended_price'] - al['baseline_price']
        al = al.sort_values('uplift', ascending=False).head(100)
        for _, row in al.iterrows():
            dist = row.get('distance_to_event_km', 99)
            alerts_list.append({
                'name':          str(row.get('name', '—'))[:35],
                'neighbourhood': str(row.get('neighbourhood_cleansed', '—'))[:25],
                'city':          str(row.get('city', '')),
                'current':       f"{row.get('baseline_price', 0):.0f}",
                'recommended':   f"{row.get('recommended_price', 0):.0f}",
                'uplift':        f"{row.get('uplift', 0):.0f}",
                'event':         str(row.get('nearest_event', '—'))[:40],
                'dist_km':       f"{dist:.1f}" if not pd.isna(dist) else '—',
                'demand':        get_demand(dist),
            })

    hood_listings_json = {}
    if 'neighbourhood_cleansed' in fm.columns and len(fm) > 0:
        for hood_name, grp in fm.groupby('neighbourhood_cleansed'):
            grp_top = grp.nlargest(10, 'recommended_price')
            hood_listings_json[str(hood_name)] = {
                'listings': [_format_listing_card(r) for r in grp_top.to_dict('records')],
                'raise':     int((grp['action'] == 'RAISE').sum()),
                'discount':  int((grp['action'] == 'DISCOUNT').sum()),
                'hold':      int((grp['action'] == 'HOLD').sum()),
                'avg_price': f"{grp['recommended_price'].mean():.0f}",
                'count':     len(grp),
            }

    monthly_opportunity = int(avg_uplift_val * 30 * len(raise_df)) if len(raise_df) > 0 else 0
    monthly_opportunity_fmt = f"${monthly_opportunity:,}"
    room_type_series = fm['room_type'].fillna('Other') if 'room_type' in fm.columns else pd.Series(['Other'] * max(total, 1))
    room_mix = {
        'Entire home': int((room_type_series == 'Entire home/apt').sum()),
        'Private room': int((room_type_series == 'Private room').sum()),
        'Shared room': int((room_type_series == 'Shared room').sum()),
        'Other': max(int(total) - int((room_type_series == 'Entire home/apt').sum()) - int((room_type_series == 'Private room').sum()) - int((room_type_series == 'Shared room').sum()), 0),
    }
    listings_donut = make_kpi_donut(room_mix, ['#3b82f6', '#60a5fa', '#6ee7b7', '#a78bfa'])
    revenue_trend = make_stat_sparkline([max(monthly_opportunity * 0.42, 0), max(monthly_opportunity * 0.58, 0), max(monthly_opportunity * 0.74, 0), max(monthly_opportunity * 0.88, 0), monthly_opportunity], '#8b5cf6', '#ede9fe')
    avg_rec_increase_pct = int((raise_avg_val - raise_baseline_val) / raise_baseline_val * 100) if raise_baseline_val > 0 else 0
    occupancy_ring = make_stat_ring(occ_rate, '#22c55e', '#dcfce7')
    top_hood_name = 'N/A'
    top_hood_uplift = 0
    if len(raise_df) > 0 and 'neighbourhood_cleansed' in raise_df.columns:
        nb_tmp = raise_df.groupby('neighbourhood_cleansed').agg(
            avg_rec2=('recommended_price', 'mean'),
            avg_base2=('baseline_price', 'mean')
        ).reset_index()
        nb_tmp['uplift2'] = nb_tmp['avg_rec2'] - nb_tmp['avg_base2']
        best = nb_tmp.loc[nb_tmp['uplift2'].idxmax()]
        top_hood_name = str(best['neighbourhood_cleansed'])[:22]
        top_hood_uplift = int(best['uplift2'])

    hood_medians = fm.groupby('neighbourhood_cleansed')['baseline_price'].median().to_dict() if 'neighbourhood_cleansed' in fm.columns else {}
    rec_table = []
    if len(raise_df) > 0:
        rt2 = raise_df.copy()
        rt2['nightly_gain'] = rt2['recommended_price'] - rt2['baseline_price']
        rt2 = rt2.nlargest(50, 'nightly_gain')
        for _, row in rt2.iterrows():
            dist = row.get('distance_to_event_km', 99)
            event = str(row.get('nearest_event', ''))[:35]
            demand = get_demand(dist)
            if not pd.isna(dist) and dist < 5:
                reason = f"Near {event[:25]} · {demand} demand"
            elif event and event != 'nan':
                reason = f"{event[:30]} · {demand} demand"
            else:
                reason = f"{demand} event demand"
            rating = row.get('review_scores_rating', 4.5)
            if pd.isna(rating):
                rating = 4.5
            base_conf = {'VERY HIGH': 92, 'HIGH': 85, 'MEDIUM': 76, 'LOW': 65}.get(demand, 75)
            conf = min(97, int(base_conf + (rating - 4.5) * 10))
            hood = row.get('neighbourhood_cleansed', '')
            hood_med = hood_medians.get(hood)
            rec_price = row.get('recommended_price', 0)
            vs_median = ''
            if hood_med and hood_med > 0:
                pct = (rec_price - hood_med) / hood_med * 100
                vs_median = f"+{pct:.0f}% vs median" if pct >= 0 else f"{pct:.0f}% vs median"
            rec_table.append({
                'name':        str(row.get('name', '-'))[:32],
                'current':     f"{row.get('baseline_price', 0):.0f}",
                'recommended': f"{rec_price:.0f}",
                'reason':      reason,
                'confidence':  conf,
                'gain':        f"{row.get('nightly_gain', 0):.0f}",
                'vs_median':   vs_median,
            })

    hold_count_val = len(hold_df)
    hold_avg_price = int(hold_df['recommended_price'].mean()) if len(hold_df) > 0 else 0
    listings_preview_json = json.dumps(listings_preview)

    return {
        'weekly_chart': weekly_chart,
        'donut_chart': donut_chart,
        'rev_event_chart': rev_event_chart,
        'revenue_event_summary': revenue_event_summary,
        'calendar_cards': calendar_cards,
        'upcoming_events': upcoming_events,
        'raise_count': raise_count,
        'discount_count': discount_count,
        'hold_count': hold_count,
        'total': total,
        'avg_price': f"{avg_price_val:.0f}" if total > 0 else "0",
        'avg_uplift': f"{avg_uplift_val:.0f}",
        'annual_opportunity': f"{annual_opportunity_val:,.0f}",
        'raise_avg': f"{raise_avg_val:.0f}",
        'raise_baseline': f"{raise_baseline_val:.0f}",
        'revenue_table': revenue_table,
        'listings_preview': listings_preview,
        'listings_preview_json': listings_preview_json,
        'occ_rate': occ_rate,
        'comp_price': comp_price,
        'missed_rev_fmt': missed_rev_fmt,
        'acceptance_rate': acceptance_rate,
        'market_new': market_new,
        'action_queue': action_queue,
        'hood_heat': hood_heat,
        'hood_options': hood_options,
        'lead_time_chart': lead_time_chart,
        'elasticity_chart': elasticity_chart,
        'review_price_chart': review_price_chart,
        'seasonal_chart': seasonal_chart,
        'pricing_charts': pricing_charts_b64,
        'alerts_list': alerts_list,
        'hood_listings_json': json.dumps(hood_listings_json),
        'hold_count_val': hold_count_val,
        'hold_avg_price': hold_avg_price,
        'nb_bar_chart': nb_bar_chart,
        'prop_type_chart': prop_type_chart,
        'dist_buckets_chart': dist_buckets_chart,
        'review_bands_chart': review_bands_chart,
        'feat_imp_chart': feat_imp_chart,
        'model_metrics': PORTFOLIO_INSIGHTS,
        'monthly_opportunity_fmt': monthly_opportunity_fmt,
        'listings_donut': listings_donut,
        'revenue_trend': revenue_trend,
        'occupancy_ring': occupancy_ring,
        'avg_rec_increase_pct': avg_rec_increase_pct,
        'top_hood_name': top_hood_name,
        'top_hood_uplift': top_hood_uplift,
        'rec_table': rec_table,
    }


try:
    _build_static_city_assets('Washington')
except Exception as _warm_err:
    print(f"Static asset warmup skipped ({_warm_err}).")




def build_judge_portfolio_mockup():
    """Small personalized demo portfolio for judge-facing dashboard mockup."""
    return {
        "user": "Sarah Chen",
        "role": "Property manager",
        "location": "Washington, DC",
        "avg_night": 189,
        "properties": [
            {"name": "Navy Yard Studio", "area": "Navy Yard", "units": 1, "now": 142, "next": 158, "action": "RAISE", "signal": "Cherry Blossom weekend demand spiking"},
            {"name": "Georgetown 2BR", "area": "Georgetown", "units": 1, "now": 188, "next": 184, "action": "HOLD", "signal": "Nationals game demand up 28%"},
            {"name": "Capitol Hill Studio", "area": "Georgetown", "units": 2, "now": 310, "next": 336, "action": "RAISE", "signal": "6 nearby comps already raised"},
            {"name": "Adams Morgan Flat", "area": "Adams Morgan", "units": 1, "now": 165, "next": 152, "action": "DISCOUNT", "signal": "4-day gap; small discount helps"},
            {"name": "Dupont Garden Apt", "area": "Navy Yard", "units": 1, "now": 221, "next": 246, "action": "RAISE", "signal": "Already near top comps"},
            {"name": "Logan Circle Condo", "area": "Logan Circle", "units": 1, "now": 199, "next": 207, "action": "HOLD", "signal": "Healthy occupancy trend"},
            {"name": "Shaw Rowhouse", "area": "Shaw", "units": 2, "now": 260, "next": 278, "action": "RAISE", "signal": "Comparable gap detected"},
        ],
        "timeline": [
            {"time": "9:00 AM", "label": "Morning scan", "title": "Raise Navy Yard Studio and Georgetown 2BR", "why": "Cherry Blossom weekend and a Nationals game are lifting demand near your listings. Comparable units have already raised prices.", "impact": "+$412 projected weekly revenue"},
            {"time": "1:30 PM", "label": "Booking pace changed", "title": "Discount Adams Morgan Flat for 48 hours", "why": "Two similar private-room units just dropped prices. You have a 4-day gap, so a small discount can help fill it.", "impact": "+8% expected occupancy lift"},
            {"time": "6:00 PM", "label": "Evening guardrail", "title": "Hold Dupont Garden Apt", "why": "It is already priced near the top of its comparable set, so PriceWise avoids a risky increase.", "impact": "Protects rating and booking probability"}
        ]
    }

def render_page(explore_results=None, explore_count=0, hood='', min_price=0, action_filter='',
                active_tab='dashboard', selected_city='Washington', sim_result=None,
                bedrooms_filter='', room_type_filter=''):
    _, fe = get_city_data(selected_city)
    static_assets = dict(_build_static_city_assets(selected_city))

    try:
        today = datetime.date.today()
        fe_dates = pd.to_datetime(fe['date'], errors='coerce')
        upcoming_hd_count = int(((fe_dates.dt.date >= today) & (fe_dates.dt.date <= today + datetime.timedelta(days=60))).sum())
    except Exception:
        upcoming_hd_count = len(fe) if len(fe) < 20 else 12

    static_assets.update({
        'history': session.get('history', []),
        'chat_welcome_message': CHAT_WELCOME_MESSAGE,
        'explore_results': explore_results,
        'explore_count': explore_count,
        'hood': hood,
        'min_price': min_price,
        'action_filter': action_filter,
        'bedrooms_filter': bedrooms_filter,
        'room_type_filter': room_type_filter,
        'active_tab': active_tab,
        'selected_city': selected_city,
        'sim_result': sim_result,
        'hoods_json': json.dumps(neighbourhoods),
        'dc_weather': _get_cached_weather(selected_city),
        'upcoming_hd_count': upcoming_hd_count,
        'portfolio_mockup': build_judge_portfolio_mockup(),
    })

    return render_template_string(HTML, **static_assets)


# HTML TEMPLATE  

# HTML TEMPLATE 

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    {{ SIDEBAR_STYLE | safe }}
<title>PriceWise</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#FAF9F7; --bg2:#F4F2EE; --card:rgba(255,255,255,0.9);
  --border:rgba(0,0,0,0.06); --border2:rgba(0,0,0,0.09);
  --text:#1D1D1F; --text2:#6E6E73; --text3:#A1A1A6;
  --gold:#93c5fd; --gold-d:#60a5fa; --gold-l:#dbeafe;
  --blue:#BFD7EA; --blue-d:#9ec4e0; --blue-l:#ebf4fb;
  --warm:#D8CFC4; --sage:#DDE6D5; --sage-d:#c5d9b8;
  --green:#22c55e; --orange:#f97316; --red:#ef4444;
  --r:20px; --r-lg:28px; --r-sm:12px;
  --sh:0 2px 12px rgba(0,0,0,0.05),0 1px 3px rgba(0,0,0,0.04);
  --sh-md:0 8px 32px rgba(0,0,0,0.09),0 2px 8px rgba(0,0,0,0.05);
  --sh-lg:0 20px 60px rgba(0,0,0,0.12),0 4px 16px rgba(0,0,0,0.07);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;-webkit-font-smoothing:antialiased}
a{text-decoration:none;color:inherit}
img{display:block;max-width:100%}
.blobs{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden}
.blob{position:absolute;border-radius:50%;filter:blur(90px);opacity:.35}
.b1{width:700px;height:700px;background:radial-gradient(circle,#e8dfd4,#f0ebe3);top:-200px;right:-150px}
.b2{width:550px;height:550px;background:radial-gradient(circle,#d3e5f0,#e8f2f8);bottom:-100px;left:-180px}
.b3{width:450px;height:450px;background:radial-gradient(circle,#dde6d5,#eaf2e4);top:45%;right:5%}
nav{position:sticky;top:0;z-index:200;background:rgba(250,249,247,.88);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border-bottom:1px solid var(--border);height:60px;padding:0 36px;display:flex;align-items:center;gap:10px}
.nav-logo{display:flex;align-items:center;gap:9px;font-size:17px;font-weight:700;letter-spacing:-.4px;flex-shrink:0;margin-right:4px}
.nav-badge{font-size:9px;font-weight:700;background:linear-gradient(135deg,#93c5fd,#60a5fa);color:#fff;padding:2px 8px;border-radius:99px;letter-spacing:.6px;text-transform:uppercase}
.nav-tabs{display:flex;align-items:center;gap:2px;flex:1;overflow-x:auto;scrollbar-width:none}
.nav-tabs::-webkit-scrollbar{display:none}
.nb{background:none;border:none;cursor:pointer;font-family:inherit;font-size:13px;font-weight:500;color:var(--text2);padding:6px 14px;border-radius:99px;transition:all .18s ease;white-space:nowrap;flex-shrink:0}
.nb:hover{background:rgba(0,0,0,0.05);color:var(--text)}
.nb.active{background:var(--text);color:#fff}
.nav-city{margin-left:auto;flex-shrink:0}
.nav-city form{display:flex}
.nav-city select{font-family:inherit;font-size:13px;font-weight:500;color:var(--text);background:#fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='11' viewBox='0 0 24 24' fill='none' stroke='%236E6E73' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E") no-repeat right 12px center;border:1px solid var(--border);border-radius:99px;padding:7px 32px 7px 14px;cursor:pointer;appearance:none;box-shadow:var(--sh);outline:none;transition:border-color .2s}
.nav-city select:focus{border-color:var(--gold)}
#stats-bar{display:flex;gap:14px;padding:24px 36px 8px;overflow-x:auto;scrollbar-width:none;position:relative;z-index:1}
#stats-bar::-webkit-scrollbar{display:none}
.sc{background:#fff;border:1px solid var(--border);border-radius:var(--r);padding:18px 22px;flex-shrink:0;min-width:155px;box-shadow:var(--sh);transition:transform .2s,box-shadow .2s;position:relative;overflow:hidden}
.sc::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--r) var(--r) 0 0}
.sc.cg::after{background:linear-gradient(90deg,#93c5fd,#60a5fa)}
.sc.cb::after{background:linear-gradient(90deg,#BFD7EA,#9ec4e0)}
.sc.cs::after{background:linear-gradient(90deg,#DDE6D5,#c5d9b8)}
.sc.cw::after{background:linear-gradient(90deg,#D8CFC4,#c4b8aa)}
.sc.cgn::after{background:linear-gradient(90deg,#22c55e,#16a34a)}
.sc.co::after{background:linear-gradient(90deg,#f97316,#ea580c)}
.sc:hover{transform:translateY(-2px);box-shadow:var(--sh-md)}
.sc-lbl{font-size:10.5px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;margin-bottom:5px}
.sc-val{font-size:26px;font-weight:700;color:var(--text);letter-spacing:-1px;line-height:1}
.sc-sub{font-size:11px;color:var(--text3);margin-top:3px}
main{position:relative;z-index:1;padding:16px 36px 64px}
.tab-pane{display:none}
.tab-pane.active{display:block;animation:fadeUp .28s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.card{background:#fff;border:1px solid var(--border);border-radius:var(--r-lg);box-shadow:var(--sh);overflow:hidden;margin-bottom:20px}
.ch{padding:20px 26px 0;display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
.ct{font-size:15px;font-weight:700;color:var(--text);letter-spacing:-.2px}
.cs2{font-size:12px;color:var(--text2);margin-top:2px}
.cb2{padding:20px 26px 26px}
.cb2 img{border-radius:12px;width:100%}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:20px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}
@media(max-width:960px){.g2,.g3,.g4{grid-template-columns:1fr}nav{padding:0 16px}main{padding:12px 16px 48px}#stats-bar{padding:16px 16px 8px}}
@media(max-width:960px){.sc-body{align-items:center!important}.sc-viz{min-width:118px!important;gap:8px!important;padding:0!important}.sc-viz-single{min-width:96px!important}.sc-spark{width:102px!important;height:40px!important}.sc-spark-lg{width:114px!important;height:46px!important}.sc-ring{width:42px!important;height:42px!important}.sc-ring-lg{width:58px!important;height:58px!important}.sc-donut{width:60px!important;height:60px!important}}
.badge{display:inline-flex;align-items:center;font-size:10px;font-weight:700;padding:3px 9px;border-radius:99px;letter-spacing:.5px}
.badge-RAISE{background:#f0fdf4;color:#15803d}
.badge-DISCOUNT{background:#fff7ed;color:#c2410c}
.badge-HOLD{background:#fef2f2;color:#b91c1c}
.aq-item{display:flex;align-items:center;gap:14px;padding:13px 0;border-bottom:1px solid var(--border)}
.aq-item:last-child{border-bottom:none;padding-bottom:0}
.aq-item:first-child{padding-top:0}
.aq-name{font-size:13px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.aq-hood{font-size:11px;color:var(--text2);margin-top:1px}
.aq-event{font-size:10.5px;color:var(--text3);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px}
.aq-price{font-size:12px;color:var(--text2);text-align:right}
.aq-price strong{font-size:15px;font-weight:700;color:var(--text)}
.aq-uplift{font-size:11px;font-weight:700;color:#15803d;background:#f0fdf4;padding:2px 9px;border-radius:99px;flex-shrink:0;margin-top:4px}
.hh-row{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)}
.hh-row:last-child{border-bottom:none}
.hh-name{font-size:13px;font-weight:500;color:var(--text);flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hh-bar{width:80px;height:5px;background:var(--bg2);border-radius:99px;overflow:hidden;flex-shrink:0}
.hh-fill{height:100%;background:linear-gradient(90deg,#93c5fd,#60a5fa);border-radius:99px;transition:width .8s ease}
.hh-price{font-size:13px;font-weight:700;color:var(--text);min-width:50px;text-align:right}
.hh-pct{font-size:10px;color:#15803d;font-weight:600;margin-top:1px;text-align:right}
.ep{background:#fff;border:1px solid var(--border);border-radius:16px;padding:14px 18px;display:flex;align-items:center;gap:14px;box-shadow:var(--sh);transition:transform .18s,box-shadow .18s}
.ep:hover{transform:translateY(-2px);box-shadow:var(--sh-md)}
.ep-icon{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;background:var(--bg2)}
.ep-name{font-size:13px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ep-meta{font-size:11px;color:var(--text2);margin-top:2px}
.ep-impact{font-size:12px;font-weight:700;color:#15803d;background:#f0fdf4;padding:2px 9px;border-radius:99px;flex-shrink:0}
.ep-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.cal-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
.tbl{width:100%;border-collapse:collapse}
.tbl th{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;padding:0 14px 13px;text-align:left;border-bottom:1px solid var(--border)}
.tbl td{font-size:13px;padding:12px 14px;border-bottom:1px solid var(--border);color:var(--text);vertical-align:middle}
.tbl tr:last-child td{border-bottom:none}
.tbl tr:hover td{background:#faf9f7}
.rtbl{width:100%;border-collapse:collapse}
.rtbl th{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;padding:0 18px 14px;text-align:left;border-bottom:1px solid var(--border)}
.rtbl td{font-size:13px;padding:14px 18px;border-bottom:1px solid var(--border)}
.rtbl tr:last-child td{border-bottom:none}
.map-pills{display:flex;gap:8px;padding:16px 22px;border-bottom:1px solid var(--border);flex-wrap:wrap;align-items:center}
.mpill{font-family:inherit;font-size:12px;font-weight:600;padding:6px 16px;border-radius:99px;border:1px solid var(--border);background:#fff;cursor:pointer;transition:all .18s;color:var(--text2)}
.mpill.active{background:var(--text);color:#fff;border-color:var(--text)}
.mpill[data-layer="Raise"].active{background:#15803d;border-color:#15803d;color:#fff}
.mpill[data-layer="Discount"].active{background:#c2410c;border-color:#c2410c;color:#fff}
.mpill[data-layer="Hold"].active{background:#b91c1c;border-color:#b91c1c;color:#fff}
.mpill[data-layer="Events"].active{background:#6d28d9;border-color:#6d28d9;color:#fff}
.filter-row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;margin-bottom:22px}
.ff{display:flex;flex-direction:column;gap:6px}
.fl{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.7px}
.fi,.fsel{font-family:inherit;font-size:13px;color:var(--text);background:#fff;border:1px solid var(--border2);border-radius:var(--r-sm);padding:9px 14px;outline:none;box-shadow:inset 0 1px 4px rgba(0,0,0,0.04);transition:border-color .2s;min-width:160px}
.fsel{padding-right:32px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='11' viewBox='0 0 24 24' fill='none' stroke='%236E6E73' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;appearance:none;cursor:pointer}
.fi:focus,.fsel:focus{border-color:var(--gold)}
.btn-p{font-family:inherit;font-size:13px;font-weight:600;color:#fff;background:linear-gradient(135deg,#0a1f44,#1d4ed8);border:none;border-radius:var(--r-sm);padding:10px 24px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.15);transition:transform .15s,box-shadow .15s}
.btn-p:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,0.22)}
.sim-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px}
@media(max-width:900px){.sim-grid{grid-template-columns:1fr 1fr}}
.sim-res{background:linear-gradient(135deg,#0a1f44 0%,#0f2a5c 60%,#1b3a73 100%);border-radius:var(--r-lg);padding:30px 36px;color:#fff;display:flex;align-items:center;gap:24px;margin-top:22px}
.sim-orb{width:54px;height:54px;border-radius:50%;background:linear-gradient(135deg,#93c5fd,#60a5fa);display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0}
.sim-price{font-size:52px;font-weight:400;letter-spacing:-2px;line-height:1}
.sim-plabel{font-size:12px;color:rgba(255,255,255,.55);margin-bottom:5px}
.sim-psub{font-size:13px;color:rgba(255,255,255,.65);margin-top:7px}
.sim-explanation{margin-top:14px;max-width:760px;padding:14px 16px;border-radius:14px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);font-size:13px;line-height:1.65;color:rgba(255,255,255,.86);white-space:normal}
.sim-insight{font-size:12px;color:rgba(255,255,255,.5);margin-top:8px;max-width:500px;line-height:1.5;white-space:pre-line}
.rmc{background:#fff;border:1px solid var(--border);border-radius:var(--r-lg);padding:26px 24px;box-shadow:var(--sh);text-align:center}
.rmc-icon{font-size:30px;margin-bottom:10px}
.rmc-val{font-size:34px;font-weight:800;letter-spacing:-1.5px;line-height:1;margin-bottom:5px}
.rmc-val.gv{color:#15803d}.rmc-val.bv{color:#1d4ed8}.rmc-val.ov{color:#92400e}
.rmc-lbl{font-size:12px;font-weight:600;color:var(--text2);margin-bottom:2px}
.rmc-sub{font-size:11px;color:var(--text3)}
.insight-banner{background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1px solid rgba(147,197,253,.35);border-radius:var(--r-lg);padding:18px 26px;margin-bottom:20px;font-size:14px;color:var(--text);line-height:1.65}
.insight-banner .hi{font-weight:800;color:#92400e}
.chat-wrap{max-width:800px;margin:0 auto}
.chat-hero{background:linear-gradient(135deg,#0a1f44 0%,#0f2a5c 60%,#1b3a73 100%);border-radius:var(--r-lg);padding:40px 44px 38px;color:#fff;margin-bottom:20px;position:relative;overflow:hidden}
.chat-hero::before{content:'';position:absolute;top:-80px;right:-80px;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle,rgba(147,197,253,.20),transparent 68%);pointer-events:none}
.chat-ai-badge{font-size:11px;font-weight:700;background:rgba(147,197,253,.14);color:#dbeafe;padding:4px 12px;border-radius:99px;display:inline-block;margin-bottom:14px;letter-spacing:.5px;border:1px solid rgba(147,197,253,.42)}
.chat-title{font-size:34px;font-weight:800;letter-spacing:-1.2px;line-height:1.18;margin-bottom:10px}
.chat-sub{font-size:14px;color:rgba(255,255,255,.6);max-width:500px;line-height:1.6}
.chat-sub span{color:#bfdbfe;font-weight:700}
.tile-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
@media(max-width:700px){.tile-grid{grid-template-columns:1fr 1fr}}
.tile{background:#fff;border:1px solid var(--border);border-radius:var(--r);padding:16px;cursor:pointer;box-shadow:var(--sh);transition:transform .18s,box-shadow .18s;text-align:left;font-family:inherit;display:flex;flex-direction:column;gap:10px;width:100%}
.tile:hover{transform:translateY(-2px);box-shadow:var(--sh-md);border-color:#9fc1e6}
.tile-icon{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:19px}
.tile-txt{font-size:13px;font-weight:600;color:var(--text)}
.msg-card{background:#fff;border:1px solid var(--border);border-radius:var(--r-lg);box-shadow:var(--sh);overflow:hidden;margin-bottom:12px}
.msgs{max-height:420px;overflow-y:auto;padding:22px 26px;display:flex;flex-direction:column;gap:14px}
.msg-u{align-self:flex-end;background:var(--text);color:#fff;padding:10px 16px;border-radius:18px 18px 4px 18px;font-size:13.5px;max-width:72%;line-height:1.5}
.msg-b-wrap{display:flex;gap:10px;align-items:flex-start}
.bot-av{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#93c5fd,#60a5fa);display:flex;align-items:center;justify-content:center;font-size:12px;color:#fff;font-weight:700;flex-shrink:0}
.msg-b{background:var(--bg2);padding:10px 16px;border-radius:4px 18px 18px 18px;font-size:13.5px;max-width:82%;line-height:1.55}
.msg-b table{font-size:12px;border-collapse:collapse;margin-top:8px;width:100%}
.msg-b th,.msg-b td{padding:7px 10px;border-bottom:1px solid var(--border);text-align:left}
.chips-row{display:flex;flex-wrap:wrap;gap:8px;padding:12px 22px;border-top:1px solid var(--border)}
.chip{font-family:inherit;font-size:12px;font-weight:500;color:var(--text2);background:var(--bg2);border:1px solid var(--border);border-radius:99px;padding:6px 14px;cursor:pointer;transition:all .18s}
.chip:hover{background:var(--text);color:#fff;border-color:var(--text)}
.inp-card{background:#fff;border:1px solid var(--border);border-radius:var(--r-lg);box-shadow:var(--sh);padding:8px 8px 8px 22px;display:flex;align-items:center;gap:8px}
.inp-card input{flex:1;border:none;outline:none;font-family:inherit;font-size:14px;color:var(--text);background:transparent}
.inp-card input::placeholder{color:var(--text3)}
.send-btn{width:42px;height:42px;border-radius:13px;border:none;cursor:pointer;background:var(--text);color:#fff;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .18s,transform .15s}
.send-btn:hover{background:#333;transform:scale(1.04)}
.send-btn svg{width:16px;height:16px;fill:#fff}
.wx-card{background:linear-gradient(135deg,#ebf4fb,#ddeef8);border:1px solid rgba(191,215,234,.45);border-radius:var(--r-lg);padding:22px 26px;box-shadow:var(--sh);margin-bottom:20px}
.wx-title{font-size:10.5px;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:.9px;margin-bottom:10px}
.wx-cur{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.wx-temp{font-size:38px;font-weight:800;letter-spacing:-1.5px;color:#0a1f44}
.wx-desc{font-size:13.5px;color:#1d4ed8;font-weight:600}
.wx-cond{font-size:11px;color:#3b82f6;margin-top:3px}
.wx-days{display:flex;gap:10px;overflow-x:auto;scrollbar-width:none;padding-bottom:2px}
.wx-days::-webkit-scrollbar{display:none}
.wx-day{background:rgba(255,255,255,.75);border-radius:14px;padding:10px 13px;text-align:center;flex-shrink:0;min-width:72px}
.wx-date{font-size:10px;color:#1d4ed8;font-weight:700;margin-bottom:3px}
.wx-dt{font-size:13px;font-weight:700;color:#0a1f44;margin:2px 0}
.wx-dd{font-size:10px;color:#3b82f6}
.wx-rain{font-size:10px;color:#1d4ed8;margin-top:2px}
.wx-tip{margin-top:13px;font-size:12.5px;color:#1d4ed8;background:rgba(255,255,255,.55);padding:9px 14px;border-radius:11px;border:1px solid rgba(191,215,234,.4)}
.srch-wrap{position:relative;margin-bottom:16px}
.srch{width:100%;font-family:inherit;font-size:13px;color:var(--text);background:#fff;border:1px solid var(--border2);border-radius:var(--r-sm);padding:10px 16px 10px 42px;outline:none;box-shadow:inset 0 1px 4px rgba(0,0,0,0.04);transition:border-color .2s}
.srch:focus{border-color:var(--gold)}
.srch-ic{position:absolute;left:14px;top:50%;transform:translateY(-50%);color:var(--text3);pointer-events:none}
.exp-card{background:#fff;border:1px solid var(--border);border-radius:var(--r);padding:18px 22px;box-shadow:var(--sh);transition:transform .18s,box-shadow .18s;display:flex;gap:18px;align-items:center;margin-bottom:10px}
.exp-card:hover{transform:translateY(-1px);box-shadow:var(--sh-md)}
.empty{color:var(--text2);font-size:13px;text-align:center;padding:40px 0}
.hero-img{position:relative;width:100%;height:340px;border-radius:var(--r-lg);overflow:hidden;margin-bottom:20px;box-shadow:var(--sh-lg)}
.hero-img img{width:100%;height:100%;object-fit:cover;object-position:center 55%;display:block;transition:transform 8s ease}
.hero-img:hover img{transform:scale(1.04)}
.hero-overlay{position:absolute;inset:0;background:linear-gradient(160deg,rgba(20,16,12,.45) 0%,rgba(20,16,12,.15) 50%,rgba(20,16,12,.55) 100%)}
.hero-text{position:absolute;bottom:32px;left:36px;color:#fff}
.hero-eyebrow{font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,255,255,.75);margin-bottom:7px}
.hero-headline{font-size:34px;font-weight:800;letter-spacing:-1.1px;line-height:1.15;margin-bottom:9px;text-shadow:0 2px 12px rgba(0,0,0,.35)}
.hero-sub{font-size:14px;color:rgba(255,255,255,.72);font-weight:400;max-width:440px;line-height:1.5}
.hero-pills{position:absolute;top:22px;right:28px;display:flex;gap:8px}
.hero-pill{font-size:11px;font-weight:600;padding:5px 13px;border-radius:99px;background:rgba(255,255,255,.18);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);color:#fff;border:1px solid rgba(255,255,255,.25);letter-spacing:.3px;display:inline-flex;align-items:center;gap:5px}
/* Stat card icon */
.sc-icon{width:32px;height:32px;border-radius:9px;display:flex;align-items:center;justify-content:center;margin-bottom:8px}
.sc-icon svg{opacity:.75}
/* Nav tab icons */
.nb{display:inline-flex;align-items:center;gap:5px}
.nb svg{opacity:.6;flex-shrink:0;transition:opacity .18s}
.nb:hover svg,.nb.active svg{opacity:1}
/* Card header icon */
.ch-icon{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
/* Map pill dot */
.mpill-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle;flex-shrink:0}
/* Action queue rank */
.aq-rank{width:20px;height:20px;border-radius:6px;background:var(--bg2);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800;color:var(--text3);flex-shrink:0;letter-spacing:-.3px}
/* Feature strip below hero */
.feature-strip{display:flex;gap:0;border-bottom:1px solid var(--border);position:relative;z-index:1;overflow-x:auto;scrollbar-width:none;background:#fff}
.feature-strip::-webkit-scrollbar{display:none}
.fstrip-item{display:flex;align-items:center;gap:10px;padding:14px 24px;font-size:12.5px;font-weight:500;color:var(--text2);white-space:nowrap;border-right:1px solid var(--border);flex-shrink:0}
.fstrip-item:first-child{padding-left:36px}
.fstrip-sq{width:10px;height:10px;border-radius:3px;flex-shrink:0}
.fstrip-val{font-weight:700;color:var(--text);font-size:14px;margin-right:3px}
/* Hero action pills (frosted glass tabs) */
.hero-actions{position:absolute;bottom:28px;left:36px;display:flex;gap:8px;flex-wrap:wrap}
.hap{font-family:inherit;font-size:12px;font-weight:600;padding:7px 16px;border-radius:99px;border:1.5px solid rgba(255,255,255,.4);background:rgba(255,255,255,.18);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);color:#fff;cursor:pointer;display:inline-flex;align-items:center;gap:6px;transition:all .18s;white-space:nowrap;letter-spacing:.2px}
.hap:hover{background:rgba(255,255,255,.88);color:#1D1D1F;border-color:rgba(255,255,255,.9)}
/* Empty state */
.empty-state{display:flex;flex-direction:column;align-items:center;gap:10px;padding:44px 0;color:var(--text3)}
.empty-state p{font-size:13px;color:var(--text2)}
/* Section divider label */
.sec-div{display:flex;align-items:center;gap:12px;margin:24px 0 14px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.1px;color:var(--text3)}
.sec-div::after{content:'';flex:1;height:1px;background:var(--border)}
/* Improved rmc icon */
.rmc-icon-wrap{width:52px;height:52px;border-radius:16px;display:flex;align-items:center;justify-content:center;margin:0 auto 12px}
/* Sort indicators */
.sort-ic{font-size:10px;color:var(--text3);margin-left:4px;user-select:none}
.tbl th{cursor:pointer}
.tbl th:last-child{cursor:default}
/* Listing modal */
.modal-overlay{position:fixed;inset:0;z-index:9000;background:rgba(0,0,0,.45);display:none;align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal-box{background:#fff;border-radius:22px;padding:32px 36px;max-width:500px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,.2);position:relative;animation:fadeUp .22s ease}
.modal-close{position:absolute;top:14px;right:14px;background:var(--bg2);border:none;border-radius:50%;width:32px;height:32px;cursor:pointer;font-size:18px;display:flex;align-items:center;justify-content:center;color:var(--text2);transition:background .15s}
.modal-close:hover{background:var(--border2)}
/* What-if slider */
.whatif-result-grid{display:flex;gap:14px;flex-wrap:wrap;margin-top:4px}
/* Neighborhood panel */
#hood-details{margin-top:4px}
/* Alerts table row hover */
#alerts-table tbody tr{cursor:default}
/* Export button link */
.btn-export{font-family:inherit;font-size:12px;font-weight:600;color:var(--text2);background:#fff;border:1px solid var(--border2);border-radius:var(--r-sm);padding:8px 16px;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:6px;transition:all .18s}
.btn-export:hover{background:var(--text);color:#fff;border-color:var(--text)}


/* ===== 2026 SaaS redesign overrides ===== */
:root{
  --pw-bg:#eef3f8;
  --pw-sidebar:rgba(255,255,255,0.94);
  --pw-card:rgba(255,255,255,0.96);
  --pw-border:#d9e3ef;
  --pw-text:#0a1f44;
  --pw-sub:#0a1f44;
  --pw-navy:#0a1f44;
  --pw-navy-2:#133a63;
  --pw-green:#16a34a;
  --pw-green-soft:#eaf7ef;
}
body{background:radial-gradient(circle at top left,#f8fbff 0%,#eef3f8 38%,#edf2f7 100%)!important;color:var(--pw-text)!important}
.blobs{display:none!important}
.app-shell{display:grid;grid-template-columns:280px 1fr;min-height:100vh;position:relative}
.side-rail{background:var(--pw-sidebar);border-right:1px solid var(--pw-border);padding:26px 18px 20px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:18px}
.side-brand{display:flex;align-items:center;gap:12px;padding:8px 8px 18px;font-weight:800;font-size:22px;color:var(--pw-navy);letter-spacing:-.03em}
.side-brand-mark{width:44px;height:44px;border-radius:14px;background:linear-gradient(135deg,var(--pw-navy),#1d6fb8);display:flex;align-items:center;justify-content:center;color:#fff;box-shadow:0 10px 24px rgba(11,58,102,.18)}
.side-sub{font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.11em;padding:8px 12px 2px}
.side-nav{display:flex;flex-direction:column;gap:6px}
.side-link{display:flex;align-items:center;gap:12px;padding:12px 14px;border-radius:16px;color:var(--pw-sub);font-size:14px;font-weight:600;transition:.18s background,.18s color,.18s transform;text-decoration:none}
.side-link:hover{background:#eef4fb;color:var(--pw-navy);transform:translateX(2px)}
.side-link.active{background:linear-gradient(135deg,var(--pw-navy),var(--pw-navy-2));color:#fff;box-shadow:0 14px 30px rgba(11,58,102,.18)}
.side-link svg{flex-shrink:0}
.side-footer{margin-top:auto;padding:14px;border:1px solid var(--pw-border);border-radius:18px;background:linear-gradient(180deg,#fff,#f8fafc)}
.side-footer-name{font-size:15px;font-weight:700;color:var(--pw-text)}
.side-footer-email{font-size:12px;color:var(--pw-sub);margin-top:2px}
.app-main{min-width:0;padding:22px 22px 34px}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:18px;padding:4px 8px 18px}
.topbar h1{font-size:42px;letter-spacing:-.04em;line-height:1.04;margin:0;color:var(--pw-text)}
.topbar p{margin-top:6px;color:var(--pw-sub);font-size:16px}
.topbar-actions{display:flex;align-items:center;gap:14px}
.topbar-icon{width:42px;height:42px;border-radius:50%;border:1px solid var(--pw-border);background:#fff;display:flex;align-items:center;justify-content:center;color:var(--pw-navy);box-shadow:0 4px 14px rgba(15,23,42,.06)}
.topbar-avatar{width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#e2e8f0,#cbd5e1);display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--pw-navy);border:3px solid #fff;box-shadow:0 10px 24px rgba(15,23,42,.1)}
.topbar-add-property{display:inline-flex;align-items:center;justify-content:center;gap:8px;height:44px;padding:0 18px;border-radius:14px;text-decoration:none;font-size:14px;font-weight:800;background:linear-gradient(135deg,var(--pw-primary),#163a66);color:#fff!important;border:1px solid rgba(255,255,255,.18);box-shadow:0 12px 24px rgba(15,39,71,.16);white-space:nowrap}
.topbar-add-property:hover{transform:translateY(-1px);box-shadow:0 16px 30px rgba(15,39,71,.2)}
#stats-bar{padding:0!important;display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;overflow:visible!important;margin-bottom:20px}
.sc{min-width:0!important;border-radius:24px!important;border:1px solid var(--pw-border)!important;box-shadow:0 10px 28px rgba(15,23,42,.06)!important;padding:22px 22px 20px!important}
.sc::after{display:none!important}
.sc-lbl{font-size:12px!important;text-transform:none!important;letter-spacing:0!important;color:#94a3b8!important;font-weight:600!important;margin-bottom:10px!important}
.sc-val{font-size:40px!important;letter-spacing:-.05em!important;color:var(--pw-text)!important}
.sc-sub{font-size:13px!important;color:var(--pw-sub)!important;margin-top:8px!important}
main{padding:0!important}
.card{border:1px solid var(--pw-border)!important;border-radius:24px!important;box-shadow:0 10px 30px rgba(15,23,42,.06)!important;margin-bottom:18px!important;overflow:hidden;background:var(--pw-card)!important}
.ch{padding:22px 24px 0!important}.cb2{padding:18px 24px 24px!important}
.ct{font-size:22px!important;letter-spacing:-.03em!important}.cs2{font-size:13px!important;color:var(--pw-sub)!important}
.sec-div{font-size:12px!important;font-weight:800!important;letter-spacing:.12em!important;color:#94a3b8!important;text-transform:uppercase!important;margin:10px 4px 14px!important}
.app-shell{display:grid;grid-template-columns:280px 1fr;min-height:100vh;gap:0}
.side-rail{background:var(--pw-sidebar)!important;border-right:1px solid var(--pw-border)!important;padding:28px 18px 20px;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);box-shadow:24px 0 60px rgba(15,23,42,.05)}
.side-brand{display:flex;align-items:center;gap:12px;font-size:20px;font-weight:800;letter-spacing:-.03em;color:var(--pw-text);padding:8px 10px;margin-bottom:22px}
.side-brand-mark{width:42px;height:42px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--pw-navy),var(--pw-navy-2));color:#fff;box-shadow:0 14px 28px rgba(10,31,68,.18)}
.side-sub{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#6c819d;padding:12px 10px 8px}
.side-nav{display:flex;flex-direction:column;gap:6px}
.side-link{display:flex;align-items:center;gap:12px;padding:12px 14px;border-radius:16px;font-size:14px;font-weight:700;color:#49607f;transition:all .18s ease}
.side-link svg{opacity:.88}
.side-link:hover{background:#f4f8fd;color:var(--pw-text);transform:translateX(2px)}
.side-link.active{background:linear-gradient(135deg,#0a1f44,#133a63);color:#fff;box-shadow:0 12px 26px rgba(10,31,68,.18)}
.side-footer{margin-top:auto;padding:18px 14px;border-radius:20px;background:linear-gradient(180deg,#f8fbff,#eef4fb);border:1px solid var(--pw-border)}
.app-main{padding:24px 26px 40px}
.topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:10px;padding:12px 4px 6px}
.topbar h1{font-size:36px;line-height:1.05;letter-spacing:-.05em;color:var(--pw-text);margin-bottom:8px}
.topbar p{font-size:14px;color:#5f7490;max-width:700px;line-height:1.55}
.topbar-actions{display:flex;align-items:center;gap:10px}
.topbar-icon,.topbar-avatar{width:44px;height:44px;border-radius:16px;display:flex;align-items:center;justify-content:center;border:1px solid var(--pw-border);background:rgba(255,255,255,.8);box-shadow:0 8px 24px rgba(15,23,42,.06)}
.topbar-avatar{font-size:13px;font-weight:800;color:var(--pw-text);background:linear-gradient(135deg,#eff5fb,#ffffff)}
#stats-bar{padding:4px 0 14px!important;display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;overflow:visible}
.sc{border-radius:24px!important;padding:22px 22px 20px!important;box-shadow:0 16px 40px rgba(15,23,42,.06)!important;background:linear-gradient(180deg,#ffffff,#fbfdff)!important;border:1px solid var(--pw-border)!important}
.sc:hover{transform:translateY(-3px)!important}
main{padding:6px 0 24px!important}
.card{border-radius:28px!important;background:var(--pw-card)!important;border:1px solid var(--pw-border)!important;box-shadow:0 18px 45px rgba(15,23,42,.06)!important}
.ch{padding:24px 28px 0!important}
.cb2{padding:22px 28px 28px!important}
.pw-dashboard-top{display:grid;grid-template-columns:1.7fr 1fr;gap:18px;margin-bottom:18px}
.pw-dashboard-bottom{display:grid;grid-template-columns:1.3fr 1fr;gap:18px;margin-bottom:18px}
.pw-hero{position:relative;border:none!important;background:linear-gradient(135deg,#0a1f44 0%,#133a63 58%,#1a4f7f 100%)!important;overflow:hidden!important}.pw-hero::before{content:'';position:absolute;right:-60px;top:-60px;width:220px;height:220px;border-radius:50%;background:rgba(255,255,255,.08)}.pw-hero::after{content:'';position:absolute;left:46%;bottom:-70px;width:180px;height:180px;border-radius:50%;background:rgba(255,255,255,.06)}
.pw-hero-inner{position:relative;z-index:1;display:flex;justify-content:space-between;align-items:flex-end;gap:24px;padding:30px}
.pw-eyebrow{font-size:12px;font-weight:800;letter-spacing:.12em;color:rgba(255,255,255,.74);text-transform:uppercase;margin-bottom:12px}
.pw-hero-title{font-size:30px;font-weight:800;letter-spacing:-.05em;color:#fff;line-height:1.08;margin-bottom:10px;max-width:720px}
.pw-hero-copy{font-size:15px;color:rgba(255,255,255,.82);max-width:700px;line-height:1.65}
.pw-price-row{display:flex;align-items:center;gap:20px;margin-top:20px;flex-wrap:wrap}
.pw-price-block small{display:block;font-size:11px;color:rgba(255,255,255,.62);margin-bottom:5px;text-transform:uppercase;letter-spacing:.08em}
.pw-price-block strong{font-size:30px;letter-spacing:-.05em;color:#fff}
.pw-arrow{font-size:30px;color:rgba(255,255,255,.45)}
.pw-green-text{color:#9ef0b8!important}
.pw-chip{display:inline-flex;align-items:center;gap:6px;padding:8px 12px;border-radius:999px;background:rgba(158,240,184,.14);border:1px solid rgba(158,240,184,.22);color:#c5f6d5;font-size:12px;font-weight:800}
.pw-btn{background:#fff;color:var(--pw-text);padding:14px 20px;border-radius:16px;font-weight:800;font-size:14px;display:inline-flex;align-items:center;justify-content:center;box-shadow:0 12px 30px rgba(15,23,42,.16)}
.pw-signal-list{display:flex;flex-direction:column;gap:14px}
.pw-signal{display:flex;gap:12px;align-items:flex-start;padding:16px;border:1px solid var(--pw-border);border-radius:20px;background:linear-gradient(180deg,#fcfdff,#f7fbff)}
.pw-signal-icon{width:42px;height:42px;border-radius:14px;background:#eff6ff;display:flex;align-items:center;justify-content:center;color:var(--pw-navy);font-weight:800;flex-shrink:0}
.pw-signal-title{font-size:15px;font-weight:700;color:var(--pw-text)}
.pw-signal-sub{font-size:13px;color:var(--pw-sub);margin-top:2px;line-height:1.45}
.pw-list-card .pw-list-row{display:grid;grid-template-columns:84px 1fr auto auto;align-items:center;gap:18px;padding:16px 0;border-bottom:1px solid var(--pw-border)}
.pw-list-card .pw-list-row:last-child{border-bottom:none;padding-bottom:0}
.pw-thumb{width:84px;height:60px;border-radius:18px;background:linear-gradient(135deg,#dbeafe,#bfdbfe);display:flex;align-items:center;justify-content:center;color:var(--pw-navy);font-weight:800;font-size:18px;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(255,255,255,.55)}
.pw-list-name{font-size:17px;font-weight:800;color:var(--pw-text);letter-spacing:-.025em}
.pw-list-meta{font-size:13px;color:var(--pw-sub);margin-top:4px}
.pw-list-price{font-size:17px;font-weight:800;color:var(--pw-text);text-align:right}
.pw-list-occ{font-size:13px;color:var(--pw-sub);text-align:right}
.pw-view{display:inline-flex;padding:10px 16px;border-radius:14px;border:1.5px solid #d4dfeb;color:var(--pw-text);font-weight:800;background:#fff;box-shadow:0 10px 20px rgba(15,23,42,.04)}
.wx-card{border-radius:24px!important;border:1px solid var(--pw-border)!important;box-shadow:0 10px 30px rgba(15,23,42,.06)!important}
.tbl tr:hover td,.rtbl tr:hover td{background:#f8fbff!important}
@media (max-width: 1180px){
  .app-shell{grid-template-columns:1fr}
  .side-rail{position:relative;height:auto;border-right:none;border-bottom:1px solid var(--pw-border)}
}
@media (max-width: 900px){
  #stats-bar,.pw-dashboard-top,.pw-dashboard-bottom{grid-template-columns:1fr!important}
  .topbar{flex-direction:column;align-items:flex-start}
  .topbar h1{font-size:32px}
  .pw-list-card .pw-list-row{grid-template-columns:60px 1fr;align-items:start}
  .pw-list-price,.pw-list-occ{grid-column:2}
}


/* === PriceWise Brand System Refresh === */
:root{
  --pw-primary:#0F2747;
  --pw-primary-strong:#0A1B33;
  --pw-secondary:#4F8BC9;
  --pw-secondary-soft:#EAF3FF;
  --pw-accent:#16A34A;
  --pw-accent-soft:#ECFDF3;
  --pw-neutral-0:#FFFFFF;
  --pw-neutral-50:#F7FAFC;
  --pw-neutral-100:#EEF3F8;
  --pw-neutral-200:#D8E2EE;
  --pw-neutral-300:#C3D0DE;
  --pw-neutral-500:#5B6B7E;
  --pw-neutral-700:#233246;
  --pw-neutral-900:#0F172A;
  --space-1:8px;
  --space-2:16px;
  --space-3:24px;
  --space-4:32px;
  --space-5:40px;
  --radius-sm:12px;
  --radius-md:18px;
  --radius-lg:24px;
  --shadow-card:0 12px 32px rgba(15,39,71,.08);
  --shadow-card-hover:0 18px 44px rgba(15,39,71,.12);
}
html,body{background:linear-gradient(180deg,#f8fbff 0%,#eef4fa 100%)!important;color:var(--pw-neutral-900)!important}
body,input,select,button,textarea{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important}
h1,h2,h3,h4,.ct,.pw-hero-title,.topbar h1,.side-brand-name,.sc-val{font-family:'Space Grotesk','Inter',sans-serif!important}
.topbar h1{font-size:40px!important;font-weight:700!important;line-height:1.05!important;letter-spacing:-.045em!important;color:var(--pw-primary)!important}
.ct{font-size:22px!important;font-weight:700!important;line-height:1.15!important;color:var(--pw-primary)!important}
.cs2,.topbar p,.side-sub,.pw-list-meta,.pw-signal-sub,.sc-sub,.wx-dd,.wx-cond,.aq-hood,.aq-event,.ep-meta{color:var(--pw-neutral-500)!important}
.side-rail{background:rgba(255,255,255,.94)!important;border-right:1px solid var(--pw-neutral-200)!important;padding:var(--space-4) var(--space-2) var(--space-3)!important;box-shadow:24px 0 64px rgba(15,39,71,.06)!important}
.side-brand{padding:0 var(--space-1) var(--space-3)!important;gap:14px!important;align-items:center!important}
.side-brand-mark{width:48px!important;height:48px!important;border-radius:16px!important;background:linear-gradient(135deg,var(--pw-primary),var(--pw-secondary))!important;box-shadow:0 14px 30px rgba(15,39,71,.18)!important}
.side-brand-lockup{display:flex;flex-direction:column;gap:2px}
.side-brand-name{font-size:22px;font-weight:700;line-height:1;color:var(--pw-primary)}
.side-brand-tag{font-size:12px;font-weight:600;letter-spacing:.01em;color:var(--pw-neutral-500)}
.side-link{min-height:48px!important;border-radius:16px!important;color:var(--pw-neutral-700)!important}
.side-link svg,.topbar-icon svg,.side-brand-mark svg{stroke-width:1.9!important}
.side-link:hover{background:var(--pw-secondary-soft)!important;color:var(--pw-primary)!important}
.side-link.active{background:linear-gradient(135deg,var(--pw-primary),#163a66)!important;color:#fff!important;box-shadow:0 14px 30px rgba(15,39,71,.18)!important}
#stats-bar,.g2,.g3,.g4,main{gap:var(--space-3)!important}
.card,.sc,.msg-card,.inp-card,.wx-card,.exp-card,.ep,.pw-signal,.side-footer,.modal-box{border-radius:var(--radius-lg)!important;border:1px solid #d8e4f0!important;box-shadow:var(--shadow-card)!important;background:rgba(255,255,255,.96)!important}
.card,.sc,.msg-card,.inp-card,.wx-card,.exp-card,.ep,.pw-signal,.side-footer,.modal-box{transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease,background .2s ease!important}
.card:hover,.sc:hover,.exp-card:hover,.ep:hover,.tile:hover,.pw-list-row:hover,.pw-signal:hover{transform:translateY(-2px)!important;border-color:#9fc1e6!important;box-shadow:var(--shadow-card-hover)!important}
.pw-dashboard-top,.pw-dashboard-bottom{display:grid!important;gap:var(--space-4)!important;align-items:stretch!important;margin-bottom:var(--space-4)!important}
.pw-dashboard-bottom{margin-top:var(--space-4)!important}
#pane-dashboard .card + .card{margin-top:0!important}
.sc{position:relative!important;background:#ffffff!important;border:1px solid #d7e3ef!important;box-shadow:0 10px 24px rgba(15,39,71,.05)!important;border-radius:22px!important;padding:18px 18px 16px!important;min-height:124px!important}
.sc-top{display:flex!important;align-items:center!important;gap:12px!important;margin-bottom:14px!important}
.sc-icon{width:38px!important;height:38px!important;display:grid!important;place-items:center!important;border-radius:14px!important;background:#eef4fb!important;border:1px solid #d8e3ef!important;color:#6b7f99!important;box-shadow:none!important}
.sc-lbl{font-size:11px!important;letter-spacing:.10em!important;text-transform:uppercase!important;color:#6b7d92!important}
.sc-val{font-size:38px!important;line-height:1!important;margin-bottom:10px!important}
.sc-sub{font-size:13px!important;line-height:1.45!important;color:#6b7d92!important}
.sc-body{display:flex!important;align-items:flex-start!important;justify-content:flex-start!important;gap:0!important;min-height:52px!important}
.sc-copy{min-width:0!important;flex:1 1 auto!important;padding-right:0!important;display:flex!important;flex-direction:column!important;align-items:flex-start!important;justify-content:flex-start!important}
.sc-viz{display:flex!important;flex-direction:row!important;align-items:center!important;justify-content:center!important;gap:16px!important;min-width:220px!important;margin-left:auto!important;flex-shrink:0!important;transform:none!important;padding:0!important;border-radius:0!important;background:none!important;border:none!important;box-shadow:none!important}.sc-viz-single{min-width:160px!important;padding:0!important}.sc-donut{width:112px!important;height:112px!important;object-fit:contain!important;display:block!important}.sc-ring-lg{width:108px!important;height:108px!important}.sc-spark-lg{width:220px!important;height:86px!important}
.sc-viz,.sc-viz-single{display:none!important;min-width:0!important;width:0!important;overflow:hidden!important}.sc-body.no-viz{justify-content:flex-start!important;align-items:flex-start!important}
.sc-ring{width:74px!important;height:74px!important;object-fit:contain!important;opacity:1!important;display:block!important;filter:none!important}
.sc-spark{width:186px!important;height:72px!important;object-fit:contain!important;display:block!important;opacity:1!important}
.pw-hero{background:linear-gradient(135deg,#08172d 0%,#0f2747 48%,#275a91 100%)!important;border:1px solid rgba(151,188,228,.34)!important;box-shadow:0 22px 52px rgba(10,27,51,.22)!important}
.pw-hero-inner{padding:36px!important;min-height:100%!important;display:flex!important;justify-content:space-between!important;gap:24px!important;align-items:stretch!important}.pw-hero-inner>div:first-child{display:flex!important;flex-direction:column!important;justify-content:flex-start!important;align-self:flex-start!important}.pw-btn{align-self:flex-end!important;margin-top:auto!important}
.pw-hero-copy{font-size:14px!important;line-height:1.65!important;color:rgba(255,255,255,.8)!important;max-width:640px!important}
.pw-price-row{margin-top:24px!important;gap:16px!important;flex-wrap:wrap!important}
.pw-price-block small{font-size:12px!important;opacity:.8!important}
.pw-price-block strong{font-size:28px!important}
.pw-signal-list{display:flex!important;flex-direction:column!important;gap:16px!important}
.pw-dashboard-top > .card:last-child{background:linear-gradient(180deg,#f7faff 0%,#eef4fb 100%)!important;border:1px solid #aac6e4!important;box-shadow:0 16px 38px rgba(15,39,71,.10)!important}
.pw-signal{background:rgba(255,255,255,.7)!important;border:1px solid #d7e2ef!important;border-left:4px solid #bfd7ea!important;border-radius:18px!important;padding:16px!important;gap:14px!important;align-items:flex-start!important}
.pw-signal:nth-child(2){border-left-color:#c7dfc0!important}
.pw-signal:last-child{border-left-color:#f2c28f!important}
.pw-signal-icon{width:42px!important;height:42px!important;display:grid!important;place-items:center!important;flex:0 0 42px!important;background:linear-gradient(180deg,#ffffff,#e7f0fb)!important;border:1px solid #d6e4f4!important;box-shadow:0 8px 18px rgba(15,39,71,.08)!important}
.pw-signal-title{font-size:15px!important;font-weight:700!important;color:var(--pw-primary)!important;margin-bottom:4px!important}
.pw-signal-sub{font-size:13px!important;line-height:1.55!important;color:#516173!important}
.pw-list-card .cb2{display:flex!important;flex-direction:column!important;gap:16px!important}
.pw-list-row{display:flex!important;align-items:center!important;justify-content:space-between!important;gap:18px!important;padding:18px!important;background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%)!important;border:1px solid #dbe4ef!important;border-left:4px solid #c8def3!important;border-radius:18px!important;box-shadow:0 10px 24px rgba(15,39,71,.06)!important}
.pw-list-main,.pw-list-side{display:flex!important;align-items:center!important;gap:14px!important}
.pw-list-main{flex:1 1 auto!important;min-width:0!important}
.pw-list-side{justify-content:flex-end!important;flex:0 0 auto!important}
.pw-list-name{font-size:15px!important;font-weight:700!important;color:var(--pw-primary)!important;margin-bottom:4px!important}
.pw-list-meta{font-size:13px!important;line-height:1.5!important;color:#617286!important}
.pw-list-price{font-size:22px!important;font-weight:700!important;color:var(--pw-primary)!important;text-align:right!important}
.pw-list-occ{font-size:12px!important;color:#617286!important;text-align:right!important}
.pw-view{min-height:42px!important;padding:10px 16px!important;font-size:13px!important;display:inline-flex!important;align-items:center!important;justify-content:center!important}
.btn-p,.pw-btn,.send-btn,.mpill.active,.nb.active{min-height:44px!important}
.btn-s,.chip,.mpill,.pw-view,.nav-city select{min-height:42px!important;padding:10px 16px!important;font-size:13px!important}
@media (max-width: 960px){
  .pw-hero-inner{padding:28px!important;align-items:flex-start!important}.pw-btn{margin-top:18px!important;align-self:flex-start!important}
  .pw-list-row,.pw-list-side{flex-direction:column!important;align-items:flex-start!important}
  .pw-list-side{width:100%!important}
  .pw-list-price,.pw-list-occ{text-align:left!important}
}
.card:hover,.sc:hover,.exp-card:hover,.ep:hover,.tile:hover{box-shadow:var(--shadow-card-hover)!important}
.sc,.ep,.pw-signal,.exp-card,.tile,.wx-day{padding:var(--space-2)!important}
.ch{padding:var(--space-3) var(--space-3) 0!important}
.cb2,.msgs{padding:var(--space-3)!important}
.tbl th,.rtbl th{padding:0 var(--space-2) 12px!important}
.tbl td,.rtbl td{padding:14px var(--space-2)!important}
.sc{background:linear-gradient(180deg,#ffffff,#f9fbfe)!important}
.sc::after{height:4px!important}
.sc-lbl{font-size:11px!important;letter-spacing:.08em!important;color:#6d8098!important;text-transform:uppercase!important}
.sc-val{font-size:34px!important;line-height:1.05!important;color:var(--pw-primary)!important;margin-bottom:8px!important}
.card,.msg-card,.inp-card,.wx-card{overflow:hidden}
.pw-hero{background:linear-gradient(135deg,var(--pw-primary-strong) 0%,var(--pw-primary) 56%,var(--pw-secondary) 100%)!important;border:none!important}
.pw-hero::before,.pw-hero::after{background:rgba(255,255,255,.09)!important}
.pw-hero-inner{padding:32px!important}
.pw-hero-title{font-size:34px!important;font-weight:700!important;max-width:760px!important}
.pw-hero-copy{font-size:15px!important;color:rgba(255,255,255,.82)!important}
.btn-p,.pw-btn,.send-btn,.mpill.active,.nb.active{background:linear-gradient(135deg,var(--pw-primary),#163a66)!important;color:#fff!important;border:none!important;box-shadow:0 12px 24px rgba(15,39,71,.16)!important}
.btn-p,.pw-btn{border-radius:14px!important;font-size:14px!important;font-weight:700!important;padding:12px 20px!important;transition:transform .18s ease, box-shadow .18s ease, background .18s ease!important}
.btn-p:hover,.pw-btn:hover,.send-btn:hover{transform:translateY(-1px)!important;box-shadow:0 16px 30px rgba(15,39,71,.2)!important}
.btn-s,.chip,.mpill,.pw-view,.nav-city select{background:#fff!important;color:var(--pw-primary)!important;border:1px solid var(--pw-neutral-200)!important;border-radius:14px!important;font-weight:600!important;box-shadow:none!important}
.chip:hover,.mpill:hover,.pw-view:hover,.btn-s:hover{background:var(--pw-secondary-soft)!important;border-color:#cfe0f2!important;color:var(--pw-primary)!important}
.send-btn{width:44px!important;height:44px!important;border-radius:14px!important}
.tile{border-radius:18px!important;padding:18px!important}
.tile-icon,.ch-icon,.topbar-icon,.topbar-avatar{border-radius:14px!important;background:var(--pw-secondary-soft)!important;color:var(--pw-primary)!important}
.badge-RAISE,.aq-uplift,.ep-impact{background:var(--pw-accent-soft)!important;color:var(--pw-accent)!important}
.badge-DISCOUNT{background:#fff7ed!important;color:#c2410c!important}
.badge-HOLD{background:#f3f4f6!important;color:#475569!important}
.wx-card{background:linear-gradient(135deg,#f5f9ff,#ebf3ff)!important}
.wx-title,.wx-desc,.wx-date,.wx-tip,.wx-rain{color:var(--pw-secondary)!important}
.wx-temp,.wx-dt{color:var(--pw-primary)!important}
.wx-day{background:rgba(255,255,255,.82)!important;border:1px solid rgba(79,139,201,.12)!important;border-radius:18px!important}
.msg-u{background:linear-gradient(135deg,var(--pw-primary),#163a66)!important;border-radius:18px 18px 6px 18px!important}
.msg-b{background:#f3f7fb!important;color:var(--pw-neutral-900)!important;border-radius:6px 18px 18px 18px!important}
.inp-card{padding:8px 8px 8px 18px!important}
.srch,.ff input,.ff select,.inp-card input{border-radius:12px!important}
.hero-pill,.hap{border-radius:999px!important}
.modal-box{padding:32px!important}
@media (max-width: 960px){
  .topbar h1{font-size:34px!important}
  .pw-hero-title{font-size:28px!important}
  .side-brand-tag{display:none}
}


/* Navy sidebar override */
.side-rail{
  background:linear-gradient(180deg,#0a1f44 0%,#102a5c 100%) !important;
  color:#ffffff !important;
  border-right:1px solid rgba(255,255,255,0.08) !important;
}
.side-brand-name,.side-sub,
.side-link,.side-link:visited,.side-link span,.side-link svg{
  color:#e8eefc !important;
}
.side-brand-tag{
  color:#ffffff !important;
}
.side-link{
  background:transparent !important;
}
.side-link:hover{
  background:rgba(255,255,255,0.08) !important;
  color:#ffffff !important;
}
.side-link.active{
  background:rgba(255,255,255,0.14) !important;
  color:#ffffff !important;
  box-shadow:inset 0 0 0 1px rgba(255,255,255,0.08) !important;
}
.side-brand-mark{
  background:rgba(255,255,255,0.12) !important;
  color:#ffffff !important;
}
 .side-brand-logo{
  width:44px;
  height:44px;
  object-fit:contain;
  display:block;
  flex-shrink:0;
}
.side-brand{
  display:flex;
  align-items:center;
  gap:12px;
}





/* === FINAL NAVY RESTORE: original PriceWise colors + personalized text only === */
:root{
  --pw-bg:#eef3f8;
  --pw-sidebar:#0a1f44;
  --pw-card:#ffffff;
  --pw-border:#d9e3ef;
  --pw-text:#0a1f44;
  --pw-sub:#5f7490;
  --pw-navy:#0a1f44;
  --pw-navy-2:#133a63;
  --pw-primary:#0a1f44;
  --pw-primary-strong:#071a39;
  --pw-secondary:#1a4f7f;
  --pw-secondary-soft:#eff6ff;
}
body{background:radial-gradient(circle at top left,#f8fbff 0%,#eef3f8 38%,#edf2f7 100%)!important;color:var(--pw-text)!important;}
.side-rail{
  background:linear-gradient(180deg,#0a1f44 0%,#102a5c 100%)!important;
  color:#ffffff!important;
  border-right:1px solid rgba(255,255,255,.08)!important;
  box-shadow:24px 0 60px rgba(15,23,42,.12)!important;
}
.side-brand-name,.side-sub,.side-link,.side-link:visited,.side-link span,.side-link svg{color:#e8eefc!important;}
.side-brand-tag{color:#cbd5f5!important;}
.side-user-card{border-bottom:1px solid rgba(255,255,255,.16)!important;}
.side-user-name{color:#ffffff!important;}
.side-user-meta{color:#cbd5f5!important;}
.side-link{background:transparent!important;color:#e8eefc!important;}
.side-link:hover{background:rgba(255,255,255,.08)!important;color:#ffffff!important;transform:translateX(2px)!important;}
.side-link.active{background:rgba(255,255,255,.14)!important;color:#ffffff!important;box-shadow:inset 0 0 0 1px rgba(255,255,255,.08)!important;}
.side-brand-mark{background:rgba(255,255,255,.12)!important;color:#ffffff!important;}
.app-main{background:transparent!important;}
.topbar h1{color:#0a1f44!important;}
.topbar p{color:#5f7490!important;}
#stats-bar .sc,.sc{background:linear-gradient(180deg,#ffffff,#fbfdff)!important;border:1px solid var(--pw-border)!important;box-shadow:0 16px 40px rgba(15,23,42,.06)!important;border-radius:24px!important;}
#stats-bar .sc-val,.sc-val{color:#0a1f44!important;}
#stats-bar .sc-sub,.sc-sub{color:#5f7490!important;}
.card{background:#ffffff!important;border:1px solid var(--pw-border)!important;box-shadow:0 18px 45px rgba(15,23,42,.06)!important;}


/* FORCE NAVY: Smart Pricing Assistant hero */
.chat-hero {
  background: linear-gradient(135deg,#0a1f44 0%,#0f2a5c 60%,#1b3a73 100%) !important;
  border: 1px solid rgba(147,197,253,.32) !important;
}
.chat-hero::before {
  background: radial-gradient(circle,rgba(147,197,253,.20),transparent 68%) !important;
}


/* --- BLUE/NAVY ACCENT OVERRIDES --- */
.chat-ai-badge {
    background: rgba(147,197,253,0.14) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.42) !important;
}
.chat-sub span {
    color: #bfdbfe !important;
    font-weight: 700 !important;
}
.nav-badge, .dc-pill {
    background: rgba(147,197,253,0.16) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.38) !important;
}
.eyebrow, .hero h1 .gold {
    color: #93c5fd !important;
}

</style>

<style id="font-force-override">
html, body,
body *, body *::before, body *::after{
  font-family: "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
  text-rendering: optimizeLegibility !important;
}

body, p, span, div, a, li, label, small, td, th, button, input, select, textarea{
  font-weight: 500 !important;
}

h1, h2, h3, h4, h5, h6,
strong, b,
.side-brand-name,
.side-link, .side-link span,
.kpi-value, .metric-value, .stat-value, .big-number,
.card-title, .section-title, .page-title,
.pw-signal-title, .pw-list-name, .sim-price, .headline{
  font-weight: 700 !important;
}

.side-brand-tag, .side-sub,
.sc-lbl, .cs2, .topbar p, .pw-list-meta, .pw-signal-sub, .sc-sub{
  font-weight: 600 !important;
}

.side-link.active{
  font-weight: 700 !important;
}


/* --- BLUE/NAVY ACCENT OVERRIDES --- */
.chat-ai-badge {
    background: rgba(147,197,253,0.14) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.42) !important;
}
.chat-sub span {
    color: #bfdbfe !important;
    font-weight: 700 !important;
}
.nav-badge, .dc-pill {
    background: rgba(147,197,253,0.16) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.38) !important;
}
.eyebrow, .hero h1 .gold {
    color: #93c5fd !important;
}

</style>

</head>
<body>
<div class="app-shell">
  <aside class="side-rail">
    <a href="/" class="side-brand" aria-label="PriceWise home">
      <img src="/static/logos.png" alt="PriceWise logo" class="side-brand-logo">
      <span class="side-brand-lockup">
        <span class="side-brand-name">PriceWise</span>
        <span class="side-brand-tag">Smart pricing intelligence</span>
      </span>
    </a>

    <div class="side-user-card">
      <div class="side-user-name">Sarah Chen</div>
      <div class="side-user-meta">Property manager · 7 units</div>
    </div>

    <div>
      <div class="side-sub">Overview</div>
      <div class="side-nav">
        <a class="side-link {{ 'active' if active_tab=='dashboard' else '' }}" href="/dashboard?tab=dashboard"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg><span>Dashboard</span></a>
        <a class="side-link {{ 'active' if active_tab=='explore' else '' }}" href="/dashboard?tab=explore"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg><span>Recommendations</span></a>
        <a class="side-link {{ 'active' if active_tab=='neighborhoods' else '' }}" href="/dashboard?tab=neighborhoods"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg><span>Market Signals</span></a>
        <a class="side-link {{ 'active' if active_tab=='calendar' else '' }}" href="/dashboard?tab=calendar"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg><span>Calendar</span></a>
        <a class="side-link {{ 'active' if active_tab=='listings' else '' }}" href="/dashboard?tab=listings"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg><span>Listings</span></a>
      </div>
    </div>

    <div>
      <div class="side-sub">Analytics</div>
      <div class="side-nav">
        <a class="side-link {{ 'active' if active_tab=='map' else '' }}" href="/dashboard?tab=map"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg><span>Map</span></a>
        <a class="side-link {{ 'active' if active_tab=='simulator' else '' }}" href="/dashboard?tab=simulator"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg><span>Simulator</span></a>
        <a class="side-link {{ 'active' if active_tab=='revenue' else '' }}" href="/dashboard?tab=revenue"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg><span>Revenue</span></a>
        <a class="side-link {{ 'active' if active_tab=='weather' else '' }}" href="/dashboard?tab=weather"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg><span>Weather</span></a>
        <a class="side-link {{ 'active' if active_tab=='chat' else '' }}" href="/dashboard?tab=chat"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg><span>Ask AI</span></a>
        <a class="side-link {{ 'active' if active_tab=='about_model' else '' }}" href="/dashboard?tab=about_model"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg><span>About Model</span></a>
      </div>
    </div>

  </aside>

  <div class="app-main">
    <div class="topbar">
      <div>
        <h1>{% if active_tab == 'dashboard' %}Good morning, Sarah{% elif active_tab == 'about_model' %}About Model{% else %}{{ active_tab.replace('_',' ')|title }}{% endif %}</h1>
        <p>{% if active_tab == 'dashboard' %}Your portfolio needs action today — 3 of your 7 properties are underpriced based on upcoming demand.{% elif active_tab == 'about_model' %}Sarah Chen’s portfolio action center: alerts, recommendations, revenue forecast, and event-driven insights.{% else %}Pricing intelligence platform for Washington DC.{% endif %}</p>
      </div>
      <div class="topbar-actions">
        <a class="topbar-add-property" href="/dashboard?tab=simulator">+ Add Property</a>
        <div class="topbar-avatar">DC</div>
      </div>
    </div>

    <div id="stats-bar">
      <div class="sc cgn">
        <div class="sc-top">
          <div class="sc-icon" style="background:#e0edff !important; color:#3b82f6 !important; border:1px solid #bfd8ff !important;" style="background:#ede9fe !important; color:#8b5cf6 !important; border:1px solid #d8b4fe !important;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M7 8h10"/><path d="M7 12h10"/><path d="M7 16h6"/></svg>
          </div>
          <div class="sc-lbl">{% if active_tab == 'dashboard' %}Portfolio avg/night{% else %}Total Listings{% endif %}</div>
        </div>
        <div class="sc-body no-viz">
          <div class="sc-copy">
            <div class="sc-val">{% if active_tab == 'dashboard' %}${{ portfolio_mockup.avg_night if portfolio_mockup.avg_night is defined else 189 }}{% else %}4161{% endif %}</div>
            <div class="sc-sub">{% if active_tab == 'dashboard' %}$18 from last week{% else %}Properties currently monitored{% endif %}</div>
          </div>
        </div>
      </div>
      <div class="sc cb">
        <div class="sc-top">
          <div class="sc-icon" style="background:#dcfce7 !important; color:#22c55e !important; border:1px solid #bbf7d0 !important;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>
          </div>
          <div class="sc-lbl">{% if active_tab == 'dashboard' %}Occupancy this month{% else %}Avg. Occupancy{% endif %}</div>
        </div>
        <div class="sc-body no-viz">
          <div class="sc-copy">
            <div class="sc-val">{% if active_tab == 'dashboard' %}74%{% else %}78%{% endif %}</div>
            <div class="sc-sub">{% if active_tab == 'dashboard' %}+6 pts vs last month{% else %}Estimated market occupancy{% endif %}</div>
          </div>
        </div>
      </div>
      <div class="sc cs">
        <div class="sc-top">
          <div class="sc-icon" style="background:#ede9fe !important; color:#8b5cf6 !important; border:1px solid #d8b4fe !important;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1v22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          </div>
          <div class="sc-lbl">{% if active_tab == 'dashboard' %}Revenue opportunity{% else %}Projected Revenue{% endif %}</div>
        </div>
        <div class="sc-body no-viz">
          <div class="sc-copy">
            <div class="sc-val">{% if active_tab == 'dashboard' %}$2,580{% else %}$1,848,707{% endif %}</div>
            <div class="sc-sub">{% if active_tab == 'dashboard' %}if 3 recs applied{% else %}This month from pricing adjustments{% endif %}</div>
          </div>
        </div>
      </div>
    </div>

    <main>
<div id="pane-dashboard" class="tab-pane {{ 'active' if active_tab=='dashboard' }}">


  <style>
  /* Navy personalized dashboard: keep the original PriceWise navy feel, only personalize the content */
  #maya-demo-card {
    background: #ffffff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 24px !important;
    box-shadow: 0 18px 45px rgba(15, 39, 71, .10) !important;
    overflow: hidden !important;
    position: relative !important;
  }
  #maya-demo-card::before,
  #maya-demo-card::after { display: none !important; }
  #maya-demo-card > div:first-child {
    background: linear-gradient(135deg,#0a1f44 0%,#133a63 60%,#1a4f7f 100%) !important;
    border-bottom: 1px solid rgba(191,219,254,.35) !important;
    color: #ffffff !important;
  }
  #maya-demo-card > div:first-child h2,
  #maya-demo-card > div:first-child div,
  #maya-demo-card > div:first-child p {
    color: #ffffff !important;
    text-shadow: none !important;
  }
  #maya-demo-card > div:first-child > div:first-child > div:first-child {
    color: #bfdbfe !important;
  }
  #maya-demo-card > div:first-child > div:last-child {
    background: rgba(255,255,255,.13) !important;
    border: 1px solid rgba(255,255,255,.22) !important;
    color: #ffffff !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.08) !important;
  }
  #maya-demo-card .cb2 {
    background: #ffffff !important;
  }
  #maya-demo-card [style*="background:#ffffff"],
  #maya-demo-card [style*="background:white"],
  #maya-demo-card [style*="background:linear-gradient(180deg"] {
    background: #ffffff !important;
    border-color: #dbe4ef !important;
    box-shadow: 0 10px 24px rgba(15,39,71,.06) !important;
    backdrop-filter: none !important;
  }
  #maya-demo-card [style*="color:#0f172a"] { color: #0a1f44 !important; }
  #maya-demo-card [style*="color:#64748b"],
  #maya-demo-card [style*="color:#475569"] { color: #5f7490 !important; }
  #maya-demo-card [style*="color:#2563eb"] { color: #2563eb !important; }
  #maya-demo-card span[style*="background:#dcfce7"] {
    color: #15803d !important;
    background: #dcfce7 !important;
    border-color: #bbf7d0 !important;
  }
  #maya-demo-card span[style*="background:#ffedd5"] {
    color: #c2410c !important;
    background: #ffedd5 !important;
    border-color: #fed7aa !important;
  }
  #maya-demo-card span[style*="background:#f1f5f9"] {
    color: #475569 !important;
    background: #f1f5f9 !important;
    border-color: #e2e8f0 !important;
  }

  .pw-action-split-card{height:100%;display:flex;flex-direction:column;}
  .pw-action-split-card .cb2{flex:1;}


  /* === Recommended Action Split: keep the card, make the whole section white === */
  .pw-action-split-card,
  .pw-action-split-card .ch,
  .pw-action-split-card .cb2,
  .pw-action-split-card img {
    background: #ffffff !important;
  }
  .pw-action-split-card {
    border: 1px solid #bfdbfe !important;
    border-radius: 22px !important;
    box-shadow: 0 18px 44px rgba(15, 39, 71, .08) !important;
    overflow: hidden !important;
  }
  .pw-action-split-card .cb2 {
    background-image: none !important;
  }
  @media (max-width: 960px){
    .pw-dashboard-top{grid-template-columns:1fr!important;}
  }

  /* FINAL OVERRIDE: make the full Recommendations for You block navy */
  #maya-demo-card,
  #maya-demo-card .cb2 {
    background: linear-gradient(135deg,#0a1f44 0%,#0f2a5c 60%,#1b3a73 100%) !important;
    color: #ffffff !important;
  }
  #maya-demo-card * {
    opacity: 1 !important;
  }


/* --- BLUE/NAVY ACCENT OVERRIDES --- */
.chat-ai-badge {
    background: rgba(147,197,253,0.14) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.42) !important;
}
.chat-sub span {
    color: #bfdbfe !important;
    font-weight: 700 !important;
}
.nav-badge, .dc-pill {
    background: rgba(147,197,253,0.16) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.38) !important;
}
.eyebrow, .hero h1 .gold {
    color: #93c5fd !important;
}

</style>

  <!-- HIGHLIGHTED ASK AI PORTFOLIO ASSISTANT -->
  <div class="card" style="margin-bottom:22px;border:1px solid #bfdbfe;border-radius:22px;overflow:hidden;background:linear-gradient(135deg,#0a1f44 0%,#0f2a5c 60%,#1b3a73 100%);box-shadow:0 18px 44px rgba(15,39,71,.14);color:#ffffff;">
    <div style="padding:22px 26px;display:grid;grid-template-columns:1.05fr 1.4fr;gap:18px;align-items:center;">
      <div>
        <div style="border:2px solid #93c5fd;background:#f8fbff;box-shadow:0 10px 28px rgba(37,99,235,0.08);border-radius:22px;padding:22px;font-size:12px;font-weight:950;letter-spacing:.1em;text-transform:uppercase;color:#60a5fa;margin-bottom:8px;text-shadow:none;display:inline-block;">
          Ask AI Portfolio Manager
        </div>
        <p style="margin:8px 0 0;color:#16345c;font-size:16px;font-weight:800;line-height:1.55;text-shadow:none;max-width:720px;">Use the assistant to explain recommendations, compare properties, and show portfolio-specific actions.</p>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end;">
        <a href="/dashboard?tab=chat" style="text-decoration:none;background:#ffffff;color:#0f2b55;border-radius:999px;padding:12px 15px;font-weight:950;font-size:13px;box-shadow:0 10px 25px rgba(0,0,0,.14);"><div style="
  background:#eff6ff;
  color:#2563eb;
  padding:10px 16px;
  border-radius:999px;
  font-weight:600;
  font-size:14px;
  border:1px solid #dbeafe;
  box-shadow:0 4px 12px rgba(37,99,235,0.1);
">
  Which properties should I raise?
</div></a>
        <a href="/dashboard?tab=chat" style="text-decoration:none;background:rgba(255,255,255,.14);color:#ffffff;border:1px solid rgba(255,255,255,.28);border-radius:999px;padding:12px 15px;font-weight:950;font-size:13px;">What’s driving demand?</a>
        <a href="/dashboard?tab=chat" style="text-decoration:none;background:rgba(255,255,255,.14);color:#ffffff;border:1px solid rgba(255,255,255,.28);border-radius:999px;padding:12px 15px;font-weight:950;font-size:13px;">Show my revenue upside</a>
      </div>
    </div>
  </div>

  <!-- PERSONALIZED NAVY RECOMMENDATIONS BLOCK -->
  <div id="maya-demo-card" class="card" style="
      margin-bottom:22px;
      border:1px solid rgba(147,197,253,.32);
      border-radius:26px;
      overflow:hidden;
      background:linear-gradient(135deg,#0a1f44 0%,#0f2a5c 60%,#1b3a73 100%);
      box-shadow:0 22px 60px rgba(15,39,71,.24);
      color:#ffffff;
      position:relative;
  ">
    <div style="
        position:absolute;
        right:-70px;
        top:-100px;
        width:280px;
        height:280px;
        border-radius:999px;
        background:rgba(255,255,255,.10);
        pointer-events:none;
    "></div>
    <div style="
        position:absolute;
        left:45%;
        bottom:-150px;
        width:320px;
        height:320px;
        border-radius:999px;
        background:rgba(255,255,255,.08);
        pointer-events:none;
    "></div>

    <div style="
        padding:26px 30px 24px 30px;
        border-bottom:1px solid rgba(255,255,255,.16);
        display:flex;
        justify-content:space-between;
        gap:18px;
        align-items:flex-start;
        flex-wrap:wrap;
        position:relative;
        z-index:1;
    ">
      <div>
        <div style="
            font-size:12px;
            font-weight:950;
            color:#b9cce4;
            text-transform:uppercase;
            letter-spacing:.10em;
            margin-bottom:6px;
        ">YOUR PORTFOLIO COMMAND CENTER</div>

        <div style="
            font-size:32px;
            line-height:1.05;
            font-weight:950;
            color:#ffffff;
            letter-spacing:-.035em;
        ">Today’s Recommendations</div>

        <div style="
            font-size:14px;
            color:#c6d5e8;
            font-weight:800;
            margin-top:6px;
        ">Insights for your 7 DC properties — powered by events, comps, and booking signals</div>
      </div>

      <div style="
          padding:10px 16px;
          border-radius:999px;
          background:rgba(255,255,255,.14);
          border:1px solid rgba(255,255,255,.24);
          color:#ffffff;
          font-weight:950;
          font-size:13px;
          box-shadow:inset 0 1px 0 rgba(255,255,255,.08);
          white-space:nowrap;
      ">🧠 AI is monitoring portfolio</div>
    </div>

    <div style="padding:24px 28px 28px 28px;position:relative;z-index:1">
      <div style="margin-bottom:18px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.22);border-radius:18px;padding:18px;box-shadow:inset 0 1px 0 rgba(255,255,255,.06);">
        <div style="font-size:12px;font-weight:950;color:#9ed0ff;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Priority insight</div>
        <div style="font-size:21px;font-weight:950;color:#ffffff;letter-spacing:-.02em;">Adams Morgan opportunity coming up</div>
        <p style="margin:8px 0 0;color:#d0deee;font-size:13.5px;font-weight:750;line-height:1.55;">A major concert weekend is expected to lift demand near your Adams Morgan listings. Comparable units are already moving prices up, so the app recommends raising selected listings 15–22% about three weeks before the demand spike.</p>
      </div>

      <div style="
          display:grid;
          grid-template-columns:minmax(0,1.2fr) minmax(340px,.9fr);
          gap:18px;
          align-items:start;
      ">
        <div style="
            background:rgba(255,255,255,.08);
            border:1px solid rgba(255,255,255,.18);
            border-radius:18px;
            padding:18px;
            box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
            color:#ffffff;
        ">
          <div style="display:flex;justify-content:space-between;align-items:end;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,.14);padding-bottom:12px">
            <div>
              <div style="font-size:18px;font-weight:950;color:#ffffff">Portfolio properties</div>
              <div style="font-size:12px;color:#c6d5e8;font-weight:800;margin-top:3px">Property-specific pricing decisions, not generic market advice</div>
            </div>
            <div style="font-size:12px;color:#c6d5e8;font-weight:900">Current → Recommended</div>
          </div>

          {% for p in portfolio_mockup.properties %}
          <div style="
              display:grid;
              grid-template-columns:1fr auto;
              gap:16px;
              align-items:center;
              padding:12px 0;
              border-top:{% if loop.first %}0{% else %}1px solid rgba(255,255,255,.13){% endif %};
          ">
            <div style="min-width:0">
              <div style="font-weight:950;color:#ffffff;font-size:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ p.name }}</div>
              <div style="font-size:12px;color:#c6d5e8;font-weight:800;margin-top:3px">{{ p.area }} · {{ p.units }} unit{% if p.units != 1 %}s{% endif %} · {{ p.signal }}</div>
            </div>
            <div style="text-align:right;min-width:130px">
              <div style="font-weight:950;color:#ffffff;font-size:16px">${{ p.now }} → ${{ p.next }}</div>
              <span style="display:inline-block;margin-top:5px;font-size:11px;font-weight:950;border-radius:999px;padding:5px 9px;{% if p.action == 'RAISE' %}color:#15803d;background:#bbf7d0;border:1px solid #86efac{% elif p.action == 'DISCOUNT' %}color:#c2410c;background:#fed7aa;border:1px solid #fdba74{% else %}color:#334155;background:#e2e8f0;border:1px solid #cbd5e1{% endif %}">{{ p.action }}</span>
            </div>
          </div>
          {% endfor %}
        </div>

        <div style="
            background:rgba(255,255,255,.09);
            border:1px solid rgba(255,255,255,.18);
            border-radius:18px;
            padding:18px;
            box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
            color:#ffffff;
        ">
          <div style="font-size:18px;font-weight:950;color:#ffffff">Why recommendations are changing</div>
          <div style="font-size:12px;color:#c6d5e8;font-weight:800;margin:3px 0 14px">The app reacts to your booking pace, upcoming DC events, and nearby comparable listings.</div>

          {% for r in portfolio_mockup.timeline %}
          <div style="
              position:relative;
              padding:15px 15px 15px 17px;
              margin-bottom:12px;
              background:rgba(255,255,255,.16);
              border:1px solid rgba(255,255,255,.18);
              border-radius:15px;
              box-shadow:0 10px 22px rgba(0,0,0,.10);
          ">
            <div style="display:flex;justify-content:space-between;gap:10px;margin-bottom:6px">
              <div style="font-size:11px;font-weight:950;color:#9ed0ff;text-transform:uppercase;letter-spacing:.04em">{{ r.label }}</div>
              <div style="font-size:11px;color:#d7e4f4;font-weight:900">{{ r.time }}</div>
            </div>
            <div style="font-weight:950;color:#ffffff;font-size:16px">{{ r.title }}</div>
            <p style="font-size:12.5px;color:#d0deee;font-weight:750;line-height:1.5;margin:7px 0">{{ r.why }}</p>
            <div style="font-size:12px;font-weight:950;color:#ffffff">{{ r.impact }}</div>
          </div>
          {% endfor %}
        </div>
      </div>
    </div>
  </div>



  <!-- DASHBOARD EXPANSION: Sarah-only market signals, listing opportunities, and demand events -->
  <style>
    .sarah-dash-grid{display:grid;grid-template-columns:1.08fr .92fr;gap:18px;margin:0 0 22px 0;}
    .sarah-dash-card{background:#ffffff;border:1px solid #dbe7f3;border-radius:22px;box-shadow:0 16px 38px rgba(15,39,71,.08);overflow:hidden;}
    .sarah-dash-head{padding:18px 20px;border-bottom:1px solid #e6eef7;display:flex;align-items:flex-start;justify-content:space-between;gap:14px;}
    .sarah-dash-title{font-size:19px;font-weight:950;color:#0a1f44;letter-spacing:-.02em;}
    .sarah-dash-sub{font-size:12.5px;color:#64748b;font-weight:750;margin-top:3px;line-height:1.4;}
    .sarah-dash-body{padding:18px 20px;}
    .sarah-signal-row{display:grid;grid-template-columns:46px 1fr auto;gap:12px;align-items:center;padding:13px 0;border-top:1px solid #edf2f7;}
    .sarah-signal-row:first-child{border-top:0;padding-top:0;}
    .sarah-icon{width:42px;height:42px;border-radius:14px;display:grid;place-items:center;font-size:19px;background:#eff6ff;border:1px solid #bfdbfe;}
    .sarah-row-title{font-size:14.5px;font-weight:950;color:#0f2747;line-height:1.22;}
    .sarah-row-meta{font-size:12px;color:#64748b;font-weight:750;margin-top:3px;line-height:1.35;}
    .sarah-pill{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:6px 10px;font-size:11px;font-weight:950;white-space:nowrap;border:1px solid transparent;}
    .sarah-pill.raise{background:#dcfce7;color:#15803d;border-color:#bbf7d0;}
    .sarah-pill.discount{background:#ffedd5;color:#c2410c;border-color:#fed7aa;}
    .sarah-pill.hold{background:#f1f5f9;color:#475569;border-color:#e2e8f0;}
    .sarah-listing-card{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center;padding:14px 0;border-top:1px solid #edf2f7;}
    .sarah-listing-card:first-child{border-top:0;padding-top:0;}
    .sarah-price{font-size:18px;font-weight:950;color:#0a1f44;text-align:right;}
    .sarah-gain{font-size:12px;font-weight:900;color:#15803d;text-align:right;margin-top:3px;}
    .sarah-event-card{border:1px solid #e6eef7;background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);border-radius:18px;padding:15px;margin-bottom:12px;}
    .sarah-event-top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px;}
    .sarah-event-name{font-size:15px;font-weight:950;color:#0a1f44;line-height:1.25;}
    .sarah-event-date{font-size:11px;font-weight:950;color:#1d4ed8;background:#dbeafe;border:1px solid #bfdbfe;border-radius:999px;padding:5px 9px;white-space:nowrap;}
    .sarah-event-note{font-size:12.5px;color:#64748b;font-weight:750;line-height:1.45;margin:0;}
    .sarah-summary-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0 0 18px 0;}
    .sarah-summary-kpi{background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);border:1px solid #dbe7f3;border-radius:18px;padding:15px 16px;box-shadow:0 10px 26px rgba(15,39,71,.06);}
    .sarah-summary-kpi .v{font-size:24px;font-weight:950;color:#0a1f44;letter-spacing:-.03em;}
    .sarah-summary-kpi .l{font-size:11.5px;color:#64748b;font-weight:850;margin-top:3px;line-height:1.35;}

    .sarah-listing-revenue-grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(330px,.85fr);gap:18px;align-items:stretch;margin-bottom:0;}
    .sarah-revenue-card{height:calc(100% - 22px);}
    .sarah-revenue-kpis{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}
    .sarah-revenue-kpis>div{border:1px solid #e6eef7;background:#f8fbff;border-radius:16px;padding:12px;}
    .sarah-rev-value{font-size:22px;font-weight:950;color:#0a1f44;letter-spacing:-.03em;}
    .sarah-rev-label{font-size:11.5px;color:#64748b;font-weight:850;margin-top:2px;}
    .sarah-forecast-chart{display:grid;gap:12px;margin:12px 0 12px;}
    .sarah-chart-row{display:grid;grid-template-columns:52px 1fr 46px;gap:10px;align-items:center;font-size:12px;font-weight:850;color:#40546b;}
    .sarah-chart-row em{font-style:normal;text-align:right;color:#0a1f44;font-weight:950;}
    .sarah-bar-track{height:20px;background:#edf4fb;border-radius:999px;position:relative;overflow:hidden;border:1px solid #dbe7f3;}
    .sarah-bar-track i,.sarah-bar-track b{position:absolute;left:0;top:0;height:100%;border-radius:999px;display:block;}
    .sarah-bar-track i.actual{background:#cbd5e1;z-index:1;}
    .sarah-bar-track b.forecast{background:linear-gradient(90deg,#2563eb,#22c55e);opacity:.82;z-index:2;height:9px;top:5px;}
    .sarah-chart-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:#64748b;font-weight:850;margin-top:8px;}
    .sarah-chart-legend span{display:inline-flex;align-items:center;gap:6px;}
    .sarah-chart-legend i{width:10px;height:10px;border-radius:999px;display:inline-block;}
    .legend-actual{background:#cbd5e1;}.legend-forecast{background:linear-gradient(90deg,#2563eb,#22c55e);}
    .sarah-rev-note{margin:14px 0 0;padding:12px 13px;border-radius:14px;background:#ecfdf5;border:1px solid #bbf7d0;color:#166534;font-size:12.5px;font-weight:850;line-height:1.45;}
    @media (max-width: 960px){.sarah-dash-grid,.sarah-summary-strip,.sarah-listing-revenue-grid{grid-template-columns:1fr!important;}.sarah-signal-row{grid-template-columns:42px 1fr;}.sarah-signal-row .sarah-pill{grid-column:2;justify-self:flex-start;}.sarah-price,.sarah-gain{text-align:left;}.sarah-revenue-card{height:auto;}}
  

/* --- BLUE/NAVY ACCENT OVERRIDES --- */
.chat-ai-badge {
    background: rgba(147,197,253,0.14) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.42) !important;
}
.chat-sub span {
    color: #bfdbfe !important;
    font-weight: 700 !important;
}
.nav-badge, .dc-pill {
    background: rgba(147,197,253,0.16) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.38) !important;
}
.eyebrow, .hero h1 .gold {
    color: #93c5fd !important;
}

</style>

  <div class="sarah-summary-strip">
    <div class="sarah-summary-kpi"><div class="v">4</div><div class="l">Listings to raise before demand peaks</div></div>
    <div class="sarah-summary-kpi"><div class="v">+$412</div><div class="l">Projected weekly upside from first changes</div></div>
    <div class="sarah-summary-kpi"><div class="v">3 weeks</div><div class="l">Best lead time for event-driven price changes</div></div>
  </div>

  <div class="sarah-dash-grid">
    <div class="sarah-dash-card">
      <div class="sarah-dash-head">
        <div><div class="sarah-dash-title">Market Signals</div><div class="sarah-dash-sub">What is moving demand around your DC portfolio right now.</div></div>
        <span class="sarah-pill raise">Live scan</span>
      </div>
      <div class="sarah-dash-body">
        <div class="sarah-signal-row"><div class="sarah-icon">🎵</div><div><div class="sarah-row-title">Concert weekend lifting Adams Morgan demand</div><div class="sarah-row-meta">Your Adams Morgan Flat is near event-driven demand. Raise early if booking pace holds.</div></div><span class="sarah-pill raise">Raise 15–22%</span></div>
        <div class="sarah-signal-row"><div class="sarah-icon">⚾</div><div><div class="sarah-row-title">Nationals game pressure near Navy Yard</div><div class="sarah-row-meta">Navy Yard Studio is under nearby comps for high-demand dates.</div></div><span class="sarah-pill raise">Raise $16</span></div>
        <div class="sarah-signal-row"><div class="sarah-icon">🏛️</div><div><div class="sarah-row-title">Georgetown is stable but price-sensitive</div><div class="sarah-row-meta">Your Georgetown 2BR is close to the top of its comp band, so avoid overcorrecting.</div></div><span class="sarah-pill hold">Hold</span></div>
      </div>
    </div>

    <div class="sarah-dash-card">
      <div class="sarah-dash-head"><div><div class="sarah-dash-title">Upcoming Demand Events</div><div class="sarah-dash-sub">When should you change prices?</div></div></div>
      <div class="sarah-dash-body">
        <div class="sarah-event-card"><div class="sarah-event-top"><div class="sarah-event-name">Major concert weekend</div><div class="sarah-event-date">21 days out</div></div><p class="sarah-event-note">Start raising Adams Morgan and Shaw listings now; watch bookings daily and cap increases if conversion slows.</p></div>
        <div class="sarah-event-card"><div class="sarah-event-top"><div class="sarah-event-name">Nationals home game cluster</div><div class="sarah-event-date">10–14 days out</div></div><p class="sarah-event-note">Raise Navy Yard Studio first because it has the clearest location advantage and room to move versus comps.</p></div>
        <div class="sarah-event-card" style="margin-bottom:0"><div class="sarah-event-top"><div class="sarah-event-name">Weekend leisure demand</div><div class="sarah-event-date">This week</div></div><p class="sarah-event-note">Hold Georgetown 2BR and Logan Circle Condo unless bookings accelerate; they are already near fair market price.</p></div>
      </div>
    </div>
  </div>

  <div class="sarah-listing-revenue-grid">
    <div class="sarah-dash-card" style="margin-bottom:22px;">
      <div class="sarah-dash-head"><div><div class="sarah-dash-title">Listings — Top pricing opportunities right now</div><div class="sarah-dash-sub">Your highest-priority listing changes.</div></div><a href="/dashboard?tab=listings" class="sarah-pill hold" style="text-decoration:none;">Open Listings tab</a></div>
      <div class="sarah-dash-body">
        <div class="sarah-listing-card"><div><div class="sarah-row-title">Navy Yard Studio</div><div class="sarah-row-meta">Raise before Nationals game cluster · under closest comps</div></div><div><div class="sarah-price">$142 → $158</div><div class="sarah-gain">+$112 weekly upside</div></div></div>
        <div class="sarah-listing-card"><div><div class="sarah-row-title">Capitol Hill Studio</div><div class="sarah-row-meta">Two-unit opportunity · nearby comps already moved up</div></div><div><div class="sarah-price">$310 → $336</div><div class="sarah-gain">+$208 weekly upside</div></div></div>
        <div class="sarah-listing-card"><div><div class="sarah-row-title">Dupont Garden Apt</div><div class="sarah-row-meta">Strong weekend demand, but keep increase moderate to protect conversion</div></div><div><div class="sarah-price">$221 → $246</div><div class="sarah-gain">+$175 weekly upside</div></div></div>
        <div class="sarah-listing-card"><div><div class="sarah-row-title">Adams Morgan Flat</div><div class="sarah-row-meta">Discount short gap now; raise again as concert weekend gets closer</div></div><div><div class="sarah-price">$165 → $152</div><div class="sarah-gain" style="color:#c2410c">Fill 4-day gap</div></div></div>
      </div>
    </div>

    <div class="sarah-dash-card sarah-revenue-card" style="margin-bottom:22px;">
      <div class="sarah-dash-head">
        <div>
          <div class="sarah-dash-title">Revenue Forecast</div>
          <div class="sarah-dash-sub">Forecast vs actual chart — projected revenue.</div>
        </div>
        <span class="sarah-pill raise">+$1.8K/mo</span>
      </div>
      <div class="sarah-dash-body">
        <div class="sarah-revenue-kpis">
          <div><div class="sarah-rev-value">$8.9K</div><div class="sarah-rev-label">Projected month</div></div>
          <div><div class="sarah-rev-value">$7.1K</div><div class="sarah-rev-label">Current pace</div></div>
        </div>
        <div class="sarah-forecast-chart" aria-label="Forecast vs actual revenue chart">
          <div class="sarah-chart-row"><span>Week 1</span><div class="sarah-bar-track"><i class="actual" style="width:58%"></i><b class="forecast" style="width:69%"></b></div><em>$1.7K</em></div>
          <div class="sarah-chart-row"><span>Week 2</span><div class="sarah-bar-track"><i class="actual" style="width:62%"></i><b class="forecast" style="width:76%"></b></div><em>$1.9K</em></div>
          <div class="sarah-chart-row"><span>Week 3</span><div class="sarah-bar-track"><i class="actual" style="width:64%"></i><b class="forecast" style="width:86%"></b></div><em>$2.2K</em></div>
          <div class="sarah-chart-row"><span>Week 4</span><div class="sarah-bar-track"><i class="actual" style="width:67%"></i><b class="forecast" style="width:94%"></b></div><em>$2.4K</em></div>
        </div>
        <div class="sarah-chart-legend"><span><i class="legend-actual"></i>Actual/current pace</span><span><i class="legend-forecast"></i>Forecast after updates</span></div>
        <p class="sarah-rev-note">Your forecast improves most after raising Navy Yard, Capitol Hill, and Dupont before event demand peaks.</p>
      </div>
    </div>
  </div>

  <!-- Dashboard-only Sarah portfolio/event/action/listing recommendation blocks removed per request. Other tabs still keep their full functionality. -->
</div>
<div id="pane-map" class="tab-pane {{ 'active' if active_tab=='map' }}">
  <div class="card" style="overflow:hidden">
    <div class="map-pills">
      <span style="font-size:12px;font-weight:600;color:var(--text2);margin-right:4px">Layers:</span>
      <button class="mpill active" data-layer="Raise" onclick="toggleMapLayer(this)"><span class="mpill-dot" style="background:#22c55e"></span>Raise</button>
      <button class="mpill active" data-layer="Discount" onclick="toggleMapLayer(this)"><span class="mpill-dot" style="background:#f97316"></span>Discount</button>
      <button class="mpill active" data-layer="Hold" onclick="toggleMapLayer(this)"><span class="mpill-dot" style="background:#ef4444"></span>Hold</button>
      <button class="mpill active" data-layer="Events" onclick="toggleMapLayer(this)"><span class="mpill-dot" style="background:#8b5cf6"></span>Events</button>
    </div>
    <div style="position:relative;height:600px">
      <div id="map-spinner" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:var(--bg);z-index:10;font-size:14px;color:var(--text2);transition:opacity .3s">Loading map\u2026</div>
      <iframe id="map-frame" src="/static/map.html" style="width:100%;height:100%;border:none" onload="hideMapSpinner()"></iframe>
    </div>
  </div>
</div>
<div id="pane-calendar" class="tab-pane {{ 'active' if active_tab=='calendar' }}">
  <div class="card">
    <div class="ch"><div><div class="ct">Event Calendar \u2014 {{ selected_city }}</div><div class="cs2">All upcoming events driving demand</div></div></div>
    <div class="cb2">
      {% if calendar_cards %}
      <div class="cal-grid">
        {% for ev in calendar_cards %}
        <div class="ep">
          <div class="ep-icon" style="font-size:18px;width:40px;height:40px;border-radius:11px">{{ ev.icon }}</div>
          <div style="flex:1;min-width:0"><div class="ep-name">{{ ev.name }}</div><div class="ep-meta">{{ ev.date_range }}</div><div class="ep-meta" style="margin-top:1px">{{ ev.venue }}</div></div>
          <div style="text-align:right;flex-shrink:0"><div class="ep-impact">{{ ev.impact }}</div><div style="font-size:10px;color:var(--text3);margin-top:4px">{{ ev.category }}</div></div>
        </div>
        {% endfor %}
      </div>
      {% else %}<div class="empty-state"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M22 12h-6l-2 3H10l-2-3H2"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 17.31 4H6.69a2 2 0 0 0-1.78 1.11z"/></svg><p>No events found for {{ selected_city }}.</p></div>{% endif %}
    </div>
  </div>
</div>
<div id="pane-listings" class="tab-pane {{ 'active' if active_tab=='listings' }}">
  <div class="card">
    <div class="ch">
      <div><div class="ct">Top Listings \u2014 {{ selected_city }}</div><div class="cs2">Click a row for details \u00b7 Click a column header to sort</div></div>
      <a href="/export/listings?city={{ selected_city }}" class="btn-export"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Export CSV</a>
    </div>
    <div class="cb2">
      <div class="srch-wrap">
        <svg class="srch-ic" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" class="srch" placeholder="Search listings\u2026" oninput="filterListings(this.value)">
      </div>
      <div style="overflow-x:auto">
        <table class="tbl" id="listings-table">
          <thead><tr>
            <th onclick="sortTable('listings-table',0,this)">Name <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('listings-table',1,this)">Type <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('listings-table',2,this)">Beds <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('listings-table',3,this)">Rating <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('listings-table',4,this)">Current <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('listings-table',5,this)">Recommended <span class="sort-ic">\u2195</span></th>
            <th style="cursor:default">Action</th>
          </tr></thead>
          <tbody>
          {% for r in listings_preview %}
          <tr onclick="showListingModal({{ loop.index0 }})" style="cursor:pointer">
            <td style="font-weight:500;max-width:240px;white-space:normal;overflow:hidden">
              <div>{{ r.name }}</div>
              <div style="font-size:11px;color:var(--text3);margin-top:2px">
                Event {% if r.get('distance_to_event_km') is not none %}{{ "%.1f"|format(r.get('distance_to_event_km')) }}km{% else %}nearby{% endif %} away: {{ r.get('nearest_event', 'Upcoming demand event') }}
              </div>
            </td>
            <td style="color:var(--text2)">{{ r.room_type }}</td>
            <td>{{ r.bedrooms }}</td>
            <td>\u2b50 {{ r.rating }}</td>
            <td>${{ r.current }}</td>
            <td style="font-weight:700">${{ r.recommended }}</td>
            <td><span class="badge badge-{{ r.action }}">{% if r.action=='RAISE' %}\u2191 Raise{% elif r.action=='DISCOUNT' %}\u2193 Discount{% else %}\u2014 Hold{% endif %}</span></td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
<div id="pane-explore" class="tab-pane {{ 'active' if active_tab=='explore' }}">
  <div class="card">
    <div class="ch"><div class="ct">Filter Listings</div></div>
    <div class="cb2">
      <form method="POST" action="/explore">
        <input type="hidden" name="city" value="{{ selected_city }}">

        <div class="filter-row">
          <div class="ff"><label class="fl">Neighbourhood</label><select name="hood" class="fsel" style="min-width:230px"><option value="" {{ 'selected' if not hood }}>All neighborhoods</option>{% for h in hood_options %}<option value="{{ h }}" {{ 'selected' if hood == h }}>{{ h }}</option>{% endfor %}</select></div>
          <div class="ff"><label class="fl">Min Price \u2014 $<span id="pv">{{ min_price or 0 }}</span></label><input type="range" name="min_price" min="0" max="500" value="{{ min_price or 0 }}" oninput="document.getElementById('pv').textContent=this.value" style="width:180px;margin-top:6px;cursor:pointer"></div>
          <div class="ff"><label class="fl">Action</label><select name="action_filter" class="fsel"><option value="">All</option><option value="RAISE" {{ 'selected' if action_filter=='RAISE' }}>RAISE</option><option value="DISCOUNT" {{ 'selected' if action_filter=='DISCOUNT' }}>DISCOUNT</option><option value="HOLD" {{ 'selected' if action_filter=='HOLD' }}>HOLD</option></select></div>
          <div class="ff"><label class="fl">Bedrooms</label><select name="bedrooms_filter" class="fsel"><option value="" {{ 'selected' if not bedrooms_filter }}>Any</option><option value="1" {{ 'selected' if bedrooms_filter=='1' }}>1</option><option value="2" {{ 'selected' if bedrooms_filter=='2' }}>2</option><option value="3" {{ 'selected' if bedrooms_filter=='3' }}>3</option><option value="4" {{ 'selected' if bedrooms_filter=='4' }}>4+</option></select></div>
          <div class="ff"><label class="fl">Room Type</label><select name="room_type_filter" class="fsel"><option value="" {{ 'selected' if not room_type_filter }}>Any</option><option value="Entire home/apt" {{ 'selected' if room_type_filter=='Entire home/apt' }}>Entire Home</option><option value="Private room" {{ 'selected' if room_type_filter=='Private room' }}>Private Room</option><option value="Shared room" {{ 'selected' if room_type_filter=='Shared room' }}>Shared Room</option><option value="Hotel room" {{ 'selected' if room_type_filter=='Hotel room' }}>Hotel Room</option></select></div>
          <button class="btn-p" type="submit" style="height:38px;align-self:flex-end">Filter</button>
        </div>
      </form>
      {% if explore_results is not none %}
      <p style="font-size:12px;color:var(--text2);margin-bottom:14px">{{ explore_count }} listings found</p>
      {% for r in explore_results %}
      <div class="exp-card">
        <div style="flex:1;min-width:0">
          <div style="font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ r.name[:40] }}</div>
          <div style="font-size:11px;color:var(--text2);margin-top:3px">{{ r.neighbourhood_cleansed[:30] }} \u00b7 {{ r.get('city','') }}</div>
          <div style="font-size:11px;color:var(--text3);margin-top:2px">
            Event {% if r.get('distance_to_event_km') is not none %}{{ "%.1f"|format(r.get('distance_to_event_km')) }}km{% else %}nearby{% endif %} away: {{ r.get('nearest_event', 'Upcoming demand event') }}
          </div>
        </div>
        <div style="text-align:right;flex-shrink:0">
          <div style="font-size:12.5px;color:var(--text2)">${{ "%.0f"|format(r.baseline_price) }} \u2192 <strong style="color:var(--text)">${{ "%.0f"|format(r.recommended_price) }}</strong></div>
          <div style="margin-top:5px"><span class="badge badge-{{ r.action }}">{{ r.action }}</span></div>
          <div style="font-size:11px;font-weight:600;color:#15803d;margin-top:4px">{{ r.demand }}</div>
        </div>
      </div>
      {% endfor %}
      {% else %}<div class="empty-state"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg><p>Use the filters above to explore listings.</p></div>{% endif %}
    </div>
  </div>
</div>
<div id="pane-neighborhoods" class="tab-pane {{ 'active' if active_tab=='neighborhoods' }}">
  <div class="card">
    <div class="ch"><div style="display:flex;align-items:center;gap:10px"><div class="ch-icon" style="background:#ede9fe"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg></div><div><div class="ct">Neighborhood Deep Dive</div><div class="cs2">Select a neighborhood to see its listings, pricing breakdown and action split</div></div></div></div>
    <div class="cb2">
      <div class="ff" style="margin-bottom:22px">
        <label class="fl">Neighborhood \u2014 {{ selected_city }}</label>
        <select class="fsel" id="hood-select" onchange="showNeighborhood(this.value)" style="max-width:380px;margin-top:6px">
          <option value="">\u2014 Choose a neighborhood \u2014</option>
          {% for h in hood_heat %}
          <option value="{{ h.name }}">{{ h.label if h.label is defined else h.name }} \u2014 ${{ h.avg_price }} avg \u00b7 {{ h.raise_pct }}% raise</option>
          {% endfor %}
        </select>
      </div>
      <div id="hood-details" style="display:none"></div>
      <div id="hood-empty" class="empty-state"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg><p>Pick a neighborhood above to explore its listings.</p></div>
    </div>
  </div>
</div>

<div id="pane-alerts" class="tab-pane {{ 'active' if active_tab=='alerts' }}">
  <div class="card">
    <div class="ch">
      <div style="display:flex;align-items:center;gap:10px">
        <div class="ch-icon" style="background:#fef9c3"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ca8a04" stroke-width="2"><path d="M22 17H2a3 3 0 0 0 3-3V9a7 7 0 0 1 14 0v5a3 3 0 0 0 3 3zm-8.27 4a2 2 0 0 1-3.46 0"/></svg></div>
        <div><div class="ct">Full Action Queue \u2014 {{ selected_city }}</div><div class="cs2">All underpriced listings sorted by nightly revenue uplift</div></div>
      </div>
      <span class="badge badge-RAISE">{{ raise_count }} to Raise</span>
    </div>
    <div class="cb2" style="padding:0">
      <div style="overflow-x:auto">
        <table class="tbl" id="alerts-table">
          <thead><tr>
            <th style="cursor:default;width:36px">#</th>
            <th onclick="sortTable('alerts-table',1,this)">Listing <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('alerts-table',2,this)">Neighborhood <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('alerts-table',3,this)">Current <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('alerts-table',4,this)">Recommended <span class="sort-ic">\u2195</span></th>
            <th onclick="sortTable('alerts-table',5,this)">Uplift/nt <span class="sort-ic">\u2195</span></th>
            <th>Nearest Event</th>
            <th onclick="sortTable('alerts-table',7,this)">Distance <span class="sort-ic">\u2195</span></th>
            <th style="cursor:default">Demand</th>
          </tr></thead>
          <tbody>
          {% for r in alerts_list %}
          <tr>
            <td style="color:var(--text3);font-size:11px;font-weight:600">{{ loop.index }}</td>
            <td style="font-weight:600;max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ r.name }}</td>
            <td style="color:var(--text2);font-size:12px">{{ r.neighbourhood }}</td>
            <td style="color:var(--text2)">${{ r.current }}</td>
            <td style="font-weight:700;color:#15803d">${{ r.recommended }}</td>
            <td><span style="font-weight:700;color:#15803d;background:#f0fdf4;padding:2px 9px;border-radius:99px;font-size:12px">+${{ r.uplift }}</span></td>
            <td style="font-size:12px;color:var(--text2);max-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ r.event }}</td>
            <td style="font-size:12px;color:var(--text2)">{{ r.dist_km }} km</td>
            <td><span class="badge badge-RAISE" style="font-size:10px">{{ r.demand }}</span></td>
          </tr>
          {% endfor %}
          {% if not alerts_list %}
          <tr><td colspan="9" style="text-align:center;padding:32px;color:var(--text2)">No underpriced listings found for {{ selected_city }}.</td></tr>
          {% endif %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<div id="pane-simulator" class="tab-pane {{ 'active' if active_tab=='simulator' }}">
  <div class="card">
    <div class="ch"><div><div class="ct">What-If Price Simulator</div><div class="cs2">Enter listing details for a custom pricing recommendation</div></div></div>
    <div class="cb2">
      <form method="POST" action="/simulate">

        <div class="sim-grid">
          <input type="hidden" name="sim_city" value="Washington">
          <div class="ff"><label class="fl">Neighbourhood</label><select name="sim_hood" id="sim-hood" class="fsel"></select></div>
          <div class="ff"><label class="fl">Room Type</label><select name="sim_room" class="fsel"><option>Entire home/apt</option><option>Private room</option><option>Shared room</option><option>Hotel room</option></select></div>
          <div class="ff"><label class="fl">Bedrooms</label><input type="number" name="sim_bedrooms" class="fi" placeholder="1" min="0" max="10" value="1" style="min-width:0"></div>
          <div class="ff"><label class="fl">Guests</label><input type="number" name="sim_accommodates" class="fi" placeholder="2" min="1" max="20" value="2" style="min-width:0"></div>
          <div class="ff"><label class="fl">Review Score</label><input type="number" name="sim_rating" class="fi" placeholder="4.5" min="1" max="5" step="0.1" value="4.5" style="min-width:0"></div>
          <div class="ff"><label class="fl">Distance to Event (km)</label><input type="number" name="sim_distance" class="fi" placeholder="5" min="0" max="50" step="0.5" value="5" style="min-width:0"></div>
          <div class="ff"><label class="fl">Event Type</label><select name="sim_event_type" class="fsel"><option>Sports</option><option>Music</option><option>Arts &amp; Theatre</option><option>None</option></select></div>
        </div>
        <button class="btn-p" type="submit" style="font-size:14px;padding:12px 30px;border-radius:14px">Get Price Recommendation \u2192</button>
      </form>
      {% if sim_result %}
      <div class="sim-res">
        <div class="sim-orb">\u2736</div>
        <div>
          <div class="sim-plabel">Recommended nightly price</div>
          <div class="sim-price">${{ sim_result.price }}</div>
          <div class="sim-psub">+${{ sim_result.monthly_uplift }}/mo above avg \u00b7 {{ sim_result.demand }} demand \u00b7 <strong>{{ sim_result.action }}</strong></div>

          <div class="sim-explanation">
            {{ sim_result.explanation }}
          </div>

        </div>
      </div>
      {% endif %}
    </div>
  </div>
</div>
<div id="pane-revenue" class="tab-pane {{ 'active' if active_tab=='revenue' }}">
  <div class="insight-banner"><b>Revenue Opportunity — {{ selected_city }}:</b> {{ raise_count }} listings are currently underpriced on event weekends. By raising prices from ${{ raise_baseline }}/night to ${{ raise_avg }}/night, operators could collectively capture <span class="hi">${{ annual_opportunity }}</span> in additional annual revenue.</div>

  <div class="card" style="padding:6px 0 0;overflow:hidden">
    <div class="cb2" style="padding:18px 18px 10px">
      <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border:1px solid #e5edf5;border-radius:22px;background:#fff;overflow:hidden">
        <div style="padding:18px 22px;display:flex;align-items:center;gap:16px;border-right:1px solid #edf2f7">
          <div class="rmc-icon-wrap" style="margin:0;background:#edf7ef;width:64px;height:64px;border-radius:999px"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#4f7f5b" stroke-width="2.2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg></div>
          <div>
            <div style="font-size:12px;font-weight:700;color:#2c4468;margin-bottom:8px">Total Potential Revenue Lift</div>
            <div style="font-size:26px;font-weight:800;color:#4f7f5b;line-height:1.1">+${{ annual_opportunity }}</div>
            <div style="font-size:12px;color:#6e819a;margin-top:8px">+{{ avg_rec_increase_pct }}% vs Current</div>
          </div>
        </div>
        <div style="padding:18px 22px;display:flex;align-items:center;gap:16px;border-right:1px solid #edf2f7">
          <div class="rmc-icon-wrap" style="margin:0;background:#edf7ef;width:64px;height:64px;border-radius:999px"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#4f7f5b" stroke-width="2.2"><path d="M20 7l-8.5 8.5L8 12"/><path d="M7 7h6v6"/></svg></div>
          <div>
            <div style="font-size:12px;font-weight:700;color:#2c4468;margin-bottom:8px">Average Price Increase</div>
            <div style="font-size:26px;font-weight:800;color:#4f7f5b;line-height:1.1">+${{ '%.2f'|format(revenue_event_summary.avg_increase) }}</div>
            <div style="font-size:12px;color:#6e819a;margin-top:8px">+{{ '%.1f'|format(revenue_event_summary.increase_pct) }}% vs Current</div>
          </div>
        </div>
        <div style="padding:18px 22px;display:flex;align-items:center;gap:16px;border-right:1px solid #edf2f7">
          <div class="rmc-icon-wrap" style="margin:0;background:#edf7ef;width:64px;height:64px;border-radius:999px"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#4f7f5b" stroke-width="2.2"><rect x="4" y="10" width="3" height="8"/><rect x="10.5" y="6" width="3" height="12"/><rect x="17" y="3" width="3" height="15"/></svg></div>
          <div>
            <div style="font-size:12px;font-weight:700;color:#2c4468;margin-bottom:8px">Events with Higher Price</div>
            <div style="font-size:26px;font-weight:800;color:#4f7f5b;line-height:1.1">{{ revenue_event_summary.higher_count }} / {{ revenue_event_summary.event_total }}</div>
            <div style="font-size:12px;color:#6e819a;margin-top:8px">Events</div>
          </div>
        </div>
        <div style="padding:18px 22px;display:flex;align-items:center;gap:16px">
          <div class="rmc-icon-wrap" style="margin:0;background:#edf7ef;width:64px;height:64px;border-radius:999px"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#4f7f5b" stroke-width="2.2"><path d="M12 17.3l-6.16 3.24 1.18-6.88L2 8.92l6.92-1.01L12 1.64l3.08 6.27L22 8.92l-5.02 4.74 1.18 6.88z"/></svg></div>
          <div>
            <div style="font-size:12px;font-weight:700;color:#2c4468;margin-bottom:8px">Highest Opportunity</div>
            <div style="font-size:26px;font-weight:800;color:#4f7f5b;line-height:1.1">+${{ '%.2f'|format(revenue_event_summary.best_gap) }}</div>
            <div style="font-size:12px;color:#6e819a;margin-top:8px">{{ revenue_event_summary.best_event }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="ch"><div class="ct">Current vs Recommended by Event Type</div></div>
    <div class="cb2"><img src="data:image/png;base64,{{ rev_event_chart }}"></div>
  </div>
  <div class="card">
    <div class="ch"><div style="display:flex;align-items:center;gap:10px"><div class="ch-icon" style="background:#fef9c3"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ca8a04" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div><div><div class="ct">What-If Revenue Calculator</div><div class="cs2">Estimate upside from adjusting {{ hold_count_val }} HOLD listings</div></div></div></div>
    <div class="cb2">
      <p style="font-size:13.5px;color:var(--text2);margin-bottom:18px">You have <strong>{{ hold_count_val }}</strong> listings currently on HOLD, averaging <strong>${{ hold_avg_price }}/night</strong>. Use the slider to model what a price increase would generate.</p>
      <div class="ff" style="margin-bottom:20px">
        <label class="fl">Price increase: <span id="whatif-pct" style="color:var(--text);font-weight:700">10</span>%</label>
        <input type="range" id="whatif-slider" min="1" max="50" value="10" oninput="calcWhatIf()" style="width:100%;max-width:420px;margin-top:8px;cursor:pointer">
      </div>
      <div id="whatif-result" class="whatif-result-grid"></div>
    </div>
  </div>

  {% if revenue_table %}
  <div class="card">
    <div class="ch"><div><div class="ct">Revenue by Neighbourhood</div><div class="cs2">Annual opportunity breakdown</div></div></div>
    <div class="cb2" style="padding:0">
      <div style="overflow-x:auto">
        <table class="rtbl">
          <thead><tr><th>Neighbourhood</th><th>Listings</th><th>Current</th><th>Recommended</th><th>Uplift/nt</th><th>Annual Opp.</th></tr></thead>
          <tbody>
          {% for r in revenue_table %}
          <tr>
            <td style="font-weight:600">{{ r.neighbourhood }}</td>
            <td style="color:var(--text2)">{{ r.count }}</td>
            <td>${{ r.baseline }}</td>
            <td style="font-weight:700;color:#15803d">${{ r.recommended }}</td>
            <td><span style="font-weight:700;color:#15803d">+${{ r.uplift }}</span></td>
            <td style="font-weight:700">${{ r.annual }}</td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  {% endif %}
</div>
<div id="pane-chat" class="tab-pane {{ 'active' if active_tab=='chat' }}">
  <div class="chat-wrap">
    <div class="chat-hero">
      <div class="chat-ai-badge">PriceWise AI</div>
      <div class="chat-title">Smart Pricing<br>Assistant</div>
      <div class="chat-sub">Ask anything about listings, revenue, events, and market trends in <span>{{ selected_city }}</span>. Try: "weather", "raise prices", "summary".</div>
    </div>
    <div class="tile-grid">
      <button class="tile" onclick="chipClick('Which listings should raise prices?')"><div class="tile-icon" style="background:#f0fdf4">\u2191</div><div class="tile-txt">Raise Prices</div></button>
      <button class="tile" onclick="chipClick('What is the DC weather forecast?')"><div class="tile-icon" style="background:#eff6ff">\U0001f324</div><div class="tile-txt">DC Weather</div></button>
      <button class="tile" onclick="chipClick('Revenue opportunity this week')"><div class="tile-icon" style="background:#fef9c3">\U0001f4b5</div><div class="tile-txt">Revenue</div></button>
      <button class="tile" onclick="chipClick('Best neighbourhoods to invest')"><div class="tile-icon" style="background:#fdf4ff">\U0001f4cd</div><div class="tile-txt">Neighbourhoods</div></button>
    </div>
    <div class="msg-card">
      <div class="msgs" id="chat-msgs">
        <div class="msg-b-wrap"><div class="bot-av">\u2736</div><div class="msg-b">{{ chat_welcome_message }}</div></div>
        {% if history %}
        {% for item in history %}
        <div class="msg-u">{{ item.q }}</div>
        <div class="msg-b-wrap"><div class="bot-av">\u2736</div><div class="msg-b">{{ item.a|safe }}</div></div>
        {% endfor %}
        {% endif %}
      </div>
      {% if history %}
      <div class="chips-row">
        <button class="chip" onclick="chipClick('What events are coming up?')">Events</button>
        <button class="chip" onclick="chipClick('What is the DC weather forecast?')">Weather</button>
        <button class="chip" onclick="chipClick('Revenue opportunity')">Revenue</button>
        <button class="chip" onclick="chipClick('summary')">Summary</button>
        <a href="/clear_chat?selected_city={{ selected_city }}" style="font-family:inherit;font-size:12px;color:var(--text2);padding:6px 14px;border-radius:99px;border:1px solid var(--border);background:#fff;cursor:pointer">Clear</a>
      </div>
      {% endif %}
    </div>
    <form method="POST" action="/chat" id="chat-form">
      <input type="hidden" name="selected_city" value="{{ selected_city }}">
      <div class="inp-card">
        <input type="text" name="question" id="q-input" placeholder="Ask about pricing, weather, events, revenue\u2026" autocomplete="off">
        <button class="send-btn" type="submit"><svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg></button>
      </div>
    </form>
  </div>
</div>

<div id="pane-weather" class="tab-pane {{ 'active' if active_tab=='weather' }}">
  <div style="max-width:900px;margin:0 auto">
    {% if dc_weather %}
    <div class="wx-card" style="margin-bottom:24px">
      <div class="wx-title">{{ dc_weather.city_label }} \u2014 Live 7-Day Forecast</div>
      <div class="wx-cur">
        <div class="wx-temp">{{ dc_weather.current_temp_c }}\u00b0C</div>
        <div>
          <div class="wx-desc">{{ dc_weather.current_description }}</div>
          <div class="wx-cond">{% if dc_weather.is_outdoor_friendly %}Great outdoor conditions \u2600{% else %}Indoor / unsettled conditions{% endif %}</div>
        </div>
      </div>
      <div class="wx-days">
        {% for d in dc_weather.days %}
        {% set desc_low = d.description.lower() %}
        {% set wx_em = '\u2600\ufe0f' if ('clear' in desc_low or 'sunny' in desc_low) else ('\u26c5' if ('partly' in desc_low or 'mostly clear' in desc_low) else ('\u2601\ufe0f' if ('cloud' in desc_low or 'overcast' in desc_low) else ('\U0001f327\ufe0f' if ('rain' in desc_low or 'drizzle' in desc_low or 'shower' in desc_low) else ('\u26c8\ufe0f' if ('thunder' in desc_low or 'storm' in desc_low) else ('\u2744\ufe0f' if ('snow' in desc_low or 'flurr' in desc_low or 'blizzard' in desc_low) else '\U0001f324\ufe0f'))))) %}
        <div class="wx-day">
          <div class="wx-date">{{ d.date[5:] }}</div>
          <div style="font-size:22px;margin:4px 0">{{ wx_em }}</div>
          <div class="wx-dt">{{ d.temp_max_c|int }}\u00b0 / {{ d.temp_min_c|int }}\u00b0</div>
          <div class="wx-dd">{{ d.description[:14] }}</div>
          <div class="wx-rain">\U0001f4a7 {{ d.precipitation_mm }}mm</div>
        </div>
        {% endfor %}
      </div>
      <div class="wx-tip">\U0001f4a1 {{ dc_weather.pricing_tip }}</div>
    </div>

    <div class="card" style="margin-bottom:20px">
      <div class="ch">
        <div style="display:flex;align-items:center;gap:10px">
          <div class="ch-icon" style="background:#eff6ff"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg></div>
          <div><div class="ct">How Weather Affects Pricing</div><div class="cs2">Conditions mapped to recommended actions for DC listings</div></div>
        </div>
      </div>
      <div class="cb2">
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <thead><tr style="border-bottom:1px solid var(--border)">
            <th style="padding:10px 14px;text-align:left;color:var(--text2);font-weight:600">Condition</th>
            <th style="padding:10px 14px;text-align:left;color:var(--text2);font-weight:600">Temperature</th>
            <th style="padding:10px 14px;text-align:left;color:var(--text2);font-weight:600">Pricing Signal</th>
            <th style="padding:10px 14px;text-align:left;color:var(--text2);font-weight:600">Reason</th>
          </tr></thead>
          <tbody>
            <tr style="border-bottom:1px solid var(--border)">
              <td style="padding:10px 14px">\u2600\ufe0f Clear / Sunny</td>
              <td style="padding:10px 14px">Above 15\u00b0C</td>
              <td style="padding:10px 14px"><span class="badge badge-RAISE">RAISE</span></td>
              <td style="padding:10px 14px;color:var(--text2)">High outdoor tourism demand, more last-minute bookings</td>
            </tr>
            <tr style="border-bottom:1px solid var(--border)">
              <td style="padding:10px 14px">\u26c5 Partly Cloudy</td>
              <td style="padding:10px 14px">10\u201320\u00b0C</td>
              <td style="padding:10px 14px"><span class="badge badge-RAISE">RAISE</span></td>
              <td style="padding:10px 14px;color:var(--text2)">Comfortable sightseeing weather, steady demand</td>
            </tr>
            <tr style="border-bottom:1px solid var(--border)">
              <td style="padding:10px 14px">\u2601\ufe0f Overcast</td>
              <td style="padding:10px 14px">Any</td>
              <td style="padding:10px 14px"><span class="badge" style="background:#f1f5f9;color:#475569;font-size:10px;padding:3px 10px;border-radius:99px;font-weight:700">HOLD</span></td>
              <td style="padding:10px 14px;color:var(--text2)">Neutral conditions, maintain current pricing</td>
            </tr>
            <tr style="border-bottom:1px solid var(--border)">
              <td style="padding:10px 14px">\U0001f327\ufe0f Rain / Drizzle</td>
              <td style="padding:10px 14px">Any</td>
              <td style="padding:10px 14px"><span class="badge badge-DISCOUNT">DISCOUNT</span></td>
              <td style="padding:10px 14px;color:var(--text2)">Reduced foot traffic, outdoor listings see lower demand</td>
            </tr>
            <tr style="border-bottom:1px solid var(--border)">
              <td style="padding:10px 14px">\u26c8\ufe0f Thunderstorm</td>
              <td style="padding:10px 14px">Any</td>
              <td style="padding:10px 14px"><span class="badge badge-DISCOUNT">DISCOUNT</span></td>
              <td style="padding:10px 14px;color:var(--text2)">Travel disruptions, last-minute cancellations likely</td>
            </tr>
            <tr>
              <td style="padding:10px 14px">\u2744\ufe0f Snow / Ice</td>
              <td style="padding:10px 14px">Below 2\u00b0C</td>
              <td style="padding:10px 14px"><span class="badge badge-DISCOUNT">DISCOUNT</span></td>
              <td style="padding:10px 14px;color:var(--text2)">Low mobility, significantly reduced visitor numbers</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="g2">
      <div class="rmc">
        <div class="rmc-icon">\U0001f327\ufe0f</div>
        <div class="rmc-lbl">Rainy Days This Week</div>
        <div class="rmc-val bv">{{ dc_weather.rainy_count }}</div>
        <div class="rmc-sub">of 7 days with rain &gt; 5mm</div>
      </div>
      <div class="rmc">
        <div class="rmc-icon">\U0001f321\ufe0f</div>
        <div class="rmc-lbl">Current Temperature</div>
        <div class="rmc-val" style="color:#1d4ed8">{{ dc_weather.current_temp_c }}\u00b0C</div>
        <div class="rmc-sub">{{ dc_weather.current_description }}</div>
      </div>
    </div>

    {% else %}
    <div class="card" style="text-align:center;padding:60px 40px">
      <div style="font-size:48px;margin-bottom:16px">\u26c5</div>
      <div style="font-weight:700;font-size:18px;margin-bottom:8px">Weather unavailable for Washington DC</div>
      <div style="color:var(--text2);font-size:14px">Could not fetch the live 7-day forecast. Please try again shortly.</div>
    </div>
    {% endif %}
  </div>
</div>



<div id="pane-about_model" class="tab-pane {{ 'active' if active_tab=='about_model' }}">
  <style>
    .mq-page{max-width:1120px;margin:0 auto;padding-bottom:28px}.mq-note{background:#fff8ea;border:1px solid #ead8b4;border-radius:22px;padding:18px 24px;margin-bottom:24px;color:#1f2a44;font-size:13px;line-height:1.45;box-shadow:0 10px 28px rgba(15,23,42,.04)}.mq-note b{font-weight:850;color:#0f1f3d}.mq-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:28px;margin-bottom:28px}.mq-card{background:#fff;border:1px solid #d8e4f0;border-radius:22px;box-shadow:0 14px 38px rgba(15,23,42,.08);padding:28px 20px;text-align:center;min-height:144px;display:flex;flex-direction:column;align-items:center;justify-content:center}.mq-icon{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:20px;font-weight:850}.mq-green{background:#eafaf1;color:#16a34a}.mq-blue{background:#eff6ff;color:#2563eb}.mq-amber{background:#fff3bf;color:#b45309}.mq-value{font-size:34px;font-weight:850;letter-spacing:-.04em;line-height:1;color:#111827;margin-bottom:8px}.mq-value.green{color:#159447}.mq-value.blue{color:#2563eb}.mq-value.brown{color:#a34108}.mq-label{font-size:12px;color:#58677f;font-weight:700;line-height:1.2}.mq-sub{font-size:10.5px;color:#94a3b8;margin-top:3px;font-weight:650}.mq-two{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-bottom:28px}.mq-panel,.mq-wide{background:#fff;border:1px solid #cfe0ef;border-radius:18px;box-shadow:0 12px 34px rgba(15,23,42,.06);padding:24px}.mq-wide{margin-bottom:28px}.mq-panel h3,.mq-title-row h3{margin:0;color:#0f2b55;font-size:19px;font-weight:850;letter-spacing:-.02em}.mq-panel .hint,.mq-title-row div div{color:#2563eb;font-size:11.5px;font-weight:700;margin-bottom:18px}.mq-title-row div div{color:#64748b;margin-bottom:0}.mq-panel p{font-size:13px;line-height:1.62;color:#334155;margin:0 0 13px}.mq-panel b{color:#0f172a;font-weight:850}.mq-red{color:#dc2626!important}.mq-blue-text{color:#2563eb!important}.mq-orange{color:#ea580c!important}.mq-title-row{display:flex;align-items:center;gap:10px;margin-bottom:20px}.mq-mini-icon{width:25px;height:25px;border-radius:50%;background:#e8f8f1;color:#2aa66a;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:13px}.mq-chart{padding:8px 8px 2px}.mq-bar-row{display:grid;grid-template-columns:170px 1fr 58px;align-items:center;gap:12px;margin:13px 0}.mq-feature{text-align:right;color:#43556f;font-size:13px;font-weight:800}.mq-track{height:24px;background:linear-gradient(90deg,rgba(226,232,240,.75),rgba(226,232,240,.18));border-left:1px solid #e2e8f0;position:relative}.mq-bar{height:100%;background:#4f7f5b;position:relative;z-index:2}.mq-bar.light{background:#a9c9b3}.mq-num{font-size:12px;font-weight:850;color:#0f2b55}.mq-axis{display:grid;grid-template-columns:170px 1fr 58px;gap:12px;margin-top:6px}.mq-axis-scale{display:flex;justify-content:space-between;color:#9aa9bd;font-size:11px;font-weight:700}.mq-axis-label{text-align:center;color:#64748b;font-size:12px;font-weight:850;margin-top:4px}.mq-table{width:100%;border-collapse:collapse;margin-top:16px}.mq-table th,.mq-table td{font-size:13px;text-align:left;padding:13px 14px;border-top:1px solid #e5edf5;color:#334155}.mq-table th{font-size:11px;color:#64748b;font-weight:850}.mq-table td:first-child{font-weight:800;color:#0f172a}@media(max-width:900px){.mq-metrics,.mq-two{grid-template-columns:1fr}.mq-bar-row,.mq-axis{grid-template-columns:130px 1fr 50px}}
  

/* --- BLUE/NAVY ACCENT OVERRIDES --- */
.chat-ai-badge {
    background: rgba(147,197,253,0.14) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.42) !important;
}
.chat-sub span {
    color: #bfdbfe !important;
    font-weight: 700 !important;
}
.nav-badge, .dc-pill {
    background: rgba(147,197,253,0.16) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.38) !important;
}
.eyebrow, .hero h1 .gold {
    color: #93c5fd !important;
}

</style>
  <div class="mq-page">
    <div class="mq-note"><b>Model Quality</b> — Gradient Boosting evaluated on a held-out 20% test set (833 listings). These numbers tell you how accurate the recommendations are.</div>
    <div class="mq-metrics">
      <div class="mq-card"><div class="mq-icon mq-green">✓</div><div class="mq-value green">19.9%</div><div class="mq-label">Mean Abs. % Error</div><div class="mq-sub">Lower is better — avg prediction error</div></div>
      <div class="mq-card"><div class="mq-icon mq-blue">$</div><div class="mq-value blue">$42.78</div><div class="mq-label">RMSE ($/night)</div><div class="mq-sub">Root mean squared error on test set</div></div>
      <div class="mq-card"><div class="mq-icon mq-amber">▥</div><div class="mq-value brown">0.8284</div><div class="mq-label">R² Score</div><div class="mq-sub">Variance explained (1.0 = perfect)</div></div>
    </div>
    <div class="mq-two">
      <div class="mq-panel"><h3>What these numbers mean</h3><div class="hint">How to interpret the model quality scores</div><p><b>MAPE 19.9%</b> means the model’s recommended price is off by 19.9% on average relative to actual market prices. For a $150/night listing that’s roughly $29 of expected error.</p><p><b>RMSE $42.78</b> is the typical dollar error in absolute terms. Recommendations outside this band deserve closer scrutiny.</p><p><b>R² 0.8284</b> shows how much of the price variation the model explains from the features: bedrooms, events, reviews, etc.</p></div>
      <div class="mq-panel"><h3>Data quality note</h3><div class="hint">Known limitations of the current dataset</div><p><b class="mq-red">Event Distances</b> for Washington DC listings were previously showing 190–200 km because events_dc.csv contains World Cup venues in other states. The pipeline now uses a lat/lon bounding box to match only genuine local events.</p><p><b class="mq-blue-text">No data leakage</b> — recommendations are generated from 5-fold out-of-fold cross-validation predictions, so every listing is priced by a model that was not trained on it.</p><p><b class="mq-orange">Threshold</b> changed from flat ±$10 to ±5% of baseline price, so the RAISE/DISCOUNT signal is proportional across price tiers.</p></div>
    </div>
    <div class="mq-wide"><div class="mq-title-row"><div class="mq-mini-icon">⌁</div><div><h3>Feature Importances</h3><div>Which inputs drive the price recommendation most</div></div></div><div class="mq-chart">
      <div class="mq-bar-row"><div class="mq-feature">bedrooms</div><div class="mq-track"><div class="mq-bar" style="width:99.8%"></div></div><div class="mq-num">0.499</div></div>
      <div class="mq-bar-row"><div class="mq-feature">accommodates</div><div class="mq-track"><div class="mq-bar" style="width:31.2%"></div></div><div class="mq-num">0.156</div></div>
      <div class="mq-bar-row"><div class="mq-feature">distance to event km</div><div class="mq-track"><div class="mq-bar" style="width:23.0%"></div></div><div class="mq-num">0.115</div></div>
      <div class="mq-bar-row"><div class="mq-feature">precipitation mm</div><div class="mq-track"><div class="mq-bar light" style="width:17.2%"></div></div><div class="mq-num">0.086</div></div>
      <div class="mq-bar-row"><div class="mq-feature">review scores rating</div><div class="mq-track"><div class="mq-bar light" style="width:14.0%"></div></div><div class="mq-num">0.070</div></div>
      <div class="mq-bar-row"><div class="mq-feature">minimum nights</div><div class="mq-track"><div class="mq-bar light" style="width:5.4%"></div></div><div class="mq-num">0.027</div></div>
      <div class="mq-bar-row"><div class="mq-feature">temp avg c</div><div class="mq-track"><div class="mq-bar light" style="width:3.2%"></div></div><div class="mq-num">0.016</div></div>
      <div class="mq-bar-row"><div class="mq-feature">type: Private room</div><div class="mq-track"><div class="mq-bar light" style="width:2.0%"></div></div><div class="mq-num">0.010</div></div>
      <div class="mq-bar-row"><div class="mq-feature">type: Shared room</div><div class="mq-track"><div class="mq-bar light" style="width:1.8%"></div></div><div class="mq-num">0.009</div></div>
      <div class="mq-bar-row"><div class="mq-feature">is peak summer</div><div class="mq-track"><div class="mq-bar light" style="width:0.8%"></div></div><div class="mq-num">0.004</div></div>
      <div class="mq-axis"><div></div><div><div class="mq-axis-scale"><span>0.0</span><span>0.1</span><span>0.2</span><span>0.3</span><span>0.4</span><span>0.5</span></div><div class="mq-axis-label">Importance</div></div><div></div></div>
    </div></div>
    <div class="mq-wide"><div class="mq-title-row"><div><h3>Training summary</h3><div>Dataset split used for evaluation</div></div></div><table class="mq-table"><thead><tr><th>Split</th><th>Listings</th><th>Purpose</th></tr></thead><tbody><tr><td>Train (80%)</td><td>3328</td><td>Used to fit model parameters</td></tr><tr><td>Test (20%)</td><td>833</td><td>Held-out evaluation — model never saw these</td></tr></tbody></table></div>
  </div>
</div>
</main>
  </div>
</div>

<div id="listing-modal" class="modal-overlay" onclick="if(event.target===this)closeLModal()">
  <div class="modal-box">
    <button class="modal-close" onclick="closeLModal()">\u00d7</button>
    <div id="modal-content"></div>
  </div>
</div>

<script>
var neighbourhoods = {{ hoods_json|safe }};
var _listingsData  = {{ listings_preview_json|safe }};
var _hoodData      = {{ hood_listings_json|safe }};
var _holdCount     = {{ hold_count_val }};
var _holdAvgPrice  = {{ hold_avg_price }};

// Table sorting 
var _sortDir = {};
function sortTable(tableId, colIdx, th) {
  var table = document.getElementById(tableId);
  if (!table) return;
  var tbody = table.querySelector('tbody');
  var rows  = Array.from(tbody.querySelectorAll('tr'));
  var key   = tableId + '-' + colIdx;
  var asc   = !_sortDir[key];
  _sortDir[key] = asc;
  rows.sort(function(a, b) {
    var av = a.cells[colIdx] ? a.cells[colIdx].textContent.trim() : '';
    var bv = b.cells[colIdx] ? b.cells[colIdx].textContent.trim() : '';
    var an = parseFloat(av.replace(/[^0-9.+-]/g, ''));
    var bn = parseFloat(bv.replace(/[^0-9.+-]/g, ''));
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
  table.querySelectorAll('.sort-ic').forEach(function(ic) { ic.textContent = '\u2195'; });
  var ic = th ? th.querySelector('.sort-ic') : null;
  if (ic) ic.textContent = asc ? '\u2191' : '\u2193';
}

// Listing modal 
function showListingModal(idx) {
  var d = _listingsData[idx];
  if (!d) return;
  var badge = d.action === 'RAISE' ? 'badge-RAISE' : (d.action === 'DISCOUNT' ? 'badge-DISCOUNT' : 'badge-HOLD');
  var label = d.action === 'RAISE' ? '\u2191 Raise' : (d.action === 'DISCOUNT' ? '\u2193 Discount' : '\u2014 Hold');
  var uplift = parseInt(d.recommended) - parseInt(d.current);
  document.getElementById('modal-content').innerHTML =
    '<div style="font-size:18px;font-weight:700;letter-spacing:-.4px;margin-bottom:18px">' + d.name + '</div>' +
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:18px">' +
    '<div><div style="font-size:10.5px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.7px">Room Type</div><div style="font-size:14px;font-weight:600;margin-top:4px">' + d.room_type + '</div></div>' +
    '<div><div style="font-size:10.5px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.7px">Bedrooms</div><div style="font-size:14px;font-weight:600;margin-top:4px">' + d.bedrooms + '</div></div>' +
    '<div><div style="font-size:10.5px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.7px">Rating</div><div style="font-size:14px;font-weight:600;margin-top:4px">\u2b50 ' + d.rating + '</div></div>' +
    '<div><div style="font-size:10.5px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.7px">Action</div><div style="margin-top:4px"><span class="badge ' + badge + '">' + label + '</span></div></div>' +
    '</div>' +
    '<div style="background:#f8fafc;border-radius:14px;padding:18px;display:flex;gap:20px;align-items:center">' +
    '<div><div style="font-size:10.5px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.7px">Current</div><div style="font-size:26px;font-weight:700;margin-top:3px">$' + d.current + '</div></div>' +
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.5" style="flex-shrink:0"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>' +
    '<div><div style="font-size:10.5px;font-weight:700;color:#15803d;text-transform:uppercase;letter-spacing:.7px">Recommended</div><div style="font-size:26px;font-weight:700;color:#15803d;margin-top:3px">$' + d.recommended + '</div></div>' +
    (uplift > 0 ? '<div style="margin-left:auto"><span style="font-weight:700;color:#15803d;background:#f0fdf4;padding:5px 13px;border-radius:99px;font-size:13px">+$' + uplift + '/nt</span></div>' : '') +
    '</div>';
  document.getElementById('listing-modal').classList.add('open');
}
function closeLModal() {
  document.getElementById('listing-modal').classList.remove('open');
}
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeLModal(); });

// Neighborhood deep-dive 
function showNeighborhood(name) {
  var el    = document.getElementById('hood-details');
  var empty = document.getElementById('hood-empty');
  if (!name) { el.style.display='none'; empty.style.display=''; return; }
  var d = _hoodData[name];
  if (!d) { el.style.display='none'; empty.style.display=''; return; }
  empty.style.display = 'none';
  var html =
    '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px">' +
    '<div class="sc cb" style="min-width:0"><div class="sc-lbl">Top properties to update</div><div class="sc-val">' + d.count + '</div></div>' +
    '<div class="sc cg" style="min-width:0"><div class="sc-lbl">Avg Price</div><div class="sc-val">$' + d.avg_price + '</div></div>' +
    '<div class="sc cgn" style="min-width:0"><div class="sc-lbl">Raise</div><div class="sc-val" style="color:#15803d">' + d.raise + '</div></div>' +
    '<div class="sc co" style="min-width:0"><div class="sc-lbl">Discount</div><div class="sc-val" style="color:#c2410c">' + d.discount + '</div></div>' +
    '</div>' +
    '<div style="font-size:13px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px">Top Listings in ' + name + '</div>' +
    '<div style="overflow-x:auto"><table class="tbl"><thead><tr>' +
    '<th style="cursor:default">Name</th><th style="cursor:default">Type</th><th style="cursor:default">Beds</th>' +
    '<th style="cursor:default">Rating</th><th style="cursor:default">Current</th><th style="cursor:default">Recommended</th><th style="cursor:default">Action</th>' +
    '</tr></thead><tbody>';
  d.listings.forEach(function(r) {
    var bg = r.action==='RAISE' ? 'badge-RAISE' : (r.action==='DISCOUNT' ? 'badge-DISCOUNT' : 'badge-HOLD');
    var lbl = r.action==='RAISE' ? '\u2191 Raise' : (r.action==='DISCOUNT' ? '\u2193 Discount' : '\u2014 Hold');
    html += '<tr><td style="font-weight:500">' + r.name + '</td><td style="color:var(--text2)">' + r.room_type +
            '</td><td>' + r.bedrooms + '</td><td>\u2b50 ' + r.rating +
            '</td><td>$' + r.current + '</td><td style="font-weight:700">$' + r.recommended +
            '</td><td><span class="badge ' + bg + '">' + lbl + '</span></td></tr>';
  });
  html += '</tbody></table></div>';
  el.innerHTML = html;
  el.style.display = 'block';
}

// Revenue what-if calculator 
function calcWhatIf() {
  var pct = parseFloat(document.getElementById('whatif-slider').value);
  document.getElementById('whatif-pct').textContent = pct;
  var uplift   = _holdAvgPrice * pct / 100;
  var monthly  = uplift * _holdCount * 20;
  var annual   = uplift * _holdCount * 240;
  var fmt = function(n) { return n.toLocaleString(undefined, {maximumFractionDigits:0}); };
  document.getElementById('whatif-result').innerHTML =
    '<div style="background:#f0fdf4;padding:14px 22px;border-radius:12px">' +
    '<div style="font-size:10.5px;font-weight:700;color:#15803d;text-transform:uppercase;letter-spacing:.7px">Uplift per listing/night</div>' +
    '<div style="font-size:28px;font-weight:800;letter-spacing:-1px;margin-top:4px">+$' + fmt(uplift) + '</div></div>' +
    '<div style="background:#eff6ff;padding:14px 22px;border-radius:12px">' +
    '<div style="font-size:10.5px;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:.7px">Monthly Revenue Gain</div>' +
    '<div style="font-size:28px;font-weight:800;letter-spacing:-1px;margin-top:4px">$' + fmt(monthly) + '</div></div>' +
    '<div style="background:#fef9c3;padding:14px 22px;border-radius:12px">' +
    '<div style="font-size:10.5px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:.7px">Annual Revenue Gain</div>' +
    '<div style="font-size:28px;font-weight:800;letter-spacing:-1px;margin-top:4px">$' + fmt(annual) + '</div></div>';
}

function updateNeighbourhoods(){
  var sel=document.getElementById('sim-hood');
  sel.innerHTML='';
  (neighbourhoods['Washington']||[]).forEach(function(h){var o=document.createElement('option');o.value=h;o.textContent=h;sel.appendChild(o);});
}
updateNeighbourhoods();
function runCountUps(){
  document.querySelectorAll('[data-count]').forEach(function(el){
    var target=parseInt(el.getAttribute('data-count').replace(/,/g,''),10);
    if(isNaN(target))return;
    var dur=900,t0=null;
    function step(ts){if(!t0)t0=ts;var p=Math.min((ts-t0)/dur,1),e=1-Math.pow(1-p,3);el.textContent=Math.round(target*e).toLocaleString();if(p<1)requestAnimationFrame(step);else el.textContent=target.toLocaleString();}
    requestAnimationFrame(step);
  });
}
function runUpliftBars(){
  document.querySelectorAll('.uplift-fill[data-pct]').forEach(function(bar){bar.style.width='0%';setTimeout(function(){bar.style.width=bar.getAttribute('data-pct')+ '%';},80);});
}
function switchTab(name,btn){
  document.querySelectorAll('.tab-pane').forEach(function(p){p.classList.remove('active');});
  document.querySelectorAll('.nb').forEach(function(b){b.classList.remove('active');});
  document.getElementById('pane-'+name).classList.add('active');
  if(btn)btn.classList.add('active');
  var sb=document.getElementById('stats-bar');
  if(name==='chat'){sb.style.display='none';}else{sb.style.display='';runCountUps();if(name==='revenue')setTimeout(runUpliftBars,120);}
}
function askQ(q){
  document.getElementById('q-input').value=q;
  document.querySelectorAll('.tab-pane').forEach(function(p){p.classList.remove('active');});
  document.querySelectorAll('.nb').forEach(function(b){b.classList.remove('active');});
  document.getElementById('pane-chat').classList.add('active');
  document.getElementById('stats-bar').style.display='none';
  document.getElementById('chat-form').submit();
}
function chipClick(q){document.getElementById('q-input').value=q;document.getElementById('chat-form').submit();}
function filterListings(q){
  q=q.toLowerCase();
  document.querySelectorAll('#listings-table tbody tr').forEach(function(row){row.style.display=row.textContent.toLowerCase().includes(q)?'':'none';});
}
function hideMapSpinner(){var s=document.getElementById('map-spinner');if(s){s.style.opacity='0';setTimeout(function(){s.style.display='none';},300);}}
var _mapActiveLayers=['Raise','Discount','Hold','Events'];
function toggleMapLayer(btn){
  var layer=btn.getAttribute('data-layer');
  var idx=_mapActiveLayers.indexOf(layer);
  if(idx>=0){_mapActiveLayers.splice(idx,1);btn.classList.remove('active');}
  else{_mapActiveLayers.push(layer);btn.classList.add('active');}
  var frame=document.getElementById('map-frame');
  if(frame&&frame.contentWindow){frame.contentWindow.postMessage({type:'pwfilter',show:_mapActiveLayers},'*');}
}
(function(){
  var active='{{ active_tab }}';
  if(active==='chat' || active==='about_model'){document.getElementById('stats-bar').style.display='none';}
  else{runCountUps();if(active==='revenue')setTimeout(runUpliftBars,200);}
})();
var cm=document.getElementById('chat-msgs');if(cm)cm.scrollTop=cm.scrollHeight;
calcWhatIf();
if('{{ active_tab }}'==='neighborhoods'){var hs=document.getElementById('hood-select');if(hs&&hs.value)showNeighborhood(hs.value);}
</script>
</body>
</html>"""


# LANDING PAGE
LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<title>PriceWise DC</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,Segoe UI,Arial,sans-serif;background:#07111f;color:white;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:32px}
.hero{max-width:980px;width:100%;text-align:center}
.badge{display:inline-block;background:rgba(96,165,250,.16);border:1px solid rgba(147,197,253,.35);color:#bfdbfe;padding:8px 14px;border-radius:999px;font-weight:800;font-size:12px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:18px}
h1{font-size:clamp(38px,6vw,72px);line-height:1.05;letter-spacing:-.05em;margin-bottom:18px}
p{color:#cbd5e1;font-size:18px;line-height:1.6;max-width:720px;margin:0 auto 30px}
.hero-actions{display:flex;gap:12px;justify-content:center;align-items:center;flex-wrap:wrap}
a{display:inline-block;background:#2563eb;color:#fff;text-decoration:none;padding:14px 24px;border-radius:999px;font-weight:800;box-shadow:0 18px 45px rgba(37,99,235,.35)}
a:nth-child(2){background:#0ea5e9;box-shadow:0 18px 45px rgba(14,165,233,.28)}
a:nth-child(3){background:#7c3aed;box-shadow:0 18px 45px rgba(124,58,237,.28)}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:36px;text-align:left}
.card{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:20px;padding:18px}
.card b{display:block;margin-bottom:8px}.card span{color:#cbd5e1;font-size:14px;line-height:1.45}
@media(max-width:760px){.cards{grid-template-columns:1fr}}


/* --- BLUE/NAVY ACCENT OVERRIDES --- */
.chat-ai-badge {
    background: rgba(147,197,253,0.14) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.42) !important;
}
.chat-sub span {
    color: #bfdbfe !important;
    font-weight: 700 !important;
}
.nav-badge, .dc-pill {
    background: rgba(147,197,253,0.16) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.38) !important;
}
.eyebrow, .hero h1 .gold {
    color: #93c5fd !important;
}

</style>
</head>
<body>
<div class="hero">
  <div class="badge">Airbnb Pricing Intelligence</div>
  <h1>PriceWise DC</h1>
  <p>Combining Sarah’s portfolio insights with real-time DC market intelligence to power smarter pricing decisions.
<div style="margin-top:16px;color:#b9cce4;font-size:14px;font-weight:700;line-height:1.45;">
  </div>
</p>
  <div class="hero-actions">
    <a href="/dashboard">Open Dashboard</a>
    <a href="/dashboard?tab=map">Market Map</a>
    <a href="/dashboard?tab=chat">Ask Market AI</a>
  </div>
  <div class="cards">
    <div class="card"><b>Sarah’s dashboard</b><span>A 7-property portfolio view for one property manager.</span></div>
    <div class="card"><b>7 units</b><span>Portfolio insights enhanced by live market signals and trends.</span></div>
    <div class="card"><b>Market tools</b><span>Market-wide data powers pricing, demand signals, and recommendations.</span></div>
  </div>
</div>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string('''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PriceWise DC</title>
  <style>
    :root{
      --gold:#93c5fd;
      --white:#ffffff;
      --muted:rgba(255,255,255,.72);
      --glass:rgba(255,255,255,.12);
      --glass-strong:rgba(255,255,255,.16);
      --stroke:rgba(255,255,255,.22);
      --dark:#07111f;
    }
    *{box-sizing:border-box}
    html,body{
      margin:0;
      min-height:100%;
      font-family:Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
      color:white;
    }
    body{
      background:
        linear-gradient(90deg,rgba(4,10,18,.72),rgba(4,10,18,.47),rgba(4,10,18,.72)),
        linear-gradient(180deg,rgba(4,10,18,.58),rgba(4,10,18,.78)),
        url("https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?auto=format&fit=crop&w=2400&q=85");
      background-size:cover;
      background-position:center;
      background-attachment:fixed;
      overflow-x:hidden;
    }
    .landing{
      min-height:100vh;
      padding:34px 56px 46px;
      position:relative;
    }
    .topbar{
      display:flex;
      align-items:center;
      justify-content:space-between;
      position:relative;
      z-index:2;
    }
    .brand{
      display:flex;
      align-items:center;
      gap:10px;
      font-weight:900;
      letter-spacing:-.02em;
      font-size:20px;
      text-shadow:0 2px 14px rgba(0,0,0,.35);
    }
    .brand-icon{
      width:22px;height:22px;display:grid;place-items:center;
      color:white;
      opacity:.94;
    }
    .dc-pill{
      font-size:11px;
      font-weight:900;
      letter-spacing:.04em;
      background:rgba(147,197,253,.16);
      color:#dbeafe;
      border:1px solid rgba(147,197,253,.38);
      border-radius:999px;
      padding:3px 9px;
      margin-left:4px;
    }
    .open-top{
      color:white;
      text-decoration:none;
      padding:12px 22px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,.28);
      background:rgba(255,255,255,.14);
      box-shadow:0 10px 30px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.12);
      backdrop-filter:blur(16px);
      font-weight:850;
      font-size:14px;
    }
    .hero{
      max-width:980px;
      margin:92px auto 0;
      text-align:center;
      position:relative;
      z-index:2;
    }
    .eyebrow{
      color:var(--gold);
      font-weight:950;
      letter-spacing:.24em;
      font-size:13px;
      text-transform:uppercase;
      margin-bottom:22px;
      text-shadow:0 2px 14px rgba(0,0,0,.35);
    }
    .hero h1{
      margin:0;
      font-size:clamp(56px,7vw,96px);
      line-height:.92;
      letter-spacing:-.065em;
      font-weight:950;
      text-shadow:0 8px 30px rgba(0,0,0,.36);
    }
    .hero h1 .gold{
      display:block;
      color:var(--gold);
      font-weight:650;
      letter-spacing:-.06em;
      margin-top:8px;
    }
    .subtitle{
      margin:34px auto 0;
      max-width:760px;
      color:rgba(255,255,255,.78);
      font-size:22px;
      line-height:1.42;
      font-weight: 400;
      text-shadow:0 2px 20px rgba(0,0,0,.36);
    }
    .search-wrap{
      margin:58px auto 0;
      width:min(760px,92vw);
    }
    .search{
      height:64px;
      border-radius:999px;
      background:rgba(255,255,255,.94);
      display:flex;
      align-items:center;
      padding:7px 8px 7px 24px;
      box-shadow:0 22px 58px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.7);
      gap:12px;
    }
    .search svg{color:#9aa6b8;flex:0 0 auto}
    .search input{
      border:0;
      outline:0;
      background:transparent;
      flex:1;
      font-size:17px;
      font-weight:750;
      color:#0f172a;
      min-width:0;
    }
    .search input::placeholder{color:#9aa6b8}
    .search button{
      border:0;
      height:50px;
      padding:0 28px;
      border-radius:999px;
      background:rgba(147,197,253,.16);
      color:#dbeafe;
      border:1px solid rgba(147,197,253,.38);
      font-weight:950;
      font-size:15px;
      cursor:pointer;
      box-shadow:0 8px 20px rgba(37,99,235,.22);
    }
    .chips{
      display:flex;
      justify-content:center;
      gap:10px;
      flex-wrap:wrap;
      margin-top:18px;
    }
    .chip{
      color:rgba(255,255,255,.72);
      background:rgba(255,255,255,.12);
      border:1px solid rgba(255,255,255,.20);
      border-radius:999px;
      padding:8px 16px;
      font-size:14px;
      font-weight:800;
      backdrop-filter:blur(10px);
    }
    .explore{
      margin-top:60px;
      color:#bfdbfe;
      font-size:13px;
      font-weight:950;
      letter-spacing:.16em;
      text-transform:uppercase;
      text-shadow:0 2px 14px rgba(0,0,0,.45);
    }
    .feature-grid{
      width:min(1060px,92vw);
      margin:20px auto 0;
      display:grid;
      grid-template-columns:repeat(5,minmax(0,1fr));
      gap:14px;
      text-align:left;
    }
    .feature-card{
      min-height:126px;
      border-radius:18px;
      background:linear-gradient(145deg,rgba(255,255,255,.17),rgba(255,255,255,.08));
      border:1px solid rgba(255,255,255,.22);
      backdrop-filter:blur(16px);
      box-shadow:0 20px 55px rgba(0,0,0,.26), inset 0 1px 0 rgba(255,255,255,.11);
      padding:18px;
      color:white;
      text-decoration:none;
      display:flex;
      flex-direction:column;
      justify-content:flex-start;
      transition:transform .18s ease, background .18s ease;
    }
    .feature-card:hover{
      transform:translateY(-4px);
      background:linear-gradient(145deg,rgba(255,255,255,.22),rgba(255,255,255,.11));
    }
    .feature-icon{
      width:38px;height:38px;
      border-radius:12px;
      display:grid;
      place-items:center;
      margin-bottom:16px;
      background:rgba(147,197,253,.18);
      color:var(--gold);
    }
    .feature-card:nth-child(2) .feature-icon{background:rgba(96,165,250,.22);color:#9cc7ff}
    .feature-card:nth-child(3) .feature-icon{background:rgba(147,197,253,.18);color:#93c5fd}
    .feature-card:nth-child(4) .feature-icon{background:rgba(255,255,255,.14);color:#d9dee7}
    .feature-card:nth-child(5) .feature-icon{background:rgba(147,197,253,.18);color:#bfdbfe}
    .feature-card:nth-child(6) .feature-icon{background:rgba(147,197,253,.18);color:#bfdbfe}
    .feature-card:nth-child(7) .feature-icon{background:rgba(99,102,241,.22);color:#818cf8}
    .feature-card:nth-child(8) .feature-icon{background:rgba(147,197,253,.18);color:#bfdbfe}
    .feature-card:nth-child(9) .feature-icon{background:rgba(147,197,253,.18);color:#93c5fd}
    .feature-card:nth-child(10) .feature-icon{background:rgba(148,163,184,.22);color:#cbd5e1}
    .feature-card:nth-child(11) .feature-icon{background:rgba(139,92,246,.22);color:#a78bfa}
    .feature-title{
      font-size:16px;
      line-height:1.05;
      font-weight:950;
      margin-bottom:6px;
      color:white;
    }
    .feature-desc{
      color:rgba(255,255,255,.68);
      font-size:13px;
      line-height:1.25;
      font-weight:750;
    }
    @media(max-width:960px){
      .landing{padding:26px 20px 40px}
      .hero{margin-top:70px}
      .subtitle{font-size:18px}
      .feature-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
      .search{height:auto;align-items:stretch;flex-wrap:wrap;border-radius:28px;padding:14px}
      .search button{width:100%}
    }
    @media(max-width:520px){
      .feature-grid{grid-template-columns:1fr}
      .topbar{gap:12px}
      .open-top{padding:10px 14px}
      .hero h1{font-size:52px}
    }
  

/* --- BLUE/NAVY ACCENT OVERRIDES --- */
.chat-ai-badge {
    background: rgba(147,197,253,0.14) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.42) !important;
}
.chat-sub span {
    color: #bfdbfe !important;
    font-weight: 700 !important;
}
.nav-badge, .dc-pill {
    background: rgba(147,197,253,0.16) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(147,197,253,0.38) !important;
}
.eyebrow, .hero h1 .gold {
    color: #93c5fd !important;
}

</style>
</head>
<body>
  <main class="landing">
    <nav class="topbar">
      <div class="brand">
        <span class="brand-icon">
          <svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <path d="M3 11.5 12 4l9 7.5"></path><path d="M5.5 10.5V20h13v-9.5"></path><path d="M9.5 20v-5h5v5"></path>
          </svg>
        </span>
        PriceWise <span class="dc-pill">DC</span>
      </div>
      <a class="open-top" href="/dashboard">Open Dashboard →</a>
    </nav>

    <section class="hero">
      <div class="eyebrow">Washington DC • Airbnb Intelligence</div>
      <h1>Price Smarter.<span class="gold">Earn More.</span></h1>
      <div class="subtitle">
        SaaS-style pricing intelligence for Washington DC, shaped by live events,
        demand signals, and neighborhood trends.
      </div>

      <div class="search-wrap">
        <form class="search" action="/dashboard" method="get">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path>
          </svg>
          <input name="q" placeholder="Search neighborhoods, events, listings..." />
          <button type="submit">Search DC</button>
        </form>
        <div class="chips">
          <span class="chip">Capitol Hill</span>
          <span class="chip">Georgetown</span>
          <span class="chip">Adams Morgan</span>
          <span class="chip">Near National Mall</span>
          <span class="chip">Dupont Circle</span>
        </div>
      </div>

      <div class="explore">Explore the platform</div>

      <div class="feature-grid">
        <a class="feature-card" href="/dashboard">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
          </div>
          <div class="feature-title">Dashboard</div>
          <div class="feature-desc">KPIs & overview at a glance</div>
        </a>

        <a class="feature-card" href="/dashboard?tab=map">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>
          </div>
          <div class="feature-title">Live Map</div>
          <div class="feature-desc">Pricing signals across DC</div>
        </a>

        <a class="feature-card" href="/dashboard?tab=calendar">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="5" width="16" height="15" rx="2"/><path d="M8 3v4M16 3v4M4 10h16"/></svg>
          </div>
          <div class="feature-title">Events</div>
          <div class="feature-desc">Upcoming DC demand drivers</div>
        </a>

        <a class="feature-card" href="/dashboard?tab=listings">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 6h13M8 12h13M8 18h13"/><path d="M3 6h.01M3 12h.01M3 18h.01"/></svg>
          </div>
          <div class="feature-title">Listings</div>
          <div class="feature-desc">Top listings by price</div>
        </a>

        <a class="feature-card" href="/dashboard?tab=neighborhoods">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2 3 7l9 5 9-5-9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg>
          </div>
          <div class="feature-title">Neighborhoods</div>
          <div class="feature-desc">Area deep-dive & heatmap</div>
        </a>

        <a class="feature-card" href="/dashboard?tab=explore">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 7h18s-3 0-3-7"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
          </div>
          <div class="feature-title">Alerts</div>
          <div class="feature-desc">Underpriced listing queue</div>
        </a>

        <a class="feature-card" href="/dashboard?tab=explore">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
          </div>
          <div class="feature-title">Explore</div>
          <div class="feature-desc">Filter & search listings</div>
        </a>


        

        <a class="feature-card" href="/dashboard?tab=simulator">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 21v-7M12 21V3M20 21v-11"/><path d="M2 14h4M10 8h4M18 14h4"/></svg>
          </div>
          <div class="feature-title">Simulator</div>
          <div class="feature-desc">What-if price predictor</div>
        </a>

        <a class="feature-card" href="/dashboard?tab=revenue">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7H14a3.5 3.5 0 0 1 0 7H6"/></svg>
          </div>
          <div class="feature-title">Revenue</div>
          <div class="feature-desc">Annual opportunity analysis</div>
        </a>

        <a class="feature-card" href="/dashboard?tab=weather">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.5 19H8a5 5 0 1 1 1.2-9.85A6 6 0 0 1 21 11.5 3.75 3.75 0 0 1 17.5 19Z"/></svg>
          </div>
          <div class="feature-title">Weather</div>
          <div class="feature-desc">DC 7-day forecast</div>
        </a>

<a class="feature-card" href="/dashboard?tab=chat">
          <div class="feature-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>
              <path d="M8 9h8M8 13h5"/>
            </svg>
          </div>
          <div class="feature-title">Ask AI</div>
          <div class="feature-desc">Chat with the pricing assistant</div>
        </a>
      </div>
    </section>
  </main>
</body>
</html>
''')

@app.route("/dashboard")
def dashboard():
    tab = request.args.get("tab", "dashboard")
    # The sidebar label is Recommendations, but this template pane is named explore.
    if tab == "recommendations":
        tab = "explore"
    return render_page(active_tab=tab)


@app.route("/chat", methods=["POST"])
def chat_route():
    q = (request.form.get("question") or request.form.get("q") or "").strip()
    city = (request.form.get("selected_city") or request.form.get("city") or "Washington").strip()
    fm, _ = get_city_data(city)
    response = answer(q, fm, session.get("history", [])) if q else "<p>Please type a question first.</p>"
    history = session.get("history", [])
    if q:
        history.append({"q": q, "a": response})
        session["history"] = history[-8:]
    return render_page(active_tab="chat", selected_city=city)


@app.route("/clear_chat")
def clear_chat_route():
    city = (request.args.get("selected_city") or request.args.get("city") or "Washington").strip()
    session.pop("history", None)
    return render_page(active_tab="chat", selected_city=city)

@app.route("/explore", methods=["GET", "POST"])
def explore_route():
    # The filter form posts here. Previously this route ignored all submitted
    # fields, so the Recommendations/Filter Listings panel never changed.
    city = request.values.get("city", "Washington")
    hood = (request.values.get("hood", "") or "").strip()
    action_filter = (request.values.get("action_filter", "") or "").strip()
    bedrooms_filter = (request.values.get("bedrooms_filter", "") or "").strip()
    room_type_filter = (request.values.get("room_type_filter", "") or "").strip()

    try:
        min_price = float(request.values.get("min_price", 0) or 0)
    except Exception:
        min_price = 0

    fm, _ = get_city_data(city)
    results = fm.copy()

    if hood and "neighbourhood_cleansed" in results.columns:
        results = results[results["neighbourhood_cleansed"].astype(str).str.contains(hood, case=False, na=False)]

    if min_price > 0 and "recommended_price" in results.columns:
        results = results[results["recommended_price"].fillna(0) >= min_price]

    if action_filter and "action" in results.columns:
        results = results[results["action"].astype(str).str.upper() == action_filter.upper()]

    if bedrooms_filter and "bedrooms" in results.columns:
        if bedrooms_filter == "4":
            results = results[results["bedrooms"].fillna(0) >= 4]
        else:
            try:
                bedrooms_value = int(bedrooms_filter)
                results = results[results["bedrooms"].fillna(0).astype(int) == bedrooms_value]
            except Exception:
                pass

    if room_type_filter and "room_type" in results.columns:
        results = results[results["room_type"].astype(str) == room_type_filter]

    results = results.head(120).copy()
    explore_results = results.to_dict("records")

    return render_page(
        active_tab="explore",
        selected_city=city,
        explore_results=explore_results,
        explore_count=len(results),
        hood=hood,
        min_price=min_price,
        action_filter=action_filter,
        bedrooms_filter=bedrooms_filter,
        room_type_filter=room_type_filter,
    )


@app.route("/recommendations", methods=["GET", "POST"])
def recommendations_route():
    """Compatibility URL: show the Recommendations tab even if a link points to /recommendations."""
    return render_page(active_tab="explore")


@app.route("/simulator", methods=["GET"])
def simulator_route():
    """Compatibility URL: show the Simulator tab even if a link points to /simulator."""
    return render_page(active_tab="simulator")


@app.route("/simulate", methods=["POST"])
def simulate_route():
    """Handle the What-If Price Simulator form."""
    city = (request.form.get("sim_city") or "Washington").strip()
    hood = (request.form.get("sim_hood") or "").strip()
    room = (request.form.get("sim_room") or "Entire home/apt").strip()
    event_type = (request.form.get("sim_event_type") or "None").strip()

    bedrooms = _safe_float(request.form.get("sim_bedrooms", 1), 1)
    accommodates = _safe_float(request.form.get("sim_accommodates", 2), 2)
    rating = _safe_float(request.form.get("sim_rating", 4.5), 4.5)
    distance = _safe_float(request.form.get("sim_distance", 5), 5)

    if event_type.lower() == "none":
        distance = 99

    lat, lon = _HOOD_CENTERS.get(hood, CITY_CENTERS.get(city, CITY_CENTERS.get("Washington", [38.9072, -77.0369])))
    fm, _ = get_city_data(city)

    payload = {
        "city": city,
        "neighbourhood_cleansed": hood,
        "room_type": room,
        "bedrooms": bedrooms,
        "accommodates": accommodates,
        "review_scores_rating": rating,
        "distance_to_event_km": distance,
        "event_type": event_type,
        "latitude": lat,
        "longitude": lon,
    }

    result = _build_ai_pricing_result(payload, fm)
    sim_result = {
        "price": int(round(result.get("price", 0))),
        "monthly_uplift": int(round(result.get("monthly_uplift", 0))),
        "annual_uplift": int(round(result.get("annual_uplift", 0))),
        "demand": result.get("demand", "LOW"),
        "action": result.get("action", "HOLD"),
        "explanation": result.get("explanation", "Pricing recommendation generated from comps, event demand, and listing details."),
    }

    return render_page(active_tab="simulator", selected_city=city, sim_result=sim_result)

if __name__ == "__main__":
    app.run(debug=True)
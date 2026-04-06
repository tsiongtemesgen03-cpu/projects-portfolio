# Airbnb Pricing Advisor – Washington DC

An AI-powered Flask web application that helps Airbnb hosts optimize pricing using machine learning, Retrieval-Augmented Generation (RAG), and event-driven demand analysis.

The system analyzes listings, reviews, and local events to generate dynamic pricing recommendations, actionable insights, and an interactive dashboard.

---

## Quick Start

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API Key
**macOS/Linux:**
```bash
export AZURE_OPENAI_API_KEY="your_api_key_here"
```

**Windows (PowerShell):**
```powershell
$env:AZURE_OPENAI_API_KEY="your_api_key_here"
```

### 4. Build RAG Index
```bash
python build_index.py
```
This creates `index.pkl` from CSV data and generates embeddings (~2-3 minutes).

### 5. Run Application
```bash
python app.py
```
Open browser: **http://localhost:5001**

**Note:** This is a Flask-based application (not Streamlit).

---

## Key Features

- RAG-powered semantic search over listings and reviews  
- Dynamic pricing recommendations (RAISE / DISCOUNT / HOLD)  
- AI chatbot (Claude) for natural language insights  
- Interactive dashboard with KPIs and charts  
- Event-aware pricing optimization  
- Neighborhood-level opportunity detection  

Note: Some charts use simulated data for demonstration purposes.

---

## Project Structure

```
airbnb-rag-advisor/
├── app.py
├── build_index.py
├── chatbot_rag.py
├── requirements.txt
├── index.pkl
├── data/
├── static/
├── templates/
├── model.py
├── simulator.py
├── recommend.py
├── fetch_*.py
├── merge_*.py
```

---

## System Architecture

### RAG Pipeline
```
CSV → Text chunks → Embeddings → index.pkl
```

- Uses SentenceTransformers (or TF-IDF fallback)
- Stores embeddings and metadata for retrieval

### Retrieval and AI
- Retrieves relevant documents via cosine similarity  
- Builds structured context  
- Sends context to Claude API  

### Flask Application
- Loads data, models, and RAG index  
- Computes KPIs and recommendations  
- Renders dashboard and charts  
- Provides chatbot and simulation endpoints  

---

## Setup

### Requirements
- Python 3.9+
- pip
- API key

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Example requirements.txt
```
flask
pandas
numpy
matplotlib
folium
scikit-learn
anthropic
sentence-transformers
python-dotenv
```

## Run the Application

```bash
python app.py
```

- URL: http://localhost:5001  
- Debug mode: enabled (development only)

---

## Dashboard Features

- Revenue opportunity estimation  
- Underpriced listing detection  
- Price recommendations  
- Event-driven demand insights  
- Neighborhood analytics  

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|--------|
| `/` | GET | Landing page |
| `/dashboard` | GET | Dashboard |
| `/explore` | POST | Filter listings |
| `/simulate` | POST | Pricing simulator |
| `/chat` | POST | AI chatbot |
| `/export/listings` | GET | Export CSV |

---

## Security Issues & Fixes

### CRITICAL

**1. Hardcoded Secret Key** (`app.py` line 14)
```python
#  INSECURE
secret_key = 'airbnb2026'

# FIXED
secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key')
```

**2. Debug Mode in Production** (`app.py` line 2724)
```python
# INSECURE
app.run(debug=True, port=5001)

# FIXED
app.run(debug=os.getenv('FLASK_DEBUG', False), port=5001)
```

**3. Unsafe Pickle Deserialization** (`app.py` line 54-65)
```python
# INSECURE: pickle.load allows arbitrary code execution
with open('index.pkl', 'rb') as f:
    data = pickle.load(f)

# Use restricted_loads or JSON instead
```

### HIGH

**4. No Input Validation** - User queries sent directly to Claude without sanitization  
**5. Missing CSRF Protection** - No Flask-WTF CSRF tokens on forms  
**6. Session Cookies Not Secure** - Missing `secure=True`, `httponly=True`, `samesite='Lax'`  
**7. Giant Inline HTML Template** - 2500+ lines in memory, makes XSS harder to audit

### MEDIUM

**8. No Error Handling** - CSV load failures crash app  
**9. Global Mutable State** - Model not thread-safe  
**10. Hardcoded API Keys in Code** - Should all use `os.getenv()`  

---

## Troubleshooting

### Port Already in Use
```bash
# Use a different port
python app.py --port 5002
```

### ImportError: Missing Dependencies
```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt
```

### index.pkl Not Found
```bash
# Rebuild the index
python build_index.py
```

### Claude API Key Error
```bash
# Check environment variable
echo $AZURE_OPENAI_API_KEY  # macOS/Linux
echo $env:AZURE_OPENAI_API_KEY  # Windows
```

### Out of Memory During Index Build
- Reduce CSV size or process in batches
- Use TF-IDF instead (set `USE_EMBEDDINGS=False` in `build_index.py`)

---

## Deployment & Production

### Before Going to Production

1. **Disable Debug Mode**
   ```bash
   export FLASK_DEBUG=False
   ```

2. **Set Strong Secret Key**
   ```bash
   export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
   ```

3. **Use WSGI Server** (not Flask dev server)
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5001 app:app
   ```

4. **Enable HTTPS & Secure Cookies** in `app.py`:
   ```python
   app.config['SESSION_COOKIE_SECURE'] = True
   app.config['SESSION_COOKIE_HTTPONLY'] = True
   app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
   ```

5. **Add Rate Limiting** to prevent abuse

6. **Review Security Checklist** in next section

---

## Recommended Improvements

- [ ] Move inline HTML (lines 2140-2520 in `app.py`) to `/templates/dashboard.html`
- [ ] Move 2000+ lines of JavaScript to `/static/dashboard.js`
- [ ] Add input validation with `bleach` library
- [ ] Replace pickle with JSON or SQLite
- [ ] Add Flask-WTF for CSRF protection
- [ ] Implement comprehensive logging
- [ ] Add unit tests and integration tests
- [ ] Use Flask blueprints to organize routes
- [ ] Add rate limiting with Flask-Limiter
- [ ] Implement proper error pages (404, 500)  

---

## Data Setup

### Input Files Required
- `listings.csv` - Airbnb listing data (must have id, name, price, reviews, neighborhood)
- `events_*.csv` or `events_*.json` - Local events data for demand analysis
- `listings_nyc.csv` - (Optional) NYC comparison data

### Generated Files
- `index.pkl` - RAG index with embeddings/TF-IDF (created by `build_index.py`)
- `merged_data.csv` - Combined listing + event data
- `pricing_recommendations.csv` - Output recommendations

### CSV Schema (Listings)
```
id, name, neighbourhood, price, latitude, longitude, reviews_per_month, ...
```

---

## Notes

- TF-IDF is used as fallback if embeddings are unavailable  
- Some analytics charts use simulated data (clearly marked in dashboard)
- Designed for local development; requires hardening for production
- Tested with Python 3.9+
- Supports both SentenceTransformers and TF-IDF embeddings  

---

## Author

Airbnb Pricing Advisor Project  
April 2026

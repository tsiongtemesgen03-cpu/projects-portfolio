"""
Run this to test the chat answer() function against a set of questions.
Usage: python test_chat.py
"""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Load the same data app.py loads
df      = pd.read_csv('pricing_recommendations_all.csv')
listings = pd.read_csv('listings.csv', usecols=['id','name','neighbourhood_cleansed','room_type','bedrooms','accommodates','review_scores_rating','latitude','longitude'])
listings['city'] = 'Washington'
fm = df.merge(listings, on='id', how='left')
if 'city_x' in fm.columns:
    fm['city'] = fm['city_x']
    fm = fm.drop(columns=['city_x','city_y'], errors='ignore')
fm = fm[fm['city'] == 'Washington'].copy()

# Import the answer function
from app import answer

QUESTIONS = [
    # Basic intent
    "Which listings should I raise prices for?",
    "What listings should I discount?",
    "Which properties are underpriced?",

    # Neighbourhood
    "What should I charge in Capitol Hill?",
    "Best listings near Columbia Heights?",

    # Room/beds
    "Show me 2-bedroom listings I should raise",
    "Any entire home listings near events?",
    "Private rooms I should discount?",

    # Events
    "What events are coming up?",
    "Which listings are near events?",

    # Edge cases / likely to fail
    "What's the average price?",
    "How many listings are in the dataset?",
    "What neighborhoods have the most demand?",
    "Should I list my place this weekend?",
    "What is the weather like?",
    "Tell me something useful",
    "hello",
]

SEP = "─" * 70

for q in QUESTIONS:
    print(f"\n{SEP}")
    print(f"Q: {q}")
    print(f"{SEP}")
    try:
        raw = answer(q, fm)
        # Strip HTML tags for readability
        import re
        text = re.sub(r'<[^>]+>', '', raw).strip()
        text = re.sub(r'\s+', ' ', text)
        print(text[:400] if text else "[empty response]")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\n{SEP}\nDone.\n")

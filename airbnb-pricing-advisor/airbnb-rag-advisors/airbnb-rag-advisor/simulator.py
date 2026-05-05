import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load merged data
df = pd.read_csv('pricing_recommendations_all.csv')
listings_dc = pd.read_csv('listings.csv', usecols=['id','name','neighbourhood_cleansed','room_type','bedrooms','accommodates','review_scores_rating','latitude','longitude'])
listings_dc['city'] = 'Washington'
listings_chicago = pd.read_csv('listings_chicago.csv', usecols=['id','name','neighbourhood_cleansed','room_type','bedrooms','accommodates','review_scores_rating','latitude','longitude'])
listings_chicago['city'] = 'Chicago'
listings = pd.concat([listings_dc, listings_chicago], ignore_index=True)
events = pd.read_csv('events_all_cities.csv')
merged = df.merge(listings, on='id', how='left')
if 'city_x' in merged.columns:
    merged['city'] = merged['city_x']
    merged = merged.drop(columns=['city_x','city_y'], errors='ignore')

# Feature engineering
merged['is_event_nearby'] = (merged['distance_to_event_km'] < 10).astype(int)
merged['is_sports'] = (merged['event_category'] == 'Sports').astype(int) if 'event_category' in merged.columns else 0
merged = pd.get_dummies(merged, columns=['room_type'], drop_first=True)

features = ['bedrooms','accommodates','review_scores_rating',
            'distance_to_event_km','is_event_nearby','is_sports']
features += [c for c in merged.columns if c.startswith('room_type_')]
features = [f for f in features if f in merged.columns]

df_model = merged.dropna(subset=features+['baseline_price'])
X = df_model[features]
y = df_model['baseline_price']

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and metadata
import pickle
with open('simulator_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': features}, f)

# Save neighbourhood list per city
hoods = merged.groupby('city')['neighbourhood_cleansed'].unique().to_dict()
hood_dict = {city: sorted(list(set(n for n in names if str(n) != 'nan'))) 
             for city, names in hoods.items()}

import json
with open('static/neighbourhoods.json', 'w') as f:
    json.dump(hood_dict, f)

print("Simulator model saved!")
print(f"Features: {features}")
print(f"DC neighbourhoods: {len(hood_dict.get('Washington', []))}")
print(f"Chicago neighbourhoods: {len(hood_dict.get('Chicago', []))}")

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import json

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# Bounding boxes for each city so we only match events that are
# geographically plausible (prevents DC listings matching to World Cup
# venues 190+ km away that were mislabelled 'Washington').
CITY_BOUNDS = {
    'Washington': {'lat': (38.7, 39.1), 'lon': (-77.3, -76.8)},
    'New York':   {'lat': (40.4, 41.0), 'lon': (-74.4, -73.6)},
    'Chicago':    {'lat': (41.6, 42.1), 'lon': (-88.1, -87.4)},
    'Los Angeles':{'lat': (33.7, 34.4), 'lon': (-118.8, -117.9)},
}

def get_local_events(events, city):
    """Return events that are within the city's bounding box.

    If a bounding box is defined for the city and zero events fall inside
    it, we return an empty DataFrame — signalling "no local events" — rather
    than falling back to city-label matches whose lat/lon may be wrong (e.g.
    Washington DC events stored with World Cup venue coordinates in other
    states). For cities without a bounding box we fall back to label matching.
    """
    bounds = CITY_BOUNDS.get(city)
    if bounds:
        lat_min, lat_max = bounds['lat']
        lon_min, lon_max = bounds['lon']
        local = events[
            (events['latitude'].between(lat_min, lat_max)) &
            (events['longitude'].between(lon_min, lon_max))
        ]
        # Return local results (may be empty — caller handles that case)
        return local
    # No bounding box defined: fall back to city-label matching
    labeled = events[events['city'] == city]
    return labeled if len(labeled) > 0 else events

def nearest_event(row, events):
    city_events = get_local_events(events, row['city'])
    if len(city_events) == 0:
        return pd.Series({
            'nearest_event': 'No local event',
            'event_date': None,
            'event_category': 'None',
            'distance_to_event_km': 100.0
        })
    distances = city_events.apply(lambda e: haversine(
        float(row['latitude']), float(row['longitude']),
        float(e['latitude']), float(e['longitude'])), axis=1)
    idx = distances.idxmin()
    return pd.Series({
        'nearest_event': city_events.loc[idx, 'event_name'],
        'event_date': city_events.loc[idx, 'date'],
        'event_category': city_events.loc[idx, 'category'],
        'distance_to_event_km': distances.min()
    })

def parse_events_json(filename, city):
    try:
        with open(filename) as f:
            d = json.load(f)
        if '_embedded' not in d:
            print(f"No events in {filename}")
            return pd.DataFrame()
        events = []
        for e in d['_embedded']['events']:
            lat = None; lon = None; city_name = None; category = None
            if '_embedded' in e and 'venues' in e['_embedded'] and len(e['_embedded']['venues']) > 0:
                v = e['_embedded']['venues'][0]
                city_name = v.get('city', {}).get('name')
                if 'location' in v:
                    lat = v['location'].get('latitude')
                    lon = v['location'].get('longitude')
            if 'classifications' in e and len(e['classifications']) > 0:
                category = e['classifications'][0].get('segment', {}).get('name')
            events.append({
                'event_name': e.get('name'),
                'date': e.get('dates', {}).get('start', {}).get('localDate'),
                'city': city_name or city,
                'latitude': lat, 'longitude': lon,
                'category': category
            })
        df = pd.DataFrame(events)
        df['city'] = city
        return df
    except Exception as ex:
        print(f"Error parsing {filename}: {ex}")
        return pd.DataFrame()

# Load all events
events_dc = pd.read_csv('events_dc.csv')
events_dc['city'] = 'Washington'
events_nyc = parse_events_json('events_nyc.json', 'New York')
events_chicago = parse_events_json('events_chicago.json', 'Chicago')
events_la = parse_events_json('events_la.json', 'Los Angeles')
events = pd.concat([events_dc, events_nyc, events_chicago, events_la], ignore_index=True)
events['latitude'] = pd.to_numeric(events['latitude'], errors='coerce')
events['longitude'] = pd.to_numeric(events['longitude'], errors='coerce')
events = events.dropna(subset=['latitude','longitude'])
events.to_csv('events_all_cities.csv', index=False)
print(f"Total events: {len(events)}")
print(events['city'].value_counts())

# Load all listings
city_files = {
    'Washington': 'listings.csv',
    'New York': 'listings_nyc.csv',
    'Chicago': 'listings_chicago.csv',
    'Los Angeles': 'listings_la.csv',
}

all_listings = []
for city, fname in city_files.items():
    try:
        df = pd.read_csv(fname, usecols=['id','name','neighbourhood_cleansed',
                         'room_type','bedrooms','accommodates',
                         'review_scores_rating','latitude','longitude','price'])
        df['city'] = city
        df['price'] = df['price'].replace(r'[\$,]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'] > 0].dropna(subset=['price','latitude','longitude'])
        all_listings.append(df)
        print(f"Loaded {len(df)} listings for {city}")
    except Exception as ex:
        print(f"Error loading {fname}: {ex}")

listings = pd.concat(all_listings, ignore_index=True)
print(f"Total listings: {len(listings)}")

# For LA which is huge, sample to keep it manageable
la_mask = listings['city'] == 'Los Angeles'
non_la = listings[~la_mask]
la_sample = listings[la_mask].sample(min(5000, la_mask.sum()), random_state=42)
listings = pd.concat([non_la, la_sample], ignore_index=True)
print(f"After LA sampling: {len(listings)} listings")

# Match each listing to nearest event in same city
print("Matching listings to events (this may take a few minutes)...")
event_features = listings.apply(lambda row: nearest_event(row, events), axis=1)
merged = pd.concat([listings, event_features], axis=1)

# Feature engineering
merged['is_event_nearby'] = (merged['distance_to_event_km'] < 10).astype(int)
merged['is_sports'] = (merged['event_category'] == 'Sports').astype(int)
merged_dummies = pd.get_dummies(merged, columns=['room_type'], drop_first=True)

features = ['bedrooms','accommodates','review_scores_rating',
            'distance_to_event_km','is_event_nearby','is_sports']
features += [c for c in merged_dummies.columns if c.startswith('room_type_')]
features = [f for f in features if f in merged_dummies.columns]

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

model_df = merged_dummies.dropna(subset=features+['price'])
X = model_df[features]
y = model_df['price']

# ── Train/test split for honest evaluation metrics ──────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
eval_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
eval_model.fit(X_train, y_train)
test_preds = eval_model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, test_preds)
rmse = np.sqrt(mean_squared_error(y_test, test_preds))
r2   = r2_score(y_test, test_preds)
print(f"\nModel evaluation (held-out 20%):")
print(f"  MAPE  = {mape:.2%}")
print(f"  RMSE  = ${rmse:.2f}")
print(f"  R²    = {r2:.3f}\n")

# ── Final model trained on full data ────────────────────────────────────────
print("Training final model on full data...")
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Out-of-fold predictions (5-fold cross-val) to avoid train-predict leakage.
# Every listing gets a prediction made by a model that has never seen it.
print("Generating out-of-fold predictions (5-fold CV)...")
oof_preds = cross_val_predict(model, X, y, cv=5)

model.fit(X, y)  # final fit for the simulator pickle

model_df = model_df.copy()
model_df['baseline_price'] = y.values
model_df['recommended_price'] = oof_preds  # OOF — no leakage
model_df['price_uplift'] = model_df['recommended_price'] - model_df['baseline_price']

# Use ±5 % of baseline as threshold instead of a flat ±$10
model_df['action'] = model_df.apply(
    lambda r: 'RAISE'    if r['price_uplift'] / r['baseline_price'] >  0.05
    else      ('DISCOUNT' if r['price_uplift'] / r['baseline_price'] < -0.05
    else       'HOLD'), axis=1)

# ── Save model metrics & feature importances ────────────────────────────────
feat_imp = dict(zip(features, model.feature_importances_.tolist()))
metrics = {
    'mape': round(mape, 4),
    'rmse': round(rmse, 2),
    'r2':   round(r2, 4),
    'mape_pct': f"{mape:.1%}",
    'n_train': len(X_train),
    'n_test':  len(X_test),
    'n_total': len(model_df),
    'feature_importances': feat_imp,
}
with open('model_metrics.json', 'w') as _f:
    json.dump(metrics, _f, indent=2)
print(f"Saved model metrics → model_metrics.json")

output = model_df[['id','city','baseline_price','recommended_price',
                   'price_uplift','action','nearest_event',
                   'event_date','event_category','distance_to_event_km']].copy()
output.to_csv('pricing_recommendations_all.csv', index=False)
print(f"Saved {len(output)} rows to pricing_recommendations_all.csv")
print(output['city'].value_counts())
print(output['action'].value_counts())

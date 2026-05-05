import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Airbnb Pricing Advisor\n### DNSC 4289/6317 - Group 8\nHana Aglan, Darshan Dullabh, David Porudominsky Rotstain, May Khin Maung Soe, Tsion Temesgen"),

    nbf.v4.new_markdown_cell("## 1. Data Collection & Loading"),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

listings = pd.read_csv('listings.csv')
events = pd.read_csv('events_dc.csv')

print("Listings shape:", listings.shape)
print("Events shape:", events.shape)
events.head()"""),

    nbf.v4.new_markdown_cell("## 2. Data Preparation & Merging"),
    nbf.v4.new_code_cell("""listings = listings[['id','latitude','longitude','price','bedrooms',
                          'accommodates','room_type','neighbourhood_cleansed',
                          'review_scores_rating','minimum_nights']]
listings['price'] = listings['price'].replace(r'[\\$,]', '', regex=True).astype(float)
listings = listings[listings['price'] > 0].dropna(subset=['price'])
print(listings.shape)
listings.head()"""),

    nbf.v4.new_code_cell("""def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def nearest_event(row):
    distances = events.apply(lambda e: haversine(row['latitude'], row['longitude'],
                             e['latitude'], e['longitude']), axis=1)
    idx = distances.idxmin()
    return pd.Series({'nearest_event': events.loc[idx,'event_name'],
                      'event_date': events.loc[idx,'date'],
                      'event_category': events.loc[idx,'category'],
                      'distance_to_event_km': distances.min()})

merged = pd.read_csv('merged_data.csv')
print("Merged shape:", merged.shape)
merged.head()"""),

    nbf.v4.new_markdown_cell("## 3. Feature Engineering"),
    nbf.v4.new_code_cell("""merged['is_event_nearby'] = (merged['distance_to_event_km'] < 10).astype(int)
merged['is_sports'] = (merged['event_category'] == 'Sports').astype(int)
merged = pd.get_dummies(merged, columns=['room_type'], drop_first=True)

features = ['bedrooms','accommodates','review_scores_rating',
            'minimum_nights','distance_to_event_km','is_event_nearby','is_sports']
features += [c for c in merged.columns if c.startswith('room_type_')]
print("Features:", features)"""),

    nbf.v4.new_markdown_cell("## 4. Model Training & Evaluation"),
    nbf.v4.new_code_cell("""from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

df = merged.dropna(subset=features+['price'])
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
for name, model in [('Linear Regression', LinearRegression()),
                    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42))]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {'MAPE': f'{mape:.2%}', 'RMSE': f'${rmse:.2f}'}
    print(f"{name}: MAPE={mape:.2%}, RMSE=${rmse:.2f}")

pd.DataFrame(results).T"""),

    nbf.v4.new_markdown_cell("## 5. Pricing Recommendations"),
    nbf.v4.new_code_cell("""recommendations = pd.read_csv('pricing_recommendations.csv')
print("Action breakdown:")
print(recommendations['action'].value_counts())
recommendations.head(10)"""),

    nbf.v4.new_markdown_cell("## 6. Visualizations"),
    nbf.v4.new_code_cell("""from IPython.display import Image
Image('pricing_charts.png')"""),
]

nb.cells = cells
with open('airbnb_pricing_advisor.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created!")                  

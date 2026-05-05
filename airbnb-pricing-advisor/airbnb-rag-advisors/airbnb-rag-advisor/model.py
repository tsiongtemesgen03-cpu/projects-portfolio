import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Load merged data
df = pd.read_csv('merged_data.csv')

# Drop rows with missing price
df = df.dropna(subset=['price'])
df = df[df['price'] > 0]

# Features
df['is_event_nearby'] = (df['distance_to_event_km'] < 10).astype(int)
df['is_sports'] = (df['event_category'] == 'Sports').astype(int)
df = pd.get_dummies(df, columns=['room_type'], drop_first=True)

features = ['bedrooms', 'accommodates', 'review_scores_rating',
            'minimum_nights', 'distance_to_event_km',
            'is_event_nearby', 'is_sports']
features += [c for c in df.columns if c.startswith('room_type_')]

df = df.dropna(subset=features)
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name}: MAPE={mape:.2%}, RMSE=${rmse:.2f}")

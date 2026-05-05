import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

df = pd.read_csv('merged_data.csv')
df = df.dropna(subset=['price'])
df = df[df['price'] > 0]

# Remove unrealistic source-price outliers before training
df = df[df['price'].between(35, 1000)]

df['is_event_nearby'] = (df['distance_to_event_km'] < 10).astype(int)
df['is_sports'] = (df['event_category'] == 'Sports').astype(int)
df = pd.get_dummies(df, columns=['room_type'], drop_first=True)

# Weather features (added by merge_data.py via DC climate normals)
weather_features = [f for f in ['temp_avg_c', 'precipitation_mm', 'is_outdoor_season', 'is_peak_summer']
                    if f in df.columns]

features = ['bedrooms', 'accommodates', 'review_scores_rating',
            'minimum_nights', 'distance_to_event_km',
            'is_event_nearby', 'is_sports'] + weather_features
features += [c for c in df.columns if c.startswith('room_type_')]

df = df.dropna(subset=features)
X = df[features]
y = df['price']

# ── Honest evaluation on held-out 20% ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
eval_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
eval_model.fit(X_train, y_train)
test_preds = eval_model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, test_preds)
rmse = np.sqrt(mean_squared_error(y_test, test_preds))
r2   = r2_score(y_test, test_preds)
print(f"Evaluation (held-out 20%): MAPE={mape:.2%}  RMSE=${rmse:.2f}  R²={r2:.3f}")

# ── Final model + out-of-fold predictions (no leakage) ──────────────────────
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
oof_preds = cross_val_predict(model, X, y, cv=5)
model.fit(X, y)

df['baseline_price'] = y.values

# Keep model signal, but anchor changes to current price so recommendations stay realistic
raw_preds = np.clip(oof_preds, 35, 650)
ratio = raw_preds / df['baseline_price']
ratio = np.clip(ratio, 0.70, 1.30)   # max -30% to +30%
df['recommended_price'] = (df['baseline_price'] * ratio).clip(35, 650)

df['price_uplift'] = df['recommended_price'] - df['baseline_price']

# ±5 % of baseline instead of a flat ±$10
df['action'] = df.apply(
    lambda r: 'RAISE'    if r['price_uplift'] / r['baseline_price'] >  0.05
    else      ('DISCOUNT' if r['price_uplift'] / r['baseline_price'] < -0.05
    else       'HOLD'), axis=1)

# ── Save metrics ─────────────────────────────────────────────────────────────
feat_imp = dict(zip(features, model.feature_importances_.tolist()))
metrics = {
    'mape': round(mape, 4), 'rmse': round(rmse, 2), 'r2': round(r2, 4),
    'mape_pct': f"{mape:.1%}", 'n_train': len(X_train), 'n_test': len(X_test),
    'n_total': len(df), 'feature_importances': feat_imp,
}
with open('model_metrics.json', 'w') as _f:
    json.dump(metrics, _f, indent=2)
print("Saved metrics → model_metrics.json")

output_cols = ['id', 'baseline_price', 'recommended_price',
               'price_uplift', 'action', 'nearest_event',
               'event_date', 'distance_to_event_km']

# Keep columns the app may already rely on, when present
for optional_col in ['city', 'event_category']:
    if optional_col in df.columns:
        output_cols.insert(1, optional_col) if optional_col == 'city' else output_cols.append(optional_col)

output = df[output_cols].copy()
output.to_csv('pricing_recommendations_all.csv', index=False)
print(output.head(10))
print(f"\nAction breakdown:\n{output['action'].value_counts()}")

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from fetch_weather_dc import get_monthly_weather_features

# Load data
listings = pd.read_csv('listings.csv')
events = pd.read_csv('events_dc.csv')

# Keep only useful listing columns
listings = listings[['id', 'latitude', 'longitude', 'price', 'bedrooms', 
                      'accommodates', 'room_type', 'neighbourhood_cleansed',
                      'review_scores_rating', 'minimum_nights']]

# Clean price column (remove $ and commas)
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)

# Function to calculate distance in km between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# DC bounding box: only match events that are actually in/near DC.
# events_dc.csv contains World Cup venues in other states whose city label
# was overridden to 'Washington', so we filter by lat/lon instead.
DC_LAT = (38.7, 39.1)
DC_LON = (-77.3, -76.8)
local_events = events[
    events['latitude'].between(*DC_LAT) &
    events['longitude'].between(*DC_LON)
]
# If no events fall within the DC bounding box, fall back to all events
# so the pipeline doesn't crash, but distances will be large (flagging the
# data quality issue rather than silently using wrong values).
match_events = local_events if len(local_events) > 0 else events
print(f"DC events in bounding box: {len(local_events)} (of {len(events)} total)")

# For each listing, find nearest event and distance
def nearest_event(row):
    distances = match_events.apply(lambda e: haversine(row['latitude'], row['longitude'],
                                   e['latitude'], e['longitude']), axis=1)
    idx = distances.idxmin()
    return pd.Series({
        'nearest_event': match_events.loc[idx, 'event_name'],
        'event_date': match_events.loc[idx, 'date'],
        'event_category': match_events.loc[idx, 'category'],
        'distance_to_event_km': distances.min()
    })

print("Merging... this may take a minute")
event_features = listings.apply(nearest_event, axis=1)
merged = pd.concat([listings, event_features], axis=1)

# Add DC weather features based on the event's month (uses climate normals, no API needed)
merged['event_month'] = pd.to_datetime(merged['event_date'], errors='coerce').dt.month.fillna(6).astype(int)
weather_features = merged['event_month'].apply(get_monthly_weather_features).apply(pd.Series)
merged = pd.concat([merged, weather_features], axis=1)
merged = merged.drop(columns=['event_month'])

merged.to_csv('merged_data.csv', index=False)
print(f"Done! Saved {len(merged)} rows to merged_data.csv")
print(merged[['id', 'nearest_event', 'temp_avg_c', 'is_outdoor_season']].head())

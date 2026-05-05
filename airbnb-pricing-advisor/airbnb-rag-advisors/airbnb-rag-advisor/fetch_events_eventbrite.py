import requests
import json
import pandas as pd
import os

TOKEN = os.environ.get("EVENTBRITE_TOKEN", "")
BASE_URL = "https://www.eventbriteapi.com/v3/events/search/"

all_events = []
page = 1

while True:
    params = {
        "token": TOKEN,
        "location.address": "Washington DC",
        "location.within": "10mi",
        "expand": "venue",
        "page": page,
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()

    if "error" in data:
        print(f"API error: {data}")
        break

    events = data.get("events", [])
    if not events:
        print("No more events.")
        break

    for e in events:
        venue = e.get("venue") or {}
        lat = venue.get("latitude")
        lon = venue.get("longitude")
        start = e.get("start") or {}
        local_dt = start.get("local", "")
        all_events.append({
            "event_name": (e.get("name") or {}).get("text"),
            "date": local_dt[:10] if local_dt else None,
            "time": local_dt[11:] if local_dt else None,
            "venue": venue.get("name"),
            "city": "Washington",
            "state": "DC",
            "latitude": lat,
            "longitude": lon,
            "category": e.get("category_id"),
            "url": e.get("url"),
        })

    pagination = data.get("pagination", {})
    page_count = pagination.get("page_count", 1)
    print(f"Page {page}/{page_count} — {len(events)} events fetched")

    if page >= page_count:
        break
    page += 1

df = pd.DataFrame(all_events)
df.to_csv("events_dc_eventbrite.csv", index=False)
print(f"\nTotal fetched: {len(df)} events")
print(df[["event_name", "date", "venue"]].head(10))

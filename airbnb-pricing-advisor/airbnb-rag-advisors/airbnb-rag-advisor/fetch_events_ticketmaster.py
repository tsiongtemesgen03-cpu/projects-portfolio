import requests
import json
import pandas as pd
import os

API_KEY = os.environ.get("TICKETMASTER_API_KEY", "")
BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

all_events = []

for page in range(0, 16):
    params = {
        "apikey": API_KEY,
        "city": "Washington",
        "stateCode": "DC",
        "countryCode": "US",
        "size": 50,
        "page": page,
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()

    if "_embedded" not in data:
        print(f"Page {page}: no events, stopping.")
        break

    events = data["_embedded"]["events"]
    for e in events:
        venue = None
        city_name = None
        state = None
        lat = None
        lon = None
        category = None

        if "_embedded" in e and "venues" in e["_embedded"] and e["_embedded"]["venues"]:
            v = e["_embedded"]["venues"][0]
            venue = v.get("name")
            city_name = (v.get("city") or {}).get("name")
            state = (v.get("state") or {}).get("stateCode")
            if "location" in v:
                lat = v["location"].get("latitude")
                lon = v["location"].get("longitude")

        if "classifications" in e and e["classifications"]:
            category = (e["classifications"][0].get("segment") or {}).get("name")

        all_events.append({
            "event_name": e.get("name"),
            "date": (e.get("dates", {}).get("start") or {}).get("localDate"),
            "time": (e.get("dates", {}).get("start") or {}).get("localTime"),
            "venue": venue,
            "city": city_name or "Washington",
            "state": state or "DC",
            "latitude": lat,
            "longitude": lon,
            "category": category,
            "url": e.get("url"),
            "source": "ticketmaster",
        })

    total_pages = data.get("page", {}).get("totalPages", 1)
    print(f"Page {page+1}/{total_pages} — {len(events)} events fetched")
    if page + 1 >= total_pages:
        break

df = pd.DataFrame(all_events)
df.to_csv("events_dc_ticketmaster_full.csv", index=False)
print(f"\nTotal Ticketmaster DC events: {len(df)}")

# Merge with existing DC data
existing = pd.read_csv("events_dc.csv")
existing["source"] = "ticketmaster"

combined = pd.concat([existing, df], ignore_index=True)
combined = combined.drop_duplicates(subset=["event_name", "date"])
combined = combined.dropna(subset=["latitude", "longitude"])
combined.to_csv("events_dc.csv", index=False)
print(f"Total DC events after merge: {len(combined)}")

import json
import pandas as pd

def parse_events(filename, city):
    with open(filename) as f:
        d = json.load(f)
    events = []
    if '_embedded' not in d:
        print(f"No events found in {filename}")
        return pd.DataFrame()
    for e in d['_embedded']['events']:
        venue = None
        city_name = None
        state = None
        lat = None
        lon = None
        category = None
        if '_embedded' in e and 'venues' in e['_embedded'] and len(e['_embedded']['venues']) > 0:
            v = e['_embedded']['venues'][0]
            venue = v.get('name')
            city_name = v.get('city', {}).get('name')
            state = v.get('state', {}).get('stateCode')
            if 'location' in v:
                lat = v['location'].get('latitude')
                lon = v['location'].get('longitude')
        if 'classifications' in e and len(e['classifications']) > 0:
            category = e['classifications'][0].get('segment', {}).get('name')
        events.append({
            'event_name': e.get('name'),
            'date': e.get('dates', {}).get('start', {}).get('localDate'),
            'time': e.get('dates', {}).get('start', {}).get('localTime'),
            'venue': venue,
            'city': city_name or city,
            'state': state,
            'latitude': lat,
            'longitude': lon,
            'category': category,
            'url': e.get('url')
        })
    return pd.DataFrame(events)

dc = pd.read_csv('events_dc.csv')
dc['city'] = 'Washington'
dc['state'] = 'DC'

nyc = parse_events('events_nyc.json', 'New York')
chicago = parse_events('events_chicago.json', 'Chicago')
la = parse_events('events_la.json', 'Los Angeles')

all_events = pd.concat([dc, nyc, chicago, la], ignore_index=True)
all_events = all_events.dropna(subset=['latitude', 'longitude'])
all_events.to_csv('events_all_cities.csv', index=False)
print(f"Total events: {len(all_events)}")
print(all_events['city'].value_counts())

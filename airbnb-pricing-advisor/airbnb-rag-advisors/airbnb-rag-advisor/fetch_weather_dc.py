"""
Weather integration using Open-Meteo (free, no API key).

Entry points:
  get_monthly_weather_features(month_int)  -- DC climate normals for training.
  get_dc_forecast()                        -- DC 7-day forecast (backward-compat wrapper).
  get_forecast_for_city(city_name)         -- 7-day forecast for any supported city.
"""

import requests

DC_LAT = 38.9072
DC_LON = -77.0369

CITY_COORDS = {
    'Washington':  {'lat': 38.9072,  'lon': -77.0369,  'tz': 'America/New_York',    'label': 'Washington DC'},
    'Chicago':     {'lat': 41.8781,  'lon': -87.6298,  'tz': 'America/Chicago',     'label': 'Chicago'},
    'New York':    {'lat': 40.7128,  'lon': -74.0060,  'tz': 'America/New_York',    'label': 'New York City'},
    'Los Angeles': {'lat': 34.0522,  'lon': -118.2437, 'tz': 'America/Los_Angeles', 'label': 'Los Angeles'},
}

# Washington DC monthly climate normals (NOAA 1991-2020 averages).
# month (1-12) -> avg temp °C, avg monthly precipitation mm
DC_CLIMATE = {
    1:  {'temp_avg_c': 3.5,  'precipitation_mm': 80},
    2:  {'temp_avg_c': 4.9,  'precipitation_mm': 72},
    3:  {'temp_avg_c': 9.3,  'precipitation_mm': 95},
    4:  {'temp_avg_c': 14.8, 'precipitation_mm': 82},
    5:  {'temp_avg_c': 19.9, 'precipitation_mm': 104},
    6:  {'temp_avg_c': 24.5, 'precipitation_mm': 91},
    7:  {'temp_avg_c': 27.1, 'precipitation_mm': 94},
    8:  {'temp_avg_c': 26.2, 'precipitation_mm': 99},
    9:  {'temp_avg_c': 22.1, 'precipitation_mm': 90},
    10: {'temp_avg_c': 15.7, 'precipitation_mm': 82},
    11: {'temp_avg_c': 9.7,  'precipitation_mm': 85},
    12: {'temp_avg_c': 5.1,  'precipitation_mm': 91},
}

# Weather code descriptions from WMO standard (Open-Meteo uses these)
WMO_DESCRIPTIONS = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Icy fog',
    51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle',
    61: 'Light rain', 63: 'Rain', 65: 'Heavy rain',
    71: 'Light snow', 73: 'Snow', 75: 'Heavy snow',
    77: 'Snow grains',
    80: 'Light showers', 81: 'Showers', 82: 'Violent showers',
    85: 'Snow showers', 86: 'Heavy snow showers',
    95: 'Thunderstorm', 96: 'Thunderstorm w/ hail', 99: 'Thunderstorm w/ heavy hail',
}


def get_monthly_weather_features(month: int) -> dict:
    """Return DC climate features for training. Month is 1-12."""
    climate = DC_CLIMATE.get(month, DC_CLIMATE[6])
    is_outdoor_season = 1 if 4 <= month <= 10 else 0  # Apr–Oct: pleasant DC tourism months
    is_peak_summer    = 1 if month in (7, 8) else 0   # July–Aug: hottest, highest tourism
    return {
        'temp_avg_c':       climate['temp_avg_c'],
        'precipitation_mm': climate['precipitation_mm'],
        'is_outdoor_season': is_outdoor_season,
        'is_peak_summer':    is_peak_summer,
    }


def _fetch_forecast(lat: float, lon: float, tz: str, city_label: str) -> dict:
    """Internal: fetch 7-day forecast from Open-Meteo for given coordinates."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":        lat,
            "longitude":       lon,
            "daily":           "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
            "current_weather": True,
            "timezone":        tz,
            "forecast_days":   7,
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        cw = data.get("current_weather", {})
        current_temp = cw.get("temperature", "?")
        current_code = cw.get("weathercode", 0)
        current_desc = WMO_DESCRIPTIONS.get(current_code, "Unknown")
        is_outdoor_friendly = current_code in (0, 1, 2) and current_temp > 10

        daily = data.get("daily", {})
        days = []
        for i, date in enumerate(daily.get("time", [])):
            code = daily["weathercode"][i]
            days.append({
                "date":             date,
                "temp_max_c":       daily["temperature_2m_max"][i],
                "temp_min_c":       daily["temperature_2m_min"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
                "description":      WMO_DESCRIPTIONS.get(code, "Unknown"),
            })

        rainy_days = [d for d in days if d["precipitation_mm"] > 5]
        nice_days  = [d for d in days if d["precipitation_mm"] <= 2 and d["temp_max_c"] > 15]
        avg_high   = sum(d["temp_max_c"] for d in days) / len(days) if days else 0

        summary_parts = [f"Current {city_label} weather: {current_desc}, {current_temp}°C."]
        if rainy_days:
            summary_parts.append(
                f"Rain expected on {len(rainy_days)} of the next 7 days "
                f"({', '.join(d['date'][5:] for d in rainy_days[:3])})."
            )
        if nice_days:
            summary_parts.append(
                f"{len(nice_days)} nice outdoor days ahead — good opportunity to raise prices."
            )
        summary_parts.append(f"7-day avg high: {avg_high:.1f}°C.")

        return {
            "city_label":          city_label,
            "current_temp_c":      current_temp,
            "current_description": current_desc,
            "is_outdoor_friendly": is_outdoor_friendly,
            "days":                days,
            "summary":             " ".join(summary_parts),
            "rainy_count":         len(rainy_days),
            "pricing_tip": (
                "Rainy week ahead — consider holding or discounting outdoor-adjacent listings."
                if len(rainy_days) >= 3
                else "Mostly clear this week — good conditions to raise prices on high-demand listings."
            ),
        }
    except Exception:
        return None


def get_forecast_for_city(city_name: str) -> dict:
    """Fetch 7-day forecast for a named city. Returns None if unsupported or fetch fails."""
    coords = CITY_COORDS.get(city_name)
    if not coords:
        return None
    return _fetch_forecast(coords['lat'], coords['lon'], coords['tz'], coords['label'])


def get_dc_forecast() -> dict:
    """Backward-compatible wrapper: fetch DC forecast."""
    return get_forecast_for_city('Washington')


if __name__ == "__main__":
    print("=== DC 7-Day Forecast ===")
    forecast = get_dc_forecast()
    if forecast:
        print(forecast["summary"])
        print()
        for d in forecast["days"]:
            print(f"  {d['date']}  {d['description']:20s}  "
                  f"High {d['temp_max_c']:.0f}°C  Rain {d['precipitation_mm']:.1f}mm")
    else:
        print("Could not fetch forecast (check network connection).")

    print()
    print("=== Monthly Climate Features (June) ===")
    print(get_monthly_weather_features(6))

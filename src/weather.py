from __future__ import annotations
import os
import requests
from dotenv import load_dotenv

# Ensure .env is loaded for the Streamlit process/import order quirks
load_dotenv(override=True)

_SESSION = requests.Session()
_BASE = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(city: str) -> str:
    """Return a short, user-friendly weather string for the given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")  # <-- read at call time
    if not api_key:
        return "Error: Set OPENWEATHER_API_KEY in .env"

    try:
        r = _SESSION.get(
            _BASE,
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=15,
        )
        if r.status_code == 401:
            return "Error: OpenWeather rejected the API key. Double-check OPENWEATHER_API_KEY in your .env."
        if r.status_code == 404:
            return f"Sorry, I couldn't find weather for '{city}'."
        r.raise_for_status()

        data = r.json()
        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels = data["main"]["feels_like"]
        hum = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        name = data.get("name", city)

        return f"Weather in {name}: {desc}. Temp {temp:.1f}°C (feels {feels:.1f}°C), humidity {hum}%, wind {wind} m/s."
    except requests.RequestException as e:
        return f"Error fetching weather: {e}"

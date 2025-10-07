import requests, os

API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

def get_weather(city: str) -> str:
    assert API_KEY, "Set OPENWEATHER_API_KEY in .env"
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    name = data.get("name", city)
    main = data["weather"][0]["description"].capitalize()
    temp = data["main"]["temp"]
    hum  = data["main"]["humidity"]
    wind = data["wind"]["speed"]
    return f"{name}: {main}. Temp {temp}°C, Humidity {hum}%, Wind {wind} m/s."

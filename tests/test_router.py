from src.graph import decide, State

def test_decides_weather_rule_only():
    s: State = {"question": "What's the weather in Delhi today?"}
    out = decide(s)
    assert out["route"] == "weather"

def test_city_extraction_optional():
    s: State = {"question": "Weather in Bengaluru?"}
    out = decide(s)
    # city may or may not parse; route must be weather
    assert out["route"] == "weather"

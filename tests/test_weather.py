import types
from src import weather as W

def test_weather_happy(monkeypatch):
    def fake_get(url, params=None, timeout=15):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {
                    "name": "Delhi",
                    "weather": [{"description": "clear sky"}],
                    "main": {"temp": 29.5, "humidity": 48},
                    "wind": {"speed": 2.5},
                }
        return R()
    monkeypatch.setattr(W, "requests", types.SimpleNamespace(get=fake_get))
    out = W.get_weather("Delhi")
    assert "Delhi" in out and "Temp" in out

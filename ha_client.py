import os
import requests

# Load Home Assistant configuration from environment
HA_BASE_URL = os.getenv("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.getenv("HA_TOKEN")

if not HA_TOKEN:
    raise RuntimeError("Missing HA_TOKEN in environment. Create a long‑lived token in Home Assistant.")

HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

def _get_state(entity_id: str):
    """Fetch a Home Assistant entity state."""
    url = f"{HA_BASE_URL}/api/states/{entity_id}"
    r = requests.get(url, headers=HEADERS, timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"HA error: {r.status_code} → {r.text}")
    return r.json()

# ---- SENSOR HELPERS ----

# These entity IDs must match what Home Assistant creates after adding Tapo integration.
# You will adjust the IDs after pairing the sensors.
ENTITY_TEMP = "sensor.druvinas_temperature"
ENTITY_HUM = "sensor.druvinas_humidity"
ENTITY_LEAK = "binary_sensor.druvinas_leak"
ENTITY_PELLETS = "binary_sensor.druvinas_pellet_level"

def get_current_temperature():
    """Return temperature in °C or None."""
    try:
        data = _get_state(ENTITY_TEMP)
        return float(data["state"])
    except Exception:
        return None

def get_current_humidity():
    """Return humidity % or None."""
    try:
        data = _get_state(ENTITY_HUM)
        return float(data["state"])
    except Exception:
        return None

def get_leak_status():
    """Return True if leak detected, False otherwise."""
    try:
        data = _get_state(ENTITY_LEAK)
        return data["state"] == "on"
    except Exception:
        return None

def get_pellet_level():
    """
    Return True = pellets OK (sensor OFF)
           False = pellets low (sensor ON)
    """
    try:
        data = _get_state(ENTITY_PELLETS)
        return data["state"] == "off"
    except Exception:
        return None

def get_full_state():
    """Return dictionary of all relevant sensors."""
    return {
        "temperature": get_current_temperature(),
        "humidity": get_current_humidity(),
        "leak_detected": get_leak_status(),
        "pellet_ok": get_pellet_level(),
    }
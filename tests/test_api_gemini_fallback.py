"""Step 5 — Gemini failure (missing key, quota, network) does not cause 500."""
import os
from unittest.mock import patch


ZONE_ID = "toronto_zone_001"


def test_missing_key_still_returns_200(client):
    """Simulate missing GEMINI_API_KEY: _get_client raises, zones.py catches it."""
    with patch.dict(os.environ, {}, clear=False):
        # Remove key for this call path by making _get_client raise
        with patch("apps.api.gemini._get_client", side_effect=RuntimeError("GEMINI_API_KEY not set")):
            r = client.get(f"/zones/{ZONE_ID}")
    assert r.status_code == 200
    assert r.json()["gemini_summary"] == ""


def test_network_error_still_returns_200(client):
    """Simulate OpenRouter network failure: generate_zone_summary raises."""
    with patch("apps.api.routes.zones.generate_zone_summary", side_effect=ConnectionError("timeout")):
        r = client.get(f"/zones/{ZONE_ID}")
    assert r.status_code == 200
    assert r.json()["gemini_summary"] == ""


def test_quota_error_still_returns_200(client):
    """Simulate quota exceeded (non-2xx from OpenRouter)."""
    with patch("apps.api.routes.zones.generate_zone_summary", side_effect=Exception("429 quota exceeded")):
        r = client.get(f"/zones/{ZONE_ID}")
    assert r.status_code == 200
    assert r.json()["gemini_summary"] == ""

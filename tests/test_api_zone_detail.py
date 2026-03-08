"""Step 3 — GET /zones/{zone_id} returns 200 with correct schema, no Gemini required."""
from unittest.mock import patch


ZONE_ID = "toronto_zone_001"


def test_zone_detail_status(client):
    with patch("apps.api.routes.zones.generate_zone_summary", side_effect=RuntimeError("no key")):
        r = client.get(f"/zones/{ZONE_ID}")
    assert r.status_code == 200


def test_zone_detail_schema(client):
    with patch("apps.api.routes.zones.generate_zone_summary", side_effect=RuntimeError("no key")):
        r = client.get(f"/zones/{ZONE_ID}")
    body = r.json()
    assert body["zone_id"] == ZONE_ID
    assert body["severity"] in ("low", "moderate", "high", "extreme")
    assert isinstance(body["mean_relative_heat"], (float, int, type(None)))
    assert isinstance(body["top_contributors"], list)
    assert isinstance(body["top_recommendations"], list)
    assert "gemini_summary" in body


def test_zone_detail_gemini_failure_returns_empty_summary(client):
    """Zone detail must return 200 with gemini_summary='' when Gemini raises."""
    with patch("apps.api.routes.zones.generate_zone_summary", side_effect=Exception("quota exceeded")):
        r = client.get(f"/zones/{ZONE_ID}")
    assert r.status_code == 200
    assert r.json()["gemini_summary"] == ""


def test_zone_detail_not_found(client):
    r = client.get("/zones/toronto_zone_999")
    assert r.status_code == 404

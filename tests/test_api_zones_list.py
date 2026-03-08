"""Step 2 — GET /zones?city_id=toronto returns valid GeoJSON FeatureCollection."""


def test_zones_list_status(client):
    r = client.get("/zones?city_id=toronto")
    assert r.status_code == 200


def test_zones_list_feature_collection(client):
    r = client.get("/zones?city_id=toronto")
    body = r.json()
    assert "zones" in body
    zones = body["zones"]
    assert zones["type"] == "FeatureCollection"
    assert isinstance(zones["features"], list)
    assert len(zones["features"]) > 0


def test_zones_list_feature_schema(client):
    r = client.get("/zones?city_id=toronto")
    features = r.json()["zones"]["features"]
    for feat in features:
        assert feat["type"] == "Feature"
        assert "geometry" in feat
        props = feat["properties"]
        assert "zone_id" in props
        assert "severity" in props
        assert props["severity"] in ("low", "moderate", "high", "extreme")
        assert "mean_relative_heat" in props
        assert "top_contributors" in props
        assert isinstance(props["top_contributors"], list)
        assert "top_recommendations" in props
        assert isinstance(props["top_recommendations"], list)

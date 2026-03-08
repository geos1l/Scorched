"""Step 4 — GET /cells?city_id=toronto returns valid GeoJSON FeatureCollection."""


def test_cells_status(client):
    r = client.get("/cells?city_id=toronto")
    assert r.status_code == 200


def test_cells_feature_collection(client):
    r = client.get("/cells?city_id=toronto")
    body = r.json()
    assert "cells" in body
    cells = body["cells"]
    assert cells["type"] == "FeatureCollection"
    assert isinstance(cells["features"], list)
    assert len(cells["features"]) > 0


def test_cells_feature_schema(client):
    r = client.get("/cells?city_id=toronto")
    features = r.json()["cells"]["features"]
    for feat in features[:10]:  # spot-check first 10
        assert feat["type"] == "Feature"
        assert "geometry" in feat
        props = feat["properties"]
        assert "cell_id" in props
        assert "predicted_heat" in props
        assert "severity" in props

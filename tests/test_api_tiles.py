"""Step 7 — /tiles/aoi/* endpoints return correct data and valid PNG images."""

PNG_MAGIC = b"\x89PNG"
TORONTO_LON = (-79.45, -79.35)
TORONTO_LAT = (43.62, 43.68)


# ---------------------------------------------------------------------------
# /tiles/aoi/info
# ---------------------------------------------------------------------------

def test_aoi_info_status(client):
    r = client.get("/tiles/aoi/info")
    assert r.status_code == 200


def test_aoi_info_shape(client):
    bounds = client.get("/tiles/aoi/info").json()["bounds"]
    assert len(bounds) == 4, "bounds must be [minLon, minLat, maxLon, maxLat]"


def test_aoi_info_valid_wgs84(client):
    bounds = client.get("/tiles/aoi/info").json()["bounds"]
    min_lon, min_lat, max_lon, max_lat = bounds
    assert TORONTO_LON[0] < min_lon < max_lon < TORONTO_LON[1], f"lon out of range: {bounds}"
    assert TORONTO_LAT[0] < min_lat < max_lat < TORONTO_LAT[1], f"lat out of range: {bounds}"


def test_aoi_info_bbox_is_ordered(client):
    min_lon, min_lat, max_lon, max_lat = client.get("/tiles/aoi/info").json()["bounds"]
    assert min_lon < max_lon
    assert min_lat < max_lat


# ---------------------------------------------------------------------------
# /tiles/aoi/mask/{tile_id}  (single tile — fast)
# ---------------------------------------------------------------------------

def test_single_tile_status(client):
    r = client.get("/tiles/aoi/mask/tile_250_213")
    assert r.status_code == 200


def test_single_tile_content_type(client):
    r = client.get("/tiles/aoi/mask/tile_250_213")
    assert "image/png" in r.headers["content-type"]


def test_single_tile_png_magic(client):
    r = client.get("/tiles/aoi/mask/tile_250_213")
    assert r.content[:4] == PNG_MAGIC


def test_single_tile_with_extension(client):
    """Accept tile_id with .npy extension too."""
    r = client.get("/tiles/aoi/mask/tile_250_213.npy")
    assert r.status_code == 200


def test_single_tile_not_found(client):
    r = client.get("/tiles/aoi/mask/tile_999_999")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /tiles/aoi/mosaic  (generates 560-tile mosaic — may take ~15-30s first call)
# ---------------------------------------------------------------------------

def test_mosaic_status(client):
    r = client.get("/tiles/aoi/mosaic", timeout=120)
    assert r.status_code == 200


def test_mosaic_content_type(client):
    r = client.get("/tiles/aoi/mosaic", timeout=120)
    assert "image/png" in r.headers["content-type"]


def test_mosaic_png_magic(client):
    r = client.get("/tiles/aoi/mosaic", timeout=120)
    assert r.content[:4] == PNG_MAGIC


def test_mosaic_cached_on_second_call(client):
    """Second call should be instant (cache hit) and return same bytes."""
    r1 = client.get("/tiles/aoi/mosaic", timeout=120)
    r2 = client.get("/tiles/aoi/mosaic", timeout=10)
    assert r1.content == r2.content

"""
GET /tiles/aoi/info           -> { bounds: [minLon, minLat, maxLon, maxLat] }
GET /tiles/aoi/mosaic         -> RGBA PNG mosaic of all AOI segmentation masks
GET /tiles/aoi/mask/{tile_id} -> RGBA PNG of a single mask tile (full 1024x1024)

Mosaic is downsampled to MOSAIC_TILE_SIZE px per tile so the image stays
manageable (~2240x1024 at 128px/tile for a 35x16 grid). First call generates
and caches in memory for the lifetime of the process.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pyproj import Transformer

log = logging.getLogger(__name__)

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]
MASK_DIR = REPO_ROOT / "data" / "processed" / "segmentation_masks"
MASK_INDEX_PATH = MASK_DIR / "mask_index.json"

# Canonical class name -> RGBA
CLASS_COLORS: dict[str, tuple[int, int, int, int]] = {
    "building":   (200,  50,  50, 180),
    "road":       (120, 120, 120, 160),
    "vegetation": ( 50, 180,  50, 170),
    "water":      (  0, 100, 200, 160),
}
FALLBACK_COLOR = (0, 0, 0, 0)  # fully transparent for unknown/background class ids

# Downsample factor for the stitched mosaic (per-tile px).
# 560 tiles x 128px = 2048x4480 canvas — fast and demo-quality.
# Per-tile /mask/{id} endpoint always returns full 1024x1024.
MOSAIC_TILE_SIZE = 128

# Pixels to feather at each tile edge in the mosaic to eliminate hard seams.
FEATHER_PX = 8

_transformer = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)

# Simple in-process cache — generated once at startup, reused for lifetime of server
_mosaic_cache: bytes | None = None


def pregenerate_mosaic() -> None:
    """Call this at server startup to populate the cache before any requests arrive."""
    global _mosaic_cache
    if _mosaic_cache is not None:
        return
    log.info("Pre-generating AOI mosaic at startup...")
    index = _load_index()
    tiles = index["tiles"]
    lut = _build_lut(index["class_map"])

    rows_cols = {name: _tile_row_col(name) for name in tiles}
    all_rows = [rc[0] for rc in rows_cols.values()]
    all_cols = [rc[1] for rc in rows_cols.values()]
    min_row, max_row = min(all_rows), max(all_rows)
    min_col, max_col = min(all_cols), max(all_cols)
    n_rows = max_row - min_row + 1
    n_cols = max_col - min_col + 1

    canvas = np.zeros((n_rows * MOSAIC_TILE_SIZE, n_cols * MOSAIC_TILE_SIZE, 4), dtype=np.uint8)
    missing = 0
    for name, (row, col) in rows_cols.items():
        npy_path = MASK_DIR / name
        if not npy_path.exists():
            missing += 1
            continue
        mask = np.load(npy_path)
        tile_arr = np.array(
            _colorize(mask, lut).resize((MOSAIC_TILE_SIZE, MOSAIC_TILE_SIZE), Image.NEAREST)
        )
        tile_arr = _feather_alpha(tile_arr, FEATHER_PX)
        r_off = (row - min_row) * MOSAIC_TILE_SIZE
        c_off = (col - min_col) * MOSAIC_TILE_SIZE
        canvas[r_off:r_off + MOSAIC_TILE_SIZE, c_off:c_off + MOSAIC_TILE_SIZE] = tile_arr

    if missing:
        log.warning("Mosaic: %d tile(s) missing on disk", missing)

    _mosaic_cache = _to_png(Image.fromarray(canvas, mode="RGBA"))
    log.info("Mosaic ready: %dx%d px, %.1f KB",
             n_cols * MOSAIC_TILE_SIZE, n_rows * MOSAIC_TILE_SIZE, len(_mosaic_cache) / 1024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_index() -> dict[str, Any]:
    with open(MASK_INDEX_PATH) as f:
        return json.load(f)


def _bounds_wgs84(index: dict[str, Any]) -> list[float]:
    """Union of all tile EPSG:3347 bounds → [minLon, minLat, maxLon, maxLat]."""
    tiles = index["tiles"]
    all_bounds = [t["bounds"] for t in tiles.values()]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    lon_min, lat_min = _transformer.transform(minx, miny)
    lon_max, lat_max = _transformer.transform(maxx, maxy)
    return [lon_min, lat_min, lon_max, lat_max]


def _build_lut(class_map: dict[str, str]) -> dict[int, tuple[int, int, int, int]]:
    """Build int class_id → RGBA tuple from mask_index class_map."""
    lut: dict[int, tuple[int, int, int, int]] = {}
    for k, name in class_map.items():
        lut[int(k)] = CLASS_COLORS.get(name, FALLBACK_COLOR)
    return lut


def _colorize(mask: np.ndarray, lut: dict[int, tuple]) -> Image.Image:
    """Convert (H, W) uint8 class mask → RGBA PIL image using vectorised ops.
    Pixels whose class_id is not in lut remain (0,0,0,0) — fully transparent."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for class_id, color in lut.items():
        rgba[mask == class_id] = color
    return Image.fromarray(rgba, mode="RGBA")


def _feather_alpha(arr: np.ndarray, feather: int) -> np.ndarray:
    """Fade the alpha channel to 0 at tile edges so adjacent tiles blend
    smoothly in the mosaic instead of showing hard seams.

    Uses per-axis distance ramps so corners are feathered correctly too.
    """
    arr = arr.copy()
    h, w = arr.shape[:2]
    # Distance from nearest horizontal / vertical edge, clamped to [0, feather]
    row_ramp = np.minimum(np.arange(h), np.arange(h - 1, -1, -1))
    col_ramp = np.minimum(np.arange(w), np.arange(w - 1, -1, -1))
    row_w = np.clip(row_ramp / feather, 0.0, 1.0)
    col_w = np.clip(col_ramp / feather, 0.0, 1.0)
    # 2-D weight = element-wise min so corners get the tightest feather
    weight_2d = np.minimum(row_w[:, None], col_w[None, :])
    arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * weight_2d).astype(np.uint8)
    return arr


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tile_row_col(name: str) -> tuple[int, int]:
    """tile_250_213.npy → (250, 213)."""
    stem = name.removesuffix(".npy")
    parts = stem.split("_")
    return int(parts[1]), int(parts[2])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/aoi/info")
def get_aoi_info():
    """Return WGS84 bounding box of the full AOI mask mosaic."""
    index = _load_index()
    bounds = _bounds_wgs84(index)
    return {"bounds": bounds}  # [minLon, minLat, maxLon, maxLat]


@router.get("/aoi/mosaic")
def get_aoi_mosaic():
    """
    Return the pre-generated RGBA PNG mosaic. Cache is populated at startup
    by pregenerate_mosaic() so this is always an instant response.
    """
    if _mosaic_cache is None:
        raise HTTPException(status_code=503, detail="Mosaic not ready yet — server still starting")
    return Response(content=_mosaic_cache, media_type="image/png")


@router.get("/aoi/mask/{tile_id}")
def get_tile_mask(tile_id: str):
    """
    Return a full-resolution (1024x1024) RGBA PNG for a single mask tile.
    Accept tile_id as stem (tile_250_213) or with extension (tile_250_213.npy).
    """
    index = _load_index()
    tiles = index["tiles"]
    key = tile_id if tile_id.endswith(".npy") else f"{tile_id}.npy"

    if key not in tiles:
        raise HTTPException(status_code=404, detail=f"Tile {tile_id!r} not in AOI mask index")

    npy_path = MASK_DIR / key
    if not npy_path.exists():
        raise HTTPException(status_code=404, detail=f"Mask file not found on disk: {key}")

    lut = _build_lut(index["class_map"])
    mask = np.load(npy_path)
    img = _colorize(mask, lut)
    return Response(content=_to_png(img), media_type="image/png")

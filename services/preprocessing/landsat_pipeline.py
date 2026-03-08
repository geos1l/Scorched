"""
Phase 4 (Real) — Landsat zonal statistics from composite TIF.

Reads the GEE-exported composite TIF and computes per-cell zonal means
for each grid cell in toronto_grid.geojson.

TIF bands (EPSG:3347, 30m resolution):
  Band 1: SR_B2  (blue reflectance)
  Band 2: SR_B3  (green reflectance)
  Band 3: SR_B4  (red reflectance)
  Band 4: SR_B5  (NIR reflectance)
  Band 5: SR_B6  (SWIR1 reflectance)
  Band 6: LST_C  (land surface temperature, Celsius)
  Band 7: NDVI   (precomputed)
  Band 8: brightness (precomputed)

INPUTS:
  data/raw/landsat/toronto_landsat_composite.tif
  data/processed/toronto_grid.geojson

OUTPUTS:
  data/processed/landsat_cell_features.parquet
  Columns: cell_id, ndvi_mean, brightness_mean, nir_mean, lst_c, relative_lst_c

USAGE:
  python -m services.preprocessing.landsat_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TIF_PATH = REPO_ROOT / "data" / "raw" / "landsat" / "toronto_landsat_composite.tif"
GRID_PATH = REPO_ROOT / "data" / "processed" / "toronto_grid.geojson"
OUT_PATH = REPO_ROOT / "data" / "processed" / "landsat_cell_features.parquet"

# Band indices (1-based in rasterio)
BAND_NIR = 4       # SR_B5
BAND_LST = 6       # LST_C
BAND_NDVI = 7      # NDVI
BAND_BRIGHT = 8    # brightness

COLS = ["cell_id", "ndvi_mean", "brightness_mean", "nir_mean", "lst_c", "relative_lst_c"]


def zonal_mean(band_data: np.ndarray, transform, cell_bounds) -> float | None:
    """
    Compute mean of band pixels that fall within cell_bounds (minx, miny, maxx, maxy).
    Returns None if no valid (non-NaN) pixels found.
    """
    minx, miny, maxx, maxy = cell_bounds
    # Convert cell bounds to pixel rows/cols
    row_min, col_min = rowcol(transform, minx, maxy)  # top-left
    row_max, col_max = rowcol(transform, maxx, miny)  # bottom-right

    # Clip to raster extent
    h, w = band_data.shape
    r0 = max(0, int(row_min))
    r1 = min(h, int(row_max) + 1)
    c0 = max(0, int(col_min))
    c1 = min(w, int(col_max) + 1)

    if r0 >= r1 or c0 >= c1:
        return None

    patch = band_data[r0:r1, c0:c1]
    valid = patch[~np.isnan(patch)]
    if len(valid) == 0:
        return None
    return float(np.mean(valid))


def main() -> None:
    log.info("Loading grid from %s...", GRID_PATH)
    grid = gpd.read_file(GRID_PATH)
    log.info("Grid: %d cells, CRS: %s", len(grid), grid.crs)

    log.info("Opening TIF: %s", TIF_PATH)
    with rasterio.open(TIF_PATH) as src:
        log.info("TIF CRS: %s, Shape: %s, Bounds: %s", src.crs, src.shape, src.bounds)

        # Ensure grid is in same CRS as TIF
        if str(grid.crs) != str(src.crs):
            log.info("Reprojecting grid from %s to %s", grid.crs, src.crs)
            grid = grid.to_crs(src.crs)

        # Read all needed bands into memory (TIF is ~100MB max)
        log.info("Reading bands...")
        nir_band = src.read(BAND_NIR).astype(np.float64)
        lst_band = src.read(BAND_LST).astype(np.float64)
        ndvi_band = src.read(BAND_NDVI).astype(np.float64)
        bright_band = src.read(BAND_BRIGHT).astype(np.float64)
        transform = src.transform

        tif_bounds = src.bounds

    log.info("Computing zonal stats for %d cells...", len(grid))

    records = []
    aoi_count = 0
    nan_count = 0

    for idx, row in grid.iterrows():
        cell_id = row["cell_id"]
        geom = row.geometry
        b = geom.bounds  # (minx, miny, maxx, maxy)

        # Quick check: cell overlaps TIF
        if (b[2] < tif_bounds.left or b[0] > tif_bounds.right or
                b[3] < tif_bounds.bottom or b[1] > tif_bounds.top):
            records.append({
                "cell_id": cell_id,
                "ndvi_mean": np.nan,
                "brightness_mean": np.nan,
                "nir_mean": np.nan,
                "lst_c": np.nan,
                "relative_lst_c": np.nan,
            })
            nan_count += 1
            continue

        ndvi = zonal_mean(ndvi_band, transform, b)
        brightness = zonal_mean(bright_band, transform, b)
        nir = zonal_mean(nir_band, transform, b)
        lst = zonal_mean(lst_band, transform, b)

        if all(v is not None for v in [ndvi, brightness, nir, lst]):
            records.append({
                "cell_id": cell_id,
                "ndvi_mean": ndvi,
                "brightness_mean": brightness,
                "nir_mean": nir,
                "lst_c": lst,
                "relative_lst_c": np.nan,  # filled after median calc
            })
            aoi_count += 1
        else:
            records.append({
                "cell_id": cell_id,
                "ndvi_mean": np.nan,
                "brightness_mean": np.nan,
                "nir_mean": np.nan,
                "lst_c": np.nan,
                "relative_lst_c": np.nan,
            })
            nan_count += 1

        if (idx + 1) % 5000 == 0:
            log.info("  Processed %d / %d cells (valid so far: %d)", idx + 1, len(grid), aoi_count)

    df = pd.DataFrame(records)

    # Compute relative_lst_c from median of valid cells
    valid_lst = df["lst_c"].dropna()
    if len(valid_lst) > 0:
        median_lst = float(valid_lst.median())
        log.info("Median LST (valid cells): %.3f °C", median_lst)
        df["relative_lst_c"] = df["lst_c"] - median_lst
    else:
        log.warning("No valid LST values found — relative_lst_c will be all NaN")

    df = df[COLS]
    df.to_parquet(OUT_PATH, index=False)

    log.info(
        "Written: %s (%d rows, %d with data, %d NaN)",
        OUT_PATH, len(df), aoi_count, nan_count,
    )
    if aoi_count > 0:
        valid = df.dropna(subset=["relative_lst_c"])
        log.info(
            "relative_lst_c stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            valid["relative_lst_c"].mean(),
            valid["relative_lst_c"].std(),
            valid["relative_lst_c"].min(),
            valid["relative_lst_c"].max(),
        )


if __name__ == "__main__":
    main()

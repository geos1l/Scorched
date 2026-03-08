"""
Phase 4 (MVP) — Landsat stub: generate plausible landsat features for AOI cells.

Real GEE Landsat pipeline is future work. For MVP, we generate random but
realistic values for cells inside the demo AOI, and NaN for all other cells.

AOI bbox (WGS84): min_lon -79.4071, min_lat 43.6354, max_lon -79.3821, max_lat 43.6534

OUTPUT:
  data/processed/landsat_cell_features.parquet
  Columns: cell_id, ndvi_mean, brightness_mean, nir_mean, lst_c, relative_lst_c

USAGE:
  python -m services.preprocessing.landsat_stub
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_PATH = REPO_ROOT / "data" / "processed" / "toronto_grid.geojson"
OUT_PATH = REPO_ROOT / "data" / "processed" / "landsat_cell_features.parquet"

AOI_BBOX_WGS84 = (-79.4071, 43.6354, -79.3821, 43.6534)  # min_lon, min_lat, max_lon, max_lat

COLS = ["cell_id", "ndvi_mean", "brightness_mean", "nir_mean", "lst_c", "relative_lst_c"]


def main() -> None:
    log.info("Loading grid from %s...", GRID_PATH)
    grid = gpd.read_file(GRID_PATH)
    log.info("Grid loaded: %d cells", len(grid))

    # Convert AOI bbox to EPSG:3347
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True)
    sw_x, sw_y = transformer.transform(AOI_BBOX_WGS84[0], AOI_BBOX_WGS84[1])
    ne_x, ne_y = transformer.transform(AOI_BBOX_WGS84[2], AOI_BBOX_WGS84[3])
    aoi_box = box(sw_x, sw_y, ne_x, ne_y)

    aoi_mask = grid.geometry.intersects(aoi_box)
    aoi_cells = grid[aoi_mask]["cell_id"].tolist()
    all_cells = grid["cell_id"].tolist()
    log.info("AOI cells: %d / %d total", len(aoi_cells), len(all_cells))

    rng = np.random.default_rng(seed=42)
    n = len(aoi_cells)

    lst_c = rng.uniform(28, 42, size=n)
    median_lst = float(np.median(lst_c))
    relative_lst_c = lst_c - median_lst

    aoi_df = pd.DataFrame({
        "cell_id": aoi_cells,
        "ndvi_mean": rng.uniform(0.15, 0.45, size=n),
        "brightness_mean": rng.uniform(0.10, 0.30, size=n),
        "nir_mean": rng.uniform(0.15, 0.35, size=n),
        "lst_c": lst_c,
        "relative_lst_c": relative_lst_c,
    })

    # Non-AOI cells get NaN (Phase 6 will impute)
    non_aoi_cells = [c for c in all_cells if c not in set(aoi_cells)]
    nan_df = pd.DataFrame({
        "cell_id": non_aoi_cells,
        "ndvi_mean": np.nan,
        "brightness_mean": np.nan,
        "nir_mean": np.nan,
        "lst_c": np.nan,
        "relative_lst_c": np.nan,
    })

    df = pd.concat([aoi_df, nan_df], ignore_index=True)
    df = df[COLS]

    df.to_parquet(OUT_PATH, index=False)
    log.info("Written: %s (%d rows, %d AOI, %d NaN)", OUT_PATH, len(df), len(aoi_cells), len(non_aoi_cells))
    log.info("relative_lst_c stats (AOI): mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
             aoi_df["relative_lst_c"].mean(), aoi_df["relative_lst_c"].std(),
             aoi_df["relative_lst_c"].min(), aoi_df["relative_lst_c"].max())


if __name__ == "__main__":
    main()

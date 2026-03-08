"""
services/preprocessing/gis_pipeline.py
Phase 3 — GIS Pipeline (Georgio)

Downloads OSM roads, parks, and water layers for Toronto via osmnx.
Computes per-cell GIS coverage features.
Merges with Julie's StatCan building coverage (statcan_buildings.parquet).

Output:  data/processed/gis_cell_features.parquet
Columns: cell_id | gis_building_coverage | gis_road_coverage
         gis_park_coverage | water_distance_m

All coverage values are floats in [0.0, 1.0].
water_distance_m is in metres, non-negative.

Run:
    python -m services.preprocessing.gis_pipeline
    # or
    python services/preprocessing/gis_pipeline.py

Dependencies:
    geopandas, osmnx, shapely, pandas, numpy, pyarrow
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# -- osmnx compat: >=1.3 renamed geometries_from_place -> features_from_place --
try:
    import osmnx as ox
    _osmnx_features = getattr(ox, "features_from_place", None) or ox.geometries_from_place
except ImportError as e:
    raise ImportError("osmnx is required. Install with: pip install osmnx") from e

# --- Logging -----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Paths & Config ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRID_PATH      = PROJECT_ROOT / "data" / "processed" / "toronto_grid.geojson"
BUILDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "statcan_buildings.parquet"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "processed" / "gis_cell_features.parquet"

CRS   = "EPSG:3347"            # Statistics Canada Lambert -- project standard
PLACE = "Toronto, Ontario, Canada"

# Approximate half-width of road surface in metres (buffer applied to centerlines).
# Urban arterials ~7 m, local streets ~4 m -> 5 m is a reasonable average half-width.
ROAD_BUFFER_M = 5.0


# --- OSM Download ------------------------------------------------------------

def _osmnx_get_features(place: str, tags: dict) -> gpd.GeoDataFrame:
    """Thin wrapper around osmnx features/geometries_from_place for compat."""
    try:
        return ox.features_from_place(place, tags=tags)
    except AttributeError:
        return ox.geometries_from_place(place, tags=tags)  # osmnx < 1.3


def download_roads(boundary_geom) -> gpd.GeoDataFrame:
    """
    Download OSM road network for Toronto, buffer centerlines to approximate
    road surface area, and clip to the city boundary.

    Returns a GeoDataFrame of buffered road polygons in EPSG:3347.
    """
    log.info("Downloading OSM road network for '%s' ...", PLACE)
    G = ox.graph_from_place(PLACE, network_type="all", retain_all=False)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges.to_crs(CRS)

    log.info("  Buffering road centerlines by %.1f m to approximate road surface ...", ROAD_BUFFER_M)
    edges = edges[["geometry"]].copy()
    edges["geometry"] = edges.geometry.buffer(ROAD_BUFFER_M)

    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_geom], crs=CRS)
    roads = gpd.clip(edges, boundary_gdf)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].reset_index(drop=True)
    log.info("  Road segments after clip: %d", len(roads))
    return roads[["geometry"]]


def download_parks(boundary_geom) -> gpd.GeoDataFrame:
    """
    Download OSM parks/leisure green-space polygons for Toronto.

    Returns a GeoDataFrame of park polygons in EPSG:3347.
    """
    log.info("Downloading OSM parks for '%s' ...", PLACE)
    raw = _osmnx_get_features(PLACE, tags={"leisure": "park"})
    parks = raw.to_crs(CRS)
    parks = parks[parks.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_geom], crs=CRS)
    parks = gpd.clip(parks[["geometry"]], boundary_gdf)
    parks = parks[parks.geometry.notna() & ~parks.geometry.is_empty].reset_index(drop=True)
    log.info("  Park polygons after clip: %d", len(parks))
    return parks[["geometry"]]


def download_water(boundary_geom) -> gpd.GeoDataFrame:
    """
    Download OSM water body polygons for Toronto (natural=water + waterway bodies).

    Returns a GeoDataFrame of water polygons in EPSG:3347.
    """
    log.info("Downloading OSM water bodies for '%s' ...", PLACE)
    raw = _osmnx_get_features(PLACE, tags={"natural": "water"})
    water = raw.to_crs(CRS)
    water = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]]

    # Also pull waterway polygon features (e.g. rivers mapped as areas)
    try:
        raw2 = _osmnx_get_features(PLACE, tags={"waterway": True})
        ww = raw2.to_crs(CRS)
        ww = ww[ww.geometry.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]]
        water = pd.concat([water, ww], ignore_index=True)
        log.info("  Merged waterway polygons into water layer.")
    except Exception:
        log.info("  No waterway polygon features found -- using natural=water only.")

    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_geom], crs=CRS)
    water = gpd.clip(water, boundary_gdf)
    water = water[water.geometry.notna() & ~water.geometry.is_empty].reset_index(drop=True)
    log.info("  Water polygons after clip: %d", len(water))
    return water[["geometry"]]


# --- Coverage Computation ----------------------------------------------------

def compute_area_coverage(
    grid: gpd.GeoDataFrame,
    features: gpd.GeoDataFrame,
    col_name: str,
) -> pd.Series:
    """
    For each cell in grid, compute the fraction of cell area covered by features.

    Strategy:
      1. Dissolve all feature geometries into a single shapely union (fast for lookup).
      2. Intersect each cell polygon with the union and divide by cell area.

    Returns a Series (same index as grid) with values in [0.0, 1.0].
    """
    log.info("Computing %s per cell ...", col_name)

    if features.empty:
        log.warning("  No features provided for %s -- all values set to 0.0.", col_name)
        return pd.Series(0.0, index=grid.index, dtype=float)

    # Build a single union of all feature geometries for O(1) intersection lookup
    log.info("  Building spatial union of %d feature geometries ...", len(features))
    feature_union = unary_union(features.geometry.values)

    coverages = np.empty(len(grid), dtype=float)
    for i, cell_geom in enumerate(grid.geometry):
        try:
            intersection = cell_geom.intersection(feature_union)
            cov = intersection.area / cell_geom.area if cell_geom.area > 0 else 0.0
        except Exception:
            cov = 0.0
        coverages[i] = min(cov, 1.0)   # guard against floating-point overshoot

    result = pd.Series(coverages, index=grid.index, dtype=float)
    log.info(
        "  %s -- mean=%.3f  max=%.3f  cells with any coverage=%d",
        col_name, result.mean(), result.max(), (result > 0).sum()
    )
    return result


def compute_water_distance(
    grid: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
) -> pd.Series:
    """
    For each cell centroid in grid, compute the distance (metres) to the nearest
    water body geometry.

    Returns a Series (same index as grid) with non-negative float values.
    """
    log.info("Computing water_distance_m per cell centroid ...")

    if water.empty:
        log.warning("  No water features -- water_distance_m set to 9999.0 for all cells.")
        return pd.Series(9999.0, index=grid.index, dtype=float)

    water_union = unary_union(water.geometry.values)
    centroids = grid.geometry.centroid

    distances = centroids.apply(lambda pt: pt.distance(water_union))
    log.info(
        "  water_distance_m -- mean=%.1f m  min=%.1f m  max=%.1f m",
        distances.mean(), distances.min(), distances.max()
    )
    return distances.astype(float)


# --- Merge & Validate --------------------------------------------------------

def merge_buildings(result: pd.DataFrame, buildings_path: Path) -> pd.DataFrame:
    """
    Load Julie's statcan_buildings.parquet and left-join gis_building_coverage
    onto the OSM result DataFrame by cell_id.

    Missing cells are filled with 0.0 and a warning is logged.
    """
    if not buildings_path.exists():
        raise FileNotFoundError(
            f"statcan_buildings.parquet not found at:\n  {buildings_path}\n"
            "Wait for Julie to complete her Phase 3 (StatCan buildings) before running this script."
        )

    log.info("Loading StatCan building coverage from %s ...", buildings_path)
    buildings_df = pd.read_parquet(buildings_path)

    required_cols = {"cell_id", "gis_building_coverage"}
    missing_cols = required_cols - set(buildings_df.columns)
    if missing_cols:
        raise ValueError(
            f"statcan_buildings.parquet is missing required columns: {missing_cols}. "
            "Coordinate with Julie to confirm column names match the Cell Schema."
        )

    before = len(result)
    result = result.merge(
        buildings_df[["cell_id", "gis_building_coverage"]],
        on="cell_id",
        how="left",
    )
    assert len(result) == before, "Merge changed row count -- duplicate cell_ids in buildings parquet?"

    n_missing = result["gis_building_coverage"].isna().sum()
    if n_missing > 0:
        log.warning(
            "  %d cells have no building coverage after merge -- filling with 0.0. "
            "Check that statcan_buildings.parquet covers all cell_ids in toronto_grid.geojson.",
            n_missing
        )
        result["gis_building_coverage"] = result["gis_building_coverage"].fillna(0.0)

    log.info(
        "  gis_building_coverage -- mean=%.3f  max=%.3f",
        result["gis_building_coverage"].mean(),
        result["gis_building_coverage"].max(),
    )
    return result


def validate_output(df: pd.DataFrame, expected_n_cells: int) -> None:
    """
    Assert output schema and value ranges match Phase 3 acceptance criteria.
    Raises AssertionError on any violation.
    """
    log.info("Validating output ...")

    required_cols = ["cell_id", "gis_building_coverage", "gis_road_coverage",
                     "gis_park_coverage", "water_distance_m"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert df["cell_id"].nunique() == len(df), \
        f"Duplicate cell_ids in output ({df['cell_id'].duplicated().sum()} duplicates)."

    assert len(df) == expected_n_cells, \
        f"Row count mismatch: output has {len(df)} rows but grid has {expected_n_cells} cells."

    for col in ["gis_building_coverage", "gis_road_coverage", "gis_park_coverage"]:
        out_of_range = ((df[col] < 0.0) | (df[col] > 1.0)).sum()
        if out_of_range > 0:
            log.warning("  %d values out of [0, 1] in %s -- clipping.", out_of_range, col)

    assert (df["water_distance_m"] >= 0.0).all(), \
        "Negative water_distance_m values found."

    log.info("  All acceptance criteria passed.")


# --- Main --------------------------------------------------------------------

def run() -> pd.DataFrame:
    # 1. Load Toronto grid (Julie Phase 1 output)
    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"toronto_grid.geojson not found at:\n  {GRID_PATH}\n"
            "Wait for Julie to complete Phase 1 before running this script."
        )

    log.info("Loading grid from %s ...", GRID_PATH)
    grid = gpd.read_file(GRID_PATH)
    if grid.crs is None or grid.crs.to_epsg() != 3347:
        log.info("  Reprojecting grid to %s ...", CRS)
        grid = grid.to_crs(CRS)
    log.info("  Grid loaded: %d cells.", len(grid))

    if "cell_id" not in grid.columns:
        raise ValueError("toronto_grid.geojson is missing 'cell_id' column.")

    n_cells = len(grid)

    # Toronto boundary as dissolved union of all grid cells
    boundary_geom = unary_union(grid.geometry.values)

    # 2. Download OSM layers
    roads = download_roads(boundary_geom)
    parks = download_parks(boundary_geom)
    water = download_water(boundary_geom)

    # 3. Compute per-cell GIS features
    grid = grid.copy()
    grid["gis_road_coverage"] = compute_area_coverage(grid, roads, "gis_road_coverage")
    grid["gis_park_coverage"] = compute_area_coverage(grid, parks, "gis_park_coverage")
    grid["water_distance_m"]  = compute_water_distance(grid, water)

    # 4. Merge StatCan building coverage (Julie Phase 3 output)
    result = grid[["cell_id", "gis_road_coverage", "gis_park_coverage",
                   "water_distance_m"]].copy()
    result = merge_buildings(result, BUILDINGS_PATH)

    # 5. Enforce canonical column order (Cell Schema)
    result = result[[
        "cell_id",
        "gis_building_coverage",
        "gis_road_coverage",
        "gis_park_coverage",
        "water_distance_m",
    ]]

    # 6. Clip coverage values to [0, 1] for safety
    for col in ["gis_building_coverage", "gis_road_coverage", "gis_park_coverage"]:
        result[col] = result[col].clip(0.0, 1.0)

    # 7. Validate
    validate_output(result, n_cells)

    # 8. Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %d rows -> %s", len(result), OUTPUT_PATH)

    # Final summary
    log.info("=" * 60)
    log.info("Phase 3 GIS Pipeline complete.")
    log.info("  Cells:                 %d", n_cells)
    log.info("  gis_building_coverage: mean=%.3f  max=%.3f",
             result["gis_building_coverage"].mean(), result["gis_building_coverage"].max())
    log.info("  gis_road_coverage:     mean=%.3f  max=%.3f",
             result["gis_road_coverage"].mean(), result["gis_road_coverage"].max())
    log.info("  gis_park_coverage:     mean=%.3f  max=%.3f",
             result["gis_park_coverage"].mean(), result["gis_park_coverage"].max())
    log.info("  water_distance_m:      mean=%.1f m  min=%.1f m  max=%.1f m",
             result["water_distance_m"].mean(),
             result["water_distance_m"].min(),
             result["water_distance_m"].max())
    log.info("=" * 60)

    return result


if __name__ == "__main__":
    run()
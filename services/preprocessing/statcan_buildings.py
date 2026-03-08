"""
Phase 3 — StatCan Building Coverage (Julie's half)

Downloads StatCan Open Database of Buildings for Ontario, clips to Toronto,
and computes gis_building_coverage (fraction of each 100m cell covered by
building footprints). Output is handed to Georgio who merges it with OSM
features into gis_cell_features.parquet.

Input:
  data/raw/statcan_buildings/ODB_v3_ON_1.zip   <- download from open.canada.ca

Output:
  data/processed/statcan_buildings.parquet
    Columns: cell_id (str), gis_building_coverage (float in [0, 1])
    One row per grid cell — all 68,394 cells present.

Run:
  python services/preprocessing/statcan_buildings.py
"""

import zipfile
import geopandas as gpd
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ODB_PATH      = PROJECT_ROOT / "data/raw/statcan_buildings/ODB_v3_ON_1.zip"
GRID_PATH     = PROJECT_ROOT / "data/processed/toronto_grid.geojson"
BOUNDARY_PATH = PROJECT_ROOT / "data/city_configs/toronto_boundary.geojson"
OUTPUT_PATH   = PROJECT_ROOT / "data/processed/statcan_buildings.parquet"

TARGET_CRS = "EPSG:3347"


# ── Step 1: Load Toronto boundary ─────────────────────────────────────────────

def load_boundary(path: Path) -> gpd.GeoDataFrame:
    """Loads Toronto boundary and reprojects to EPSG:3347."""
    print("Loading Toronto boundary...")
    boundary = gpd.read_file(path).dissolve().to_crs(TARGET_CRS)
    print(f"  CRS: {boundary.crs}")
    return boundary


# ── Step 2: Load + clip ODB buildings ─────────────────────────────────────────

def load_buildings(odb_path: Path, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Loads the ODB GeoPackage from inside the zip and clips it to Toronto.

    The ODB is distributed as a .zip containing a .gpkg. GDAL's /vsizip/
    virtual filesystem lets geopandas read it directly without unzipping.
    The ODB already uses EPSG:3347 so no reprojection is needed.
    We clip early to reduce from ~millions of Ontario buildings to Toronto only.
    """
    print(f"Loading ODB buildings from: {odb_path}")
    print("  (This may take 1-2 minutes for a large provincial file...)")

    # Resolve the .gpkg filename inside the zip
    with zipfile.ZipFile(odb_path) as zf:
        gpkg_name = next(n for n in zf.namelist() if n.endswith(".gpkg"))

    vsi_path = f"/vsizip/{odb_path}/{gpkg_name}"
    buildings = gpd.read_file(vsi_path)
    print(f"  Loaded {len(buildings):,} buildings (province-wide)")
    print(f"  Buildings CRS: {buildings.crs}")

    # Reproject if needed (older ODB versions may differ)
    if buildings.crs is None or buildings.crs.to_epsg() != 3347:
        print("  Reprojecting buildings to EPSG:3347...")
        buildings = buildings.to_crs(TARGET_CRS)

    # Clip to Toronto — reduces dataset significantly before heavy computation
    print("  Clipping to Toronto boundary...")
    toronto_poly = boundary.geometry.union_all()
    buildings_toronto = buildings[buildings.geometry.intersects(toronto_poly)].copy()
    buildings_toronto = gpd.clip(buildings_toronto, boundary)

    print(f"  Buildings after clip: {len(buildings_toronto):,}")
    return buildings_toronto


# ── Step 3: Load grid ─────────────────────────────────────────────────────────

def load_grid(grid_path: Path) -> gpd.GeoDataFrame:
    """Loads the toronto_grid.geojson produced in Phase 1."""
    print(f"Loading grid from: {grid_path}")
    grid = gpd.read_file(grid_path)
    print(f"  Grid cells: {len(grid):,}  |  CRS: {grid.crs}")
    return grid


# ── Step 4: Compute building coverage per cell ────────────────────────────────

def compute_building_coverage(
    buildings: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    For each grid cell:
        gis_building_coverage = sum(building fragment area inside cell) / cell area

    Uses gpd.overlay(how="intersection") to compute building fragments per cell,
    sums fragment areas per cell_id, divides by cell area. Cells with no buildings
    are filled with 0.0 via a left join. Result is clamped to [0, 1].
    """
    print("Computing building coverage per cell...")
    print("  Running spatial overlay (intersection)... this may take a few minutes")

    grid = grid.copy()
    grid["cell_area_m2"] = grid.geometry.area

    # Each row in result = one building fragment clipped to one cell
    intersected = gpd.overlay(
        buildings[["geometry"]],
        grid[["cell_id", "cell_area_m2", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    print(f"  Intersection produced {len(intersected):,} building fragments")

    intersected["fragment_area_m2"] = intersected.geometry.area

    building_area_per_cell = (
        intersected
        .groupby("cell_id")["fragment_area_m2"]
        .sum()
        .reset_index()
        .rename(columns={"fragment_area_m2": "total_building_area_m2"})
    )

    merged = grid[["cell_id", "cell_area_m2"]].merge(
        building_area_per_cell, on="cell_id", how="left"
    )
    merged["total_building_area_m2"] = merged["total_building_area_m2"].fillna(0.0)
    merged["gis_building_coverage"] = (
        merged["total_building_area_m2"] / merged["cell_area_m2"]
    ).clip(0.0, 1.0)

    print(f"  Coverage — min: {merged['gis_building_coverage'].min():.4f}  "
          f"max: {merged['gis_building_coverage'].max():.4f}  "
          f"mean: {merged['gis_building_coverage'].mean():.4f}")
    print(f"  Cells with 0 buildings: {(merged['gis_building_coverage'] == 0).sum():,}")

    return merged[["cell_id", "gis_building_coverage"]]


# ── Step 5: Validate ──────────────────────────────────────────────────────────

def validate(result: pd.DataFrame, grid: gpd.GeoDataFrame) -> None:
    """Checks Phase 3 acceptance criteria from JULIE.md."""
    print("\nValidating output...")

    missing = set(grid["cell_id"]) - set(result["cell_id"])
    assert len(missing) == 0, f"FAIL: {len(missing)} cell_ids missing from grid"
    print(f"  All {len(result):,} cell_ids present")

    dupes = result["cell_id"].duplicated().sum()
    assert dupes == 0, f"FAIL: {dupes} duplicate cell_ids"
    print(f"  No duplicate cell_ids")

    out_of_range = result[
        (result["gis_building_coverage"] < 0.0) |
        (result["gis_building_coverage"] > 1.0)
    ]
    assert len(out_of_range) == 0, f"FAIL: {len(out_of_range)} coverage values out of [0,1]"
    print(f"  All coverage values in [0.0, 1.0]")

    nulls = result["gis_building_coverage"].isna().sum()
    assert nulls == 0, f"FAIL: {nulls} null values"
    print(f"  No null values")

    print("  All checks passed.\n")


# ── Step 6: Save ──────────────────────────────────────────────────────────────

def save(result: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"  Rows: {len(result):,}  |  Columns: {list(result.columns)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 3: StatCan Building Coverage ===\n")

    if not ODB_PATH.exists():
        print(f"ERROR: ODB zip not found at {ODB_PATH}")
        print("Move your downloaded ODB_v3_ON_1.zip to that path and re-run.")
        return

    if not GRID_PATH.exists():
        print(f"ERROR: Grid not found at {GRID_PATH}")
        print("Run Phase 1 (grid.py) first.")
        return

    boundary  = load_boundary(BOUNDARY_PATH)
    buildings = load_buildings(ODB_PATH, boundary)
    grid      = load_grid(GRID_PATH)
    result    = compute_building_coverage(buildings, grid)
    validate(result, grid)
    save(result, OUTPUT_PATH)

    print("\nPhase 3 complete.")
    print(f"Hand off to Georgio: send him {OUTPUT_PATH}")
    print("He merges it with OSM features into gis_cell_features.parquet")


if __name__ == "__main__":
    main()

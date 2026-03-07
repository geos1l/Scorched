import geopandas as gpd
import os
import numpy as np
from pathlib import Path
from shapely.geometry import box

# Target Coordinate Reference System
TARGET_CRS = "EPSG:3347"
CELL_SIZE_M = 100  # 100m x 100m grid
CITY_ID = "toronto"

# Resolve all paths from repo root so script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical boundary path (JULIE.md spec). Derived from .shp on first run.
BOUNDARY_GEOJSON = PROJECT_ROOT / "data/city_configs/toronto_boundary.geojson"
BOUNDARY_SHP = PROJECT_ROOT / "data/city_configs/citygcs_regional_mun_wgs84.shp"

# Output
OUTPUT_PATH = PROJECT_ROOT / "data/processed/toronto_grid.geojson"


# STEP 1: Load Toronto Boundary
def load_boundary() -> gpd.GeoDataFrame:
    """
    Loads the Toronto boundary in WGS84, reprojects to EPSG:3347.
    Prefers the canonical toronto_boundary.geojson; derives it from the
    .shp on first run and saves it so all downstream phases can use it.
    """
    if BOUNDARY_GEOJSON.exists():
        print(f"Loading boundary from: {BOUNDARY_GEOJSON}")
        gdf = gpd.read_file(BOUNDARY_GEOJSON)
    else:
        print(f"toronto_boundary.geojson not found — deriving from shapefile.")
        os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
        gdf = gpd.read_file(BOUNDARY_SHP)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        # Save canonical boundary GeoJSON for downstream phases
        gdf.to_file(BOUNDARY_GEOJSON, driver="GeoJSON")
        print(f"  Saved canonical boundary to: {BOUNDARY_GEOJSON}")

    # Dissolve to single polygon and reproject
    boundary = gdf.dissolve().to_crs(TARGET_CRS)

    print(f"  Boundary CRS: {boundary.crs}")
    print(f"  Boundary bounds: {boundary.total_bounds}")
    return boundary


# STEP 2: Generate 100m x 100m grid clipped to boundary
def generate_grid(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates a regular 100m x 100m grid over the bounding box,
    clips each cell to the Toronto boundary, and assigns stable cell_ids.

    cell_id format: toronto_{row:03d}_{col:03d}
      - row 000 = northernmost row (top of map)
      - col 000 = westernmost column (left of map)
    """
    dissolved = boundary.union_all()
    minx, miny, maxx, maxy = dissolved.bounds

    # Snap origin to clean multiples of CELL_SIZE_M for stable, reproducible ids
    origin_x = np.floor(minx / CELL_SIZE_M) * CELL_SIZE_M
    origin_y = np.floor(miny / CELL_SIZE_M) * CELL_SIZE_M

    n_cols = int(np.ceil((maxx - origin_x) / CELL_SIZE_M))
    n_rows = int(np.ceil((maxy - origin_y) / CELL_SIZE_M))

    print(f"  Grid extent: {n_rows} rows x {n_cols} cols ({n_rows * n_cols} candidate cells)")

    cells = []
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            x0 = origin_x + col_idx * CELL_SIZE_M
            y0 = origin_y + row_idx * CELL_SIZE_M
            cell_geom = box(x0, y0, x0 + CELL_SIZE_M, y0 + CELL_SIZE_M)

            if not cell_geom.intersects(dissolved):
                continue

            clipped = cell_geom.intersection(dissolved)
            if clipped.is_empty:
                continue

            # row 000 = northernmost: invert row index
            display_row = n_rows - 1 - row_idx
            cell_id = f"{CITY_ID}_{display_row:03d}_{col_idx:03d}"
            cells.append({
                "cell_id": cell_id,
                "city_id": CITY_ID,
                "geometry": clipped,
            })

    grid = gpd.GeoDataFrame(cells, crs=TARGET_CRS)
    return grid


def main():
    print("=== Phase 1: City Grid Generation ===\n")

    # 1. Load Toronto boundary
    boundary = load_boundary()

    # 2. Generate grid
    print("\nGenerating 100m x 100m grid...")
    grid = generate_grid(boundary)
    print(f"  {len(grid)} cells clipped to Toronto boundary")

    # 3. Validate
    assert grid.crs.to_epsg() == 3347, "CRS must be EPSG:3347"
    assert grid["cell_id"].nunique() == len(grid), "cell_ids must be unique"
    assert grid["cell_id"].str.match(rf"^{CITY_ID}_\d{{3}}_\d{{3}}$").all(), \
        "cell_id format must be toronto_XXX_XXX"
    print("  Validation passed")

    # 4. Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"\n  Saved: {OUTPUT_PATH}")
    print("=== Phase 1 complete ===")


if __name__ == "__main__":
    main()
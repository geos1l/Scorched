"""
Phase 2B — Aggregate segmentation masks to grid cell percentages.

INPUTS:
  data/processed/segmentation_masks/mask_index.json   (from inference.py)
  data/processed/segmentation_masks/*.npy              (from inference.py)
  data/processed/toronto_grid.geojson                 (from Julie)

OUTPUT:
  data/processed/segmentation_cell_features.parquet
  Columns: cell_id, seg_building_pct, seg_road_pct, seg_vegetation_pct,
           seg_water_pct, seg_land_pct, seg_unlabeled_pct

  seg_land_pct    = residual (1 - sum of mapped class percentages)
  seg_unlabeled_pct = 0.0 (model has no unlabeled class; Phase 6 may flag gaps)

  Cells with zero overlapping mask pixels get NaN — Phase 6 imputes them.

USAGE:
  python -m services.segmentation.aggregate
  python -m services.segmentation.aggregate --validate    # validate existing output only
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MASK_DIR = REPO_ROOT / "data" / "processed" / "segmentation_masks"
GRID_PATH = REPO_ROOT / "data" / "processed" / "toronto_grid.geojson"
OUT_PATH = REPO_ROOT / "data" / "processed" / "segmentation_cell_features.parquet"

# Schema field name -> output column name
FIELD_TO_COL = {
    "building": "seg_building_pct",
    "road": "seg_road_pct",
    "vegetation": "seg_vegetation_pct",
    "water": "seg_water_pct",
}
SEG_COLS = list(FIELD_TO_COL.values()) + ["seg_land_pct", "seg_unlabeled_pct"]


def validate_output(path: Path) -> None:
    """CHECKPOINT 3 — validate the output parquet schema and value ranges."""
    log.info("=== CHECKPOINT 3: Validating %s ===", path)
    df = pd.read_parquet(path)

    assert "cell_id" in df.columns, "Missing cell_id column"
    for col in SEG_COLS:
        assert col in df.columns, f"Missing column: {col}"

    non_null = df.dropna(subset=SEG_COLS)
    log.info("Total cells: %d | Cells with coverage: %d | No-coverage (NaN): %d",
             len(df), len(non_null), len(df) - len(non_null))

    # All values in [0, 1]
    for col in SEG_COLS:
        col_data = non_null[col]
        assert col_data.min() >= -0.001, f"{col} has values below 0: {col_data.min()}"
        assert col_data.max() <= 1.001, f"{col} has values above 1: {col_data.max()}"

    # Row sums ~1.0
    row_sums = non_null[SEG_COLS].sum(axis=1)
    bad = (row_sums - 1.0).abs() > 0.02
    if bad.any():
        log.warning("%d rows sum to != 1.0 (max deviation: %.4f)", bad.sum(), (row_sums - 1.0).abs().max())
    else:
        log.info("All non-null rows sum to 1.0 ✓")

    log.info("Column means (non-null cells):\n%s", non_null[SEG_COLS].mean().to_string())
    log.info(
        "CHECKPOINT 3 COMPLETE\n"
        ">>> USER CHECK: Do the column means above look plausible for Toronto?\n"
        "    Rough expectations: seg_building_pct ~0.2-0.4, seg_road_pct ~0.1-0.25,\n"
        "    seg_vegetation_pct ~0.2-0.4, seg_water_pct ~0.01-0.05 city-wide average."
    )


def main(validate_only: bool = False) -> None:
    if validate_only:
        validate_output(OUT_PATH)
        return

    # --- Load grid ---
    if not GRID_PATH.exists():
        raise FileNotFoundError(f"toronto_grid.geojson not found at {GRID_PATH} — needed from Julie")
    grid = gpd.read_file(GRID_PATH)
    if grid.crs is None or grid.crs.to_epsg() != 3347:
        raise ValueError(f"Grid CRS must be EPSG:3347, got {grid.crs}")
    log.info("Grid loaded: %d cells", len(grid))

    # --- Load mask index ---
    mask_index_path = MASK_DIR / "mask_index.json"
    if not mask_index_path.exists():
        raise FileNotFoundError(f"mask_index.json not found — run inference.py first")
    with open(mask_index_path) as f:
        mask_index = json.load(f)

    # class_map: {model_class_int -> schema_field_name}
    class_map: dict[int, str] = {int(k): v for k, v in mask_index["class_map"].items()}
    log.info("Class map from inference: %s", class_map)

    # --- Build cell lookup ---
    cell_ids: list[str] = grid["cell_id"].tolist()
    # 1-indexed so 0 can mean "no cell" in rasterized grid
    cell_id_to_int = {cid: i + 1 for i, cid in enumerate(cell_ids)}
    int_to_cell_id = {v: k for k, v in cell_id_to_int.items()}

    # Accumulators
    cell_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cell_total_pixels: dict[str, int] = defaultdict(int)

    # --- Process each mask tile ---
    tile_entries = mask_index["tiles"]
    log.info("Processing %d mask tiles", len(tile_entries))

    for tile_num, (mask_name, meta) in enumerate(tile_entries.items()):
        mask_path = MASK_DIR / mask_name
        if not mask_path.exists():
            log.warning("Mask file not found: %s — skipping", mask_path)
            continue

        mask: np.ndarray = np.load(mask_path)  # H×W uint8
        H, W = mask.shape
        minx, miny, maxx, maxy = meta["bounds"]

        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, W, H)
        tile_bbox = box(minx, miny, maxx, maxy)

        # Find grid cells intersecting this tile's bounding box
        intersecting = grid[grid.geometry.intersects(tile_bbox)]
        if intersecting.empty:
            continue

        # Rasterize cell polygons onto tile pixel grid (cell integer id per pixel)
        shapes = [
            (geom, cell_id_to_int[cid])
            for geom, cid in zip(intersecting.geometry, intersecting["cell_id"])
            if cid in cell_id_to_int
        ]
        if not shapes:
            continue

        cell_raster = rasterio.features.rasterize(
            shapes,
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.int32,
        )

        # For each cell present in this tile, tally class pixel counts
        cell_ints = np.unique(cell_raster)
        cell_ints = cell_ints[cell_ints != 0]

        for cell_int in cell_ints:
            cell_id = int_to_cell_id[cell_int]
            pixel_mask = cell_raster == cell_int
            cell_pixels = mask[pixel_mask]

            cell_total_pixels[cell_id] += len(cell_pixels)

            for class_idx, count in zip(*np.unique(cell_pixels, return_counts=True)):
                schema_field = class_map.get(int(class_idx))
                if schema_field:
                    cell_class_counts[cell_id][schema_field] += int(count)
                # Unmapped pixels silently count toward residual (seg_land_pct)

        if (tile_num + 1) % 50 == 0:
            log.info("  Processed %d / %d tiles, %d cells accumulated",
                     tile_num + 1, len(tile_entries), len(cell_class_counts))

    log.info("Mask processing complete. %d cells have coverage.", len(cell_class_counts))

    # --- Build output dataframe ---
    rows = []
    for cell_id in cell_ids:
        total = cell_total_pixels.get(cell_id, 0)
        counts = cell_class_counts.get(cell_id, {})

        if total == 0:
            # No mask pixels for this cell — will be imputed in Phase 6
            row: dict = {"cell_id": cell_id}
            for col in SEG_COLS:
                row[col] = np.nan
        else:
            known_sum = 0.0
            row = {"cell_id": cell_id}
            for field, col in FIELD_TO_COL.items():
                pct = counts.get(field, 0) / total
                row[col] = float(pct)
                known_sum += pct

            # Residual: everything the model didn't map to a known class
            row["seg_land_pct"] = float(max(0.0, 1.0 - known_sum))
            row["seg_unlabeled_pct"] = 0.0

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure correct column order
    df = df[["cell_id"] + SEG_COLS]

    df.to_parquet(OUT_PATH, index=False)
    log.info("Written: %s (%d rows)", OUT_PATH, len(df))

    validate_output(OUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Validate existing output only")
    args = parser.parse_args()
    main(validate_only=args.validate)

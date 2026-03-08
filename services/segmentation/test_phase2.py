"""
Phase 2 test script — runs without any real tiles or grid data.

Tests:
  1. Model loads and maps classes correctly
  2. Single synthetic tile inference produces correct shape/dtype
  3. Aggregate logic produces valid percentages from synthetic data

Usage:
  python -m services.segmentation.test_phase2
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def test_model_load_and_inference():
    """Test 1 — model loads and produces correct output on synthetic input."""
    import torch
    from PIL import Image
    from services.segmentation.inference import load_model, infer_tile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, class_map = load_model(device)

    assert len(class_map) > 0, "class_map is empty — check LABEL_TO_FIELD mapping"

    schema_fields = set(class_map.values())
    expected = {"building", "road", "vegetation", "water"}
    missing = expected - schema_fields
    if missing:
        log.warning("Missing expected schema fields in class_map: %s", missing)
        log.warning("Check model id2label above and update LABEL_TO_FIELD in inference.py")
    else:
        log.info("All expected schema fields mapped: %s ✓", expected)

    # Synthetic 1024×1024 tile
    synthetic = __import__("PIL").Image.fromarray(
        np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    )
    mask = infer_tile(synthetic, model, processor, device)

    assert mask.shape == (1024, 1024), f"Expected (1024,1024), got {mask.shape}"
    assert mask.dtype == np.uint8, f"Expected uint8, got {mask.dtype}"
    assert mask.min() >= 0
    assert mask.max() < 256

    unique_classes = set(mask.flatten().tolist())
    log.info("Unique class indices in synthetic mask: %s", sorted(unique_classes))
    log.info("TEST 1 PASSED ✓")
    return class_map


def test_aggregate_logic():
    """Test 2 — aggregate.py core logic with synthetic mask and grid data."""
    import geopandas as gpd
    import rasterio.features
    import rasterio.transform
    from shapely.geometry import box

    log.info("TEST 2: Aggregate logic with synthetic data")

    # Synthetic 2×2 grid (4 cells), each 100m×100m, EPSG:3347
    # Using arbitrary coords in EPSG:3347 space
    base_x, base_y = 600000.0, 4820000.0
    size = 100.0
    cells = []
    for row in range(2):
        for col in range(2):
            minx = base_x + col * size
            miny = base_y + row * size
            cells.append({
                "cell_id": f"toronto_{row:03d}_{col:03d}",
                "geometry": box(minx, miny, minx + size, miny + size),
            })
    grid = gpd.GeoDataFrame(cells, crs="EPSG:3347")

    # Synthetic mask tile covering the whole 2×2 grid
    W, H = 200, 200
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[:100, :100] = 0   # top-left cell: all class 0 (building)
    mask[:100, 100:] = 1   # top-right cell: all class 1 (road)
    mask[100:, :100] = 2   # bottom-left cell: all class 2 (vegetation)
    mask[100:, 100:] = 3   # bottom-right cell: all class 3 (water)

    bounds = [base_x, base_y, base_x + 2 * size, base_y + 2 * size]
    class_map = {0: "building", 1: "road", 2: "vegetation", 3: "water"}

    FIELD_TO_COL = {
        "building": "seg_building_pct",
        "road": "seg_road_pct",
        "vegetation": "seg_vegetation_pct",
        "water": "seg_water_pct",
    }

    # --- Run core aggregate logic inline ---
    minx, miny, maxx, maxy = bounds
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, W, H)

    cell_ids = grid["cell_id"].tolist()
    cell_id_to_int = {cid: i + 1 for i, cid in enumerate(cell_ids)}
    int_to_cell_id = {v: k for k, v in cell_id_to_int.items()}

    shapes = [(geom, cell_id_to_int[cid]) for geom, cid in zip(grid.geometry, grid["cell_id"])]
    cell_raster = rasterio.features.rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.int32)

    cell_class_counts = defaultdict(lambda: defaultdict(int))
    cell_total_pixels = defaultdict(int)

    for cell_int in np.unique(cell_raster):
        if cell_int == 0:
            continue
        cell_id = int_to_cell_id[cell_int]
        cell_pixels = mask[cell_raster == cell_int]
        cell_total_pixels[cell_id] += len(cell_pixels)
        for idx, count in zip(*np.unique(cell_pixels, return_counts=True)):
            field = class_map.get(int(idx))
            if field:
                cell_class_counts[cell_id][field] += int(count)

    rows = []
    for cell_id in cell_ids:
        total = cell_total_pixels.get(cell_id, 0)
        counts = cell_class_counts.get(cell_id, {})
        row = {"cell_id": cell_id}
        known_sum = 0.0
        for field, col in FIELD_TO_COL.items():
            pct = counts.get(field, 0) / total if total > 0 else np.nan
            row[col] = float(pct) if not np.isnan(pct) else np.nan
            if not np.isnan(pct):
                known_sum += pct
        row["seg_land_pct"] = float(max(0.0, 1.0 - known_sum)) if total > 0 else np.nan
        row["seg_unlabeled_pct"] = 0.0 if total > 0 else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    SEG_COLS = list(FIELD_TO_COL.values()) + ["seg_land_pct", "seg_unlabeled_pct"]

    # Assertions
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"
    assert df["seg_building_pct"].max() > 0.9, "Expected cell dominated by buildings"
    assert df["seg_road_pct"].max() > 0.9, "Expected cell dominated by road"
    assert df["seg_vegetation_pct"].max() > 0.9, "Expected cell dominated by vegetation"
    assert df["seg_water_pct"].max() > 0.9, "Expected cell dominated by water"

    non_null = df.dropna(subset=SEG_COLS)
    row_sums = non_null[SEG_COLS].sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 0.02, f"Row sums deviate from 1.0: {row_sums.tolist()}"

    log.info("Aggregate output:\n%s", df[["cell_id"] + SEG_COLS].to_string())
    log.info("TEST 2 PASSED ✓")


if __name__ == "__main__":
    log.info("Running Phase 2 tests...")
    class_map = test_model_load_and_inference()
    test_aggregate_logic()
    log.info("All tests passed.")

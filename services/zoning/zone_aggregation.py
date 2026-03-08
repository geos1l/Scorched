"""
Phase 9 — Cluster hot cells into zone polygons.

INPUTS:
  data/processed/predictions.parquet   cell_id, predicted_heat, severity (from Phase 7)
  data/processed/toronto_grid.geojson  cell geometries

OUTPUTS:
  data/processed/zones_raw.geojson     Zone polygons passed to Julie for recommendations.py
  (Julie adds top_contributors + top_recommendations → outputs final zones.geojson)

Zone schema (partial — Julie completes it):
  zone_id            string   toronto_zone_001
  city_id            string   toronto
  geometry           GeoJSON  Dissolved polygon (EPSG:3347)
  severity           string   dominant severity across member cells
  mean_relative_heat float    mean predicted_heat across member cells

USAGE:
  python -m services.zoning.zone_aggregation
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "predictions.parquet"
GRID_PATH = REPO_ROOT / "data" / "processed" / "toronto_grid.geojson"
OUT_PATH = REPO_ROOT / "data" / "processed" / "zones_raw.geojson"

HOT_SEVERITIES = {"high", "extreme"}
CITY_ID = "toronto"


def dominant_severity(severities: pd.Series) -> str:
    order = {"extreme": 3, "high": 2, "moderate": 1, "low": 0}
    return max(severities, key=lambda s: order.get(s, 0))


def cluster_adjacent(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Assign cluster_id to adjacent hot cells using a tiny buffer to detect adjacency.
    Returns gdf with a new 'cluster_id' column.
    """
    # Buffer by 1m to make adjacent cells (shared edge) overlap slightly
    buffered = gdf.geometry.buffer(1.0)

    # Union-find via spatial join to group overlapping (= adjacent) cells
    from shapely.strtree import STRtree

    tree = STRtree(buffered.tolist())
    n = len(gdf)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, geom in enumerate(buffered):
        candidates = tree.query(geom)
        for j in candidates:
            if j != i and geom.intersects(buffered.iloc[j]):
                union(i, j)

    gdf = gdf.copy()
    gdf["cluster_id"] = [find(i) for i in range(n)]
    return gdf


def main() -> None:
    log.info("Loading predictions from %s", PREDICTIONS_PATH)
    preds = pd.read_parquet(PREDICTIONS_PATH)

    log.info("Loading grid from %s", GRID_PATH)
    grid = gpd.read_file(GRID_PATH)
    assert grid.crs.to_epsg() == 3347, "Grid must be EPSG:3347"

    # Join predictions to grid geometries
    merged = grid.merge(preds, on="cell_id", how="inner")
    log.info("Merged: %d cells", len(merged))

    # Filter to hot/extreme cells only
    hot = merged[merged["severity"].isin(HOT_SEVERITIES)].copy()
    log.info("Hot cells (high + extreme): %d", len(hot))

    if len(hot) == 0:
        log.warning("No hot cells found — check predictions.parquet severity values")
        return

    # Cluster adjacent hot cells
    hot = cluster_adjacent(hot)
    log.info("Clusters found: %d", hot["cluster_id"].nunique())

    # Dissolve by cluster_id
    zones = hot.dissolve(by="cluster_id", aggfunc={
        "predicted_heat": "mean",
        "severity": dominant_severity,
        "cell_id": "count",
    }).reset_index(drop=True)

    zones = zones.rename(columns={
        "predicted_heat": "mean_relative_heat",
        "cell_id": "cell_count",
    })

    # Assign stable zone_ids
    zones = zones.sort_values("mean_relative_heat", ascending=False).reset_index(drop=True)
    zones["zone_id"] = [f"{CITY_ID}_zone_{i+1:03d}" for i in range(len(zones))]
    zones["city_id"] = CITY_ID
    zones["gemini_summary"] = ""  # Farill fills this in Phase 10

    # Keep only schema columns (Julie adds top_contributors + top_recommendations)
    out_cols = ["zone_id", "city_id", "geometry", "severity", "mean_relative_heat", "gemini_summary", "cell_count"]
    zones = zones[out_cols]

    log.info("Zones created: %d", len(zones))
    log.info("Severity breakdown:\n%s", zones["severity"].value_counts().to_string())

    zones.to_file(OUT_PATH, driver="GeoJSON")
    log.info("Written: %s", OUT_PATH)
    log.info(
        ">>> HANDOFF TO JULIE: zones_raw.geojson is ready.\n"
        "    She adds top_contributors + top_recommendations → outputs zones.geojson"
    )


if __name__ == "__main__":
    main()

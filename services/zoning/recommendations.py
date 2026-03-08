"""
Phase 9 — Recommendation Engine (Julie's half)

Loads Georgio's zone polygons (zones_raw.geojson), joins cell-level features
to each zone, derives top_contributors and top_recommendations using threshold
rules, and outputs the final zones.geojson.

Inputs:
  data/processed/zones_raw.geojson    <- Georgio's zone_aggregation.py output
  data/processed/features.parquet     <- cell-level features (Phase 6)
  data/processed/toronto_grid.geojson <- cell geometries for spatial join

Output:
  data/processed/zones.geojson
    Adds to each zone: top_contributors, top_recommendations, gemini_summary ("")
    gemini_summary is left empty — Farill fills it in Phase 10.

Run:
  python services/zoning/recommendations.py
"""

import sys
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
ZONES_RAW_PATH = PROJECT_ROOT / "data/processed/zones_raw.geojson"
FEATURES_PATH  = PROJECT_ROOT / "data/processed/features.parquet"
GRID_PATH      = PROJECT_ROOT / "data/processed/toronto_grid.geojson"
OUTPUT_PATH    = PROJECT_ROOT / "data/processed/zones.geojson"

TARGET_CRS = "EPSG:3347"

# ── Feature columns used in contributor rules ──────────────────────────────────

ZONE_FEATURE_COLS = [
    "seg_vegetation_pct",
    "seg_road_pct",
    "seg_building_pct",
    "gis_park_coverage",
    "gis_road_coverage",
    "gis_building_coverage",
    "water_distance_m",
]


# ── Load inputs ────────────────────────────────────────────────────────────────

def load_inputs():
    missing = [p for p in [ZONES_RAW_PATH, FEATURES_PATH, GRID_PATH] if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: missing input: {p}")
        if ZONES_RAW_PATH in missing:
            print("\nWaiting on Georgio: run zone_aggregation.py to produce zones_raw.geojson first.")
        sys.exit(1)

    zones    = gpd.read_file(ZONES_RAW_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    grid     = gpd.read_file(GRID_PATH)

    print(f"zones_raw        : {len(zones):,} zones")
    print(f"features         : {len(features):,} cells")
    print(f"grid             : {len(grid):,} cells")

    # Ensure consistent CRS
    if zones.crs is None or zones.crs.to_epsg() != 3347:
        zones = zones.to_crs(TARGET_CRS)
    if grid.crs is None or grid.crs.to_epsg() != 3347:
        grid = grid.to_crs(TARGET_CRS)

    return zones, features, grid


# ── Aggregate features per zone ────────────────────────────────────────────────

def aggregate_features_per_zone(
    zones: gpd.GeoDataFrame,
    features: pd.DataFrame,
    grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Spatially joins grid cells to zones, then averages cell-level features
    across all cells that fall within each zone polygon.

    Returns a DataFrame with zone_id and mean feature values.
    """
    print("\nJoining cell features to zones...")

    # Attach feature data to grid cell geometries
    cells_with_features = grid[["cell_id", "geometry"]].merge(
        features[["cell_id"] + ZONE_FEATURE_COLS],
        on="cell_id",
        how="inner",
    )
    cells_gdf = gpd.GeoDataFrame(cells_with_features, geometry="geometry", crs=TARGET_CRS)

    # Spatial join: find which zone each cell centroid falls in
    cell_centroids = cells_gdf.copy()
    cell_centroids["geometry"] = cell_centroids.geometry.centroid

    joined = gpd.sjoin(cell_centroids, zones[["zone_id", "geometry"]], how="left", predicate="within")

    # Drop cells that didn't land in any zone
    joined = joined.dropna(subset=["zone_id"])

    # Average features per zone
    zone_features = (
        joined.groupby("zone_id")[ZONE_FEATURE_COLS]
        .mean()
        .reset_index()
    )

    matched = joined["zone_id"].nunique()
    print(f"  {matched:,} / {len(zones):,} zones have matched cells")

    return zone_features


# ── Contributor rules ──────────────────────────────────────────────────────────

def derive_contributors(row: pd.Series) -> list:
    """
    Derives top_contributors for a zone based on threshold rules from JULIE.md.
    Uses mean feature values across all cells in the zone.
    """
    contributors = []

    if row.get("seg_vegetation_pct", 1) < 0.10 and row.get("gis_park_coverage", 1) < 0.05:
        contributors.append("low vegetation")

    if row.get("seg_road_pct", 0) > 0.35 or row.get("gis_road_coverage", 0) > 0.40:
        contributors.append("road-dominant impervious surface")

    if row.get("seg_building_pct", 0) > 0.40 and row.get("gis_building_coverage", 0) > 0.35:
        contributors.append("dense built form")

    if row.get("water_distance_m", 0) > 1000:
        contributors.append("minimal water proximity")

    # Ensure at least one contributor
    if not contributors:
        contributors.append("urban heat accumulation")

    return contributors


# ── Recommendation rules ───────────────────────────────────────────────────────

def derive_recommendations(contributors: list, row: pd.Series) -> list:
    """
    Assigns exactly 3 recommendations based on contributor combination.
    Rules from JULIE.md — deterministic, priority-ordered.
    """
    has_road     = "road-dominant impervious surface" in contributors
    has_building = "dense built form" in contributors
    has_low_veg  = "low vegetation" in contributors

    if has_road and has_low_veg:
        return ["cool pavement", "street trees", "shade structures"]

    if has_building:
        return ["cool roofs", "green roofs", "targeted canopy"]

    if has_low_veg:
        # High open lot + low vegetation
        return ["permeable landscaping", "tree canopy", "shade structures"]

    if has_road:
        return ["cool pavement", "street trees", "shade structures"]

    # Fallback
    return ["tree canopy", "green roofs", "shade structures"]


# ── Apply rules to all zones ───────────────────────────────────────────────────

def apply_rules(zones: gpd.GeoDataFrame, zone_features: pd.DataFrame) -> gpd.GeoDataFrame:
    """Merges zone features into zone GeoDataFrame, applies contributor and recommendation rules."""
    print("\nApplying contributor and recommendation rules...")

    zones = zones.merge(zone_features, on="zone_id", how="left")

    contributors_list   = []
    recommendations_list = []

    for _, row in zones.iterrows():
        contributors = derive_contributors(row)
        recommendations = derive_recommendations(contributors, row)
        contributors_list.append(contributors)
        recommendations_list.append(recommendations)

    zones["top_contributors"]    = contributors_list
    zones["top_recommendations"] = recommendations_list
    zones["gemini_summary"]      = ""   # Farill fills this in Phase 10

    return zones


# ── Validate ───────────────────────────────────────────────────────────────────

def validate(zones: gpd.GeoDataFrame) -> None:
    print("\nValidating output...")

    required = {"zone_id", "city_id", "geometry", "severity",
                "mean_relative_heat", "top_contributors",
                "top_recommendations", "gemini_summary"}
    missing = required - set(zones.columns)
    assert not missing, f"FAIL: missing columns: {missing}"
    print(f"  All required columns present")

    bad_recs = zones[zones["top_recommendations"].apply(len) != 3]
    assert len(bad_recs) == 0, f"FAIL: {len(bad_recs)} zones don't have exactly 3 recommendations"
    print(f"  All zones have exactly 3 recommendations")

    empty_contributors = zones[zones["top_contributors"].apply(len) == 0]
    assert len(empty_contributors) == 0, f"FAIL: {len(empty_contributors)} zones have no contributors"
    print(f"  All zones have at least 1 contributor")

    valid_severities = {"low", "moderate", "high", "extreme"}
    bad_sev = zones[~zones["severity"].isin(valid_severities)]
    assert len(bad_sev) == 0, f"FAIL: invalid severity values: {zones['severity'].unique()}"
    print(f"  All severity values valid")

    print(f"  All checks passed. {len(zones):,} zones ready.\n")


# ── Save ───────────────────────────────────────────────────────────────────────

def save(zones: gpd.GeoDataFrame) -> None:
    """
    Serialises list columns (top_contributors, top_recommendations) to JSON strings
    for GeoJSON compatibility, then saves.
    """
    out = zones.copy()
    out["top_contributors"]    = out["top_contributors"].apply(json.dumps)
    out["top_recommendations"] = out["top_recommendations"].apply(json.dumps)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"  Zones: {len(out):,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 9: Recommendation Engine ===\n")

    zones, features, grid = load_inputs()
    zone_features = aggregate_features_per_zone(zones, features, grid)
    zones = apply_rules(zones, zone_features)

    # Drop intermediate feature columns before saving (not part of zone schema)
    drop_cols = [c for c in ZONE_FEATURE_COLS if c in zones.columns]
    zones = zones.drop(columns=drop_cols)

    validate(zones)
    save(zones)

    print("Phase 9 complete.")
    print("Tell Georgio + Farill: data/processed/zones.geojson is ready.")
    print("  Georgio loads it into the API.")
    print("  Farill adds gemini_summary in Phase 10.")


if __name__ == "__main__":
    main()

"""
Phase 4 — GEE Landsat Level 2 pull + compositing for Toronto.

Owner: Julie
Output: data/raw/landsat/toronto_landsat_composite.tif
  - CRS: EPSG:3347
  - Bands exported: SR_B2 (Blue), SR_B3 (Green), SR_B4 (Red),
                    SR_B5 (NIR), SR_B6 (SWIR), LST_C, NDVI, brightness
  - Consumed by: Georgio (cell aggregation -> landsat_cell_features.parquet)

Band name -> cell schema mapping (for Georgio):
  SR_B5      -> nir_mean
  LST_C      -> lst_c
  NDVI       -> ndvi_mean
  brightness -> brightness_mean

Authentication:
  Personal OAuth (required — service accounts lack GEE storage quota for exports).
  First run will open a browser to authenticate. Subsequent runs use cached credentials.

Run:
  python services/preprocessing/gee_pipeline.py              # exports to Google Drive
  python services/preprocessing/gee_pipeline.py --local      # downloads directly (testing only)

After Drive export completes:
  Download toronto_landsat_composite.tif from Google Drive > CanCoolAI/ and place at:
  data/raw/landsat/toronto_landsat_composite.tif
  Then tell Georgio it is ready.
"""

import ee
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
CONFIG_PATH   = PROJECT_ROOT / "data/city_configs/toronto.json"
BOUNDARY_PATH = PROJECT_ROOT / "data/city_configs/toronto_boundary.geojson"
OUTPUT_DIR    = PROJECT_ROOT / "data/raw/landsat"
OUTPUT_FILE   = OUTPUT_DIR / "toronto_landsat_composite.tif"

# ── Band identifiers in LANDSAT/LC08/C02/T1_L2 ────────────────────────────────
SR_BANDS  = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"]   # Blue Green Red NIR SWIR
ST_BANDS  = ["ST_B10"]                                         # Thermal
QA_BAND   = "QA_PIXEL"
ALL_BANDS = SR_BANDS + ST_BANDS + [QA_BAND]

# Landsat Collection 2 Level-2 scale factors (USGS documentation)
SR_SCALE  = 0.0000275
SR_OFFSET = -0.2
ST_SCALE  = 0.00341802
ST_OFFSET = 149.0          # DN -> Kelvin; subtract 273.15 for Celsius

# QA_PIXEL bit positions (Landsat Collection 2)
QA_DILATED_CLOUD_BIT = 1
QA_CLOUD_BIT         = 3
QA_CLOUD_SHADOW_BIT  = 4

# Export at native Landsat resolution; Georgio resamples to 100m grid cells
EXPORT_SCALE_M = 30


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ── GEE authentication ─────────────────────────────────────────────────────────

def authenticate_gee() -> None:
    """
    Authenticate with GEE using personal OAuth.
    Service accounts cannot be used — they lack storage quota for Drive exports.
    First run opens a browser. Subsequent runs use cached credentials.
    GEE_PROJECT is read from .env (required by GEE since 2024).
    """
    project = os.environ.get("GEE_PROJECT")
    if not project:
        raise RuntimeError("GEE_PROJECT not set. Add GEE_PROJECT=<your-gcp-project-id> to .env")
    try:
        ee.Initialize(project=project)
        print(f"GEE initialized (project: {project}).")
    except Exception:
        print("No cached credentials — running ee.Authenticate()...")
        ee.Authenticate()
        ee.Initialize(project=project)
        print(f"GEE authenticated interactively (project: {project}).")


# ── Toronto region of interest ─────────────────────────────────────────────────

def get_toronto_roi(config: dict) -> ee.Geometry:
    """
    Returns the Toronto ROI as an ee.Geometry.
    Prefers toronto_boundary.geojson (exact city boundary).
    Falls back to bbox from toronto.json if boundary file is absent.
    """
    if BOUNDARY_PATH.exists():
        import geopandas as gpd
        gdf = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326").dissolve()
        geojson = json.loads(gdf.geometry.to_json())
        geometry = geojson["features"][0]["geometry"]
        roi = ee.Geometry(geometry)
        print("ROI loaded from toronto_boundary.geojson.")
    else:
        bbox = config["bbox_epsg4326"]
        roi = ee.Geometry.Rectangle([
            bbox["min_lon"], bbox["min_lat"],
            bbox["max_lon"], bbox["max_lat"],
        ])
        print("WARNING: toronto_boundary.geojson not found — using bbox from toronto.json.")
    return roi


# ── Cloud masking ──────────────────────────────────────────────────────────────

def apply_cloud_mask(image: ee.Image) -> ee.Image:
    """
    Masks clouds, cloud shadows, and dilated clouds using QA_PIXEL bit flags.
    Bit 1: Dilated cloud | Bit 3: Cloud | Bit 4: Cloud shadow.
    """
    qa = image.select(QA_BAND)
    mask = (
        qa.bitwiseAnd(1 << QA_CLOUD_BIT).eq(0)
        .And(qa.bitwiseAnd(1 << QA_CLOUD_SHADOW_BIT).eq(0))
        .And(qa.bitwiseAnd(1 << QA_DILATED_CLOUD_BIT).eq(0))
    )
    return image.updateMask(mask)


# ── Scale factors ──────────────────────────────────────────────────────────────

def apply_scale_factors(image: ee.Image) -> ee.Image:
    """
    Applies Landsat C02 L2 scale factors:
      SR bands -> surface reflectance (0.0-1.0)
      ST_B10   -> Land Surface Temperature in Celsius
    Returns image with SR_B2-SR_B6 and LST_C bands (QA_PIXEL dropped).
    """
    optical = image.select(SR_BANDS).multiply(SR_SCALE).add(SR_OFFSET)
    lst_c = (
        image.select("ST_B10")
        .multiply(ST_SCALE)
        .add(ST_OFFSET)
        .subtract(273.15)
        .rename("LST_C")
    )
    return optical.addBands(lst_c)


# ── Derived bands ──────────────────────────────────────────────────────────────

def add_derived_bands(image: ee.Image) -> ee.Image:
    """
    Adds NDVI and brightness proxy bands.
    Called after apply_scale_factors so band values are in reflectance units.
    SR_B4 = Red, SR_B5 = NIR, SR_B3 = Green, SR_B2 = Blue.
    """
    nir        = image.select("SR_B5")
    red        = image.select("SR_B4")
    green      = image.select("SR_B3")
    blue       = image.select("SR_B2")
    ndvi       = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    brightness = red.add(green).add(blue).divide(3).rename("brightness")
    return image.addBands([ndvi, brightness])


# ── Collection filtering + compositing ────────────────────────────────────────

def build_composite(config: dict, roi: ee.Geometry) -> tuple:
    """
    Filters Landsat 8 C02 T1_L2 to Toronto, applies cloud mask + scale factors,
    composites using median reducer, and clips to ROI.

    Date range driven by toronto.json composite_year + composite_months.
    Cloud cover threshold starts at 20%; relaxes to 40% if fewer than 3 scenes found.

    Returns (composite_image, raw_collection).
    """
    year   = config.get("composite_year", 2024)
    months = config.get("composite_months", [6, 7, 8])
    start  = f"{year}-{months[0]:02d}-01"
    last_m = months[-1]
    end    = f"{year + 1}-01-01" if last_m == 12 else f"{year}-{last_m + 1:02d}-01"

    collection_id = config.get("landsat_collection", "LANDSAT/LC08/C02/T1_L2")

    print(f"Collection : {collection_id}")
    print(f"Date range : {start} -> {end}")

    def filter_collection(cloud_pct):
        return (
            ee.ImageCollection(collection_id)
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
            .select(ALL_BANDS)
        )

    raw = filter_collection(20)
    scene_count = raw.size().getInfo()
    print(f"Scenes (cloud < 20%): {scene_count}")

    if scene_count < 3:
        print("  Fewer than 3 scenes — relaxing cloud cover threshold to 40%.")
        raw = filter_collection(40)
        scene_count = raw.size().getInfo()
        print(f"Scenes (cloud < 40%): {scene_count}")

    if scene_count == 0:
        raise RuntimeError(
            f"No Landsat scenes found for Toronto between {start} and {end}. "
            "Check GEE authentication, date range, and toronto.json composite_year."
        )

    processed = (
        raw
        .map(apply_cloud_mask)
        .map(apply_scale_factors)
        .map(add_derived_bands)
    )
    composite = processed.median().clip(roi)
    return composite, raw


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_composite(composite: ee.Image, roi: ee.Geometry) -> None:
    """
    Checks band presence and LST plausibility before export.
    Uses a coarse scale to keep getInfo() fast.
    """
    band_names = composite.bandNames().getInfo()
    print(f"\nComposite bands: {band_names}")

    required = {"SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "LST_C", "NDVI", "brightness"}
    missing  = required - set(band_names)
    if missing:
        raise ValueError(f"Composite is missing required bands: {missing}")

    lst_stats = (
        composite.select("LST_C")
        .reduceRegion(
            reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
            geometry=roi,
            scale=500,
            maxPixels=1e7,
            bestEffort=True,
        )
        .getInfo()
    )
    print(f"LST_C stats (C): {lst_stats}")

    lst_mean = lst_stats.get("LST_C_mean")
    if lst_mean is not None and not (5 < lst_mean < 65):
        print(
            f"WARNING: LST mean {lst_mean:.1f}C is outside expected range (5-65C). "
            "Verify scale factors and QA masking."
        )


# ── Export: GEE Drive batch task (default) ─────────────────────────────────────

def export_to_drive(composite: ee.Image, roi: ee.Geometry) -> ee.batch.Task:
    """
    Submits a GEE batch export to Google Drive folder 'CanCoolAI'.
    After the task completes (check https://code.earthengine.google.com/tasks),
    download the file and place it at:
      data/raw/landsat/toronto_landsat_composite.tif
    """
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description="toronto_landsat_composite",
        folder="CanCoolAI",
        fileNamePrefix="toronto_landsat_composite",
        region=roi,
        scale=EXPORT_SCALE_M,
        crs="EPSG:3347",
        maxPixels=1e10,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    print(f"\nGEE batch export submitted.")
    print(f"  Task ID : {task.id}")
    print(f"  Monitor : https://code.earthengine.google.com/tasks")
    print(f"  When complete, download from Google Drive > CanCoolAI/ and move to:")
    print(f"  {OUTPUT_FILE}")
    return task


# ── Export: local download via geemap (dev/testing fallback) ───────────────────

def export_local(composite: ee.Image, roi: ee.Geometry) -> None:
    """
    Downloads the composite directly as a GeoTIFF using geemap.
    May time out for full Toronto extent — use only for testing on a small bbox.
    Requires: pip install geemap
    """
    try:
        import geemap
    except ImportError:
        raise ImportError("geemap required for local export. Install with: pip install geemap")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading composite locally to: {OUTPUT_FILE}")
    geemap.ee_export_image(
        composite,
        filename=str(OUTPUT_FILE),
        scale=EXPORT_SCALE_M,
        region=roi,
        crs="EPSG:3347",
        file_per_band=False,
    )
    print(f"Saved: {OUTPUT_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(local: bool = False) -> None:
    print("=== Phase 4: GEE Landsat Pull + Compositing ===\n")

    authenticate_gee()

    config = load_config()
    print(
        f"Config: {config['city_id']} | "
        f"year={config.get('composite_year')} | "
        f"months={config.get('composite_months')} | "
        f"CRS={config['crs']}"
    )

    roi = get_toronto_roi(config)

    print("\nBuilding Landsat composite...")
    composite, raw_collection = build_composite(config, roi)

    dates = raw_collection.aggregate_array("DATE_ACQUIRED").getInfo()
    print(f"Scene dates: {sorted(dates)}")

    validate_composite(composite, roi)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if local:
        export_local(composite, roi)
    else:
        export_to_drive(composite, roi)

    print("\n=== Phase 4 complete ===")
    if not local:
        print("Next steps:")
        print("  1. Wait for GEE task to finish (check link above)")
        print("  2. Download toronto_landsat_composite.tif from Google Drive > CanCoolAI/")
        print(f"  3. Move it to: {OUTPUT_FILE}")
        print("  4. Tell Georgio — he aggregates it to landsat_cell_features.parquet")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 — GEE Landsat pull + compositing")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Download GeoTIFF directly via geemap (testing only — may time out for full Toronto)",
    )
    args = parser.parse_args()
    main(local=args.local)

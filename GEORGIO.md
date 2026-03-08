# CanCool AI — Georgio's Workstream
> Read 00_PROJECT_CORE.md first, then this file. Do not read Farill.md or Julie.md.

---

## Your Role

Backend engineer + GIS pipeline + ML co-lead + segmentation inference.
You own repo structure, all data aggregation, model evaluation, zone clustering, and the FastAPI backend.

---

## Files You Own

```
apps/api/
├── main.py                  <- FastAPI app entry, CORS, startup
└── routes/
    ├── cities.py            <- GET /cities
    ├── zones.py             <- GET /zones, GET /zones/{id}, POST /selection
    │                           (Farill adds Gemini endpoint here — coordinate)
    └── cells.py             <- GET /cells debug endpoint

services/segmentation/
├── inference.py             <- SegFormer batch inference runner
└── aggregate.py             <- Tile masks -> cell % per grid cell

services/preprocessing/
└── gis_pipeline.py          <- StatCan ODB + OSM processing + cell coverage

services/training/
└── evaluate.py              <- Model evaluation + severity buckets + predictions.parquet

services/zoning/
└── zone_aggregation.py      <- Hot cell clustering -> zone polygons
```

**Do not touch:**
- `apps/web/` (Farill)
- `apps/api/gemini.py` (Farill)
- `services/preprocessing/grid.py` (Julie)
- `services/preprocessing/gee_pipeline.py` (Julie)
- `services/segmentation/inference_prep.py` (Julie)
- `services/training/features.py` (Julie)
- `services/training/train.py` (Julie)
- `services/zoning/recommendations.py` (Julie)

---

## What You Consume

### From Julie (Phase 1)
- `data/processed/toronto_grid.geojson` — you need this before Phase 2 aggregation, Phase 3 GIS join, Phase 4 cell aggregation

### From Julie (Phase 2)
- `data/raw/orthophoto_tiles/` — preprocessed PNG tiles ready for SegFormer inference (jgerbscheid model)
- Confirm naming convention with Julie before she starts so your aggregate.py matches

### From Julie (Phase 3)
- `data/processed/statcan_buildings.parquet` — building coverage per cell
- You merge this with your OSM features into gis_cell_features.parquet

### From Julie (Phase 4)
- `data/raw/landsat/toronto_landsat_composite.tif` — composited Landsat GeoTIFF in EPSG:3347
- You aggregate this to grid cells and compute all raster features

### From Julie (Phase 6)
- Joined + fused DataFrame from features.py
- You validate, handle missing values, output final features.parquet

### From Julie (Phase 7)
- `models/xgboost_heat_model.json` — trained model
- Train/test split indices
- You load model, run evaluation, compute predictions

---

## What You Produce

| Output | Where | Consumed By |
|--------|-------|-------------|
| Repo folder structure | root | Everyone — do this first |
| `segmentation_cell_features.parquet` | `data/processed/` | Julie (features.py) |
| `gis_cell_features.parquet` | `data/processed/` | Julie (features.py) |
| `landsat_cell_features.parquet` | `data/processed/` | Julie (features.py) |
| `features.parquet` | `data/processed/` | Julie (train.py) |
| `predictions.parquet` | `data/processed/` | Phase 8 API + Phase 9 zoning |
| `feature_importance.png` | `data/processed/` | README + pitch |
| Zone polygon GeoDataFrame | passed to Julie | Julie (recommendations.py) |
| `zones.geojson` | `data/processed/` | FastAPI + Farill frontend |
| FastAPI running on Vultr | public URL | Farill frontend |

---

## Your Phases

### Phase 1 — Repo Structure
Starts immediately. No dependencies.

- Create full monorepo folder structure:
```
HackCanada-2026/
├── apps/web/
├── apps/api/routes/
├── services/segmentation/
├── services/preprocessing/
├── services/training/
├── services/zoning/
├── data/raw/orthophoto_tiles/
├── data/raw/landsat/
├── data/processed/
├── data/city_configs/
└── models/
```
- Add `.gitkeep` to empty dirs
- Commit `data/city_configs/toronto.json` with locked Phase 0 config
- Push so Julie and Farill can branch immediately

**Acceptance criteria:**
- Full structure visible in GitHub
- toronto.json committed
- Both teammates can pull and see the structure

---

### Phase 2 — Segmentation Inference + Aggregation (your half)

> INPUT REQUIRED: `data/raw/orthophoto_tiles/` populated by Julie first.

**Model:** `jgerbscheid/segformer_b1-nlver_finetuned-1024-1024` — pretrained aerial-image SegFormer, inference-only
**Primary useful classes:** buildings, roads/pavement, vegetation, water
**Note:** seg_land_pct and seg_unlabeled_pct should be derived as residual if model doesn't output them directly — GIS fills any gaps

- Write `services/segmentation/inference.py`:
  - Load SegFormer from HuggingFace: `jgerbscheid/segformer_b1-nlver_finetuned-1024-1024`
  - Loop over all tiles in `data/raw/orthophoto_tiles/`
  - Run batch inference on each preprocessed tile
  - Output class mask per tile to `data/processed/segmentation_masks/`
  - Primary output classes: buildings, roads/pavement, vegetation, water
  - Derive seg_land_pct as residual (1 - sum of known classes) if not directly output

- Write `services/segmentation/aggregate.py`:
  - Load toronto_grid.geojson
  - Georeference each mask to EPSG:3347
  - For each grid cell, compute class % from overlapping mask pixels:
    - `seg_building_pct`, `seg_road_pct`, `seg_vegetation_pct`
    - `seg_water_pct`, `seg_land_pct`, `seg_unlabeled_pct`
  - Output: `data/processed/segmentation_cell_features.parquet`
    - Columns: `cell_id`, all seg_ columns

**Acceptance criteria:**
- Every cell_id from toronto_grid.geojson has a row
- All seg_ percentages are floats between 0.0 and 1.0
- Sum of all seg_ columns per row is approximately 1.0

---

### Phase 3 — GIS Pipeline (your half)

> INPUT REQUIRED: `toronto_grid.geojson` (Julie, Phase 1) + `statcan_buildings.parquet` (Julie, Phase 3).

- Write `services/preprocessing/gis_pipeline.py`:
  - Download OSM layers for Toronto via osmnx:
    - Roads: `ox.graph_from_place("Toronto, Ontario, Canada")`
    - Parks: `ox.geometries_from_place(..., tags={"leisure": "park"})`
    - Water: `ox.geometries_from_place(..., tags={"natural": "water"})`
  - Reproject all layers to EPSG:3347
  - Clip all layers to Toronto boundary
  - For each grid cell compute:
    - `gis_road_coverage` = % cell area covered by road network
    - `gis_park_coverage` = % cell area covered by parks/green space
    - `water_distance_m` = distance from cell centroid to nearest water body
  - Load Julie's `statcan_buildings.parquet`
  - Merge building coverage with OSM features on `cell_id`
  - Output: `data/processed/gis_cell_features.parquet`
    - Columns: `cell_id`, `gis_building_coverage`, `gis_road_coverage`, `gis_park_coverage`, `water_distance_m`

**Acceptance criteria:**
- Every cell_id has a row
- All coverage values are floats between 0.0 and 1.0
- water_distance_m is in metres, non-negative

---

### Phase 4 — Landsat Cell Aggregation (your half)

> INPUT REQUIRED: `data/raw/landsat/toronto_landsat_composite.tif` from Julie first.

- Load composited Landsat GeoTIFF from `data/raw/landsat/`
- For each grid cell in toronto_grid.geojson, compute mean value per band:
  - `ndvi_mean` = (NIR - Red) / (NIR + Red)
  - `brightness_mean` = proxy from visible bands
  - `nir_mean` = mean NIR band value
  - `lst_c` = mean land surface temperature in Celsius
- Compute city-wide median LST across all cells
- Compute `relative_lst_c = cell_lst_c - city_median_lst` ← this is the ML label
- Output: `data/processed/landsat_cell_features.parquet`
  - Columns: `cell_id`, `ndvi_mean`, `brightness_mean`, `nir_mean`, `lst_c`, `relative_lst_c`

**Acceptance criteria:**
- Every cell_id has a row
- relative_lst_c is centered roughly around 0 (median cell = 0)
- lst_c values are in plausible range for Toronto summer (20-50°C)

---

### Phase 6 — Feature Validation + Output (your half)

> INPUT REQUIRED: All three parquet files must exist first.
> Julie runs join + fusion in features.py and passes you the DataFrame.

- Validate join completeness:
  - Every cell_id must appear in all three source files
  - Log any cell_ids missing from one or more branches
- Handle missing values:
  - Impute numeric columns with column median
  - Flag imputed cells with a boolean column `has_missing_data`
- Confirm schema matches 00_PROJECT_CORE.md exactly — field names, types
- Output: `data/processed/features.parquet`

**Coordinate with Julie:**
- Both of you must sign off on schema before Phase 7 starts

---

### Phase 7 — Model Evaluation + Predictions (your half)

> INPUT REQUIRED: `models/xgboost_heat_model.json` and split indices from Julie first.

- Write `services/training/evaluate.py`:
  - Load model from `models/xgboost_heat_model.json`
  - Load held-out test set using Julie's split indices
  - Compute evaluation metrics: MAE, RMSE, R²
  - Generate feature importance chart → save to `data/processed/feature_importance.png`
  - Run predictions on ALL cells in features.parquet
  - Compute severity buckets from predicted relative_lst_c:
    - `extreme`: > +5.0°C above median
    - `high`:    +2.0 to +5.0°C
    - `moderate`: 0.0 to +2.0°C
    - `low`:     < 0.0°C
  - Append `predicted_heat` and `severity` columns
  - Output: `data/processed/predictions.parquet`

**Acceptance criteria:**
- predictions.parquet has `cell_id`, `predicted_heat`, `severity` columns
- All four severity values present in output
- feature_importance.png is readable

---

### Phase 8 — FastAPI Backend

> INPUT REQUIRED: predictions.parquet (Phase 7) + zones.geojson (Phase 9).
> Note: you can build and test the API with dummy data before zones.geojson exists.

- Initialize FastAPI app in `apps/api/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
```
- Load zones.geojson and predictions.parquet at startup
- Implement routes (shapes defined in 00_PROJECT_CORE.md):
  - `apps/api/routes/cities.py` → GET /cities
  - `apps/api/routes/zones.py` → GET /zones, GET /zones/{zone_id} (leave gemini_summary as empty string — Farill adds it)
  - `apps/api/routes/cells.py` → GET /cells
  - POST /selection in zones.py
- Deploy to Vultr:
  - Spin up Vultr instance
  - Install dependencies, run uvicorn
  - Share public URL with Farill

**Coordinate with Farill:**
- Tell Farill the Vultr URL when deployed
- Confirm response shapes match API contracts in 00_PROJECT_CORE.md before she wires frontend

**Acceptance criteria:**
- All endpoints return correct shapes
- CORS allows requests from Farill's frontend origin
- API publicly accessible via Vultr URL

---

### Phase 9 — Zone Aggregation (your half)

> INPUT REQUIRED: predictions.parquet with severity column (Phase 7).

- Write `services/zoning/zone_aggregation.py`:
  - Load predictions.parquet
  - Filter cells where severity = `high` or `extreme`
  - Convert cell geometries to GeoDataFrame
  - Dissolve adjacent hot cells using GeoPandas: `gdf.dissolve(by='cluster_id')`
  - Assign stable zone_ids: `toronto_zone_{number}` starting from 001
  - Compute per zone:
    - `mean_relative_heat` = mean of relative_lst_c across member cells
    - `severity` = dominant severity class across member cells
  - Pass zone GeoDataFrame to Julie for recommendations.py

**Hand off to Julie:**
- Tell Julie when zone polygons are ready
- She adds top_contributors and top_recommendations
- She outputs final zones.geojson

**Acceptance criteria:**
- Zones are dissolved polygons, not individual cells
- Every zone has zone_id, city_id, geometry, severity, mean_relative_heat
- zone_id format matches naming convention in 00_PROJECT_CORE.md

---

### Phase 11 — Deployment + Submission (your part)

- Confirm Vultr deployment is stable and publicly accessible
- Make GitHub repo public before 10AM Sunday
- Verify no commits in repo dated before hackathon start
- Test POST /selection if drag-rectangle was implemented
- Add Vultr deployment instructions to README

---

## Merge Points

| When | What happens |
|------|-------------|
| After Phase 1 repo structure committed | Julie and Farill pull and branch |
| When Julie populates orthophoto_tiles/ | You start inference.py |
| When Julie sends statcan_buildings.parquet | You merge into gis_cell_features.parquet |
| When Julie exports landsat GeoTIFFs | You start cell aggregation |
| After features.parquet output | Julie starts train.py immediately |
| After predictions.parquet output | You start Phase 8 API + Phase 9 zones simultaneously |
| After zones.geojson output | Load into API, tell Farill to swap dummy data |
| After base API deployed on Vultr | Tell Farill the URL, she updates frontend |

# CanCool AI — Project Context & Build Reference
> Hack Canada 2026 | Urban Heat Island Detection & Intervention System
> Read this file before writing any code. All data contracts, API shapes, and file ownership are fixed.

---

## Project Summary

CanCool AI is a geospatial ML system that identifies heat-prone urban zones across Canadian cities. It combines satellite image segmentation (CV), thermal satellite data, and open geospatial layers to predict which city zones are hottest, explain why, and recommend the top 3 cooling interventions per zone. The MVP targets Toronto with a clickable map UI, a supervised XGBoost model, and Gemini-generated plain-English summaries.

---

## Team & Role Ownership

| Person | Primary Role | Owns |
|--------|-------------|------|
| **Farill** | Frontend + Gemini API + API integration | `apps/web/`, `apps/api/gemini.py`, Gemini endpoint in `apps/api/routes/zones.py` |
| **Julie** | GEE data pull + ML co-lead + Recommendations | `services/preprocessing/gee_pipeline.py`, `services/preprocessing/grid.py`, `services/segmentation/inference_prep.py`, `services/training/train.py`, `services/zoning/recommendations.py` |
| **Georgio** | Backend + GIS + ML co-lead + Segmentation inference | `apps/api/`, `services/preprocessing/gis_pipeline.py`, `services/segmentation/inference.py`, `services/segmentation/aggregate.py`, `services/training/evaluate.py`, `services/zoning/zone_aggregation.py` |

**Rule:** Do not modify files outside your ownership without telling the team first.

---

## Tech Stack (Definitive — No Alternatives)

```
Segmentation model  : SegFormer (HuggingFace: jgerbscheid/segformer_b1-nlver_finetuned-1024-1024) — pretrained aerial-image checkpoint, inference-only
Segmentation imagery: High-resolution Canadian orthophoto/aerial imagery for Toronto (NOT Sentinel-2)
Thermal imagery     : Google Earth Engine — Landsat Level 2 (thermal/LST branch only)
GIS data            : StatCan Open Database of Buildings + OSM (roads, parks, water)
ML model            : XGBoost Regressor
Backend             : FastAPI (Python)
Frontend            : Next.js + TypeScript
Map rendering       : Mapbox GL JS
Storage             : GeoJSON + Parquet (MVP)
AI explanations     : Gemini API
Deployment          : Vultr (backend) + Vercel (frontend)
```

---

## Repo Structure

```
HackCanada-2026/
├── CONTEXT.md                          <- this file
├── apps/
│   ├── web/                            <- Next.js frontend (Farill)
│   │   ├── components/
│   │   │   ├── Map.tsx                 <- Mapbox map component
│   │   │   ├── DetailPanel.tsx         <- Right-side zone info panel
│   │   │   └── ZoneLayer.tsx           <- GeoJSON zone polygon overlay
│   │   ├── pages/
│   │   │   └── index.tsx
│   │   └── .env.local                  <- MAPBOX_TOKEN, GEMINI_API_KEY
│   └── api/                            <- FastAPI backend (Georgio + Farill)
│       ├── main.py                     <- App entry, CORS, startup (Georgio)
│       ├── gemini.py                   <- Gemini API calls (Farill)
│       └── routes/
│           ├── cities.py               <- GET /cities (Georgio)
│           ├── zones.py                <- GET /zones, GET /zones/{id} (Georgio + Farill)
│           └── cells.py                <- GET /cells debug endpoint (Georgio)
├── services/
│   ├── segmentation/
│   │   ├── inference_prep.py           <- GEE tile pull + preprocessing for SegFormer (Julie)
│   │   ├── inference.py                <- SegFormer batch inference runner (Georgio)
│   │   └── aggregate.py                <- Tile masks -> cell percentages (Georgio)
│   ├── preprocessing/
│   │   ├── gee_pipeline.py             <- GEE Landsat pull + compositing (Julie)
│   │   ├── gis_pipeline.py             <- StatCan ODB + OSM processing (Georgio)
│   │   └── grid.py                     <- City grid generation (Julie)
│   ├── training/
│   │   ├── features.py                 <- Merge all branches -> features.parquet (Julie)
│   │   ├── train.py                    <- XGBoost training (Julie)
│   │   └── evaluate.py                 <- Metrics + severity buckets + predictions (Georgio)
│   └── zoning/
│       ├── zone_aggregation.py         <- Hot cell clustering -> zones (Georgio)
│       └── recommendations.py          <- Contributor tags + rule engine (Julie)
├── data/
│   ├── raw/                            <- Downloaded source data (do not edit)
│   ├── processed/
│   │   ├── toronto_grid.geojson                   <- OUTPUT of Phase 1 (Julie)
│   │   ├── segmentation_cell_features.parquet     <- OUTPUT of Phase 2 (Georgio)
│   │   ├── gis_cell_features.parquet              <- OUTPUT of Phase 3 (Georgio)
│   │   ├── landsat_cell_features.parquet          <- OUTPUT of Phase 4 (Georgio)
│   │   ├── features.parquet                       <- OUTPUT of Phase 6 (merged)
│   │   ├── predictions.parquet                    <- OUTPUT of Phase 7 (Georgio)
│   │   └── zones.geojson                          <- OUTPUT of Phase 9
│   └── city_configs/
│       └── toronto.json                <- Phase 0 locked config
└── models/
    └── xgboost_heat_model.json         <- OUTPUT of Phase 7 (Julie)
```

---

## Phase 0 — Locked Config (toronto.json)

```json
{
  "city_id": "toronto",
  "boundary_source": "data/city_configs/toronto_boundary.geojson",
  "crs": "EPSG:3347",
  "grid_size_m": 100,
  "segmentation_imagery_source": "toronto_orthophoto",
  "landsat_source": "landsat_level2",
  "building_source": "statcan_odb",
  "road_source": "osm",
  "park_source": "osm",
  "water_source": "osm",
  "heat_label_window": ["2024-06-01", "2024-08-31"],
  "ml_target": "relative_lst_c"
}
```

---

## Data Contracts (FIXED — Do Not Change Field Names)

### Cell Schema — `features.parquet`
One row per 100m x 100m grid cell. Central join key for all three branches.

```
cell_id               string    e.g. "toronto_001_002"
city_id               string    "toronto"
geometry              GeoJSON   Cell polygon in EPSG:3347

# Branch A — Segmentation (Julie sources orthophoto tiles, Georgio runs inference)
# Primary model: jgerbscheid/segformer_b1-nlver_finetuned-1024-1024 (pretrained aerial, inference-only)
# Primary useful classes: buildings, roads/pavement, vegetation, water
# GIS branch cross-checks and gap-fills any missing/ambiguous surface categories
seg_building_pct      float     % cell area classified as buildings by SegFormer
seg_road_pct          float     % cell area classified as roads/pavement
seg_vegetation_pct    float     % cell area classified as vegetation
seg_water_pct         float     % cell area classified as water
seg_land_pct          float     % residual open/bare land (derived or model output — GIS fills if absent)
seg_unlabeled_pct     float     % cell area unlabeled (may be zero depending on model output)

# Branch B — Thermal/Raster (Julie pulls + composites, Georgio aggregates to cells)
ndvi_mean             float     (NIR - Red) / (NIR + Red)
brightness_mean       float     Proxy from spectral bands
nir_mean              float     Mean NIR band value
lst_c                 float     Raw land surface temperature (Celsius)
relative_lst_c        float     cell_lst - city_median_lst  <- ML LABEL

# Branch C — GIS (Julie: buildings, Georgio: roads/parks/water)
gis_building_coverage float     % cell area from StatCan ODB footprints
gis_road_coverage     float     % cell area from OSM road network
gis_park_coverage     float     % cell area from OSM parks/green space
water_distance_m      float     Distance from cell centroid to nearest water (metres)
```

### Zone Schema — `zones.geojson`
One row per aggregated hot zone. Output of Phase 9.

```
zone_id               string    e.g. "toronto_zone_014"
city_id               string    "toronto"
geometry              GeoJSON   Dissolved polygon of adjacent hot cells
severity              string    "low" | "moderate" | "high" | "extreme"
mean_relative_heat    float     Mean relative_lst_c across zone cells
top_contributors      string[]  e.g. ["low vegetation", "high road coverage"]
top_recommendations   string[]  Top 3 interventions from rule engine
gemini_summary        string    Plain-English explanation — added in Phase 10 (Farill)
```

---

## API Endpoints (FIXED — Frontend and Backend Must Match Exactly)

```
GET  /cities
     Response: { cities: [{ city_id: string, name: string }] }

GET  /zones?city_id=toronto
     Response: { zones: ZoneFeatureCollection }  <- GeoJSON FeatureCollection

GET  /zones/{zone_id}
     Response: {
       zone_id: string,
       severity: string,
       mean_relative_heat: number,
       top_contributors: string[],
       top_recommendations: string[],
       gemini_summary: string
     }

GET  /cells?city_id=toronto
     Response: { cells: CellFeatureCollection }  <- debug only

POST /selection
     Body:    { geometry: GeoJSON Polygon }
     Response: {
       severity: string,
       mean_relative_heat: number,
       top_contributors: string[],
       top_recommendations: string[]
     }
```

---

## Build Phases — Ownership & Task Split

### Phase 0 — Lock Config (All Three, Together, Before Any Code)
- Commit `data/city_configs/toronto.json` with locked values above
- One person commits, everyone pulls before writing any code

---

### Phase 1 — Repo Structure & City Grid

**Julie**
- Write `services/preprocessing/grid.py`
- Load Toronto boundary -> project to EPSG:3347 -> generate 100m grid -> clip to boundary -> assign stable cell_ids
- Output: `data/processed/toronto_grid.geojson`

**Georgio**
- Create full monorepo folder structure (all dirs and placeholder files listed in repo structure above)
- Commit initial structure so Julie and Farill can branch off immediately

> MERGE POINT: toronto_grid.geojson must exist before Phase 2 (segmentation alignment), Phase 3 (GIS cell join), Phase 4 (Landsat cell join), and Phase 5 (frontend dummy data). Complete Phase 1 before anything else.

---

### Phase 2 — Segmentation Pipeline (Branch A)

**Model:** `jgerbscheid/segformer_b1-nlver_finetuned-1024-1024` — pretrained aerial-image SegFormer, inference-only (no training or fine-tuning for MVP)
**Imagery:** High-resolution Canadian orthophoto/aerial imagery for Toronto (NOT Sentinel-2 from GEE)
**Primary classes:** buildings, roads/pavement, vegetation, water. GIS cross-checks and gap-fills missing surface categories.
**Note:** One good orthophoto vintage is sufficient — temporal diversity matters more for Landsat than for orthophotos.

**Julie**
- Write `services/segmentation/inference_prep.py`
- Source high-resolution orthophoto/aerial imagery for Toronto (e.g. City of Toronto open orthophoto tiles or equivalent Canadian source)
- Tile imagery to match model input spec, preprocess (resize + normalize per model card)
- Output: preprocessed tiles ready for inference in `data/raw/orthophoto_tiles/`, consistent PNG naming: `tile_{row}_{col}.png`

**Georgio**
- Write `services/segmentation/inference.py` — load `jgerbscheid/segformer_b1-nlver_finetuned-1024-1024`, run batch inference on preprocessed tiles, output class masks to `data/processed/segmentation_masks/`
- Write `services/segmentation/aggregate.py` — georeference masks to EPSG:3347, aggregate tile masks → compute class % per grid cell using toronto_grid.geojson
- For seg_land_pct and seg_unlabeled_pct: derive as residual if model does not output these classes directly; GIS will fill gaps
- Output: `data/processed/segmentation_cell_features.parquet`

> MERGE POINT: Julie finishes tile preprocessing first, then hands off to Georgio for inference and aggregation. Coordinate on tile format (PNG, consistent naming) before starting.

---

### Phase 3 — GIS Data Pipeline (Branch C)

**Julie**
- Download StatCan Open Database of Buildings for Toronto
- Reproject building footprints to EPSG:3347, clip to Toronto boundary
- Compute `gis_building_coverage` per grid cell
- Output: `data/processed/statcan_buildings.parquet` — hand off to Georgio for final merge

**Georgio**
- Download OSM layers via osmnx: roads, parks, water
- Reproject and clip all layers to EPSG:3347 and Toronto boundary
- Compute `gis_road_coverage`, `gis_park_coverage`, `water_distance_m` per grid cell
- Merge Julie's statcan_buildings.parquet with OSM features
- Output: `data/processed/gis_cell_features.parquet`

> MERGE POINT: Julie sends statcan_buildings.parquet to Georgio who runs the final merge. Agree on column names before starting — they must match the Cell Schema above exactly.

---

### Phase 4 — Thermal Data Pipeline (Branch B)

**Julie**
- Write `services/preprocessing/gee_pipeline.py`
- Pull Landsat Level 2 from GEE for Toronto (Jun-Aug 2024, low cloud cover, 3-8 scenes)
- Extract bands: LST, NIR, SWIR, Red, Green, Blue + QA mask
- Apply cloud mask, composite scenes using median
- Export composited rasters as GeoTIFF to `data/raw/landsat/` in EPSG:3347

**Georgio**
- Aggregate composited Landsat GeoTIFFs to 100m grid cells using toronto_grid.geojson
- Compute per cell: `ndvi_mean`, `brightness_mean`, `nir_mean`, `lst_c`
- Compute city-wide median LST
- Compute `relative_lst_c = cell_lst - city_median_lst` <- this is the ML label
- Output: `data/processed/landsat_cell_features.parquet`

> MERGE POINT: Julie exports GeoTIFFs to data/raw/landsat/, Georgio aggregates to cells. Coordinate on GeoTIFF CRS (EPSG:3347) and file naming before starting.

---

### Phase 5 — Frontend Scaffold (Parallel — No Dependencies)

**Farill** (entire phase, runs fully parallel to Phases 2-4)
- Initialize Next.js + TypeScript in `apps/web/`
- Install: mapbox-gl, tailwindcss, shadcn/ui
- Create Mapbox account, store token in `.env.local`
- Build `Map.tsx`: dark style, centered on Toronto [43.6532, -79.3832], zoom 11
- Build `ZoneLayer.tsx`: loads zones from dummy zones.geojson, colors by severity
  - extreme: #C0392B | high: #E67E22 | moderate: #F1C40F | low: #27AE60
- Build `DetailPanel.tsx`: renders on zone click
  - Shows: severity badge, mean_relative_heat, top_contributors list, top_recommendations list, gemini_summary placeholder
- Wire click handler: zone click -> fetch GET /zones/{zone_id} -> populate DetailPanel
- Add loading states and error handling throughout

> MERGE POINT: After Phase 8 (Backend API) is complete, Farill updates API base URL from dummy data to live FastAPI endpoints and runs integration tests.

---

### Phase 6 — Feature Engineering (Merge of Branches A + B + C)

> INPUT REQUIRED: segmentation_cell_features.parquet (Phase 2) + gis_cell_features.parquet (Phase 3) + landsat_cell_features.parquet (Phase 4) must all exist before this phase starts.

**Julie**
- Write `services/training/features.py`
- Load all three parquet files, join on cell_id
- Compute fusion features:
  - `building_disagreement = abs(seg_building_pct - gis_building_coverage)`
  - `road_disagreement = abs(seg_road_pct - gis_road_coverage)`
  - `green_consensus = mean(seg_vegetation_pct, gis_park_coverage)`

**Georgio**
- Validate join completeness — every cell_id must appear in all three parquet files
- Handle missing values: flag and impute cells with incomplete data
- Output final merged file: `data/processed/features.parquet`

> MERGE POINT: Julie writes join and fusion logic, Georgio validates and outputs final features.parquet. Both must confirm schema matches Cell Schema contract exactly before Phase 7 starts.

---

### Phase 7 — Model Training & Validation

**Julie**
- Write `services/training/train.py`
- Load features.parquet, define feature columns and target (relative_lst_c)
- Split by scene date — hold out latest scenes as test set to avoid temporal leakage
- Train XGBoost Regressor
- Save model artifact: `models/xgboost_heat_model.json`

**Georgio**
- Write `services/training/evaluate.py`
- Load model and held-out test set, compute MAE, RMSE, R²
- Generate and save feature importance chart: `data/processed/feature_importance.png`
- Run predictions on all cells
- Compute severity buckets from predicted relative heat:
  - extreme: > +5°C | high: +2 to +5°C | moderate: 0 to +2°C | low: < 0°C
- Output: `data/processed/predictions.parquet` with severity column appended

> MERGE POINT: Julie outputs trained model, Georgio loads it for evaluation and produces predictions.parquet. Final predictions.parquet feeds Phase 8 (API) and Phase 9 (Zone Aggregation) simultaneously.

---

### Phase 8 — Backend API

**Georgio**
- Initialize FastAPI app in `apps/api/main.py`
- Add CORS middleware (allow Next.js frontend origin)
- Load zones.geojson and predictions.parquet at startup
- Implement all routes except Gemini endpoint:
  - `apps/api/routes/cities.py` -> GET /cities
  - `apps/api/routes/zones.py` -> GET /zones, GET /zones/{zone_id} (without gemini_summary for now)
  - `apps/api/routes/cells.py` -> GET /cells
  - POST /selection in zones.py
- Deploy to Vultr — backend must be publicly accessible via URL

**Farill**
- Write `apps/api/gemini.py` — Gemini API call function
- Add gemini_summary generation to GET /zones/{zone_id}
- Prompt template: structured zone data -> 2-3 sentence plain-English city planner explanation
- Test Gemini responses across all severity types

> MERGE POINT: Georgio builds and deploys base API first. Farill plugs in gemini.py after base routes are working. Once both are done, Farill updates frontend base URL from dummy data to live Vultr endpoint.

---

### Phase 9 — Zone Aggregation & Recommendation Engine

> INPUT REQUIRED: predictions.parquet with severity column (Phase 7) must exist before this phase starts.

**Georgio**
- Write `services/zoning/zone_aggregation.py`
- Load predictions.parquet, identify cells with severity = high or extreme
- Merge adjacent hot cells using GeoPandas dissolve
- Assign stable zone_ids, compute mean_relative_heat and severity per zone

**Julie**
- Write `services/zoning/recommendations.py`
- Derive top_contributors per zone using threshold rules:
  - seg_vegetation_pct < 0.10 AND gis_park_coverage < 0.05 -> "low vegetation"
  - seg_road_pct > 0.35 OR gis_road_coverage > 0.40 -> "road-dominant impervious surface"
  - seg_building_pct > 0.40 AND gis_building_coverage > 0.35 -> "dense built form"
  - water_distance_m > 1000 -> "minimal water proximity"
- Assign top 3 interventions per zone:
  - high road + low vegetation -> cool pavement, street trees, shade structures
  - high building + dense footprint -> cool roofs, green roofs, targeted canopy
  - high open lot + low vegetation -> permeable landscaping, tree canopy, shade structures
- Output: `data/processed/zones.geojson` (gemini_summary field left empty — Farill fills in Phase 10)

> MERGE POINT: Georgio outputs zone polygons, Julie adds contributors and recommendations. Julie's recommendations.py takes Georgio's zone output as input. Final zones.geojson feeds Phase 8 backend and Phase 10.

---

### Phase 10 — Gemini Integration

> INPUT REQUIRED: zones.geojson (Phase 9) + FastAPI running (Phase 8)

**Farill** (entire phase)
- Finalize `apps/api/gemini.py` with real zones data
- For each zone: build structured prompt -> call Gemini API -> store gemini_summary
- Two options, pick based on time:
  - Option A (simpler): precompute all summaries, bake into zones.geojson
  - Option B (better): generate on-demand inside GET /zones/{zone_id}
- Render gemini_summary in `DetailPanel.tsx`

---

### Phase 11 — Integration, Polish & Submission (All Three)

**Farill**
- Verify all API endpoints wired correctly in frontend
- Add segmentation preview card in DetailPanel: RGB tile + colored mask overlay
- Test all zone click interactions end-to-end
- Write README.md
- Record demo video
- Submit Devpost — check prize tracks: Gemini API, Vultr, Stan

**Julie**
- Validate model metrics are presentable — add MAE/RMSE/R² to README
- Export feature importance chart for slides and README
- Confirm segmentation masks accessible via API preview endpoint

**Georgio**
- Confirm Vultr deployment is stable and API publicly accessible
- Make GitHub repo public before 10AM Sunday
- Verify no commits exist from before hackathon start
- Test POST /selection if drag-rectangle was implemented

---

## Prize Tracks to Submit For

| Prize | Requirement |
|-------|-------------|
| Overall 1st/2nd/3rd | Build well |
| MLH Best Use of Gemini API | Gemini integrated in Phase 10 |
| MLH Best Use of Vultr | Backend deployed on Vultr in Phase 8 — MUST keep, non-negotiable |
| Stan $350/$150 cash | 3 LinkedIn posts Fri to Sat tagging @Stanley on LinkedIn |

---

## Fallback Cut Order

Cut in this order if time is short. Vultr is non-negotiable — do not cut.

1. PostGIS — stay on GeoJSON + Parquet only
2. POST /selection drag-rectangle endpoint
3. Gemini API summaries (lose Gemini prize)
4. Reduce scope to downtown Toronto only
5. Last resort only: drop supervised ML, use rule-based heat scoring — SegFormer CV story still intact

---

## Non-MVP Scalability Notes

- **Multi-city**: city_config.json pattern already supports this — add new config per city, no rewrites needed
- **PostGIS**: migration path is loading processed GeoJSON into PostGIS with ST_GeomFromGeoJSON
- **Live heat updates**: current system is batch/periodic — live version needs streaming GEE or daily Landsat pulls
- **Digital Twin simulation**: persistent what-if scenario state — user adds trees to a zone, model predicts new heat score
- **Zoning law generation**: extended Gemini prompt using City of Toronto open zoning data to draft bylaws per zone
- **Custom segmentation model**: train on labeled Canadian aerial imagery datasets for higher accuracy than pretrained SegFormer

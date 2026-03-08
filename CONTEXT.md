# CanCool AI — Project Context & Build Reference
> Hack Canada 2026 | Urban Heat Island Detection & Intervention System
> Read this file before writing any code. All data contracts, API shapes, and file ownership are fixed.

---

## Project Summary

CanCool AI is a geospatial ML system that identifies heat-prone urban zones across Canadian cities. It combines satellite image segmentation (CV), thermal satellite data, and open geospatial layers to predict which city zones are hottest, explain why, and recommend the top 3 cooling interventions per zone. The MVP targets Toronto with a clickable map UI, a supervised XGBoost model, and Gemini-generated plain-English summaries.

---

## DEMO AOI (area of interest) — MVP SCOPE (READ THIS FIRST)

> **Time constraint: ~8 hours remaining.** Full-city Toronto pipelines (68k tiles, 709k road geometries) take hours. The MVP runs the entire pipeline end-to-end on a **small demo region** so we can show seg → GIS → ML → zones → API → frontend working.

**Demo AOI center:** 43.6444°N, 79.3946°W (Parkdale / Liberty Village — dense built + parks + water proximity = good heat contrast)
**Demo AOI bbox (WGS84):** min_lon -79.4071, min_lat 43.6354, max_lon -79.3821, max_lat 43.6534 (~2 km × 2 km)
**Grid:** Full `toronto_grid.geojson` is kept on disk. All scripts **filter to cells inside the demo AOI bbox**.
**Tile cap for inference:** 200 tiles max.
**GIS:** `gis_cell_features.parquet` already exists for all 68k cells (completed). Filter to AOI cell_ids.
**Landsat:** Stubbed for MVP (plausible synthetic values for AOI cells). Real GEE pipeline is future work.
**Frontend:** Map auto-zooms to demo AOI on first load; user can still pan/zoom freely.

### What this means for each person

- **Georgio:** Add `--limit` to inference.py; run inference on 200 AOI tiles; aggregate; build stub landsat parquet; merge features; evaluate; zone_aggregation; API; deploy.
- **Julie:** features.py + train.py + recommendations.py all operate on the AOI subset. No full-city runs needed.
- **Farill:** Swap dummy-zones.geojson for real zones from AOI; auto-zoom map to AOI; wire API; Gemini; polish.

---

## Team & Role Ownership

| Person | Primary Role | Owns |
|--------|-------------|------|
| **Farill** | Frontend + Gemini API + API integration | `apps/web/`, `apps/api/gemini.py`, Gemini endpoint in `apps/api/routes/zones.py` |
| **Julie** | GEE data pull + ML co-lead + Recommendations | `services/preprocessing/gee_pipeline.py`, `services/preprocessing/grid.py`, `services/preprocessing/statcan_buildings.py`, `services/training/train.py`, `services/zoning/recommendations.py` |
| **Georgio** | Backend + GIS + ML co-lead + Segmentation inference | `apps/api/`, `services/preprocessing/tile_uploader.py`, `services/preprocessing/gis_pipeline.py`, `services/segmentation/inference.py`, `services/segmentation/aggregate.py`, `services/training/evaluate.py`, `services/zoning/zone_aggregation.py` |

**Rule:** Do not modify files outside your ownership without telling the team first.

---

## Tech Stack (Definitive — No Alternatives)

```
Segmentation model  : SegFormer (HuggingFace: jgerbscheid/segformer_b1-nlver_finetuned-1024-1024) — pretrained aerial-image checkpoint, inference-only
Segmentation imagery: ESRI World Imagery MapServer (Toronto Open Data retired) — tiles fetched by tile_uploader.py, stored in Vultr Object Storage bucket "torontotiles" (ewr1); credentials in .env
Thermal imagery     : Google Earth Engine — Landsat Level 2 (thermal/LST branch only); GEE requires personal OAuth; export goes to Google Drive then manually moved to data/raw/landsat/
GIS data            : StatCan Open Database of Buildings + OSM (roads, parks, water)
ML model            : XGBoost Regressor
Backend             : FastAPI (Python)
Frontend            : Next.js + TypeScript
Map rendering       : Mapbox GL JS
Storage             : GeoJSON + Parquet (MVP); orthophoto tiles + tile_index.json live in Vultr bucket not local
AI explanations     : Gemini API
Deployment          : Vultr (backend + object storage) + Vercel (frontend)
```

---

## Repo Structure

```
HackCanada-2026/
├── CONTEXT.md                          <- this file
├── scripts/
│   └── vultr_setup.sh                  <- Vultr bucket setup script (Georgio)
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
│   │   ├── inference.py                <- SegFormer batch inference runner, reads from Vultr bucket (Georgio)
│   │   └── aggregate.py                <- Tile masks -> cell percentages (Georgio)
│   ├── preprocessing/
│   │   ├── tile_uploader.py            <- Fetch ESRI tiles, upload to Vultr bucket, build tile_index.json (Georgio)
│   │   ├── statcan_buildings.py        <- StatCan ODB buildings -> gis_building_coverage (Julie)
│   │   ├── gee_pipeline.py             <- GEE Landsat pull + compositing (Julie)
│   │   ├── gis_pipeline.py             <- OSM roads/parks/water + merge with StatCan (Georgio)
│   │   └── grid.py                     <- City grid generation (Julie)
│   ├── training/
│   │   ├── features.py                 <- Merge all branches -> features.parquet (Julie)
│   │   ├── train.py                    <- XGBoost training (Julie)
│   │   └── evaluate.py                 <- Metrics + severity buckets + predictions (Georgio)
│   └── zoning/
│       ├── zone_aggregation.py         <- Hot cell clustering -> zones (Georgio)
│       └── recommendations.py          <- Contributor tags + rule engine (Julie)
├── data/
│   ├── raw/
│   │   └── landsat/                    <- GeoTIFFs exported from GEE via Google Drive (Julie)
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

> NOTE: Orthophoto tiles and tile_index.json live in Vultr Object Storage bucket "torontotiles" (ewr1), NOT in data/raw/orthophoto_tiles/. See changes.md for current build vs plan.
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

## Build Phases — MVP Sprint (Demo AOI Scoped)

> All phases below are scoped to the **demo AOI** (~2 km × 2 km, ~400 cells, 200 tiles). Full-city runs are future work.

### Phases 0–1: DONE
- toronto.json committed, repo structure created, toronto_grid.geojson generated (68,394 cells).
- statcan_buildings.parquet completed.
- gis_cell_features.parquet completed for all 68k cells.
- Tile upload to Vultr in progress (~75% done, ~51k of 68k); continues in background.

---

### Phase 2 (MVP) — Segmentation on Demo AOI

**Georgio**
- Add `--limit N` and `--aoi-bbox` flags to `inference.py` so it only processes tiles inside the demo AOI, capped at 200
- Run inference on ~200 AOI tiles from Vultr bucket → masks to `data/processed/segmentation_masks/`
- Run `aggregate.py` → `segmentation_cell_features.parquet` (68k rows; only AOI cells have non-NaN seg data)

**Estimated time:** ~1–2 h on CPU for 200 tiles. Account for this in your project planning to assign up tasks during that wait time.

---

### Phase 3 (MVP) — GIS: ALREADY DONE

`gis_cell_features.parquet` exists for all 68,394 cells. For the MVP, filter to AOI cell_ids when building features.parquet. No re-run needed.

---

### Phase 4 (MVP) — Landsat: STUB

**Georgio**
- Write a small script to generate `landsat_cell_features.parquet` with **stub values** for AOI cells only:
  - `ndvi_mean`: random 0.15–0.45 (plausible Toronto summer range)
  - `brightness_mean`: random 0.10–0.30
  - `nir_mean`: random 0.15–0.35
  - `lst_c`: random 28–42°C (Toronto summer surface temps)
  - `relative_lst_c`: `lst_c - median(lst_c)` across AOI cells (centered around 0)
- This is the ML training label. Stub values let the pipeline run; real GEE export replaces this post-demo.

**Estimated time:** ~15 min to write + run.

---

### Phase 5 (MVP) — Frontend: MOSTLY DONE + updates

**Farill**
- Map, ZoneLayer, DetailPanel already built with dummy data
- **Update:** Map auto-zooms to demo AOI bbox on first load (`map.fitBounds([[-79.4071, 43.6354], [-79.3821, 43.6534]])`)
- **Update:** When real zones.geojson arrives from Georgio, swap dummy-zones.geojson for it
- **Update:** Heat gradient coloring on zones (existing severity colors work; optionally add continuous gradient from mean_relative_heat)

---

### Phase 6 (MVP) — Feature Engineering (AOI subset)

**Julie**
- `features.py`: Load all three parquets, **inner join on cell_id** (only AOI cells that have seg + gis + landsat data survive)
- Compute fusion features: building_disagreement, road_disagreement, green_consensus
- Pass to Georgio for validation

**Georgio**
- Validate + impute + output `features.parquet`

**Estimated time:** ~5–10 min.

---

### Phase 7 (MVP) — Model Training (AOI subset)

**Julie**
- `train.py`: Load features.parquet (~200–400 rows), train XGBoost on AOI cells
- Target: `relative_lst_c`
- Simple train/test split (e.g. 80/20 random — no temporal leakage concern with stub labels)
- Save model: `models/xgboost_heat_model.json`

**Georgio**
- `evaluate.py`: Load model, predict on all AOI cells, compute severity buckets, output `predictions.parquet`

**Estimated time:** ~5 min total.

---

### Phase 8 (MVP) — Backend API

**Georgio**
- FastAPI in `apps/api/main.py`, CORS, load zones.geojson + predictions.parquet at startup
- Implement: GET /cities, GET /zones, GET /zones/{zone_id}, GET /cells
- Deploy to Vultr, share URL with Farill

**Farill**
- Write `apps/api/gemini.py`, add Gemini call to GET /zones/{zone_id}
- Render gemini_summary in DetailPanel

**Estimated time:** ~1–2 h.

---

### Phase 9 (MVP) — Zone Aggregation + Recommendations (AOI subset)

**Georgio**
- `zone_aggregation.py`: Load predictions.parquet, filter high/extreme cells, dissolve adjacent → zone polygons
- Pass to Julie

**Julie**
- `recommendations.py`: Add top_contributors + top_recommendations per zone using threshold rules
- Output: `data/processed/zones.geojson`

**Estimated time:** ~15–30 min.

---

### Phase 10 (MVP) — Gemini Integration

**Farill**
- Wire Gemini summaries (on-demand per zone click, or precomputed at startup)
- Render in DetailPanel

**Estimated time:** ~30 min.

---

### Phase 11 (MVP) — Integration, Polish & Submission

**Farill**
- Verify API endpoints wired in frontend
- Segmentation preview card in DetailPanel (RGB tile + colored mask overlay) — if time allows
- README, demo video, Devpost submission

**Julie**
- Validate model metrics for README
- Feature importance chart for pitch

**Georgio**
- Confirm Vultr deployment stable
- Make GitHub repo public before 10AM Sunday
- Verify no pre-hackathon commits

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
4. Segmentation preview card in DetailPanel
5. Last resort only: drop supervised ML, use rule-based heat scoring — SegFormer CV story still intact

---

## Future Scaling (post-demo)

- **Full-city Toronto:** Run all pipelines on 68k cells. Tile upload ~75% done; gis_cell_features.parquet already done. Inference + aggregate + real Landsat + full train = the "real" run.
- **Real Landsat:** Replace stub landsat_cell_features with GEE export + cell aggregation
- **Optimize gis_pipeline:** Replace unary_union + 68k intersection with overlay/sjoin (minutes instead of hours)
- **Multi-city:** city_config.json pattern already supports this
- **PostGIS / live heat updates / Digital Twin / custom segmentation model** — see Non-MVP Scalability Notes below

## Non-MVP Scalability Notes

- **Multi-city**: city_config.json pattern already supports this — add new config per city, no rewrites needed
- **PostGIS**: migration path is loading processed GeoJSON into PostGIS with ST_GeomFromGeoJSON
- **Live heat updates**: current system is batch/periodic — live version needs streaming GEE or daily Landsat pulls
- **Digital Twin simulation**: persistent what-if scenario state — user adds trees to a zone, model predicts new heat score
- **Zoning law generation**: extended Gemini prompt using City of Toronto open zoning data to draft bylaws per zone
- **Custom segmentation model**: train on labeled Canadian aerial imagery datasets for higher accuracy than pretrained SegFormer
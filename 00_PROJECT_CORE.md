# CanCool AI — Project Core
> Hack Canada 2026 | Read this every session before writing any code.

---

## What We Are Building

CanCool AI is a geospatial ML system that identifies heat-prone urban zones in Toronto. It combines satellite image segmentation (CV), thermal satellite data, and open GIS layers to predict which zones are hottest, explain the surface-level causes, and recommend the top 3 cooling interventions per zone. Users interact via a clickable map that shows zone severity, contributors, and Gemini-generated plain-English summaries.

---

## MVP Scope

- One city: Toronto
- 100m x 100m grid cells across the city
- Supervised XGBoost model predicting relative heat per cell
- SegFormer segmentation visibly in the pipeline
- Clickable zone map with severity + contributors + recommendations + Gemini summary
- FastAPI backend deployed on Vultr
- Next.js + Mapbox GL JS frontend

---

## Architecture (10 bullets)

- Branch A: Julie sources high-resolution Canadian orthophoto/aerial imagery for Toronto → Georgio runs pretrained SegFormer inference → seg composition % per cell
- Branch B: Julie pulls Landsat Level 2 from GEE → Georgio aggregates to cells → LST + NDVI + spectral features per cell
- Branch C: Julie downloads StatCan ODB buildings → Georgio downloads OSM roads/parks/water → GIS coverage % per cell (GIS also cross-checks and gap-fills segmentation outputs)
- All three branches merge in Phase 6 into features.parquet (cell_id as join key)
- Julie trains XGBoost on features.parquet with relative_lst_c as target
- Georgio evaluates model, appends severity buckets, outputs predictions.parquet
- Georgio clusters hot cells into zone polygons, Julie adds contributor tags + recommendations → zones.geojson
- Georgio builds FastAPI backend, deploys to Vultr
- Farill builds Next.js + Mapbox frontend, wires to API, adds Gemini summaries
- Gemini API called per zone to generate plain-English explanation for city planners

---

## Canonical Schemas (DO NOT CHANGE FIELD NAMES)

### Cell — `features.parquet`
```
cell_id               string    e.g. "toronto_001_002"
city_id               string    "toronto"
geometry              GeoJSON   Cell polygon in EPSG:3347
seg_building_pct      float     Branch A
seg_road_pct          float     Branch A
seg_vegetation_pct    float     Branch A
seg_water_pct         float     Branch A
seg_land_pct          float     Branch A (residual/derived — GIS fills gaps if model omits)
seg_unlabeled_pct     float     Branch A (residual — may be zero depending on model output)
ndvi_mean             float     Branch B
brightness_mean       float     Branch B
nir_mean              float     Branch B
lst_c                 float     Branch B
relative_lst_c        float     Branch B <- ML LABEL
gis_building_coverage float     Branch C
gis_road_coverage     float     Branch C
gis_park_coverage     float     Branch C
water_distance_m      float     Branch C
```

### Zone — `zones.geojson`
```
zone_id               string
city_id               string
geometry              GeoJSON   Dissolved polygon
severity              string    "low" | "moderate" | "high" | "extreme"
mean_relative_heat    float
top_contributors      string[]
top_recommendations   string[]
gemini_summary        string    Added by Farill in Phase 10
```

---

## API Contracts (DO NOT CHANGE)

```
GET  /cities
     Response: { cities: [{ city_id: string, name: string }] }

GET  /zones?city_id=toronto
     Response: { zones: ZoneFeatureCollection }

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
     Response: { cells: CellFeatureCollection }

POST /selection
     Body:    { geometry: GeoJSON Polygon }
     Response: { severity, mean_relative_heat, top_contributors, top_recommendations }
```

---

## Naming Conventions

- All parquet files use snake_case field names exactly as listed in schemas above
- All GeoJSON uses EPSG:3347 CRS
- cell_id format: `{city_id}_{row}_{col}` e.g. `toronto_001_002`
- zone_id format: `{city_id}_zone_{number}` e.g. `toronto_zone_014`
- Severity values are lowercase strings exactly: `low`, `moderate`, `high`, `extreme`

---

## Do Not Change These Without Team Discussion

- Cell schema field names
- Zone schema field names
- API endpoint paths and response shapes
- CRS (always EPSG:3347)
- Grid size (100m x 100m)
- ML target (relative_lst_c)
- City (Toronto for MVP)

---

## Team File Ownership

| Person | Owns |
|--------|------|
| Farill | `apps/web/`, `apps/api/gemini.py`, Gemini endpoint in `apps/api/routes/zones.py` |
| Julie | `services/preprocessing/gee_pipeline.py`, `services/preprocessing/grid.py`, `services/preprocessing/statcan_buildings.py`, `services/segmentation/inference_prep.py`, `services/training/train.py`, `services/zoning/recommendations.py` |
| Georgio | `apps/api/`, `services/preprocessing/gis_pipeline.py`, `services/segmentation/inference.py`, `services/segmentation/aggregate.py`, `services/training/evaluate.py`, `services/zoning/zone_aggregation.py` |

Do not modify files outside your ownership without telling the team first.

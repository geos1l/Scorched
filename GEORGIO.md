# CanCool AI — Georgio's Workstream
> Read 00_PROJECT_CORE.md first, then this file. Do not read Farill.md or Julie.md.
> **MVP is scoped to a demo AOI (~2 km × 2 km, ~400 cells, 200 tiles). All tasks below operate on that subset, not full Toronto.**

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

## Your Phases (MVP — Demo AOI scoped)

> **Demo AOI bbox (WGS84):** min_lon -79.4071, min_lat 43.6354, max_lon -79.3821, max_lat 43.6534
> All scripts filter to cells inside this bbox. Tile cap: 200.

### Phases 1 + 3: DONE
- Repo structure, toronto_grid.geojson, statcan_buildings.parquet, gis_cell_features.parquet all complete for 68k cells.
- Tile upload to Vultr ~75% done (continues in background).
- No re-runs needed for these.

---

### Phase 2 (MVP) — Segmentation Inference on Demo AOI

- Add `--limit N` and `--aoi-bbox` flags to `inference.py`:
  - Filter tile_index.json to tiles whose bounds intersect the demo AOI bbox
  - Cap at 200 tiles
  - Write mask_index.json at the end (not only on full completion)
- Run inference: `python -m services.segmentation.inference --limit 200 --aoi-bbox "-79.4071,43.6354,-79.3821,43.6534"`
- Run `aggregate.py` → `segmentation_cell_features.parquet` (68k rows; only AOI cells have non-NaN seg)

**Estimated time:** ~1–2 h on CPU for 200 tiles.

---

### Phase 4 (MVP) — Landsat: STUB

- Write a small script (or add to existing) to generate `landsat_cell_features.parquet` with stub values for AOI cells:
  - Filter toronto_grid.geojson to AOI bbox → get list of cell_ids
  - `ndvi_mean`: random in [0.15, 0.45]
  - `brightness_mean`: random in [0.10, 0.30]
  - `nir_mean`: random in [0.15, 0.35]
  - `lst_c`: random in [28, 42] (Toronto summer surface temps)
  - `relative_lst_c`: `lst_c - median(lst_c)` across AOI cells

**Estimated time:** ~15 min.

---

### Phase 6 (MVP) — Feature Validation + Output

- Julie writes features.py (inner join of seg + gis + landsat on cell_id, plus fusion features)
- You validate join, impute missing, output `features.parquet`

**Estimated time:** ~5–10 min.

---

### Phase 7 (MVP) — Model Evaluation + Predictions

- Julie trains XGBoost, saves `models/xgboost_heat_model.json`
- You write `evaluate.py`: load model, predict on AOI cells, compute severity buckets, output `predictions.parquet`

**Estimated time:** ~5 min.

---

### Phase 8 (MVP) — FastAPI Backend

- FastAPI + CORS in `apps/api/main.py`
- Load zones.geojson + predictions.parquet at startup
- Implement: GET /cities, GET /zones, GET /zones/{zone_id}, GET /cells
- Deploy to Vultr, share URL with Farill

**Estimated time:** ~1–2 h.

---

### Phase 9 (MVP) — Zone Aggregation

- `zone_aggregation.py`: Load predictions.parquet, filter high/extreme, dissolve adjacent → zone polygons
- Pass to Julie for recommendations.py → she outputs zones.geojson

**Estimated time:** ~15–30 min.

---

### Phase 11 (MVP) — Deployment + Submission

- Vultr stable + publicly accessible
- GitHub repo public before 10AM Sunday
- No pre-hackathon commits

---

## Merge Points (MVP)

| When | What happens |
|------|-------------|
| After inference + aggregate done | You produce seg parquet; start stub landsat |
| After all three parquets exist | Julie starts features.py |
| After features.parquet output | Julie trains; you evaluate → predictions.parquet |
| After predictions.parquet | You start zone_aggregation + API simultaneously |
| After Julie outputs zones.geojson | Load into API, tell Farill to swap dummy data |
| After API deployed on Vultr | Tell Farill the URL |

---

## Future Work (post-demo)

- Full 68k inference + aggregate (tile upload continues in background)
- Real GEE Landsat export + cell aggregation (replaces stub)
- Optimize gis_pipeline (overlay/sjoin instead of unary_union)
- Full-city train/evaluate/zones
- POST /selection drag-rectangle endpoint

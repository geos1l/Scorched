# Build Plan Changes (What Was Actually Built vs Original Plan)

> Reference doc for implementing the Vultr-based tile workflow and other deviations from the original MD build plan. Use this when updating inference.py, docs, or file ownership.

---

## ML target (relative_lst_c) — where it comes from

- **The target/label** (the number we train the model to predict) is **relative_lst_c** = `cell_lst_c - city_median_lst`. That value is **computed from Landsat only** (Branch B / GEE): LST per cell from the Landsat composite, then subtract the city-wide median.
- **The features** (inputs the model uses to predict that target) come from **all three branches**: Branch A (ESRI/segmentation → seg_building_pct, seg_road_pct, seg_vegetation_pct, seg_water_pct, seg_land_pct, seg_unlabeled_pct), Branch B (Landsat → ndvi_mean, brightness_mean, nir_mean; lst_c/relative_lst_c are the label, not features), Branch C (GIS → gis_building_coverage, gis_road_coverage, gis_park_coverage, water_distance_m). So the **model** is trained on feature composition from all three; the **label** we fit to is from Landsat/GEE only.
- **GEE** is used only for **Branch B (Phase 4)** — Landsat pull and composite. It is not used for segmentation imagery (Branch A uses ESRI World Imagery).

---

## Phase 2 — Who does what (current)

**Julie:** Sourced imagery (ESRI World Imagery). Tiles were initially downloading to her device but stopped at a small number; she **deleted** those and switched to **routing directly to the Vultr server** (source → Vultr, not source → device → Vultr). After **~2GB** had been uploaded to the bucket she **stopped** (ran out of credits). She did **not** generate or store all 68k locally. Bucket `torontotiles` currently has **~2GB**; tile prep is **incomplete**. Bucket max is 1TB; this task should use under 300GB.

**Georgio (for the time being):**
1. **Create** a tile-fetch + upload script (it **does not exist** — Julie deleted it). The script must: fetch tiles from ESRI World Imagery MapServer (`https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export`), generate PNGs (1250×1250 → 1024×1024 resize per plan), and **upload incrementally** to the Vultr bucket (fetch a batch → upload to bucket → repeat; we **don’t have** the PNGs, so they must be generated and uploaded in batches so we never need to store all 68k locally). Build/update `tile_index.json` (bounds in EPSG:3347 per tile) and upload it to the bucket. Use one paginated LIST to get existing bucket keys and skip tiles already in the bucket (do not HEAD per file).
2. **Run** that script to fill the bucket (or run it on a Vultr instance so fetch + upload happens on the server).
3. Update `inference.py` to pull tiles and `tile_index.json` from the Vultr bucket (boto3, credentials from `.env`), load PNGs only (no NPY), run SegFormer, write masks to `data/processed/segmentation_masks/`.
4. Run `aggregate.py` as-is (reads masks + grid → `segmentation_cell_features.parquet`).
5. Ensure the pipeline runs on a machine with bucket access (e.g. Vultr instance with `.env` configured).

**We don’t have the PNGs.** The tile-fetch script was deleted by Julie. Claude/Georgio must **implement** the script and run it to generate tiles from Esri and upload them incrementally to Vultr; there is no existing script and no local copy of the full tile set.

---

## Phase 2 — Segmentation Tile Prep: What Changed

### 1. Imagery source

| | Plan | Reality |
|---|------|--------|
| **Source** | "Try City of Toronto Open Data orthophoto tiles first; Alternative: ArcGIS REST tile endpoint" | Toronto Open Data aerial imagery is **retired** and no longer available. Using **ESRI World Imagery MapServer export**: `https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export` — publicly accessible, no API key, confirmed working. Same 8cm/pixel GSD, same 1250×1250 → 1024×1024 resize logic. **Note:** ESRI World Imagery is a commercial product; licensing may apply for production or public demos. |

### 2. Tile storage location

| | Plan | Reality |
|---|------|--------|
| **Storage** | Save tiles locally to `data/raw/orthophoto_tiles/` | Full set: 68,394 PNGs at ~1.8MB each ≈ **~126GB**. Target is **Vultr Object Storage** — bucket `torontotiles`, endpoint `ewr1.vultrobjects.com`. Max bucket 1TB; this task under 300GB. **Upload incomplete:** Julie routed source→Vultr, then ran out of credits; **~2GB** is in the bucket now. Georgio needs to upload the rest. Credentials in `.env` (added to `.gitignore`). |

**Adding the remaining tiles:** We do **not** have the PNGs; they must be **generated** by a script that fetches from Esri and **uploads incrementally** to the bucket (fetch batch → upload → repeat). The script should LIST the bucket **once** (paginated) to get existing keys, then for each tile: if key already in bucket, skip; else fetch from Esri, save as PNG, upload to bucket. **Do not** issue a HEAD (or LIST) per file. When the script finishes, ensure `tile_index.json` is in the bucket and covers all tiles (script should build it from grid bounds in EPSG:3347).

**Georgio (inference.py):** To access tiles, **pull from the Vultr bucket using boto3 (S3-compatible)**. Credentials are in `.env`:

```
VULTR_ACCESS_KEY=...
VULTR_SECRET_KEY=...
VULTR_BUCKET=torontotiles
VULTR_ENDPOINT=https://ewr1.vultrobjects.com
```

### 3. No NPY files

| | Plan / original code | Reality |
|---|----------------------|--------|
| **Formats** | Save a `.npy` float32 ImageNet-normalized array alongside each PNG. | **Dropped entirely.** The HuggingFace pipeline for `jgerbscheid/segformer_b1-nlver_finetuned-1024-1024` normalizes internally. Only PNGs are uploaded. **inference.py should load PNGs directly** and let the pipeline handle normalization. |

### 4. tile_index.json location

| | Plan | Reality |
|---|------|--------|
| **Location** | Saved to `data/raw/orthophoto_tiles/tile_index.json` (local) | **Uploaded to the Vultr bucket** as `tile_index.json` alongside the tiles. Same format — maps each `tile_{row:03d}_{col:03d}.png` to its EPSG:3347 bounding box: |

```json
{
  "tile_001_002.png": {"bounds": [minx, miny, maxx, maxy], "crs": "EPSG:3347"}
}
```

**tile_index.json provenance:** The tile-fetch script (to be created) must **build** `tile_index.json` from the grid: e.g. use `toronto_grid.geojson` (or the same tile grid the script uses to define which tiles to fetch) to get bounds in EPSG:3347 per tile, then write the JSON and upload it to the bucket. There is no existing index to reuse — Julie’s script was deleted.

---

## Phase 3 — StatCan Buildings: What Changed

| | Plan | Reality |
|---|------|--------|
| **File** | JULIE.md listed only `grid.py` and `gee_pipeline.py` under `services/preprocessing/`. Phase 3 was described but no file was listed. | File created at **`services/preprocessing/statcan_buildings.py`** (not gee_pipeline.py). Georgio's file ownership table in `00_PROJECT_CORE.md` lists `gis_pipeline.py` under his ownership — **`statcan_buildings.py` is Julie's** and should be added to her file list. |


---

## Phase 4 — GEE: What Changed (for when Julie resumes)

GEE is used **only for Branch B (Landsat)** — thermal/spectral composite that yields `lst_c`, `relative_lst_c` (the ML label), `ndvi_mean`, etc. It is not used for segmentation imagery (Branch A = ESRI World Imagery).

| | Plan | Reality |
|---|------|--------|
| **Auth** | Doesn't specify authentication method. | GEE **service accounts cannot export to Google Drive** — they have no storage quota. Julie must use **personal OAuth** (`ee.Authenticate()`) to run `gee_pipeline.py`. Teammates cannot run Phase 4 without their own GEE-enabled Google account. The export outputs GeoTIFFs to **Google Drive**, which must then be **manually moved** to `data/raw/landsat/`. |

---

## Summary for Georgio

| What | Was | Now |
|------|-----|-----|
| **Tile source** | Toronto Open Data (unavailable) | ESRI World Imagery MapServer |
| **Tile storage** | `data/raw/orthophoto_tiles/` (local) | Vultr S3 bucket `torontotiles` (target; only ~2GB uploaded so far — Georgio to create fetch+upload script and fill bucket incrementally; we don’t have the PNGs) |
| **File formats** | PNG + NPY | PNG only |
| **tile_index.json** | Local file | Uploaded to Vultr bucket |
| **Credentials** | n/a | `.env` at project root |

---

## What to do (implementation checklist)

0. **Create and run the tile-fetch + upload script** (script does **not** exist — Julie deleted it). We **don’t have** the PNGs; the script must fetch from Esri World Imagery MapServer, generate PNGs (1250×1250 → 1024×1024), and upload **incrementally** to bucket `torontotiles`: LIST bucket once (paginated) for existing keys, then for each tile (e.g. from `toronto_grid.geojson` or a matching tile grid), if key already in bucket skip, else fetch → save PNG → upload. Do not HEAD per file. Script must also build `tile_index.json` (bounds EPSG:3347 per tile) and upload it to the bucket. Run the script (locally or on a Vultr instance) until the bucket has all tiles.

1. **inference.py**
   - Use **boto3** (S3-compatible) to list and download tiles from Vultr bucket `torontotiles` (or stream in batches). Read credentials from `.env`: `VULTR_ACCESS_KEY`, `VULTR_SECRET_KEY`, `VULTR_BUCKET`, `VULTR_ENDPOINT`.
   - Download (or stream) **tile_index.json** from the bucket; same format as before.
   - Load **PNGs only** (no .npy); let the HuggingFace processor handle normalization.
   - **When loading the model:** HuggingFace `config.id2label` may be a list (index = class id) or a dict with string keys (e.g. `"0"`, `"1"`). Normalize so your class_map uses integer keys — e.g. if it’s a list, build `{i: id2label[i] for i in range(len(id2label))}`; if dict with string keys, use `int(k)` when iterating.
   - **Verify** `data/city_configs/toronto.json`: `segformer_model` should be `jgerbscheid/segformer_b1-nlver_finetuned-1024-1024` (two 1024s); fix if it’s truncated (e.g. `...1024-102`).
   - Run inference on each tile; write masks to `data/processed/segmentation_masks/` as before (local or on same Vultr instance that has bucket access).

2. **Docs / ownership**
   - Add `services/preprocessing/statcan_buildings.py` to Julie's file list in `00_PROJECT_CORE.md` (or equivalent ownership table).
   - Ensure `.env` is in `.gitignore` (credentials must not be committed).

3. **Environment**
   - Add `boto3` (and optionally `python-dotenv` for loading `.env`) to requirements if not already present.
   - Document in README or CONTEXT: tiles live in Vultr bucket; inference runs on a machine with bucket access and `.env` configured.

"""
Orthophoto tile uploader: ESRI World Imagery → Vultr Object Storage.

Streams tiles directly from ESRI MapServer to the Vultr bucket — nothing
is saved to local disk. Safe to re-run: lists existing bucket keys once
(paginated) and skips tiles already in the bucket.

Source:  https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export
Bucket:  torontotiles @ ewr1.vultrobjects.com (S3-compatible)
Auth:    .env at project root (VULTR_ACCESS_KEY, VULTR_SECRET_KEY, VULTR_BUCKET, VULTR_ENDPOINT)

Grid: loaded directly from data/processed/toronto_grid.geojson (Julie's Phase 1 output).
  ~68k cells clipped to Toronto boundary — avoids requesting out-of-coverage areas that 500.

Each tile: 1250×1250px exported from ESRI (8cm/pixel), resized to 1024×1024, uploaded as PNG.
Tile naming: tile_{row:03d}_{col:03d}.png  (row/col parsed from cell_id in geojson)

tile_index.json uploaded to bucket root mapping filename → EPSG:3347 bounds.

Usage:
  python -m services.preprocessing.tile_uploader                 # upload all missing tiles
  python -m services.preprocessing.tile_uploader --dry-run       # count missing, don't upload
  python -m services.preprocessing.tile_uploader --workers 2     # parallel workers (default 2)
  python -m services.preprocessing.tile_uploader --index-only    # only regenerate tile_index.json

  # 5-way split (run 5 processes at once; no overlap — each shard gets 1/5 of current missing by index):
  python -m services.preprocessing.tile_uploader --workers 16 --shard 0 --total-shards 5
  python -m services.preprocessing.tile_uploader --workers 16 --shard 1 --total-shards 5
  python -m services.preprocessing.tile_uploader --workers 16 --shard 2 --total-shards 5
  python -m services.preprocessing.tile_uploader --workers 16 --shard 3 --total-shards 5
  python -m services.preprocessing.tile_uploader --workers 16 --shard 4 --total-shards 5
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import geopandas as gpd
import requests
from botocore.client import Config
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

VULTR_ACCESS_KEY = os.environ["VULTR_ACCESS_KEY"]
VULTR_SECRET_KEY = os.environ["VULTR_SECRET_KEY"]
VULTR_BUCKET = os.environ["VULTR_BUCKET"]
VULTR_ENDPOINT = os.environ["VULTR_ENDPOINT"]

GRID_PATH = REPO_ROOT / "data" / "processed" / "toronto_grid.geojson"

ESRI_URL = "https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
ESRI_EXPORT_PX = 1024  # request at final size directly — no resize needed
TILE_PX = 1024
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2
RETRY_BACKOFF = [3, 6]  # seconds between retries — failed tiles are retried on re-run


# ---------------------------------------------------------------------------
# Grid — load from toronto_grid.geojson
# ---------------------------------------------------------------------------

def load_cells() -> list[dict]:
    """
    Load grid cells from toronto_grid.geojson.
    Returns list of {key, row, col, minx, miny, maxx, maxy}.
    cell_id format: toronto_{row:03d}_{col:03d}
    """
    log.info("Loading grid from %s...", GRID_PATH)
    gdf = gpd.read_file(GRID_PATH)
    log.info("Grid loaded: %d cells", len(gdf))

    cells = []
    for _, row in gdf.iterrows():
        cell_id: str = row["cell_id"]
        # Parse row/col from cell_id: toronto_001_002 -> row=1, col=2
        parts = cell_id.split("_")
        r, c = int(parts[-2]), int(parts[-1])
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        cells.append({
            "key": f"tile_{r:03d}_{c:03d}.png",
            "row": r,
            "col": c,
            "minx": bounds[0],
            "miny": bounds[1],
            "maxx": bounds[2],
            "maxy": bounds[3],
        })
    return cells


# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------

def make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=VULTR_ENDPOINT,
        aws_access_key_id=VULTR_ACCESS_KEY,
        aws_secret_access_key=VULTR_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def list_existing_keys(s3) -> set[str]:
    log.info("Listing existing bucket objects (one paginated LIST)...")
    existing = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=VULTR_BUCKET):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])
    log.info("Bucket contains %d existing objects", len(existing))
    return existing


# ---------------------------------------------------------------------------
# Tile fetch + upload
# ---------------------------------------------------------------------------

def fetch_tile_from_esri(minx: float, miny: float, maxx: float, maxy: float) -> bytes:
    """Fetch tile from ESRI, resize 1250→1024, return PNG bytes."""
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "3347",
        "size": f"{ESRI_EXPORT_PX},{ESRI_EXPORT_PX}",
        "format": "png32",
        "transparent": "false",
        "f": "image",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ESRI_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            # Verify image and return as-is (requested at 1024px directly)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF[attempt - 1]
            log.warning("ESRI attempt %d/%d failed (%s) — retrying in %ds", attempt, MAX_RETRIES, e, wait)
            time.sleep(wait)


def process_tile(cell: dict) -> tuple[str, bool, str | None]:
    key = cell["key"]
    try:
        s3 = make_s3_client()
        png_bytes = fetch_tile_from_esri(cell["minx"], cell["miny"], cell["maxx"], cell["maxy"])
        s3.upload_fileobj(
            io.BytesIO(png_bytes),
            VULTR_BUCKET,
            key,
            ExtraArgs={"ContentType": "image/png"},
        )
        return key, True, None
    except Exception as e:
        return key, False, str(e)


# ---------------------------------------------------------------------------
# tile_index.json
# ---------------------------------------------------------------------------

def build_and_upload_tile_index(s3, cells: list[dict]) -> None:
    log.info("Building tile_index.json for %d tiles...", len(cells))
    index = {
        c["key"]: {
            "bounds": [c["minx"], c["miny"], c["maxx"], c["maxy"]],
            "crs": "EPSG:3347",
        }
        for c in cells
    }
    data = json.dumps(index, indent=2).encode()
    s3.upload_fileobj(
        io.BytesIO(data),
        VULTR_BUCKET,
        "tile_index.json",
        ExtraArgs={"ContentType": "application/json"},
    )
    log.info("tile_index.json uploaded (%d entries)", len(index))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, workers: int = 16, index_only: bool = False,
         shard: int = 0, total_shards: int = 1) -> None:
    cells = load_cells()
    s3 = make_s3_client()

    if index_only:
        build_and_upload_tile_index(s3, cells)
        return

    existing_keys = list_existing_keys(s3)
    missing = [c for c in cells if c["key"] not in existing_keys]

    # Partition current missing set by index (i % total_shards == shard). No overlap:
    # shard 0 gets indices 0,5,10,..., shard 1 gets 1,6,11,..., etc. Safe sequential or parallel.
    if total_shards > 1:
        missing = [c for i, c in enumerate(missing) if i % total_shards == shard]
        log.info("Shard %d/%d — handling %d tiles", shard + 1, total_shards, len(missing))
    else:
        log.info("Total tiles: %d | Already uploaded: %d | Missing: %d",
                 len(cells), len(cells) - len(missing), len(missing))

    if dry_run:
        log.info("DRY RUN — no uploads performed")
        return

    if not missing:
        log.info("All tiles in this shard already uploaded.")
        return

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_tile, cell): cell for cell in missing}
        with tqdm(total=len(missing), unit="tile") as pbar:
            for future in as_completed(futures):
                key, success, err = future.result()
                if success:
                    pbar.set_postfix_str(key)
                else:
                    failed.append(key)
                    log.warning("FAILED: %s — %s", key, err)
                pbar.update(1)

    log.info("Done. Success: %d | Failed: %d", len(missing) - len(failed), len(failed))
    if failed:
        log.warning("Failed tiles (re-run to retry): %d tiles", len(failed))

    # Only shard 0 regenerates tile_index.json — run after all shards finish
    if shard == 0:
        build_and_upload_tile_index(s3, cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--total-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()
    main(dry_run=args.dry_run, workers=args.workers, index_only=args.index_only,
         shard=args.shard, total_shards=args.total_shards)

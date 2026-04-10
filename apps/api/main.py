"""
Phase 8 — FastAPI backend entry point.

Loads zones.geojson and predictions.parquet at startup.
All route shapes defined in 00_PROJECT_CORE.md.

USAGE:
  uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env from repo root so GEMINI_API_KEY etc. are available before routes (e.g. gemini) import
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from apps.api.routes import cities, zones, cells, tiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ZONES_PATH = REPO_ROOT / "data" / "processed" / "zones.geojson"
PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "predictions.parquet"
GRID_PATH = REPO_ROOT / "data" / "processed" / "toronto_grid.geojson"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading data...")

    if ZONES_PATH.exists():
        zones_gdf = gpd.read_file(ZONES_PATH)
        if zones_gdf.crs and zones_gdf.crs.to_epsg() != 4326:
            zones_gdf = zones_gdf.to_crs(epsg=4326)
        app.state.zones_gdf = zones_gdf
        log.info("Loaded zones.geojson: %d zones (CRS: WGS84)", len(app.state.zones_gdf))
    else:
        log.warning("zones.geojson not found — using empty GeoDataFrame")
        app.state.zones_gdf = gpd.GeoDataFrame()

    if PREDICTIONS_PATH.exists():
        app.state.predictions = pd.read_parquet(PREDICTIONS_PATH)
        log.info("Loaded predictions.parquet: %d rows", len(app.state.predictions))
    else:
        log.warning("predictions.parquet not found — using empty DataFrame")
        app.state.predictions = pd.DataFrame()

    if GRID_PATH.exists():
        grid_gdf = gpd.read_file(GRID_PATH)
        if grid_gdf.crs and grid_gdf.crs.to_epsg() != 4326:
            grid_gdf = grid_gdf.to_crs(epsg=4326)
        app.state.grid_gdf = grid_gdf
        log.info("Loaded toronto_grid.geojson: %d cells (CRS: WGS84)", len(app.state.grid_gdf))
    else:
        log.warning("toronto_grid.geojson not found — using empty GeoDataFrame")
        app.state.grid_gdf = gpd.GeoDataFrame()

    # Pre-generate segmentation mosaic in a thread so the first HTTP request
    # is instant (avoids proxy timeout on /tiles/aoi/mosaic).
    log.info("Pre-generating segmentation mosaic (background thread)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, tiles.pregenerate_mosaic)
    log.info("Startup complete.")

    yield
    log.info("Shutting down.")


app = FastAPI(title="CanCool AI", lifespan=lifespan)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cities.router)
app.include_router(zones.router)
app.include_router(cells.router)
app.include_router(tiles.router, prefix="/tiles", tags=["tiles"])


@app.get("/health")
def health():
    return {"status": "ok"}

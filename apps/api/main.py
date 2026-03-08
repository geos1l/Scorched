"""
Phase 8 — FastAPI backend entry point.

Loads zones.geojson and predictions.parquet at startup.
All route shapes defined in 00_PROJECT_CORE.md.

USAGE:
  uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import cities, zones, cells

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
        app.state.zones_gdf = gpd.read_file(ZONES_PATH)
        log.info("Loaded zones.geojson: %d zones", len(app.state.zones_gdf))
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
        app.state.grid_gdf = gpd.read_file(GRID_PATH)
        log.info("Loaded toronto_grid.geojson: %d cells", len(app.state.grid_gdf))
    else:
        log.warning("toronto_grid.geojson not found — using empty GeoDataFrame")
        app.state.grid_gdf = gpd.GeoDataFrame()

    yield
    log.info("Shutting down.")


app = FastAPI(title="CanCool AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cities.router)
app.include_router(zones.router)
app.include_router(cells.router)


@app.get("/health")
def health():
    return {"status": "ok"}

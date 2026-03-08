"""
GET /cells?city_id=toronto  -> CellFeatureCollection (debug endpoint)
Returns grid cells with predicted_heat and severity joined in.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/cells")
def get_cells(request: Request, city_id: str = "toronto"):
    grid: gpd.GeoDataFrame = request.app.state.grid_gdf
    predictions: pd.DataFrame = request.app.state.predictions

    if grid.empty:
        return {"cells": {"type": "FeatureCollection", "features": []}}

    # Filter by city_id if column exists
    if "city_id" in grid.columns:
        grid = grid[grid["city_id"] == city_id]

    # Join predictions
    if not predictions.empty:
        grid = grid.merge(predictions[["cell_id", "predicted_heat", "severity"]],
                          on="cell_id", how="left")

    features = []
    for _, row in grid.iterrows():
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__ if row.geometry else None,
            "properties": {
                "cell_id": row.cell_id,
                "predicted_heat": float(row.predicted_heat) if pd.notna(row.get("predicted_heat")) else None,
                "severity": row.get("severity") if pd.notna(row.get("severity")) else None,
            },
        })

    return {"cells": {"type": "FeatureCollection", "features": features}}

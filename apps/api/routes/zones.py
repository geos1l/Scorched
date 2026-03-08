"""
GET /zones?city_id=toronto       -> ZoneFeatureCollection
GET /zones/{zone_id}             -> zone detail
POST /selection                  -> { geometry } -> summary for selection area

gemini_summary is loaded from zones.geojson — Farill populates it via gemini.py.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from shapely.geometry import shape

from ..gemini import generate_zone_summary

router = APIRouter()


def _to_list(x):
    """Ensure value is a JSON-serializable list."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x) if not isinstance(x, list) else x


def _zone_feature(row) -> dict:
    mrh = row.get("mean_relative_heat") if hasattr(row, "get") else row.mean_relative_heat
    if pd.isna(mrh):
        mrh = None
    else:
        mrh = float(mrh)
    return {
        "type": "Feature",
        "geometry": row.geometry.__geo_interface__ if row.geometry else None,
        "properties": {
            "zone_id": row.zone_id,
            "city_id": row.city_id,
            "severity": row.severity,
            "mean_relative_heat": mrh,
            "top_contributors": _to_list(row.top_contributors if hasattr(row, "top_contributors") else []),
            "top_recommendations": _to_list(row.top_recommendations if hasattr(row, "top_recommendations") else []),
            "gemini_summary": str(row.gemini_summary) if hasattr(row, "gemini_summary") and pd.notna(row.gemini_summary) else "",
        },
    }


@router.get("/zones")
def get_zones(request: Request, city_id: str = "toronto"):
    gdf: gpd.GeoDataFrame = request.app.state.zones_gdf
    if gdf.empty:
        return {"zones": {"type": "FeatureCollection", "features": []}}
    filtered = gdf[gdf["city_id"] == city_id] if "city_id" in gdf.columns else gdf
    features = [_zone_feature(row) for _, row in filtered.iterrows()]
    return {"zones": {"type": "FeatureCollection", "features": features}}


@router.get("/zones/{zone_id}")
def get_zone(zone_id: str, request: Request):
    gdf: gpd.GeoDataFrame = request.app.state.zones_gdf
    if gdf.empty:
        raise HTTPException(status_code=404, detail="No zones loaded")
    match = gdf[gdf["zone_id"] == zone_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")
    row = match.iloc[0]
    mrh = row.mean_relative_heat
    mrh = None if pd.isna(mrh) else float(mrh)
    contributors = _to_list(row.top_contributors if hasattr(row, "top_contributors") else [])
    recommendations = _to_list(row.top_recommendations if hasattr(row, "top_recommendations") else [])
    summary = str(row.gemini_summary) if hasattr(row, "gemini_summary") and pd.notna(row.gemini_summary) else ""

    if not summary:
        summary = generate_zone_summary({
            "severity": row.severity,
            "mean_relative_heat": mrh,
            "top_contributors": contributors,
            "top_recommendations": recommendations,
        })

    return {
        "zone_id": row.zone_id,
        "severity": row.severity,
        "mean_relative_heat": mrh,
        "top_contributors": contributors,
        "top_recommendations": recommendations,
        "gemini_summary": summary,
    }


class SelectionBody(BaseModel):
    geometry: dict


@router.post("/selection")
def post_selection(body: SelectionBody, request: Request):
    gdf: gpd.GeoDataFrame = request.app.state.zones_gdf
    predictions = request.app.state.predictions
    grid: gpd.GeoDataFrame = request.app.state.grid_gdf

    selection_shape = shape(body.geometry)

    # Find cells within selection
    if not grid.empty and not predictions.empty:
        hits = grid[grid.geometry.intersects(selection_shape)]
        cell_ids = hits["cell_id"].tolist()
        pred_hits = predictions[predictions["cell_id"].isin(cell_ids)].dropna(subset=["predicted_heat"])

        if not pred_hits.empty:
            mean_heat = float(pred_hits["predicted_heat"].mean())
            dominant = pred_hits["severity"].value_counts().idxmax()
            # Collect contributors/recommendations from any overlapping zones
            contributors: list = []
            recommendations: list = []
            if not gdf.empty:
                zone_hits = gdf[gdf.geometry.intersects(selection_shape)]
                for _, zrow in zone_hits.iterrows():
                    if hasattr(zrow, "top_contributors"):
                        contributors.extend(zrow.top_contributors or [])
                    if hasattr(zrow, "top_recommendations"):
                        recommendations.extend(zrow.top_recommendations or [])
            return {
                "severity": dominant,
                "mean_relative_heat": mean_heat,
                "top_contributors": list(dict.fromkeys(contributors))[:3],
                "top_recommendations": list(dict.fromkeys(recommendations))[:3],
            }

    return {
        "severity": "unknown",
        "mean_relative_heat": None,
        "top_contributors": [],
        "top_recommendations": [],
    }

# Scorched

Geospatial ML heat zone detector for Toronto. Identifies heat-prone urban zones by combining satellite image segmentation, thermal data, and open GIS layers to predict which areas are hottest, explain the surface-level causes, and recommend cooling interventions.

Built at [Hack Canada 2026](https://devpost.com/software/scorched).

## Stack

- **Backend:** Python, FastAPI, XGBoost, SegFormer, GeoPandas
- **Frontend:** Next.js, Mapbox GL JS, Tailwind CSS
- **AI:** Gemini for plain-English zone summaries
- **Infra:** Vultr (API + object storage), Vercel (frontend)

## How It Works

1. High-res orthophoto tiles are segmented using a pretrained SegFormer model to classify buildings, roads, vegetation, and water
2. Segmentation outputs are combined with GIS features (StatCan buildings, OSM roads/parks/water) and Landsat thermal data
3. An XGBoost model predicts relative heat per 100m x 100m grid cell
4. Hot cells are clustered into zones with severity ratings, top contributors, and recommended interventions
5. Users explore results on an interactive map with clickable zones and AI-generated summaries

## Team

- **Georgio** — Backend, GIS pipeline, ML eval, segmentation inference, FastAPI
- **Julie** — Data sourcing, preprocessing, model training, zoning recommendations
- **Farill** — Frontend, Mapbox UI, Gemini integration

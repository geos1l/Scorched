#!/bin/bash
# Run this on the Vultr instance after SSH-ing in.
# Sets up Python env and kicks off the tile uploader.

set -e

# 1. System deps
apt-get update -y && apt-get install -y python3 python3-venv python3-pip git

# 2. Clone repo
git clone https://github.com/geos1l/HackCanada-2026.git
cd HackCanada-2026

# 3. Venv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install geopandas boto3 python-dotenv requests tqdm Pillow

# 4. Drop .env (paste your credentials here before running)
cat > .env << 'EOF'
VULTR_ACCESS_KEY=YOUR_ACCESS_KEY
VULTR_SECRET_KEY=YOUR_SECRET_KEY
VULTR_BUCKET=torontotiles
VULTR_ENDPOINT=https://ewr1.vultrobjects.com
EOF

# 5. Dry run first to confirm tile count
python -m services.preprocessing.tile_uploader --dry-run

# 6. Full run — 16 workers, stays alive in background
nohup python -m services.preprocessing.tile_uploader --workers 16 > tile_upload.log 2>&1 &
echo "Uploader running in background. PID: $!"
echo "Tail logs with: tail -f tile_upload.log"

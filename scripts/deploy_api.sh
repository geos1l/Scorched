#!/bin/bash
# Deploy CanCool AI FastAPI backend on a fresh Vultr Ubuntu instance.
# Run as root after SSH-ing in.
#
# Usage:
#   bash deploy_api.sh
#
# After deploy, API is live at http://<instance-ip>:8000
# Share that URL with Farill.

set -e

echo "=== CanCool AI — API Deployment ==="

# 1. System deps
apt-get update -y && apt-get install -y python3 python3-venv python3-pip git

# 2. Find repo root (allow running from inside repo or from parent)
if [ -f "apps/api/main.py" ] || [ -f "scripts/deploy_api.sh" ]; then
  REPO_ROOT="$(pwd)"
  echo "Already in repo at $REPO_ROOT — pulling latest..."
  git pull
else
  if [ -d "HackCanada-2026" ]; then
    echo "Repo exists — pulling latest..."
    cd HackCanada-2026 && git pull && cd ..
  else
    git clone https://github.com/geos1l/HackCanada-2026.git
  fi
  cd HackCanada-2026
  REPO_ROOT="$(pwd)"
fi
cd "$REPO_ROOT"

# 3. Venv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r apps/api/requirements.txt

# 4. Kill any existing API process
pkill -f "uvicorn apps.api.main" || true

# 5. Start API with nohup
nohup python -m uvicorn apps.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  > api.log 2>&1 &

echo "API starting... PID: $!"
sleep 3
echo ""

# 6. Health check
if curl -s http://localhost:8000/health | grep -q "ok"; then
  echo "=== API is live at http://$(curl -s ifconfig.me):8000 ==="
  echo "Share this URL with Farill."
else
  echo "ERROR: API did not start. Check api.log:"
  tail -20 api.log
fi

#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "Installing dependencies..."
pip install -r requirements.txt -q
[ -f .env ] && export $(grep -v '^#' .env | xargs) 2>/dev/null || true
echo "Starting OpenCQL at http://localhost:8501"
streamlit run opencql_demo.py --server.port 8501 --browser.gatherUsageStats false

#!/usr/bin/env bash
cd "$(dirname "$0")"
pip install -r requirements.txt -q
[ -f .env ] && export $(grep -v "^#" .env | xargs) 2>/dev/null || true
streamlit run opencql_demo.py --server.port 8501

#!/usr/bin/env bash
# Launch OpenCQL Workbench
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
export OPENCQL_DSL_PATH="$SCRIPT_DIR/opencql_dsl_extracted/pkg_dsl"
export OPENCQL_INSPECTOR_PATH="$SCRIPT_DIR/inspector_extracted/inspector_clean"
export OPENCQL_WORKBENCH_DIR="$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/.env" ] && export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠  Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY=sk-ant-..."
fi

echo "Starting OpenCQL Workbench at http://localhost:8501"
streamlit run "$SCRIPT_DIR/opencql_demo.py" --server.port 8501 --browser.gatherUsageStats false

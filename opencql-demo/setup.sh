#!/usr/bin/env bash
# ============================================================
# OpenCQL Workbench — Complete Machine Setup
# Works on: macOS (Intel + Apple Silicon) and Ubuntu/Debian
# Run: bash setup.sh
# ============================================================
set -e
GREEN="\033[0;32m"; BLUE="\033[0;34m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; BOLD="\033[1m"; NC="\033[0m"
log()  { echo -e "${BLUE}[setup]${NC} $1"; }
ok()   { echo -e "${GREEN}[  ok ]${NC} $1"; }
warn() { echo -e "${YELLOW}[ warn]${NC} $1"; }

echo -e "\n${BOLD}OpenCQL Workbench — Machine Setup${NC}\n$(printf '=%.0s' {1..40})\n"

OS="$(uname -s)"; ARCH="$(uname -m)"
log "Detected: $OS / $ARCH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Step 1: Python ───────────────────────────────────────────
log "Step 1: Checking Python 3.9+..."
if command -v python3 &>/dev/null; then
    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    ok "Python $PYVER found at $(which python3)"
    if [ "$(echo $PYVER | cut -d. -f1)" -lt 3 ] || [ "$(echo $PYVER | cut -d. -f2)" -lt 11 ]; then
        warn "Python 3.9+ required. Installing..."
        if [ "$OS" = "Darwin" ]; then
            brew install python@3.12
        else
            sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
        fi
    fi
else
    log "Python not found. Installing..."
    if [ "$OS" = "Darwin" ]; then
        command -v brew &>/dev/null || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install python@3.12
    else
        sudo apt-get update -q && sudo apt-get install -y python3 python3-pip python3-venv python3-dev
    fi
fi
PYTHON=$(command -v python3)

# ── Step 2: Virtual Environment ──────────────────────────────
log "Step 2: Setting up virtual environment..."
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
else
    ok "venv already exists at $VENV_DIR"
fi
# Activate
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
ok "Activated: $PYTHON"

# ── Step 3: Install packages ─────────────────────────────────
log "Step 3: Installing Python packages..."
$PIP install --upgrade pip --quiet
$PIP install streamlit anthropic plotly "lark>=1.1" numpy google-generativeai --quiet
ok "Core packages installed"

# opencql-dsl
if [ -d "$SCRIPT_DIR/opencql_dsl_extracted/pkg_dsl" ]; then
    $PIP install -e "$SCRIPT_DIR/opencql_dsl_extracted/pkg_dsl" --quiet
    ok "opencql-dsl installed"
else
    warn "opencql_dsl_extracted/ not found — run from the unzipped workbench directory"
fi

# opencql-inspector
if [ -d "$SCRIPT_DIR/inspector_extracted/inspector_clean" ]; then
    $PIP install -e "$SCRIPT_DIR/inspector_extracted/inspector_clean" --quiet
    ok "opencql-inspector installed"
else
    warn "inspector_extracted/ not found"
fi

# ── Step 4: Create opencql CLI command ────────────────────────
log "Step 4: Creating opencql CLI..."
CLI_PATH="$VENV_DIR/bin/opencql"
cat > "$CLI_PATH" << CLIPY
#!/usr/bin/env python3
"""opencql CLI — run, inspect, version"""
import sys, os, argparse, math

_DIR = os.environ.get("OPENCQL_WORKBENCH_DIR", os.path.dirname(os.path.abspath(__file__)) + "/../..")
_DSL = os.environ.get("OPENCQL_DSL_PATH", os.path.join(_DIR, "opencql_dsl_extracted/pkg_dsl"))
_INS = os.environ.get("OPENCQL_INSPECTOR_PATH", os.path.join(_DIR, "inspector_extracted/inspector_clean"))
for p in [_DSL, _INS]:
    if os.path.exists(p): sys.path.insert(0, p)

def _uw(x):
    try:
        from lark import Tree
        return x.children[0] if isinstance(x, Tree) and len(x.children)==1 else x
    except: return x

def cmd_run(args):
    from opencql_dsl.runtime import CQLRuntime
    rt = CQLRuntime(default_model=args.model)
    print(rt.execute(open(args.file).read(), query=args.query or ""))

def cmd_inspect(args):
    from opencql_inspector import ContextInspector
    from opencql_dsl.runtime import CQLRuntime
    inspector = ContextInspector()
    rt = CQLRuntime(default_model="mock")
    cql = open(args.file).read()
    try:
        tree = rt.parser.parse(cql)
        plan = rt.compiler.transform(tree)
        stmts = [_uw(s) for s in plan.get("statements", [])]
        for s in stmts:
            if isinstance(s,dict) and s.get("type")=="context_def":
                rt._context_defs[s["name"]] = [_uw(c) for c in s.get("clauses",[])]
        infer = next((_uw(s) for s in stmts if isinstance(_uw(s),dict) and _uw(s).get("type")=="infer"), None)
        if not infer: print("No INFER found"); return
        goal = infer.get("goal", args.query or "")
        clauses = rt._context_defs.get(infer.get("context",""), [])
        gl = next((c.get("value") for c in clauses if isinstance(c,dict) and c.get("type")=="limit"), 4000)
        smap = {}
        for c in clauses:
            if not isinstance(c,dict) or c.get("type")!="retrieve": continue
            src = c["source"]; store = rt.registry.get(src)
            if not store: smap[src]=[]; continue
            wh = c.get("where",{}); thr = float(wh.get("value",0)) if wh.get("field")=="similarity" else 0
            results = store.search(goal, top_k=c.get("top",10), threshold=thr)
            adm,used=[],0; budget = c.get("token_limit") or gl
            for doc,score in results:
                t = max(1,math.ceil(len(doc.get("text",""))/3.8))
                if used+t>budget: break
                adm.append({"text":doc.get("text",""),"source":src}); used+=t
            smap[src]=adm
        chunks=[{"text":c["text"],"source":src} for src,cs in smap.items() for c in cs]
        report = inspector.inspect(chunks, query=goal, token_budget=gl, sources_expected=list(smap.keys()))
    except Exception as e:
        print(f"Parse error: {e}"); report = inspector.inspect([], query="", token_budget=2000)
    icons = {"PASS":"✅","CRITICAL":"🚨","WARN":"⚠️"}
    print(f"\n{icons.get(report.status,'?')}  {report.status}  Q=(ρ={report.rho} τ={report.tau:.2f} δ={report.delta} κ={report.kappa})")
    print(f"   Tokens: {report.tokens_used}/{report.token_budget}")
    for s in report.sources_admitted: print(f"   ✓ {s}")
    for s in report.sources_starved:  print(f"   ✗ {s}  ← STARVED")
    for w in report.warnings: print(f"   ⚠  {w['type']}: {w['msg']}")
    sys.exit({"PASS":0,"WARN":1,"CRITICAL":2}.get(report.status,2))

def cmd_version():
    try:
        from opencql_dsl.runtime import CQLRuntime; print("opencql-dsl:       v0.3.0 ✓")
    except: print("opencql-dsl:       not installed")
    try:
        from opencql_inspector import ContextInspector; print("opencql-inspector: v0.3.0 ✓")
    except: print("opencql-inspector: not installed")

p = argparse.ArgumentParser(prog="opencql")
sub = p.add_subparsers(dest="cmd")
r = sub.add_parser("run");       r.add_argument("file"); r.add_argument("--query","-q",default=""); r.add_argument("--model",default="claude-3-haiku-20240307")
i = sub.add_parser("inspect");   i.add_argument("file"); i.add_argument("--query","-q",default="")
sub.add_parser("version")
args = p.parse_args()
{"run":cmd_run,"inspect":cmd_inspect,"version":cmd_version}.get(args.cmd, lambda _: p.print_help())(args)
CLIPY
chmod +x "$CLI_PATH"
ok "opencql CLI created at $CLI_PATH"

# ── Step 5: Shell activation helpers ────────────────────────
log "Step 5: Creating shell helpers..."

# Detect shell config
if [ -f "$HOME/.zshrc" ]; then SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then SHELL_RC="$HOME/.bashrc"
else SHELL_RC="$HOME/.bash_profile"; fi

ACTIVATE_LINE="source $VENV_DIR/bin/activate  # opencql workbench"
if ! grep -q "opencql workbench" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# OpenCQL Workbench venv" >> "$SHELL_RC"
    echo "alias opencql-workbench='cd $SCRIPT_DIR && source $VENV_DIR/bin/activate'" >> "$SHELL_RC"
    ok "Added alias 'opencql-workbench' to $SHELL_RC"
else
    ok "Shell aliases already configured"
fi

# ── Step 6: run_workbench.sh launcher ───────────────────────
cat > "$SCRIPT_DIR/run_workbench.sh" << RUN
#!/usr/bin/env bash
# Launch OpenCQL Workbench
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/.venv/bin/activate"
export OPENCQL_DSL_PATH="\$SCRIPT_DIR/opencql_dsl_extracted/pkg_dsl"
export OPENCQL_INSPECTOR_PATH="\$SCRIPT_DIR/inspector_extracted/inspector_clean"
export OPENCQL_WORKBENCH_DIR="\$SCRIPT_DIR"

[ -f "\$SCRIPT_DIR/.env" ] && export \$(grep -v '^#' "\$SCRIPT_DIR/.env" | xargs)

if [ -z "\$ANTHROPIC_API_KEY" ]; then
    echo "⚠  Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY=sk-ant-..."
fi

echo "Starting OpenCQL Workbench at http://localhost:8501"
streamlit run "\$SCRIPT_DIR/opencql_workbench.py" --server.port 8501 --browser.gatherUsageStats false
RUN
chmod +x "$SCRIPT_DIR/run_workbench.sh"
ok "Created run_workbench.sh"

# .env template
[ ! -f "$SCRIPT_DIR/.env" ] && cat > "$SCRIPT_DIR/.env" << ENV
ANTHROPIC_API_KEY=sk-ant-your-key-here
ENV
ok ".env created (add your API key)"

# ── Verify ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Verification${NC}"
$PYTHON -c "import sys; print(f'  python:       {sys.version.split()[0]}')"
$PIP --version | awk '{print "  pip:          "$2}'
$PYTHON -c "import streamlit; print(f'  streamlit:    {streamlit.__version__}')" 2>/dev/null || warn "streamlit issue"
$PYTHON -c "import sys; sys.path.insert(0,'$SCRIPT_DIR/opencql_dsl_extracted/pkg_dsl'); from opencql_dsl.runtime import CQLRuntime; print('  opencql-dsl:  ✓')" 2>/dev/null || warn "dsl not found"
$PYTHON -c "import sys; sys.path.insert(0,'$SCRIPT_DIR/inspector_extracted/inspector_clean'); from opencql_inspector import ContextInspector; print('  inspector:    ✓')" 2>/dev/null || warn "inspector not found"
"$VENV_DIR/bin/opencql" version 2>/dev/null | sed 's/^/  /' || warn "opencql CLI issue"

echo ""
echo -e "${BOLD}Done! ✓${NC}"
echo ""
echo "To use:"
echo "  cd $(basename $SCRIPT_DIR)"
echo "  bash run_workbench.sh           # launch Streamlit workbench"
echo "  source .venv/bin/activate       # activate venv manually"
echo "  opencql version                 # check install"
echo "  opencql inspect billing.cql     # diagnose a CQL file"
echo "  opencql run billing.cql -q 'refund policy?'"
echo ""

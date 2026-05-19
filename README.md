# OpenCQL — Context Quality Demo

Pre-inference context quality inspection for RAG pipelines.  
Live demo: [your-app.streamlit.app](https://share.streamlit.io)

## Deploy to Streamlit Cloud (5 minutes)

### Step 1 — Push to GitHub

```bash
# Create a new GitHub repo, then:
git init
git add .
git commit -m "OpenCQL demo"
git remote add origin https://github.com/YOUR_USERNAME/opencql-demo.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch: `main` → file: `opencql_demo.py`
5. Click **"Deploy"**

### Step 3 — Add API Keys (Secrets)

1. On your deployed app → **Settings** → **Secrets**
2. Paste this (replace with your real key):

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Get your free Groq key at **console.groq.com** — no credit card needed.

### Step 4 — Share the URL

```
https://YOUR_APP_NAME.streamlit.app
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run opencql_demo.py
```

---

## Repo Structure

```
opencql-demo/
├── opencql_demo.py          # Streamlit app
├── requirements.txt         # pip dependencies
├── opencql_inspector/       # bundled — zero external deps
│   ├── __init__.py
│   └── inspector.py
└── .streamlit/
    └── secrets.toml         # API keys (never commit!)
```

---

## What it demos

**6 context quality problems:**
| # | Metric | Problem |
|---|--------|---------|
| 🔴 P1 | δ=1 | Source Starvation |
| 🔴 P2 | δ=2 | Multi-Source Starvation |
| ⚡ P3 | κ>0.3 | Contradictions |
| 📉 P4 | ρ<0.3 | Low Coverage |
| ⚠️ P5 | τ>0.95 | Budget Exhaustion |
| 💀 P6 | δ+κ+ρ | Worst Case |

**4 tabs:** Inspect (Developer) · CI Tests (QA) · Business Impact (PM) · Integrate

---

## Free LLM options

| Provider | Model | Key |
|----------|-------|-----|
| **Groq** ⭐ | Llama 3.3 70B | console.groq.com |
| Gemini | 2.5 Flash | aistudio.google.com |
| Claude | Haiku | console.anthropic.com ($5 free) |

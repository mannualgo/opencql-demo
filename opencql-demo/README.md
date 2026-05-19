# OpenCQL — Context Quality Demo

## Structure
```
├── opencql_demo.py               # Streamlit app
├── requirements.txt              # pip deps (includes local packages)
├── packages/
│   ├── opencql-inspector/        # Pre-inference context quality inspector
│   │   ├── pyproject.toml
│   │   └── opencql_inspector/
│   └── opencql-dsl/              # SQL-like DSL for context assembly
│       ├── pyproject.toml
│       └── opencql_dsl/
└── .streamlit/
    └── config.toml
```

## Run Locally
```bash
cp .env.example .env       # add your API key
bash run_local.sh
```

## Deploy to Streamlit Cloud
1. Push to GitHub
2. share.streamlit.io → New app → select repo
3. Settings → Secrets → add GROQ_API_KEY

## Free API Keys
- Groq (recommended): console.groq.com
- Gemini: aistudio.google.com

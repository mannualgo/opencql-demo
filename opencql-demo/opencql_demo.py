"""
OpenCQL — Comprehensive Demo
Developer · QA · Product Manager · Context Quality Problems · CI Tests
Run: streamlit run opencql_demo.py
"""
from __future__ import annotations
import sys, os, math, time, re, textwrap
from collections import defaultdict

# opencql-inspector and opencql-dsl installed as packages
# (see packages/ directory and requirements.txt)

import streamlit as st

st.set_page_config(page_title="OpenCQL", page_icon="⬡",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Inspector ─────────────────────────────────────────────────────────────────
try:
    from opencql_inspector import ContextInspector
    _inspector = ContextInspector()
    PKG_OK = True
except Exception as e:
    PKG_OK = False; _PKG_ERR = str(e)

# ── Core helpers ──────────────────────────────────────────────────────────────
def tok(t):  return max(1, math.ceil(len(t or "") / 3.8))
def sim(q, t):
    a = set(re.findall(r"[a-z]{3,}", (q or "").lower()))
    b = re.findall(r"[a-z]{3,}", (t or "").lower())
    return round(sum(1 for w in b if w in a) / len(a | set(b)), 4) if a and b else 0.0

def assemble(docs, query, global_limit, per_limits=None, threshold=0.0, doc_filter=None, source_filter=None):
    if doc_filter:
        k, v = list(doc_filter.items())[0]
        docs = [d for d in docs if str(d.get(k, "")) == v or k not in d]
    if source_filter:
        filtered = []
        for d in docs:
            sf = source_filter.get(d["source"])
            if sf:
                k, v = list(sf.items())[0]
                if str(d.get(k, "")) == v or k not in d:
                    filtered.append(d)
            else:
                filtered.append(d)
        docs = filtered
    by_src = defaultdict(list)
    for d in docs:
        s = sim(query, d["text"])
        if s >= threshold:
            by_src[d["source"]].append({**d, "_sc": s, "_tk": tok(d["text"])})
    # Ensure sources with per_limits appear even if no docs pass threshold
    for src in (per_limits or {}):
        if src not in by_src:
            by_src[src] = []
    smap, rem = {}, global_limit
    for src, chunks in by_src.items():
        chunks.sort(key=lambda x: x["_sc"], reverse=True)
        budget = (per_limits or {}).get(src, rem)
        used, adm = 0, []
        for c in chunks:
            if used + c["_tk"] > budget: break
            adm.append(c); used += c["_tk"]
        smap[src] = adm; rem -= used
    return smap

def run_inspector(smap, global_limit, query, sources_expected, doc_store=None, max_age_days=None):
    if not PKG_OK: return None
    chunks = []
    for src, cs in smap.items():
        for c in cs:
            chunk = {"text": c["text"], "source": src}
            # Preserve metadata fields for staleness/redundancy detection
            for k in ("age_days","ver","cat","status"):
                if k in c: chunk[k] = c[k]
            chunks.append(chunk)
    if max_age_days:
        from opencql_inspector.inspector import ContextInspector as _CI
        _insp = _CI(max_age_days=max_age_days)
    else:
        _insp = _inspector
    return _insp.inspect(chunks, query=query,
                          token_budget=global_limit,
                          sources_expected=sources_expected,
                          doc_store=doc_store)

def call_claude(system, smap, goal, api_key):
    import anthropic
    ctx = "\n\n".join(c["text"] for cs in smap.values() for c in cs)
    body = f"Context:\n{ctx}\n\n---\n{goal}" if ctx else goal
    msg = anthropic.Anthropic(api_key=api_key).messages.create(
        model="claude-3-haiku-20240307", max_tokens=450,
        system=system, messages=[{"role": "user", "content": body}])
    return msg.content[0].text

def call_gemini(system, smap, goal, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    ctx = "\n\n".join(c["text"] for cs in smap.values() for c in cs)
    prompt = f"{system}\n\nContext:\n{ctx}\n\n---\n{goal}" if ctx else f"{system}\n\n{goal}"
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

def call_groq(system, smap, goal, api_key):
    from openai import OpenAI
    ctx = "\n\n".join(c["text"] for cs in smap.values() for c in cs)
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", max_tokens=500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Context:\n{ctx}\n\n---\n{goal}" if ctx else goal}
        ]
    )
    return resp.choices[0].message.content

def call_llm(provider, system, smap, goal, api_key):
    if "Gemini"  in provider: return call_gemini(system, smap, goal, api_key)
    if "Groq"    in provider: return call_groq(system, smap, goal, api_key)
    return call_claude(system, smap, goal, api_key)

def warn_msg(w): return w.get("message", w.get("msg", ""))
def warn_type(w): return w.get("type", "")

# ════════════════════════════════════════════════════════════════════════════════
# AMEX DATA — consistent across all problems and tabs
# ════════════════════════════════════════════════════════════════════════════════
BENEFITS = [
    {"source":"amex.benefits","text":"Platinum Card: $200 annual airline fee credit — select one airline, auto-applied to incidentals each calendar year.","cat":"travel"},
    {"source":"amex.benefits","text":"Platinum Card: $200 annual Uber Cash — $15/month + $20 in December, loaded automatically to Uber account.","cat":"travel"},
    {"source":"amex.benefits","text":"Platinum Card: Centurion Lounge + Priority Pass Select — unlimited lounge access, unlimited guest visits worldwide.","cat":"travel"},
    {"source":"amex.benefits","text":"Platinum Card: $240 digital entertainment credit — $20/month for Disney+, Hulu, ESPN+, Peacock, and NYT.","cat":"lifestyle"},
    {"source":"amex.benefits","text":"Platinum Card: $189 CLEAR Plus credit — full annual reimbursement, biometric airport security at 50+ airports.","cat":"travel"},
    {"source":"amex.benefits","text":"Platinum Card: Marriott Bonvoy Gold Elite + Hilton Honors Gold status — automatic, no minimum nights required.","cat":"hotel"},
    {"source":"amex.benefits","text":"Platinum Card: 5x Membership Rewards points on flights booked directly with airlines or amextravel.com.","cat":"rewards"},
]
OFFERS = [
    {"source":"amex.offers","text":"Amex Offer ACTIVE spend Amazon get statement credit valid December add card now."},
    {"source":"amex.offers","text":"Amex Offer ACTIVE spend Delta Air Lines get credit direct ticket purchases airline."},
    {"source":"amex.offers","text":"Amex Offer ACTIVE spend Hilton Hotels get credit participating hotel properties travel."},
    {"source":"amex.offers","text":"Amex Offer ACTIVE spend DoorDash get credit food delivery orders one time use."},
]
CUSTOMER = [
    {"source":"amex.customer","text":"Priya Sharma Platinum Card January 2022 annual spend business travel fine dining."},
    {"source":"amex.customer","text":"Priya spending travel forty percent dining twenty five percent electronics other categories."},
    {"source":"amex.customer","text":"Priya Membership Rewards points accumulated airline credit mostly used small remainder left."},
    {"source":"amex.customer","text":"Priya offers Amazon activated Delta activated Hilton activated DoorDash not yet activated."},
]
POLICY_OLD = [
    {"source":"amex.policy","text":"Purchase protection: customers may return items within 30 days covers damage theft.","ver":"v2024"},
    {"source":"amex.policy","text":"Return protection: extend store return period to 30 days maximum claim per item.","ver":"v2024"},
]
POLICY_NEW = [
    {"source":"amex.policy","text":"Purchase protection: customers may return items within 90 days covers damage theft.","ver":"v2025"},
    {"source":"amex.policy","text":"Return protection: extend store return period to 90 days maximum claim per item.","ver":"v2025"},
]
# Interleaved: OLD[0], NEW[0], OLD[1], NEW[1] — ensures admitted pairs always contradict
POLICY_MIXED = [
    {"source":"amex.policy","text":"Purchase protection: customers may return items within 30 days covers damage theft.","ver":"v2024"},
    {"source":"amex.policy","text":"Purchase protection: customers may return items within 90 days covers damage theft.","ver":"v2025"},
    {"source":"amex.policy","text":"Return protection: extend store return period to 30 days maximum claim per item.","ver":"v2024"},
    {"source":"amex.policy","text":"Return protection: extend store return period to 90 days maximum claim per item.","ver":"v2025"},
]
ALL_SOURCES = ["amex.benefits", "amex.offers", "amex.customer"]
# Global doc pool — used when CQL references sources outside current problem
BENEFITS_EXTENDED = BENEFITS + [
    {"source":"amex.benefits","text":"Platinum Card Global Lounge Collection four thousand lounges one hundred fifty countries."},
    {"source":"amex.benefits","text":"Platinum Card Saks Fifth Avenue credit biannual up to fifty dollars each period shopping."},
    {"source":"amex.benefits","text":"Platinum Card Equinox credit towards gym membership monthly health fitness wellness."},
    {"source":"amex.benefits","text":"Platinum Card Fine Hotels Resorts breakfast included arrival upgrade when available."},
    {"source":"amex.benefits","text":"Platinum Card TSA PreCheck Global Entry credit application fee covered every four years."},
    {"source":"amex.benefits","text":"Platinum Card no foreign transaction fees international purchases currency conversion."},
    {"source":"amex.benefits","text":"Platinum Card purchase protection covers eligible items damage theft ninety days."},
    {"source":"amex.benefits","text":"Platinum Card extended warranty doubles manufacturers warranty two additional years."},
    {"source":"amex.benefits","text":"Platinum Card return protection allows returns merchants declined ninety days purchase."},
]
# Semantic redundancy demo: 5 near-duplicate airline credit docs
REDUNDANT_BENEFITS = [
    {"source":"amex.benefits","text":"Platinum Card annual airline fee credit incidentals selected airline year account."},
    {"source":"amex.benefits","text":"Platinum Card airline incidentals credit annual selected airline benefit year."},
    {"source":"amex.benefits","text":"Platinum annual airline credit incidentals fees selected airline reimbursed year."},
    {"source":"amex.benefits","text":"Amex Platinum annual airline credit incidentals selected airline card benefit year."},
    {"source":"amex.benefits","text":"Platinum airline credit annual incidentals fees selected airline account benefit."},
    {"source":"amex.benefits","text":"Centurion Lounge access worldwide unlimited Priority Pass Select card."},
    {"source":"amex.benefits","text":"Hotel Gold Status Marriott Bonvoy Hilton Honors automatic granted cardholders."},
]

# Staleness demo: old docs (age_days>30) mixed with current
STALE_POLICY = [
    {"source":"amex.policy","text":"Policy 2024 return within thirty days covers damage theft eligible items.","age_days":400,"ver":"v2024"},
    {"source":"amex.policy","text":"Policy 2024 return within thirty days covers theft damage items eligible.","age_days":380,"ver":"v2024"},
    {"source":"amex.policy","text":"Policy 2025 return within ninety days extended coverage Platinum cardholders.","age_days":5,"ver":"v2025"},
]

ALL_DOCS = BENEFITS + OFFERS + CUSTOMER + POLICY_OLD + POLICY_NEW + POLICY_MIXED
DEFAULT_QUERY = "platinum card offers airline lounge credit Priya"
DEFAULT_GOAL  = "What Amex offers and benefits should Priya use this month?"
DEFAULT_SYSTEM = "You are Priya's personal Amex concierge. Use her profile to give specific, personalised advice."

# ════════════════════════════════════════════════════════════════════════════════
# PROBLEM DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════
import math
# Compute P5 budget dynamically
_BEN_TOTAL = sum(max(1,math.ceil(len(d["text"])/3.8)) for d in BENEFITS)
_P5_BUDGET = _BEN_TOTAL + 5

PROBLEMS = [
  # ── ISOLATED: one metric fires, others at zero ────────────────────────────
  {
    "id":"p1","idx":0,"icon":"🔴","metric":"δ=1","label":"CRITICAL",
    "title":"Source Starvation",
    "sub":"One source gets zero tokens",
    "col":"#ef4444","bg":"#fef2f2","bdr":"#fca5a5",
    "what":"amex.benefits consumes the entire 65-token budget. amex.customer gets zero chunks. The LLM never sees Priya — answers generically. δ=1. κ=0 (no contradictions). Only δ fires.",
    "fix":"Add LIMIT TOKENS per RETRIEVE. Each source gets a guaranteed budget regardless of what other sources consume.",
    "docs": BENEFITS+CUSTOMER,
    "sources":["amex.benefits","amex.customer"],
    "doc_store_keys": ["amex.benefits","amex.customer"],
    "query":"platinum card offers airline lounge credit Priya",
    "goal":"What Amex benefits should Priya use this month?",
    "system":"You are Priya's personal Amex concierge.",
    "b_gl":65,  "b_lim":{},             "b_thr":0.05, "b_filter":None,
    "f_gl":1500,"f_lim":{"amex.benefits":350,"amex.customer":250},"f_thr":0.05,"f_filter":None,
    "broken_cql":"""\
-- ❌ ISOLATED δ=1: only starvation fires (κ=0, ρ reasonable)
-- benefits fills 65-token budget → customer gets zero chunks
-- LLM sees no customer profile → generic answer

CONTEXT broken AS (
  WITH SYSTEM "You are Priya's Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4

  LIMIT TOKENS 65
)
INFER USING CONTEXT broken
  GOAL "What Amex benefits should Priya use this month?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: per-source LIMIT TOKENS → δ=0
-- customer always gets 250 tokens guaranteed

CONTEXT fixed AS (
  WITH SYSTEM "You are Priya's personal Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 350

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 250

  LIMIT TOKENS 1500
)
INFER USING CONTEXT fixed
  GOAL "What Amex benefits should Priya use this month?"
  EXPLAIN""",
  },
  {
    "id":"p2","idx":1,"icon":"🔴","metric":"δ=2","label":"CRITICAL",
    "title":"Multi-Source Starvation",
    "sub":"Two sources starved simultaneously",
    "col":"#f87171","bg":"#fff5f5","bdr":"#fca5a5",
    "what":"Three sources share a 45-token budget. benefits takes 45 tokens. Both offers AND customer get zero chunks (δ=2). κ=0, ρ measures only admitted. Only δ fires.",
    "fix":"All three RETRIEVE blocks need their own LIMIT TOKENS. Sum of per-source limits must be less than global limit.",
    "docs": BENEFITS+OFFERS+CUSTOMER,
    "sources":["amex.benefits","amex.offers","amex.customer"],
    "doc_store_keys":["amex.benefits","amex.offers","amex.customer"],
    "query":"platinum card offers airline lounge credit Priya",
    "goal":"What offers and benefits should Priya use this month?",
    "system":"You are Priya's personal Amex concierge.",
    "b_gl":45,  "b_lim":{},             "b_thr":0.05, "b_filter":None,
    "f_gl":1500,"f_lim":{"amex.benefits":300,"amex.offers":250,"amex.customer":200},"f_thr":0.05,"f_filter":None,
    "broken_cql":"""\
-- ❌ ISOLATED δ=2: two sources starved (κ=0)
-- benefits takes all 45 tokens
-- offers and customer both get zero chunks

CONTEXT broken AS (
  WITH SYSTEM "You are Priya's Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 4

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4

  LIMIT TOKENS 45
)
INFER USING CONTEXT broken
  GOAL "What offers and benefits should Priya use this month?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: all three sources get guaranteed budgets → δ=0

CONTEXT fixed AS (
  WITH SYSTEM "You are Priya's personal Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 300

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 250

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 200

  LIMIT TOKENS 1500
)
INFER USING CONTEXT fixed
  GOAL "What offers and benefits should Priya use this month?"
  EXPLAIN""",
  },
  {
    "id":"p3","idx":2,"icon":"⚡","metric":"κ=0.67","label":"WARN",
    "title":"Contradictions",
    "sub":"Conflicting facts in context — δ=0",
    "col":"#d97706","bg":"#fffbeb","bdr":"#fcd34d",
    "what":"Old policy (30-day return) and new policy (90-day) are both loaded. κ=0.67. δ=0 (policy source contributes chunks). Only κ fires — this is a pure contradiction problem.",
    "fix":"FILTER BY ver='v2025' loads only current policy. Contradicting 30-day docs excluded. κ drops to 0.",
    "docs": POLICY_MIXED,
    "sources":["amex.policy"],
    "doc_store_keys":["amex.policy"],
    "query":"purchase return protection days policy",
    "goal":"What is the return protection period for Priya's Platinum Card?",
    "system":"You are a billing support agent. State exact number of days.",
    "b_gl":1500,"b_lim":{"amex.policy":800},"b_thr":0.05,"b_filter":None,
    "f_gl":1500,"f_lim":{"amex.policy":800},"f_thr":0.05,"f_filter":{"ver":"v2025"},
    "broken_cql":"""\
-- ❌ ISOLATED κ=0.67: only contradiction fires (δ=0)
-- OLD (30-day) + NEW (90-day) policy both loaded
-- LLM will synthesise wrong answer: "30 or 90 days?"

CONTEXT broken AS (
  WITH SYSTEM "You are a billing support agent."

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 8
    LIMIT TOKENS 800

  LIMIT TOKENS 1500
)
INFER USING CONTEXT broken
  GOAL "What is the return protection period for Priya's Platinum Card?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: FILTER BY ver="v2025" → only current policy → κ=0

CONTEXT fixed AS (
  WITH SYSTEM "You are a billing support agent."

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 800
  FILTER BY ver = "v2025"

  LIMIT TOKENS 1500
)
INFER USING CONTEXT fixed
  GOAL "What is the return protection period for Priya's Platinum Card?"
  EXPLAIN""",
  },
  {
    "id":"p4","idx":3,"icon":"📉","metric":"ρ=0.14","label":"WARN",
    "title":"Low Coverage",
    "sub":"Threshold too strict — content silently excluded",
    "col":"#7c3aed","bg":"#f5f3ff","bdr":"#c4b5fd",
    "what":"WHERE similarity > 0.30 is above most documents' scores. Only 1 of 7 benefits docs is admitted (ρ=0.14). δ=0 (benefits contributes 1 chunk). Only ρ fires — a silent quality issue.",
    "fix":"Lower threshold to 0.05–0.15. If ρ < 0.3, the threshold is too strict. Run EXPLAIN after every schema change.",
    "docs": BENEFITS,
    "sources":["amex.benefits"],
    "doc_store_keys":["amex.benefits"],
    "query":"platinum card offers airline lounge credit Priya",
    "goal":"What Amex benefits does the Platinum Card offer?",
    "system":"You are an Amex concierge. List all relevant benefits.",
    "b_gl":1500,"b_lim":{"amex.benefits":800},"b_thr":0.30,"b_filter":None,
    "f_gl":1500,"f_lim":{"amex.benefits":800},"f_thr":0.05,"f_filter":None,
    "broken_cql":"""\
-- ❌ ISOLATED ρ=0.14: only coverage fires (δ=0, κ=0)
-- threshold 0.30 is above most document similarity scores
-- Only 1 of 7 docs admitted — 6 relevant docs SILENTLY excluded
-- LLM gives incomplete answer. Status still PASS!

CONTEXT broken AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.30
    TOP 10
    LIMIT TOKENS 800

  LIMIT TOKENS 1500
)
INFER USING CONTEXT broken
  GOAL "What Amex benefits does the Platinum Card offer?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: lower threshold → more docs admitted → ρ rises

CONTEXT fixed AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7
    LIMIT TOKENS 800

  LIMIT TOKENS 1500
)
INFER USING CONTEXT fixed
  GOAL "What Amex benefits does the Platinum Card offer?"
  EXPLAIN""",
  },
  {
    "id":"p5","idx":4,"icon":"⚠️","metric":"τ=0.97","label":"WARN",
    "title":"Budget Exhaustion",
    "sub":"Context window nearly full — δ=0 κ=0",
    "col":"#d97706","bg":"#fffbeb","bdr":"#fcd34d",
    "what":f"All 7 benefits docs fill {_P5_BUDGET-5} of {_P5_BUDGET} tokens (τ=0.97). δ=0 (benefits contributes). κ=0 (no contradictions). Only τ fires — next source change triggers immediate starvation.",
    "fix":f"Increase LIMIT TOKENS to 2000+. Per-source limits prevent any one source from monopolising the budget.",
    "docs": BENEFITS,
    "sources":["amex.benefits"],
    "doc_store_keys":["amex.benefits"],
    "query":"platinum card offers airline lounge credit Priya",
    "goal":"What benefits does the Platinum Card offer?",
    "system":"You are an Amex concierge.",
    "b_gl":_P5_BUDGET,"b_lim":{"amex.benefits":_P5_BUDGET},"b_thr":0.05,"b_filter":None,
    "f_gl":2000,"f_lim":{"amex.benefits":500},"f_thr":0.05,"f_filter":None,
    "broken_cql":f"""\
-- ❌ ISOLATED τ=0.97: only budget exhaustion fires (δ=0, κ=0)
-- All 7 benefits docs consume {_P5_BUDGET-5} of {_P5_BUDGET} tokens
-- τ=0.97 → WARN. One more source → immediate starvation.

CONTEXT broken AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7
    LIMIT TOKENS {_P5_BUDGET}

  LIMIT TOKENS {_P5_BUDGET}
)
INFER USING CONTEXT broken
  GOAL "What benefits does the Platinum Card offer?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: larger budget → τ drops to safe range

CONTEXT fixed AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7
    LIMIT TOKENS 500

  LIMIT TOKENS 2000
)
INFER USING CONTEXT fixed
  GOAL "What benefits does the Platinum Card offer?"
  EXPLAIN""",
  },
  # ── MIXED: multiple metrics fire ─────────────────────────────────────────────
  {
    "id":"mx1","idx":5,"icon":"🟠","metric":"δ+κ","label":"CRITICAL",
    "title":"Mix 1: Starvation + Contradiction",
    "sub":"Two metrics fire: δ=1 and κ=0.67",
    "col":"#ea580c","bg":"#fff7ed","bdr":"#fdba74",
    "what":"budget=95: all 4 policy docs admitted (κ=0.67) but customer gets zero tokens (δ=1). Both starvation AND contradiction fire together. Real production scenario.",
    "fix":"Two independent fixes: (1) LIMIT TOKENS per RETRIEVE for δ. (2) FILTER BY ver='v2025' for κ. Each fix addresses one metric independently.",
    "docs": POLICY_MIXED+CUSTOMER,
    "sources":["amex.policy","amex.customer"],
    "doc_store_keys":["amex.policy","amex.customer"],
    "query":"purchase return protection days policy",
    "goal":"What return protection does Priya have and how does it apply to her account?",
    "system":"You are a billing support agent for Priya.",
    "b_gl":95,  "b_lim":{},             "b_thr":0.05,"b_filter":None,
    "f_gl":1500,"f_lim":{"amex.policy":500,"amex.customer":200},"f_thr":0.0,"f_filter":{"ver":"v2025"},
    "broken_cql":"""\
-- ❌ MIXED δ+κ: two metrics fire simultaneously
-- budget=95: policy (88tok) fills it → customer STARVED (δ=1)
-- old+new policy loaded → contradictions (κ=0.67)

CONTEXT broken AS (
  WITH SYSTEM "You are a billing support agent."

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 8

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4

  LIMIT TOKENS 95
)
INFER USING CONTEXT broken
  GOAL "What return protection does Priya have?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ TWO FIXES needed — one per metric:
-- Fix δ: LIMIT TOKENS per source + thr=0 (customer has sim=0 on policy query)
-- Fix κ: FILTER BY ver="v2025"

CONTEXT fixed AS (
  WITH SYSTEM "You are a billing support agent for Priya."

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 500
  FILTER BY ver = "v2025"

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 200

  LIMIT TOKENS 1500
)
INFER USING CONTEXT fixed
  GOAL "What return protection does Priya have?"
  EXPLAIN""",
  },
  {
    "id":"mx2","idx":6,"icon":"💀","metric":"δ+κ+ρ","label":"CRITICAL",
    "title":"Mix 2: The Worst Case",
    "sub":"Three metrics fire: δ=1, κ=1.0, ρ=0.25",
    "col":"#94a3b8","bg":"#f8fafc","bdr":"#cbd5e1",
    "what":"budget=50: only 2 policy docs admitted (ρ=0.25) but one old + one new → κ=1.0. Customer starved (δ=1). Status CRITICAL. Three independent problems firing at once — untested pipeline.",
    "fix":"Three independent fixes: (1) LIMIT TOKENS per RETRIEVE for δ. (2) FILTER BY ver='v2025' for κ. (3) Lower threshold or increase budget for ρ.",
    "docs": POLICY_MIXED+CUSTOMER,
    "sources":["amex.policy","amex.customer"],
    "doc_store_keys":["amex.policy","amex.customer"],
    "query":"purchase return protection days policy",
    "goal":"What return protection does Priya have and how does it apply to her account?",
    "system":"You are a billing support agent for Priya.",
    "b_gl":50,  "b_lim":{},             "b_thr":0.05,"b_filter":None,
    "f_gl":1500,"f_lim":{"amex.policy":500,"amex.customer":200},"f_thr":0.0,"f_filter":{"ver":"v2025"},
    "broken_cql":"""\
-- ❌ WORST CASE: three metrics fire simultaneously
-- budget=50: only 2 policy docs fit → ρ=0.25 (low coverage)
-- 1 old + 1 new policy admitted → κ=1.0 (contradictions)
-- customer gets zero → δ=1 (starvation)
-- Status: CRITICAL. No error. HTTP 200.

CONTEXT broken AS (
  WITH SYSTEM "You are a billing agent."

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 8

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4

  LIMIT TOKENS 50
)
INFER USING CONTEXT broken
  GOAL "What return protection does Priya have?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ THREE FIXES — each targets one metric:
-- (1) LIMIT TOKENS per source + thr=0 → fixes δ
-- (2) FILTER BY ver="v2025"           → fixes κ
-- (3) Larger budget                   → fixes ρ

CONTEXT fixed AS (
  WITH SYSTEM "You are a billing support agent for Priya."

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 500
  FILTER BY ver = "v2025"

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 200

  LIMIT TOKENS 1500
)
INFER USING CONTEXT fixed
  GOAL "What return protection does Priya have?"
  EXPLAIN""",
  },
  {
    "id":"p6","idx":7,"icon":"👻","metric":"LITM","label":"WARN",
    "title":"Lost in the Middle",
    "sub":"Critical chunks in dead zone — LLM ignores them",
    "col":"#0369a1","bg":"#f0f9ff","bdr":"#7dd3fc",
    "what":"16 benefits chunks loaded. Positions 5-10 fall into the attention dead zone (A < 0.45). LLM focus drops 40% for middle chunks — TSA PreCheck, hotel credits, extended warranty silently ignored even though they are in context.",
    "fix":"Use TOP 5 to load only the most relevant chunks. All 5 chunks stay in high-attention positions (0-4). Inspector fires P5_LITM per dead-zone chunk.",
    "docs": BENEFITS_EXTENDED,
    "sources":["amex.benefits"],
    "doc_store_keys":["amex.benefits"],
    "query":"platinum card airline lounge credit benefit Priya",
    "goal":"What are all the key benefits of the Platinum Card?",
    "system":"You are an Amex concierge. List all key benefits.",
    "b_gl":2000,"b_lim":{"amex.benefits":2000},"b_thr":0.05,"b_filter":None,
    "f_gl":2000,"f_lim":{"amex.benefits":500},"f_thr":0.05,"f_filter":None,
    "broken_cql":"""\
-- ❌ LITM: 16 chunks loaded — positions 5-10 in dead zone
-- A(5,16)=0.45 A(6,16)=0.43 A(7,16)=0.42 — below 0.45 threshold
-- LLM answers with first 5 benefits only. Ignores TSA, hotel, warranty.

CONTEXT broken AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 16
    LIMIT TOKENS 2000

  LIMIT TOKENS 2000
)
INFER USING CONTEXT broken
  GOAL "What are all the key benefits of the Platinum Card?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: TOP 5 — all chunks in high-attention positions 0-4

CONTEXT fixed AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 500

  LIMIT TOKENS 2000
)
INFER USING CONTEXT fixed
  GOAL "What are all the key benefits of the Platinum Card?"
  EXPLAIN""",
  },
  {
    "id":"p7","idx":8,"icon":"🔁","metric":"η=0.38","label":"WARN",
    "title":"Semantic Redundancy",
    "sub":"Near-duplicate chunks waste token budget",
    "col":"#059669","bg":"#f0fdf4","bdr":"#bbf7d0",
    "what":"5 of 7 chunks are paraphrases of the same airline credit benefit (η=0.38). 8 of 21 chunk pairs are near-duplicates. Token budget wasted — LLM reads the same fact 5 times instead of learning about lounge access, hotel status, or rewards.",
    "fix":"Deduplicate at ingestion using embedding similarity. Or lower TOP N to 3 to force the retriever to pick more diverse chunks.",
    "docs": REDUNDANT_BENEFITS,
    "sources":["amex.benefits"],
    "doc_store_keys":["amex.benefits"],
    "query":"platinum card airline lounge credit benefit",
    "goal":"What are the key Platinum Card benefits?",
    "system":"You are an Amex concierge. List all unique benefits.",
    "b_gl":1500,"b_lim":{"amex.benefits":800},"b_thr":0.05,"b_filter":None,
    "f_gl":1500,"f_lim":{"amex.benefits":300},"f_thr":0.05,"f_filter":None,
    "broken_cql":"""\
-- ❌ SEMANTIC REDUNDANCY: η=0.38
-- 5 of 7 chunks are paraphrases of airline credit
-- 120 tokens wasted repeating one fact
-- LLM never learns about lounge, hotel status, rewards

CONTEXT broken AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7
    LIMIT TOKENS 800

  LIMIT TOKENS 1500
)
INFER USING CONTEXT broken
  GOAL "What are the key Platinum Card benefits?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: TOP 3 forces diverse chunk selection
-- η drops as near-duplicates no longer all admitted

CONTEXT fixed AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 300

  LIMIT TOKENS 1500
)
INFER USING CONTEXT fixed
  GOAL "What are the key Platinum Card benefits?"
  EXPLAIN""",
  },
  {
    "id":"p8","idx":9,"icon":"⏰","metric":"σ=0.67","label":"WARN","max_age_days":30,
    "title":"Staleness",
    "sub":"Old documents — κ=0 but answer still wrong",
    "col":"#92400e","bg":"#fffbeb","bdr":"#fcd34d",
    "what":"2 of 3 policy chunks are 380-400 days old (v2024: 30-day return). Current policy (v2025: 90-day) exists but is outvoted. σ=0.67 (2/3 chunks stale). LLM tells customer '30 days' — the wrong answer today.",
    "fix":"Tag documents with age_days and ver metadata. Use FILTER BY ver='v2025' or ContextInspector(max_age_days=30) to detect stale content before calling the LLM.",
    "docs": STALE_POLICY,
    "sources":["amex.policy"],
    "doc_store_keys":["amex.policy"],
    "query":"return protection days policy coverage",
    "goal":"How many days does Platinum return protection cover?",
    "system":"You are a billing agent. State the exact number of days.",
    "b_gl":1500,"b_lim":{"amex.policy":800},"b_thr":0.05,"b_filter":None,
    "f_gl":1500,"f_lim":{"amex.policy":800},"f_thr":0.05,"f_filter":{"ver":"v2025"},
    "broken_cql":"""\
-- ❌ STALENESS: σ=0.67, κ=0.33
-- 2/3 chunks are v2024 (380-400 days old, age_days>30 threshold)
-- LLM says "30 days" — old policy, wrong answer today
-- Status PASS. HTTP 200. No error raised.

CONTEXT broken AS (
  WITH SYSTEM "You are a billing agent."

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 800

  LIMIT TOKENS 1500
)
INFER USING CONTEXT broken
  GOAL "How many days does Platinum return protection cover?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: FILTER BY ver="v2025" excludes all stale docs
-- σ=0.0, κ=0.0, correct answer: 90 days

CONTEXT fixed AS (
  WITH SYSTEM "You are a billing agent."

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 800
  FILTER BY ver = "v2025"

  LIMIT TOKENS 1500
)
INFER USING CONTEXT fixed
  GOAL "How many days does Platinum return protection cover?"
  EXPLAIN""",
  },
]


# ════════════════════════════════════════════════════════════════════════════════
# CI TEST DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════
CI_TESTS = [
    {
        "name": "test_customer_profile_never_starved",
        "desc": "amex.customer must always contribute at least one chunk",
        "assertion": "assert report.delta == 0, f'STARVATION: {report.sources_starved}'",
        "check": lambda r, sm: r.delta == 0,
        "fail_msg": lambda r, sm: f"amex.customer starved — {r.sources_starved}",
        "fix": "Add LIMIT TOKENS N to each RETRIEVE block",
    },
    {
        "name": "test_offers_context_always_present",
        "desc": "amex.offers must contribute chunks — personalisation requires it",
        "assertion": "assert 'amex.offers' not in report.sources_starved",
        "check": lambda r, sm: "amex.offers" not in r.sources_starved,
        "fail_msg": lambda r, sm: "amex.offers contributed 0 chunks — offer recommendations impossible",
        "fix": "Add LIMIT TOKENS N to amex.offers RETRIEVE block",
    },
    {
        "name": "test_budget_utilisation_safe",
        "desc": "Token budget utilisation must stay below 90% (safety headroom)",
        "assertion": "assert report.tau <= 0.90, f'Budget {report.tau:.0%} — starvation risk'",
        "check": lambda r, sm: r.tau <= 0.90,
        "fail_msg": lambda r, sm: f"τ={r.tau:.3f} — budget {r.tau:.0%} consumed. One source change triggers starvation.",
        "fix": "Increase global LIMIT TOKENS to give all sources breathing room",
    },
    {
        "name": "test_no_policy_contradictions",
        "desc": "Contradiction density κ must be below 0.30",
        "assertion": "assert report.kappa < 0.30, f'Contradictions κ={report.kappa}'",
        "check": lambda r, sm: r.kappa < 0.30,
        "fail_msg": lambda r, sm: f"κ={r.kappa:.3f} — conflicting facts in context. LLM will synthesise wrong answers.",
        "fix": "Add FILTER BY version = 'current' to exclude outdated documents",
    },
    {
        "name": "test_minimum_coverage",
        "desc": "Coverage ρ must be >= 0.20 (threshold not too strict)",
        "assertion": "assert report.rho >= 0.20, f'Coverage ρ={report.rho}'",
        "check": lambda r, sm: r.rho >= 0.20,
        "fail_msg": lambda r, sm: f"ρ={r.rho:.3f} — threshold too strict. Most relevant content excluded.",
        "fix": "Lower WHERE similarity threshold to 0.05–0.15",
    },
    {
        "name": "test_inspector_passes",
        "desc": "Overall Inspector status must be PASS",
        "assertion": "assert report.status == 'PASS', f'STATUS: {report.status}'",
        "check": lambda r, sm: r.status == "PASS",
        "fail_msg": lambda r, sm: f"Inspector status: {r.status}. Q=(ρ={r.rho} τ={r.tau:.2f} δ={r.delta} κ={r.kappa})",
        "fix": "Fix all warnings above — each maps to a specific CQL change",
    },
]

# ════════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,600;0,700&family=Syne:wght@600;700;800&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

:root {
  --bg:    #f8fafc;
  --panel: #ffffff;
  --p2:    #f1f5f9;
  --bdr:   #e2e8f0;
  --b2:    #cbd5e1;
  --txt:   #1e293b;
  --mid:   #64748b;
  --dim:   #e2e8f0;
  --blue:  #2563eb;
  --green: #16a34a;
  --red:   #dc2626;
  --amber: #d97706;
  --purple:#7c3aed;
  --code:  #f8fafc;
}

.stApp,.main,section.main { background: var(--bg) !important; }
html,body { background: var(--bg); }

h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; color:#0f172a !important; }
p,li,label,div,span { font-family:'JetBrains Mono',monospace !important; font-size:12px; color:var(--txt); }

[data-testid="collapsedControl"] { display:none; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#fff; border-bottom:2px solid var(--bdr); padding:0 4px; gap:0; }
.stTabs [data-baseweb="tab"] { font-family:'Syne',sans-serif !important; font-size:12.5px !important; font-weight:600 !important; color:#94a3b8 !important; padding:11px 18px !important; border-bottom:2px solid transparent !important; }
.stTabs [data-baseweb="tab"]:hover { color:#475569 !important; }
.stTabs [aria-selected="true"] { color:#1e293b !important; font-weight:800 !important; border-bottom:2px solid var(--blue) !important; background:transparent !important; }
.stTabs [data-baseweb="tab"] p { color:inherit !important; font-size:inherit !important; font-family:inherit !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:0 !important; background:transparent; }

/* Buttons */
.stButton>button { background:var(--blue) !important; color:#fff !important; border:none !important; border-radius:6px !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:11.5px !important; padding:7px 16px !important; }
.stButton>button:hover { filter:brightness(1.1) !important; }

/* Inputs */
.stTextArea textarea { background:#f8fafc !important; color:#1e293b !important; border:1px solid var(--b2) !important; border-radius:6px !important; font-family:'JetBrains Mono',monospace !important; font-size:11.5px !important; line-height:1.8 !important; }
.stTextArea textarea:focus { border-color:var(--blue) !important; box-shadow:0 0 0 2px #2563eb20 !important; }
.stTextInput input,.stSelectbox>div>div { background:#fff !important; color:#1e293b !important; border:1px solid var(--b2) !important; border-radius:6px !important; font-family:'JetBrains Mono',monospace !important; font-size:11px !important; }

/* Metrics */
[data-testid="stMetric"] { background:#fff !important; border:1px solid var(--bdr) !important; border-radius:8px !important; padding:10px 14px !important; }
[data-testid="stMetricLabel"] { color:var(--mid) !important; font-size:9px !important; text-transform:uppercase; letter-spacing:1.5px; font-family:'Syne',sans-serif !important; }
[data-testid="stMetricValue"] { color:#0f172a !important; font-family:'JetBrains Mono',monospace !important; font-size:20px !important; }

/* Expanders */
.stExpander { background:#fff !important; border:1px solid var(--bdr) !important; border-radius:8px !important; overflow:hidden; }
.stExpander summary svg { display:none !important; }
.stExpander [data-testid="stExpanderToggleIcon"] { display:none !important; }
.stExpander summary::before { content:"▶"; color:#94a3b8; font-size:9px; margin-right:8px; transition:transform .2s; }
details[open] .stExpander summary::before { transform:rotate(90deg); }
.stExpander summary { color:#1e293b !important; font-size:12px !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; padding:10px 14px !important; background:#fff !important; list-style:none !important; cursor:pointer; display:flex !important; align-items:center !important; }
.stExpander summary::-webkit-details-marker { display:none; }
.stExpander summary:hover { background:var(--p2) !important; }
.stExpander summary p,.stExpander summary span { color:#1e293b !important; font-size:12px !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; margin:0 !important; }
.stExpander [data-testid="stExpanderDetails"] { background:#fff !important; padding:10px 14px !important; }

/* Divider */
hr { border-color:var(--bdr) !important; }
.stCheckbox label,.stRadio label { font-size:11px !important; color:var(--mid) !important; }
#MainMenu,footer,.stDeployButton { display:none !important; }

/* Custom components */
.sl { font-size:8px; letter-spacing:2.5px; text-transform:uppercase; color:#94a3b8; font-family:'Syne',sans-serif; font-weight:700; margin-bottom:6px; display:block; }
.qbar { height:4px; background:#e2e8f0; border-radius:2px; overflow:hidden; margin-top:2px; }
.qfill { height:100%; border-radius:2px; transition:width .4s ease; }
.chip { background:#fff; border:1px solid var(--bdr); border-radius:5px; padding:5px 10px; margin:3px 0; font-size:10px; }
.banner { border-radius:10px; padding:12px 16px; margin:6px 0; }
.ans { background:#f8fafc; border:1px solid var(--bdr); border-radius:8px; padding:14px; font-size:12.5px; line-height:1.9; color:#1e293b; }
.log { background:#f8fafc; border:1px solid var(--bdr); border-radius:6px; padding:8px 12px; font-family:'JetBrains Mono',monospace; font-size:10px; line-height:1.9; max-height:110px; overflow-y:auto; margin:5px 0; }
.tag { display:inline-block; padding:1px 7px; border-radius:3px; font-size:9px; font-weight:700; margin:1px; }
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════════
if "sel"     not in st.session_state: st.session_state.sel    = 0
if "cql"     not in st.session_state: st.session_state.cql   = PROBLEMS[0]["broken_cql"]
if "result"  not in st.session_state: st.session_state.result = None
if "logs"    not in st.session_state: st.session_state.logs   = []
if "extra"   not in st.session_state: st.session_state.extra  = []
if "ci_res"  not in st.session_state: st.session_state.ci_res = None
if "pm_res"  not in st.session_state: st.session_state.pm_res = None

P = PROBLEMS[st.session_state.sel]

# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════
hc1, hc2, hc3 = st.columns([3, 5, 2])
with hc1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:5px 0">
      <div style="width:30px;height:30px;background:linear-gradient(135deg,#1e3a8a,#3b82f6);border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:15px;color:white">⬡</div>
      <div>
        <div style="font-family:Syne,sans-serif;font-weight:800;font-size:16px;color:#e2e8f0;line-height:1">OpenCQL</div>
        <div style="font-size:8px;color:#3b82f6;font-weight:700;letter-spacing:2px">CONTEXT QUALITY DEMO</div>
      </div>
    </div>""", unsafe_allow_html=True)
with hc2:
    r = st.session_state.result
    if r and r.get("report"):
        rep = r["report"]
        sc  = {"PASS":"#22c55e","CRITICAL":"#ef4444","WARN":"#f59e0b"}.get(rep.status,"#64748b")
        bg  = {"PASS":"#f0fdf4","CRITICAL":"#fef2f2","WARN":"#1a1100"}.get(rep.status,"#f8fafc")
        em  = {"PASS":"✅","CRITICAL":"🚨","WARN":"⚠️"}.get(rep.status,"")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 14px;background:{bg};border:1px solid {sc}44;border-radius:8px;margin:3px 0">
          <span style="font-size:15px">{em}</span>
          <span style="font-weight:800;color:{sc};font-size:13px;font-family:Syne,sans-serif">{rep.status}</span>
          <span style="color:#64748b;font-size:9px">ρ={rep.rho} · τ={rep.tau:.2f} · δ={rep.delta} · κ={rep.kappa}</span>
          <span style="color:#94a3b8;font-size:9px">{rep.tokens_used}/{rep.token_budget} tok</span>
        </div>""", unsafe_allow_html=True)
with hc3:
    provider = st.selectbox("", [
        "Groq — Llama 3.3 70B (Free)",
        "Gemini 2.5 Flash (Free)",
        "Claude Haiku (Paid)",
    ], key="provider_sel", label_visibility="collapsed")
    def _get_key(env_name):
        try: return st.secrets.get(env_name, os.getenv(env_name, ""))
        except: return os.getenv(env_name, "")
    if "Groq"   in provider: default_key = _get_key("GROQ_API_KEY");       placeholder = "gsk_... (console.groq.com)"
    elif "Gemini" in provider: default_key = _get_key("GEMINI_API_KEY");   placeholder = "AIza... (aistudio.google.com)"
    else:                      default_key = _get_key("ANTHROPIC_API_KEY"); placeholder = "sk-ant-..."
    api_key = st.text_input("", value=default_key, type="password",
                             placeholder=placeholder, label_visibility="collapsed")
    c = "#22c55e" if api_key else "#64748b"
    model_name = provider.split(" (")[0].replace("Groq — ","")
    lbl = f"● {model_name} connected" if api_key else "○ No key — Inspector only"
    st.markdown(f'<div style="color:{c};font-size:9px">{lbl}</div>', unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
t_inspect, t_ci, t_pm, t_integrate, t_guide = st.tabs([
    "Inspect — Developer / QA",
    "CI Tests — GitHub Actions",
    "PM View — Business Impact",
    "Integrate — Add to Pipeline",
    "Quality Guide",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: INSPECT
with t_inspect:
    c_left, c_mid, c_right = st.columns([22, 40, 32], gap="medium")

    # ── LEFT: Problem Gallery ─────────────────────────────────────────────────
    with c_left:
        st.markdown('<span class="sl">Context Quality Problems</span>', unsafe_allow_html=True)
        for prob in PROBLEMS:
            i = prob["idx"]
            if st.button(f"{prob['icon']}  {prob['title']}", key=f"p_{i}", use_container_width=True):
                st.session_state.sel           = i
                st.session_state.cql           = prob["broken_cql"]
                st.session_state["cql_ta"]     = prob["broken_cql"]   # ← force widget update
                st.session_state.result        = None
                st.session_state.logs          = []
                st.rerun()
            mc = prob["col"]
            st.markdown(f'<div style="margin:-7px 0 5px 2px;font-size:9px;color:{mc}">{prob["metric"]} · {prob["sub"]}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-weight:800;font-size:11px;color:#e2e8f0;margin-bottom:5px">{P["icon"]} {P["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:{P["bg"]};border:1px solid {P["bdr"]};border-radius:7px;padding:8px 11px;font-size:10px;line-height:1.7;color:{P["col"]};margin-bottom:7px">{P["what"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:7px;padding:8px 11px;font-size:10px;line-height:1.7;color:#4ade80">{P["fix"]}</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:9px"></div>', unsafe_allow_html=True)
        ba, bb = st.columns(2)
        with ba:
            if st.button("⚡ Broken", key="lb", use_container_width=True):
                st.session_state.cql       = P["broken_cql"]
                st.session_state["cql_ta"] = P["broken_cql"]
                st.session_state.result    = None
                st.session_state.logs      = []
                st.rerun()
        with bb:
            if st.button("✅ Fixed", key="lf", use_container_width=True):
                st.session_state.cql       = P["fixed_cql"]
                st.session_state["cql_ta"] = P["fixed_cql"]
                st.session_state.result    = None
                st.session_state.logs      = []
                st.rerun()

    # ── MID: CQL Editor ───────────────────────────────────────────────────────
    with c_mid:
        st.markdown('<span class="sl">CQL Editor</span>', unsafe_allow_html=True)

        cql_val = st.text_area("cql", value=st.session_state.cql, height=380,
                                label_visibility="collapsed", key="cql_ta")
        st.session_state.cql = cql_val

        mc1, mc2, mc3 = st.columns([3, 3, 4])
        with mc1: run_btn = st.button("▶  Run", key="run", use_container_width=True)
        with mc2: inspect_only = st.checkbox("Inspector only", key="insonly")
        with mc3: goal_ov = st.text_input("", key="gov", placeholder="Override GOAL (optional)", label_visibility="collapsed")

        if st.session_state.logs:
            lh = '<div class="log">'
            for k, m in st.session_state.logs:
                c = {"ok":"#22c55e","warn":"#f59e0b","err":"#ef4444"}.get(k,"#64748b")
                lh += f'<div style="color:{c}">{m}</div>'
            lh += "</div>"
            st.markdown(lh, unsafe_allow_html=True)

        st.divider()

        # ── CQL Reference ─────────────────────────────────────────────────────
        with st.expander("CQL Reference"):
            st.markdown("""<pre style="font-family:'JetBrains Mono',monospace;font-size:10.5px;line-height:2;background:transparent;color:#c0cce0;margin:0">
<b style="color:#7ab4f5">CONTEXT</b> name <b style="color:#7ab4f5">AS</b> (
  <b style="color:#7ab4f5">WITH SYSTEM</b> <span style="color:#7ec8e3">"You are..."</span>
  <b style="color:#7ab4f5">RETRIEVE FROM</b> source
    <b style="color:#7ab4f5">WHERE</b> similarity > <span style="color:#a8c97a">0.05</span>
    <b style="color:#7ab4f5">TOP</b> <span style="color:#a8c97a">5</span>
    <b style="color:#7ab4f5">LIMIT TOKENS</b> <span style="color:#a8c97a">300</span>   ← prevents δ
  <b style="color:#7ab4f5">FILTER BY</b> field = <span style="color:#7ec8e3">"v2025"</span>  ← prevents κ
  <b style="color:#7ab4f5">JOIN</b> source2 <b style="color:#7ab4f5">SEMANTIC ON</b> cat ← cross-source join
  <b style="color:#7ab4f5">INJECT HISTORY LAST</b> <span style="color:#a8c97a">3</span> <b style="color:#7ab4f5">TURNS</b>    ← conversation
  <b style="color:#7ab4f5">PARTITION BY</b> cat (<span style="color:#7ec8e3">"a","b"</span>)  ← MapReduce
  <b style="color:#7ab4f5">LIMIT TOKENS</b> <span style="color:#a8c97a">2000</span>          ← global cap
)
<b style="color:#7ab4f5">INFER</b>
  <b style="color:#7ab4f5">USING CONTEXT</b> name
  <b style="color:#7ab4f5">GOAL</b> <span style="color:#7ec8e3">"..."</span>
  <b style="color:#7ab4f5">AGGREGATE BY</b> <span style="color:#7ec8e3">"synthesis"</span>  ← merge partitions
  <b style="color:#7ab4f5">EXPLAIN</b>                  ← Inspector fires pre-call</pre>""", unsafe_allow_html=True)

        # ── Full DSL Examples ──────────────────────────────────────────────────
        with st.expander("Full DSL Examples — Joins · MapReduce · Aggregate"):
            dx1, dx2, dx3, dx4 = st.tabs(["FILTER + JOIN", "PARTITION BY", "AGGREGATE Strategies", "Complete Example"])

            with dx1:
                st.code("""\
-- FILTER BY: post-retrieval metadata filter
CONTEXT filtered AS (
  WITH SYSTEM "You are a billing agent."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 6
    LIMIT TOKENS 400
  FILTER BY cat = "travel"          -- only travel benefits

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 300
  FILTER BY status = "active"       -- only active offers

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 200

  LIMIT TOKENS 2000
)

-- JOIN SEMANTIC: retrieval conditioned on another source
CONTEXT joined AS (
  WITH SYSTEM "You are Priya's concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 400

  JOIN amex.offers SEMANTIC ON cat   -- match offers to benefit category
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 250

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 200

  LIMIT TOKENS 2000
)

INFER USING CONTEXT joined
  GOAL "What travel offers match Priya's Platinum benefits?"
  EXPLAIN""", language="sql")

            with dx2:
                st.code("""\
-- PARTITION BY: MapReduce over categories
-- Runs one INFER per partition value in parallel
-- Results merged by AGGREGATE BY strategy

CONTEXT partitioned AS (
  WITH SYSTEM "You are an Amex product analyst."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 6
    LIMIT TOKENS 400

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 350

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 200

  -- Creates 4 parallel INFER calls:
  -- one each for travel, dining, lifestyle, rewards
  PARTITION BY cat ("travel", "dining", "lifestyle", "rewards")

  LIMIT TOKENS 2000
)

INFER USING CONTEXT partitioned
  GOAL "What are the key benefits and offers in this category?"
  AGGREGATE BY "synthesis"
  EXPLAIN

-- Output: one synthesised answer across all 4 partitions
-- Much richer than a single flat INFER
-- Inspector checks each partition independently: Q per shard""", language="sql")

            with dx3:
                st.code("""\
-- AGGREGATE BY strategies (used with PARTITION BY)

-- synthesis (default)
-- LLM merges all partition answers into one coherent response
-- Best for: comprehensive summaries, combining insights
INFER USING CONTEXT partitioned
  GOAL "What benefits exist across all categories?"
  AGGREGATE BY "synthesis"

-- vote
-- Pick the most common answer across partitions
-- Best for: yes/no questions, classification, sentiment
INFER USING CONTEXT partitioned
  GOAL "Is the Platinum Card worth its annual fee?"
  AGGREGATE BY "vote"

-- concat
-- Join all partition answers sequentially (labeled by partition)
-- Best for: structured reports, category-by-category breakdowns
INFER USING CONTEXT partitioned
  GOAL "List top benefits in this category."
  AGGREGATE BY "concat"

-- first
-- Return only the first partition's answer
-- Best for: debugging, spot-checks, quick validation
INFER USING CONTEXT partitioned
  GOAL "Give one example benefit."
  AGGREGATE BY "first"

-- reduce
-- Progressive merging: answer(n) + answer(n+1) -> merged(n+1)
-- Best for: long lists, large partition counts (5+)
INFER USING CONTEXT partitioned
  GOAL "Build a comprehensive benefits guide."
  AGGREGATE BY "reduce\"""", language="sql")

            with dx4:
                st.code("""\
-- COMPLETE EXAMPLE: all DSL features combined
-- FILTER + JOIN + INJECT HISTORY + PARTITION + AGGREGATE + EXPLAIN

CONTEXT full_pipeline AS (
  WITH SYSTEM
    "You are Priya's expert Amex concierge. Give specific,
    personalised advice using her account details and history."

  -- Travel benefits only
  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 350
  FILTER BY cat = "travel"

  -- Active offers semantically joined to benefit categories
  JOIN amex.offers SEMANTIC ON cat
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 280
  FILTER BY status = "active"

  -- Customer profile — always guaranteed budget
  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 250

  -- Current month policy — no contradictions
  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 2
    LIMIT TOKENS 150
  FILTER BY ver = "v2025"

  -- Last 2 turns of conversation
  INJECT HISTORY LAST 2 TURNS LIMIT TOKENS 150

  -- MapReduce: analyse by spending category
  PARTITION BY cat ("travel", "dining", "rewards", "lifestyle")

  LIMIT TOKENS 2000
)

INFER
  USING CONTEXT full_pipeline
  GOAL "What should Priya prioritise this month for max value?"
  AGGREGATE BY "synthesis"
  EXPLAIN

-- Pre-call Inspector output:
-- Q = (rho=0.82  tau=0.71  delta=0  kappa=0.0)
-- STATUS: PASS
-- benefits: 5 chunks · 147 tok
-- offers:   4 chunks · 112 tok
-- customer: 4 chunks · 107 tok
-- policy:   2 chunks ·  46 tok""", language="sql")

        # ── Source Manager ─────────────────────────────────────────────────────
        with st.expander("Sources"):
            all_d = ALL_DOCS + st.session_state.extra
            by_s = defaultdict(list)
            for d in all_d: by_s[d["source"]].append(d)
            for src, docs in by_s.items():
                st.markdown(f'<div style="font-size:10px;font-weight:700;color:#60a5fa;margin:5px 0 3px">{src} <span style="color:#64748b;font-weight:400">· {len(docs)} docs</span></div>', unsafe_allow_html=True)
                for d in docs:
                    meta = {k:v for k,v in d.items() if k not in ("source","text")}
                    tags = "".join(f'<span class="tag" style="background:#0f1e3a;color:#60a5fa">{k}:{v}</span>' for k,v in meta.items())
                    st.markdown(f'<div class="chip">{d["text"][:65]}…&nbsp;{tags}</div>', unsafe_allow_html=True)
            st.divider()
            na, nb = st.columns(2)
            with na: new_src = st.text_input("Source", key="asrc", placeholder="domain.name")
            with nb: new_meta = st.text_input("Metadata", key="amet", placeholder="ver:v2025")
            new_txt = st.text_area("Text", key="atxt", height=55, placeholder="Document text…")
            if st.button("+ Add", key="addbtn"):
                if new_txt.strip() and new_src.strip():
                    md = {}
                    for pair in new_meta.split(","):
                        pts = pair.strip().split(":")
                        if len(pts)==2: md[pts[0].strip()]=pts[1].strip()
                    st.session_state.extra.append({"source":new_src.strip(),"text":new_txt.strip(),**md})
                    st.success("Added"); st.rerun()

    # ── RIGHT: Inspector + Answer + Math ──────────────────────────────────────
    with c_right:
        ri1, ri2, ri3 = st.tabs(["Inspector", "Answer", "Math"])

        # ── Inspector ─────────────────────────────────────────────────────────
        with ri1:
            r = st.session_state.result
            if not r:
                st.markdown('<div style="text-align:center;padding:50px 10px;color:#1a2535"><div style="font-size:34px;opacity:.2;margin-bottom:10px">◈</div><div style="font-size:11px;color:#2a3550">Run a query to see Inspector results</div></div>', unsafe_allow_html=True)
            else:
                rep = r.get("report")
                if rep:
                    sc={"PASS":"#22c55e","CRITICAL":"#ef4444","WARN":"#f59e0b"}.get(rep.status,"#64748b")
                    em={"PASS":"✅","CRITICAL":"🚨","WARN":"⚠️"}.get(rep.status,"")
                    bg={"PASS":"#f0fdf4","CRITICAL":"#fef2f2","WARN":"#fffbeb"}.get(rep.status,"#f8fafc")
                    bd={"PASS":"#16a34a","CRITICAL":"#dc2626","WARN":"#d97706"}.get(rep.status,"#94a3b8")
                    st.markdown(f"""
                    <div class="banner" style="background:{bg};border:2px solid {bd}">
                      <div style="display:flex;align-items:center;gap:10px">
                        <span style="font-size:22px">{em}</span>
                        <div>
                          <div style="font-size:17px;font-weight:800;color:{sc};font-family:Syne,sans-serif">{rep.status}</div>
                          <div style="font-size:9px;color:#64748b">{rep.tokens_used}/{rep.token_budget} tok</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    for sym,name,val,ok,cb,cg,note in [
                        ("ρ","Coverage",rep.rho,rep.rho>=0.20,"#7c3aed","#a78bfa","% retrieved → admitted"),
                        ("τ","Utilisation",rep.tau,rep.tau<0.90,"#1d4ed8","#60a5fa","tokens used / budget"),
                        ("δ","Starvation",min(rep.delta/max(len(P["sources"]),1),1),rep.delta==0,"#991b1b","#ef4444","sources with 0 chunks"),
                        ("κ","Contradiction",rep.kappa,rep.kappa<0.30,"#92400e","#f59e0b","conflicting fact pairs"),
                        ("η","Redundancy",getattr(rep,"eta",0),getattr(rep,"eta",0)<0.30,"#065f46","#10b981","near-duplicate chunk pairs"),
                        ("σ","Staleness",getattr(rep,"sigma",0),getattr(rep,"sigma",0)<0.30,"#78350f","#d97706","outdated chunks in context"),
                    ]:
                        bc=cg if ok else cb
                        av={"ρ":rep.rho,"τ":rep.tau,"δ":rep.delta,"κ":rep.kappa,"η":getattr(rep,"eta",0),"σ":getattr(rep,"sigma",0)}[sym]
                        st.markdown(f"""
                        <div style="margin:7px 0">
                          <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <div style="display:flex;align-items:baseline;gap:5px">
                              <span style="color:{bc};font-size:14px;font-family:Georgia,serif;font-weight:700">{sym}</span>
                              <span style="color:#64748b;font-size:9px">{name}</span>
                            </div>
                            <span style="color:{bc};font-size:12px;font-family:JetBrains Mono,monospace;font-weight:700">{av if sym=='δ' else f'{av:.3f}'}</span>
                          </div>
                          <div class="qbar"><div class="qfill" style="width:{min(100,val*100):.1f}%;background:{bc}"></div></div>
                          <div style="font-size:8px;color:#94a3b8;margin-top:1px">{note}</div>
                        </div>""", unsafe_allow_html=True)
                    for w in rep.warnings:
                        wt=warn_type(w); wm=warn_msg(w)
                        wc="#ef4444" if "STARV" in wt else "#f59e0b"
                        st.markdown(f'<div style="background:#fef2f2;border:1px solid {wc}33;border-radius:5px;padding:7px 10px;margin:5px 0"><span style="color:{wc};font-size:9px;font-weight:700">{wt}</span><br><span style="color:#64748b;font-size:9px">{wm}</span></div>', unsafe_allow_html=True)
                    st.divider()
                    st.markdown('<span class="sl">Source Breakdown</span>', unsafe_allow_html=True)
                    smap = r.get("smap", {})
                    _breakdown_srcs = r.get("insp_sources") or list(smap.keys()) or P["sources"]
                    for src in _breakdown_srcs:
                        cs=smap.get(src,[]); ok_s=len(cs)>0
                        sc2="#22c55e" if ok_s else "#ef4444"
                        tt=sum(tok(c["text"]) for c in cs)
                        avg=round(sum(c.get("_sc",0) for c in cs)/max(len(cs),1),3) if cs else 0
                        st.markdown(f"""
                        <div class="chip" style="border-color:{'#bbf7d0' if ok_s else '#fca5a5'}">
                          <div style="display:flex;align-items:center;gap:7px">
                            <span style="color:{sc2};font-size:13px">{"✓" if ok_s else "✗"}</span>
                            <div style="flex:1">
                              <div style="color:{sc2};font-weight:700;font-size:10px">{src}</div>
                              <div style="color:#64748b;font-size:9px">{len(cs)} chunks · {tt} tok · sim̄={avg}</div>
                            </div>
                            {"<span style='color:#ef4444;font-size:9px;font-weight:700'>STARVED</span>" if not ok_s else ""}
                          </div>
                        </div>""", unsafe_allow_html=True)
                        for c in cs[:2]:
                            st.markdown(f'<div style="padding:2px 4px 2px 24px;font-size:9px;color:#94a3b8;border-left:1px solid #0f1520;margin:1px 0">{c["text"][:72]}…</div>', unsafe_allow_html=True)
                        if len(cs)>2: st.markdown(f'<div style="padding:1px 4px 1px 24px;font-size:9px;color:#94a3b8">+{len(cs)-2} more</div>', unsafe_allow_html=True)

        # ── Answer ────────────────────────────────────────────────────────────
        with ri2:
            r=st.session_state.result
            if not r: st.markdown('<div style="text-align:center;padding:50px 10px;color:#1a2535;font-size:11px">Run a query to see the answer</div>', unsafe_allow_html=True)
            elif r.get("inspect_only"): st.info("Inspector only — uncheck to get LLM answer.")
            elif r.get("no_key"): st.markdown('<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:18px;font-size:11px;color:#64748b;text-align:center">Add API key (top-right) to see the LLM answer</div>', unsafe_allow_html=True)
            elif r.get("api_error"): st.error(r["api_error"])
            else:
                rep=r.get("report"); ans=r.get("answer","")
                if rep:
                    sc={"PASS":"#22c55e","CRITICAL":"#ef4444","WARN":"#f59e0b"}.get(rep.status,"#64748b")
                    bg={"PASS":"#f0fdf4","CRITICAL":"#fef2f2","WARN":"#fffbeb"}.get(rep.status,"#f8fafc")
                    st.markdown(f'<div style="margin-bottom:7px"><span style="background:{bg};color:{sc};border:1px solid {sc}33;padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700">{rep.status}</span> <span style="color:#94a3b8;font-size:9px">δ={rep.delta}</span></div>', unsafe_allow_html=True)
                    if rep.delta>0:
                        st.markdown(f'<div style="background:#fef2f2;border:1px solid #fca5a533;border-radius:6px;padding:9px 12px;font-size:10px;color:#f87171;margin-bottom:8px">⚠ Starvation δ={rep.delta} — missing: <b>{", ".join(rep.sources_starved)}</b><br><span style="color:#dc2626">Answer lacks customer/offer context.</span></div>', unsafe_allow_html=True)
                if ans: st.markdown(f'<div class="ans">{ans}</div>', unsafe_allow_html=True)

        # ── Math Formulation ─────────────────────────────────────────────────
        with ri3:
            pmath = PROBLEMS[st.session_state.sel]
            pid   = pmath["id"]

            st.markdown(
                f'<div style="font-family:Syne,sans-serif;font-weight:800;font-size:13px;'
                f'color:#e2e8f0;margin-bottom:8px">{pmath["icon"]} {pmath["title"]}</div>',
                unsafe_allow_html=True)

            st.markdown('<span class="sl">Plain English</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:{pmath["bg"]};border:1px solid {pmath["bdr"]};'
                f'border-radius:7px;padding:10px 13px;font-size:11px;line-height:1.8;'
                f'color:{pmath["col"]};margin-bottom:10px">{pmath["what"]}</div>',
                unsafe_allow_html=True)

            # Per-problem math
            MATH = {
              "p1": dict(
                metric="δ — Source Starvation",
                formula="δ  =  |{ i : Aᵢ = ∅ }|",
                syms=[("i","each registered source"),("Aᵢ","chunks admitted from source i"),
                      ("Aᵢ = ∅","source i contributed zero chunks"),("δ","count of starved sources")],
                calc="W=65. benefits fills 65 tok. A_customer = { }\ndelta = |{customer}| = 1  →  CRITICAL",
                thr="delta > 0  →  CRITICAL",
                fix="With per-source limits:\nbudget(customer) = 250  always\nbenefits cannot overflow into customer budget"),
              "p2": dict(
                metric="δ — Multi-Source Starvation (δ=2)",
                formula="δ  =  |{ i : Aᵢ = ∅ }|  =  2",
                syms=[("i ∈ {ben,off,cust}","3 registered sources"),("A_offers = { }","offers starved"),
                      ("A_customer = { }","customer starved"),("δ = 2","two sources, 0 chunks each")],
                calc="W=45. benefits fills 45 tok.\nA_offers={ }, A_customer={ }\ndelta = 2  →  CRITICAL",
                thr="delta >= 1  →  CRITICAL",
                fix="Sum of per-source limits < W ensures delta=0\nlimit_b + limit_o + limit_c = 900 < 2000"),
              "p3": dict(
                metric="κ — Contradiction Density",
                formula="κ  =  Σ Φ(cₐ,cᵦ)  /  C(n,2)",
                syms=[("n","total admitted chunks"),("C(n,2)","unique pairs = n*(n-1)/2"),
                      ("Φ(cₐ,cᵦ)","1 if pair contradicts on a numeric fact"),("κ","contradiction fraction")],
                calc="n=4: 2 chunks say 30-day, 2 say 14-day\nContradicting pairs = 4.  C(4,2) = 6\nkappa = 4/6 = 0.667  →  WARN (> 0.30)",
                thr="kappa > 0.30  →  WARN",
                fix="FILTER BY ver=v2025 loads only v2025 docs\nAll same version: Phi=0 for all pairs\nkappa = 0  →  PASS"),
              "p4": dict(
                metric="ρ — Coverage",
                formula="ρ  =  |A|  /  Σᵢ |Rᵢ|",
                syms=[("A","set of admitted chunks"),("Rᵢ","chunks retrieved from source i"),
                      ("Σ|Rᵢ|","total retrieved across all sources"),("ρ","admitted / retrieved ratio")],
                calc="Retrieved: 7 benefit docs\nThreshold 0.25 > all similarity scores\n|A|=0.  rho = 0/7 = 0.0  →  WARN (< 0.20)",
                thr="rho < 0.20  →  WARN",
                fix="Lower threshold: 0.25 → 0.05\nsim(q, doc_i) >= 0.05 for 5/7 docs\nrho = 5/7 = 0.71  →  PASS"),
              "p5": dict(
                metric="τ — Budget Utilisation",
                formula="τ  =  Σ tok(A)  /  W",
                syms=[("tok(c)","token length of chunk c"),("Σ tok(A)","total tokens in context window"),
                      ("W","global LIMIT TOKENS budget"),("τ","fraction of budget consumed")],
                calc="W=190. benefits=163 tok, customer=26 tok\nSum = 189.  tau = 189/190 = 0.995  →  WARN\nAdding one chunk → starvation",
                thr="tau > 0.95  →  WARN  (starvation imminent)",
                fix="Increase W to 2000\ntau = (196+107)/2000 = 0.15\nSafe headroom for all sources"),
              "p6": dict(
                metric="Q = (ρ, τ, δ, κ) — All Failing",
                formula="status  =  f(δ, τ, κ, ρ)",
                syms=[("ρ = |A|/Σ|Rᵢ|","coverage fraction"),("τ = Σtok/W","utilisation fraction"),
                      ("δ = |{i:Aᵢ=∅}|","starvation count"),("κ = ΣΦ/C(n,2)","contradiction density")],
                calc="thr=0.25 → rho=0 (nothing passes)\nW=45 → delta=2 (two sources starved)\nold+new policy → kappa=0.67\nAll three metrics fail simultaneously",
                thr="CRITICAL if delta>0\nWARN if tau>0.95 or kappa>0.30",
                fix="(1) thr 0.25->0.05  →  fixes rho\n(2) per-source limits  →  fixes delta\n(3) FILTER BY ver=v2025  →  fixes kappa\nEach fix is fully independent"),
            }

            m = MATH.get(pid, MATH["p1"])

            st.markdown('<span class="sl" style="margin-top:6px;display:block">Metric</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:11px;color:#a78bfa;font-weight:700;margin-bottom:6px">{m["metric"]}</div>',
                unsafe_allow_html=True)

            # Formula
            st.markdown(
                '<div style="background:#f5f3ff;border:2px solid #c4b5fd;border-radius:8px;'
                'padding:12px 16px;margin:4px 0 10px;text-align:center">'
                f'<span style="font-size:16px;color:#c4b5fd;font-family:Georgia,serif;'
                f'font-weight:700;letter-spacing:1px">{m["formula"]}</span></div>',
                unsafe_allow_html=True)

            # Symbols
            st.markdown('<span class="sl">Symbol Definitions</span>', unsafe_allow_html=True)
            for sym, defn in m["syms"]:
                st.markdown(
                    '<div style="display:flex;gap:8px;padding:3px 0;border-bottom:1px solid #0f1520">'
                    f'<span style="color:#7ec8e3;font-family:JetBrains Mono,monospace;font-size:9px;'
                    f'min-width:115px;flex-shrink:0">{sym}</span>'
                    f'<span style="color:#6b7280;font-size:9px">{defn}</span></div>',
                    unsafe_allow_html=True)

            # Example calculation
            st.markdown('<span class="sl" style="margin-top:10px;display:block">Example</span>', unsafe_allow_html=True)
            st.code(m["calc"], language=None)

            # Threshold
            st.markdown('<span class="sl">Threshold</span>', unsafe_allow_html=True)
            st.markdown(
                '<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:6px;'
                f'padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:10px;color:#f59e0b">{m["thr"]}</div>',
                unsafe_allow_html=True)

            # Fix
            st.markdown('<span class="sl" style="margin-top:8px;display:block">Mathematical Fix</span>', unsafe_allow_html=True)
            st.code(m["fix"], language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CI TESTS
# ════════════════════════════════════════════════════════════════════════════════
with t_ci:
    st.markdown("")
    cc1, cc2 = st.columns([5, 4], gap="large")

    with cc1:
        st.markdown("""<div style="font-family:Syne,sans-serif;font-weight:800;font-size:16px;color:#e2e8f0;margin-bottom:4px">CI Pipeline Integration</div>
        <div style="font-size:11px;color:#4a5a7a;margin-bottom:16px">assert report.delta == 0 — deterministic, &lt;5ms, no LLM call</div>""", unsafe_allow_html=True)

        st.markdown('<span class="sl">Test Suite — tests/test_context_quality.py</span>', unsafe_allow_html=True)
        st.code(textwrap.dedent("""\
        # tests/test_context_quality.py
        # Runs in < 50ms total. No LLM calls. No API costs.
        # Add to your CI pipeline: pytest tests/test_context_quality.py

        import pytest
        from opencql_inspector import ContextInspector
        from your_pipeline import assemble_context  # your existing code

        inspector = ContextInspector()
        SOURCES   = ["amex.benefits", "amex.offers", "amex.customer"]
        BUDGET    = 2000

        # ── Core quality gate ─────────────────────────────────────────────
        def test_customer_profile_never_starved():
            \"\"\"amex.customer must always contribute chunks.\"\"\"
            chunks = assemble_context(query="amex offers benefits Priya")
            report = inspector.inspect(chunks, sources_expected=SOURCES,
                                        token_budget=BUDGET)
            assert report.delta == 0, (
                f"STARVATION detected: {report.sources_starved}\\n"
                f"Fix: add LIMIT TOKENS N to each starved RETRIEVE block"
            )

        def test_offers_always_present():
            \"\"\"Offers context required for personalised recommendations.\"\"\"
            chunks = assemble_context(query="amex offers")
            report = inspector.inspect(chunks, sources_expected=SOURCES,
                                        token_budget=BUDGET)
            assert "amex.offers" not in report.sources_starved, (
                "amex.offers contributed 0 chunks — offer recs impossible"
            )

        def test_budget_utilisation_safe():
            \"\"\"Token budget must not exceed 90% — starvation risk.\"\"\"
            chunks = assemble_context(query="amex benefits credits")
            report = inspector.inspect(chunks, token_budget=BUDGET)
            assert report.tau <= 0.90, (
                f"Budget {report.tau:.0%} consumed. One source change "
                f"away from starvation."
            )

        def test_no_policy_contradictions():
            \"\"\"Contradiction density must stay below 0.30.\"\"\"
            chunks = assemble_context(query="return protection policy")
            report = inspector.inspect(chunks, token_budget=BUDGET)
            assert report.kappa < 0.30, (
                f"Contradictions κ={report.kappa:.3f} — conflicting policy "
                f"versions in context. Add FILTER BY version='current'"
            )

        def test_minimum_coverage():
            \"\"\"Coverage ρ must be >= 0.20 (threshold not too strict).\"\"\"
            chunks = assemble_context(query="amex benefits")
            report = inspector.inspect(chunks, token_budget=BUDGET)
            assert report.rho >= 0.20, (
                f"Coverage ρ={report.rho:.3f} — threshold too strict. "
                f"Lower WHERE similarity threshold to 0.05–0.15"
            )

        @pytest.mark.parametrize("query", [
            "amex offers benefits Priya travel",
            "platinum card lounge credit airline",
            "DoorDash Delta Hilton Amazon offers",
            "membership rewards points redemption",
        ])
        def test_all_queries_pass_inspection(query):
            \"\"\"Every standard query must pass Inspector.\"\"\"
            chunks = assemble_context(query=query)
            report = inspector.inspect(chunks, sources_expected=SOURCES,
                                        token_budget=BUDGET)
            assert report.delta == 0, (
                f"Query '{query}' → STARVATION: {report.sources_starved}"
            )
        """), language="python")

        st.markdown('<span class="sl" style="margin-top:12px;display:block">GitHub Actions</span>', unsafe_allow_html=True)
        st.code(textwrap.dedent("""\
        # .github/workflows/context-quality.yml
        name: OpenCQL Context Quality Gate
        on: [push, pull_request]
        jobs:
          inspect:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - run: pip install opencql-inspector google-generativeai
              - name: Run context quality tests
                run: pytest tests/test_context_quality.py -v
              - name: Block on starvation
                if: failure()
                run: echo "::error::Context quality gate failed"
        """), language="yaml")

    with cc2:
        st.markdown("""<div style="font-family:Syne,sans-serif;font-weight:800;font-size:16px;color:#e2e8f0;margin-bottom:4px">Live Test Runner</div>
        <div style="font-size:11px;color:#4a5a7a;margin-bottom:16px">Tests run against broken and fixed pipelines in real time</div>""", unsafe_allow_html=True)

        run_against = st.radio("Run tests against:", ["Broken pipeline", "Fixed pipeline"], horizontal=True, key="ci_target")
        run_ci = st.button("▶  Run Test Suite", key="run_ci", use_container_width=True)

        if run_ci:
            broken = (run_against == "Broken pipeline")
            P_ci = PROBLEMS[0]  # Use P1 (starvation) as the canonical pipeline under test
            docs = P_ci["docs"] + st.session_state.extra
            query = P_ci["query"]
            sources = P_ci["sources"]

            if broken:
                gl = P_ci["b_gl"]; lim = P_ci["b_lim"]; thr = P_ci["b_thr"]
            else:
                gl = P_ci["f_gl"]; lim = P_ci["f_lim"]; thr = P_ci["f_thr"]

            with st.spinner("Running tests…"):
                smap = assemble(docs, query, gl, lim, thr)
                report = run_inspector(smap, gl, query, sources)

            results = []
            for test in CI_TESTS:
                t0 = time.perf_counter()
                passed = test["check"](report, smap) if report else False
                elapsed = round((time.perf_counter()-t0)*1000, 2)
                results.append({
                    "name": test["name"], "passed": passed, "elapsed": elapsed,
                    "fail_msg": test["fail_msg"](report, smap) if report and not passed else "",
                    "desc": test["desc"],
                })

            st.session_state.ci_res = {
                "results": results, "report": report, "smap": smap,
                "broken": broken, "elapsed_total": sum(r["elapsed"] for r in results),
            }

        ci = st.session_state.ci_res
        if ci:
            results = ci["results"]
            n_pass = sum(1 for r in results if r["passed"])
            n_fail = len(results) - n_pass
            total_ms = round(ci["elapsed_total"], 1)

            # Pytest header
            label_col = "#ef4444" if n_fail else "#22c55e"
            label = f"{n_fail} failed, {n_pass} passed" if n_fail else f"{n_pass} passed"
            pipeline_label = "BROKEN" if ci["broken"] else "FIXED"
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;font-family:'JetBrains Mono',monospace">
              <div style="font-size:10px;color:#64748b;margin-bottom:6px">platform linux — Python 3.9 — pytest 7.x  ·  {pipeline_label} pipeline</div>
              <div style="font-size:10px;color:#64748b;margin-bottom:8px">collected {len(results)} items</div>
              <div style="border-top:1px solid #0c1220;margin:6px 0;padding-top:8px">""", unsafe_allow_html=True)

            for r in results:
                p = r["passed"]
                status_html = '<span style="color:#22c55e;font-weight:700">PASSED</span>' if p else '<span style="color:#ef4444;font-weight:700">FAILED</span>'
                st.markdown(f"""
                <div class="pytest-line" style="color:{'#3a4a5a' if p else '#e2e8f0'}">
                  {status_html} &nbsp; {r['name']} &nbsp;
                  <span style="color:#94a3b8;font-size:9px">({r['elapsed']:.2f}ms)</span>
                </div>
                {f'<div style="color:#fbbf24;font-size:9px;padding-left:12px;margin-bottom:3px">    AssertionError: {r["fail_msg"]}</div>' if not p and r.get("fail_msg") else ""}
                """, unsafe_allow_html=True)

            st.markdown(f"""
              </div>
              <div style="border-top:1px solid #0c1220;margin-top:8px;padding-top:8px;font-size:11px;font-weight:700;color:{label_col}">
                {label} &nbsp; in {total_ms:.1f}ms &nbsp; <span style="color:#94a3b8;font-size:9px">(no LLM calls)</span>
              </div>
            </div>""", unsafe_allow_html=True)

            if n_fail:
                st.markdown(f"""
                <div style="background:#fef2f2;border:1px solid #fca5a544;border-radius:8px;padding:12px;margin-top:10px;font-size:11px;color:#f87171">
                  <b>Broken pipeline failed {n_fail} tests.</b><br>
                  Load the ✅ Fixed CQL template and re-run — all tests will pass.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f0fdf4;border:1px solid #bbf7d044;border-radius:8px;padding:12px;margin-top:10px;font-size:11px;color:#4ade80">
                  <b>All tests pass.</b> This pipeline is safe to deploy.<br>
                  These checks run on every PR. Context bugs caught before production.
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: PM VIEW
# ════════════════════════════════════════════════════════════════════════════════
with t_pm:
    st.markdown("")
    st.markdown("""
    <div style="font-family:Syne,sans-serif;font-weight:800;font-size:18px;color:#e2e8f0;margin-bottom:4px">Business Impact</div>
    <div style="font-size:12px;color:#4a5a7a;margin-bottom:20px">Same customer. Same question. Different context quality. Completely different answers.</div>
    """, unsafe_allow_html=True)

    run_pm = st.button("▶  Run Comparison", key="run_pm", use_container_width=False)
    if not api_key:
        st.markdown('<div style="color:#f59e0b;font-size:11px;margin:4px 0">⚠ Add API key above (Gemini = free at aistudio.google.com). Inspector results shown without key.</div>', unsafe_allow_html=True)

    if run_pm:
        P_pm = PROBLEMS[0]
        docs = P_pm["docs"]
        query = P_pm["query"]
        goal  = P_pm["goal"]
        system = P_pm["system"]
        sources = P_pm["sources"]

        _ds_pm = {k:[d for d in docs if d["source"]==k] for k in sources}
        with st.spinner("Running broken pipeline…"):
            smap_b = assemble(docs, query, P_pm["b_gl"], P_pm["b_lim"], P_pm["b_thr"])
            rep_b  = run_inspector(smap_b, P_pm["b_gl"], query, sources, _ds_pm)
            ans_b  = call_llm(provider, system, smap_b, goal, api_key) if api_key else None

        with st.spinner("Running fixed pipeline…"):
            smap_f = assemble(docs, query, P_pm["f_gl"], P_pm["f_lim"], P_pm["f_thr"])
            rep_f  = run_inspector(smap_f, P_pm["f_gl"], query, sources, _ds_pm)
            ans_f  = call_llm(provider, system, smap_f, goal, api_key) if api_key else None

        st.session_state.pm_res = {
            "smap_b":smap_b,"rep_b":rep_b,"ans_b":ans_b,
            "smap_f":smap_f,"rep_f":rep_f,"ans_f":ans_f,
        }

    pm = st.session_state.pm_res
    if pm:
        pc1, pc2 = st.columns(2, gap="large")
        with pc1:
            rep_b = pm["rep_b"]
            st.markdown("""
            <div style="background:#fef2f2;border:2px solid #fca5a5;border-radius:10px;padding:14px 18px;margin-bottom:12px">
              <div style="font-size:18px;margin-bottom:6px">🚨</div>
              <div style="font-family:Syne,sans-serif;font-weight:800;font-size:14px;color:#ef4444">Without OpenCQL</div>
              <div style="font-size:10px;color:#5a2020;margin-top:3px">Unmanaged context assembly</div>
            </div>""", unsafe_allow_html=True)

            if rep_b:
                for src in sources:
                    cs = pm["smap_b"].get(src,[])
                    ok = bool(cs)
                    c = "#22c55e" if ok else "#ef4444"
                    st.markdown(f'<div style="font-size:10px;color:{c};padding:3px 0">{"✓" if ok else "✗"} {src} → {len(cs)} chunks {"STARVED" if not ok else ""}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:10px;color:#5a2a2a;margin-top:6px">δ={rep_b.delta} · τ={rep_b.tau:.2f} · {rep_b.tokens_used}/{rep_b.token_budget} tok</div>', unsafe_allow_html=True)

            if pm.get("ans_b"):
                st.markdown('<div style="font-size:10px;color:#5a3a3a;margin:10px 0 4px">Customer sees:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ans" style="border-color:#7f1d1d33;font-size:12px">{pm["ans_b"]}</div>', unsafe_allow_html=True)
                st.markdown('<div style="color:#ef4444;font-size:10px;margin-top:6px">⚠ Generic answer — no personalisation. No mention of Priya\'s name, points, or specific offers.</div>', unsafe_allow_html=True)
            elif not api_key:
                st.markdown(f'<div style="background:#fef2f2;border:1px solid #fca5a544;border-radius:6px;padding:10px;font-size:10px;color:#f87171;margin-top:10px">Inspector: <b>{rep_b.status if rep_b else "N/A"}</b> δ={rep_b.delta if rep_b else "?"}<br><br>With API key: see the generic, wrong answer the customer would receive.</div>', unsafe_allow_html=True)

        with pc2:
            rep_f = pm["rep_f"]
            st.markdown("""
            <div style="background:#f0fdf4;border:2px solid #bbf7d0;border-radius:10px;padding:14px 18px;margin-bottom:12px">
              <div style="font-size:18px;margin-bottom:6px">✅</div>
              <div style="font-family:Syne,sans-serif;font-weight:800;font-size:14px;color:#22c55e">With OpenCQL</div>
              <div style="font-size:10px;color:#16a34a;margin-top:3px">Managed context — all sources guaranteed</div>
            </div>""", unsafe_allow_html=True)

            if rep_f:
                for src in sources:
                    cs = pm["smap_f"].get(src,[])
                    ok = bool(cs)
                    c = "#22c55e" if ok else "#ef4444"
                    tt = sum(tok(c["text"]) for c in cs)
                    st.markdown(f'<div style="font-size:10px;color:{c};padding:3px 0">{"✓" if ok else "✗"} {src} → {len(cs)} chunks · {tt} tok</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:10px;color:#16a34a;margin-top:6px">δ={rep_f.delta} · τ={rep_f.tau:.2f} · {rep_f.tokens_used}/{rep_f.token_budget} tok</div>', unsafe_allow_html=True)

            if pm.get("ans_f"):
                st.markdown('<div style="font-size:10px;color:#16a34a;margin:10px 0 4px">Customer sees:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ans" style="border-color:#14532d33;font-size:12px">{pm["ans_f"]}</div>', unsafe_allow_html=True)
                st.markdown('<div style="color:#22c55e;font-size:10px;margin-top:6px">✓ Personalised — mentions Priya by name, her specific offers, her remaining credits.</div>', unsafe_allow_html=True)
            elif not api_key:
                st.markdown(f'<div style="background:#f0fdf4;border:1px solid #bbf7d044;border-radius:6px;padding:10px;font-size:10px;color:#4ade80;margin-top:10px">Inspector: <b>{rep_f.status if rep_f else "N/A"}</b> δ={rep_f.delta if rep_f else "?"}<br><br>With API key: see the personalised answer mentioning Priya\'s specific offers and credits.</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div style="font-family:Syne,sans-serif;font-weight:800;font-size:14px;color:#e2e8f0;margin-bottom:12px">Business Metrics Impact</div>', unsafe_allow_html=True)
        bm1, bm2, bm3, bm4 = st.columns(4)
        metrics = [
            ("Customer Satisfaction", "Generic answers → escalation calls", "Personalised answers → self-service resolution", "#ef4444", "#22c55e"),
            ("Offer Activation Rate", "Can't recommend offers (source starved)", "Specific offers recommended by name", "#ef4444", "#22c55e"),
            ("Compliance Auditability", "Raw Python — not auditable", "CQL files — version-controlled spec", "#f59e0b", "#22c55e"),
            ("Bug Detection Time", "3 months (production complaint)", "Request #1 (Inspector CRITICAL)", "#ef4444", "#22c55e"),
        ]
        for col, (title, bad, good, bc, gc) in zip([bm1,bm2,bm3,bm4], metrics):
            with col:
                st.markdown(f"""
                <div style="background:var(--panel);border:1px solid var(--bdr);border-radius:8px;padding:12px;height:140px">
                  <div style="font-family:Syne,sans-serif;font-weight:700;font-size:11px;color:#e2e8f0;margin-bottom:8px">{title}</div>
                  <div style="font-size:9px;color:{bc};margin-bottom:5px">✗ {bad}</div>
                  <div style="font-size:9px;color:{gc}">✓ {good}</div>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: INTEGRATE
# ════════════════════════════════════════════════════════════════════════════════
with t_integrate:
    st.markdown("")
    st.markdown("""<div style="font-family:Syne,sans-serif;font-weight:800;font-size:16px;color:#e2e8f0;margin-bottom:4px">Add to Your Pipeline</div>
    <div style="font-size:11px;color:#4a5a7a;margin-bottom:16px">Three levels of adoption — start with Level 1 today, zero pipeline changes required</div>""", unsafe_allow_html=True)

    ia, ib, ic = st.tabs(["Level 1 — Drop-in Inspector", "Level 2 — CQL Spec File", "Level 3 — Full DSL Runtime"])

    with ia:
        st.markdown('<span class="sl">Add 3 lines to your existing pipeline — no other changes</span>', unsafe_allow_html=True)
        st.code(textwrap.dedent("""\
        # Before: your existing code
        chunks = retrieve_from_vector_db(query)
        answer = llm.call(system_prompt, chunks, goal)

        # After: add Inspector between retrieval and LLM call
        from opencql_inspector import ContextInspector

        inspector = ContextInspector()
        report    = inspector.inspect(
            chunks,
            sources_expected=["amex.benefits", "amex.offers", "amex.customer"],
            token_budget=2000,
        )

        if report.delta > 0:
            raise ContextQualityError(
                f"Source starvation: {report.sources_starved}\\n"
                f"Fix: add LIMIT TOKENS per RETRIEVE\\n"
                f"Q = (ρ={report.rho} τ={report.tau:.2f} δ={report.delta} κ={report.kappa})"
            )

        answer = llm.call(system_prompt, chunks, goal)  # only called if PASS
        """), language="python")

        st.markdown('<span class="sl" style="margin-top:12px;display:block">In CI (GitHub Actions)</span>', unsafe_allow_html=True)
        st.code(textwrap.dedent("""\
        # pytest runs this in < 5ms — no LLM cost, no API call
        def test_no_starvation():
            chunks = assemble_context(query="amex offers benefits Priya")
            report = inspector.inspect(chunks, sources_expected=SOURCES, token_budget=2000)
            assert report.delta == 0, f"STARVED: {report.sources_starved}"
        """), language="python")

    with ib:
        st.markdown('<span class="sl">Write a .cql spec describing your intended context assembly</span>', unsafe_allow_html=True)
        st.code(textwrap.dedent("""\
        # pipelines/amex_concierge.cql
        # This file IS the specification for what the LLM should receive.
        # Readable by developers, architects, PMs, and compliance teams.

        CONTEXT amex_concierge AS (
          WITH SYSTEM "You are Priya's personal Amex concierge."

          RETRIEVE FROM amex.benefits
            WHERE similarity > 0.05
            TOP 5
            LIMIT TOKENS 350    -- hard cap: benefits can never take more

          RETRIEVE FROM amex.offers
            WHERE similarity > 0.05
            TOP 4
            LIMIT TOKENS 300    -- offers always guaranteed budget

          RETRIEVE FROM amex.customer
            WHERE similarity > 0.05
            TOP 4
            LIMIT TOKENS 250    -- customer profile always guaranteed

          LIMIT TOKENS 2000
        )

        INFER
          USING CONTEXT amex_concierge
          GOAL "What Amex offers and benefits should Priya use this month?"
          EXPLAIN
        """), language="sql")

        st.markdown('<span class="sl" style="margin-top:12px;display:block">Use the spec as a test</span>', unsafe_allow_html=True)
        st.code(textwrap.dedent("""\
        # CI test against the spec
        from opencql_inspector import ContextInspector
        from opencql_dsl.runtime  import CQLRuntime

        inspector = ContextInspector()
        rt        = CQLRuntime(default_model="mock")

        cql    = open("pipelines/amex_concierge.cql").read()
        result = rt.build_context(cql, query="amex offers benefits Priya")
        report = inspector.inspect(result.chunks,
                                    token_budget=result.global_limit,
                                    sources_expected=result.sources_expected)

        assert report.delta == 0   # PASS: all sources contributed
        assert report.tau   < 0.90  # PASS: budget not exhausted
        """), language="python")

    with ic:
        st.markdown('<span class="sl">Full DSL runtime — replace your Python assembly entirely</span>', unsafe_allow_html=True)
        st.code(textwrap.dedent("""\
        from opencql_dsl.runtime  import CQLRuntime
        from opencql_inspector    import ContextInspector
        import anthropic

        rt        = CQLRuntime(default_model="claude-3-haiku-20240307")
        inspector = ContextInspector()

        # Register your sources
        benefits_store = rt.registry.get_or_create("amex.benefits")
        benefits_store.add_documents(your_benefits_docs)

        offers_store = rt.registry.get_or_create("amex.offers")
        offers_store.add_documents(your_offers_docs)

        customer_store = rt.registry.get_or_create("amex.customer")
        customer_store.add_documents(your_customer_docs)

        # Execute CQL — Inspector fires automatically on EXPLAIN
        result = rt.execute(
            open("pipelines/amex_concierge.cql").read(),
            query="amex offers benefits Priya",
        )

        print(result)  # Claude's answer — only reached if Inspector PASS
        """), language="python")

        st.markdown('<span class="sl" style="margin-top:12px;display:block">Installation</span>', unsafe_allow_html=True)
        st.code(textwrap.dedent("""\
        # From local source
        pip install -e ./opencql_dsl_extracted/pkg_dsl
        pip install -e ./inspector_extracted/inspector_clean

        # For Claude (paid)
        export ANTHROPIC_API_KEY=sk-ant-your-key
        # For Gemini Flash (free)
        export GEMINI_API_KEY=AIza-your-key

        # Run this workbench
        streamlit run opencql_demo.py
        """), language="bash")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5: QUALITY GUIDE
# ════════════════════════════════════════════════════════════════════════════════
with t_guide:
    st.markdown("""
    <div style="max-width:900px;margin:0 auto">
    <div style="font-family:Syne,sans-serif;font-weight:800;font-size:20px;color:#0f172a;margin-bottom:4px">
    Context Quality Parameters — Q = (ρ, τ, δ, κ)</div>
    <div style="font-size:12px;color:#64748b;margin-bottom:24px">
    OpenCQL Inspector computes these four metrics <b>pre-inference</b> — before any LLM call.
    Each targets a different failure mode in RAG pipelines.
    </div></div>""", unsafe_allow_html=True)

    PARAMS = [
        {
            "sym": "δ",
            "name": "Delta — Source Starvation",
            "color": "#dc2626",
            "bg": "#fef2f2",
            "bdr": "#fca5a5",
            "formula": "δ = |{ i : Aᵢ = ∅ }|",
            "what": "Counts how many expected sources contributed zero chunks. A source is 'starved' when the global token budget is exhausted before it gets to process any documents.",
            "ideal": "δ = 0 always. Any δ ≥ 1 is CRITICAL and will produce wrong answers.",
            "ideal_color": "#dc2626",
            "ranges": [
                ("δ = 0", "PASS", "#16a34a", "All sources contributed. Safe to call LLM."),
                ("δ = 1", "CRITICAL", "#dc2626", "One source starved. LLM is missing a key data source."),
                ("δ ≥ 2", "CRITICAL", "#dc2626", "Multiple sources starved. Answer will be severely incomplete."),
            ],
            "examples": [
                ("🏥 Healthcare", "Patient medication history (source B) gets starved when lab results (source A) fill the context. LLM recommends a drug that conflicts with patient's existing medication — because it never saw the medication list."),
                ("💳 Finance (Amex)", "Customer profile starved when benefits docs fill 65-token budget. LLM gives generic advice instead of Priya-specific recommendations."),
                ("⚖️ Legal", "Case law (source A) exhausts budget. Contract terms (source B) never loaded. AI drafts a clause that violates the actual contract."),
            ],
            "fix": "Add LIMIT TOKENS per RETRIEVE block. Each source gets a guaranteed budget slice regardless of what others consume.",
            "fix_code": """RETRIEVE FROM amex.benefits
  WHERE similarity > 0.05
  TOP 5
  LIMIT TOKENS 350   ← guaranteed budget

RETRIEVE FROM amex.customer
  WHERE similarity > 0.05
  TOP 3
  LIMIT TOKENS 250   ← guaranteed budget""",
        },
        {
            "sym": "κ",
            "name": "Kappa — Contradiction Density",
            "color": "#d97706",
            "bg": "#fffbeb",
            "bdr": "#fcd34d",
            "formula": "κ = Σ Φ(cₐ, cᵦ) / C(n, 2)",
            "what": "Measures the fraction of chunk pairs that contain conflicting facts (same topic, different numbers). C(n,2) = n*(n-1)/2 is all unique pairs. Φ = 1 if the pair contradicts on a numeric value.",
            "ideal": "κ = 0 ideal. κ > 0.30 triggers WARN. LLM will hallucinate or synthesise a wrong middle value.",
            "ideal_color": "#d97706",
            "ranges": [
                ("κ = 0.0", "PASS", "#16a34a", "No contradictions. All chunks agree on facts."),
                ("0 < κ ≤ 0.10", "PASS", "#16a34a", "Minor overlap. Usually acceptable."),
                ("0.10 < κ ≤ 0.30", "WARN", "#d97706", "Some conflicting chunks. Review before production."),
                ("κ > 0.30", "WARN", "#dc2626", "High contradiction density. LLM will give wrong answer."),
            ],
            "examples": [
                ("📋 Policy versioning", "Old policy (30-day return) and new policy (90-day return) both loaded. LLM tells customer '60 days' — averaging the two. Complaint ensues."),
                ("💊 Drug dosage", "Two versions of a dosage guideline in context: 500mg (2022) and 750mg (2024). LLM recommends 625mg — a dangerous synthesis."),
                ("🏠 Mortgage rates", "Two rate sheets — Monday's (6.8%) and Wednesday's (7.1%) — both loaded. LLM quotes 6.95% to a customer. Neither is correct."),
            ],
            "fix": "Use FILTER BY to load only the current version. Or pre-process your vector store to tag documents with version metadata.",
            "fix_code": """RETRIEVE FROM amex.policy
  WHERE similarity > 0.05
  TOP 4
  LIMIT TOKENS 600
FILTER BY ver = "v2025"   ← only current policy""",
        },
        {
            "sym": "ρ",
            "name": "Rho — Coverage",
            "color": "#7c3aed",
            "bg": "#f5f3ff",
            "bdr": "#c4b5fd",
            "formula": "ρ = |A| / Σᵢ |Rᵢ|",
            "what": "Measures what fraction of available documents actually made it into the context. |A| = admitted chunks. Σ|Rᵢ| = total docs available across all sources. Low ρ means your threshold is too strict — relevant content is being silently excluded.",
            "ideal": "ρ ≥ 0.40 is healthy. ρ < 0.20 is a WARN — consider lowering WHERE similarity threshold.",
            "ideal_color": "#7c3aed",
            "ranges": [
                ("ρ ≥ 0.70", "PASS", "#16a34a", "Most available content admitted. Good retrieval."),
                ("0.40 ≤ ρ < 0.70", "PASS", "#16a34a", "Healthy coverage. Threshold is well-calibrated."),
                ("0.20 ≤ ρ < 0.40", "WARN", "#d97706", "Coverage low. Consider reducing similarity threshold."),
                ("ρ < 0.20", "WARN", "#dc2626", "Very low coverage. Threshold likely too strict."),
            ],
            "examples": [
                ("🔍 Support chatbot", "Threshold 0.40 set during dev on good embeddings. After migration to cheaper embedding model, similarity scores drop. Now 1/10 support articles admitted. Bot says 'I don't know' for 90% of valid questions."),
                ("📦 Product catalogue", "Seasonal products have unusual descriptions — low cosine similarity. Threshold 0.35 silently excludes 8 of 10 relevant products. Sales AI misses upsell opportunities."),
                ("💳 Benefits query", "threshold=0.30 admits only 1 of 7 Platinum benefits. LLM gives an incomplete answer — mentions only the airline credit, misses lounge, hotel status, entertainment credit."),
            ],
            "fix": "Lower WHERE similarity threshold. Start at 0.05 and increase until PASS. Use EXPLAIN to check ρ before every threshold change.",
            "fix_code": """-- Before (too strict):
  WHERE similarity > 0.30   ← ρ = 0.14

-- After (calibrated):
  WHERE similarity > 0.05   ← ρ = 0.71""",
        },
        {
            "sym": "τ",
            "name": "Tau — Budget Utilisation",
            "color": "#0369a1",
            "bg": "#f0f9ff",
            "bdr": "#7dd3fc",
            "formula": "τ = Σ tok(A) / W",
            "what": "Measures how much of the global token budget W is consumed. Σtok(A) = total tokens of admitted chunks. τ close to 1.0 means the context window is almost full — the next source added to the pipeline will immediately cause starvation (δ > 0).",
            "ideal": "0.60 ≤ τ ≤ 0.85 is the sweet spot. τ > 0.95 is a WARN — you're one source away from starvation.",
            "ideal_color": "#0369a1",
            "ranges": [
                ("τ < 0.30", "PASS", "#16a34a", "Budget mostly unused. Consider retrieving more content."),
                ("0.30 ≤ τ ≤ 0.85", "PASS", "#16a34a", "Healthy utilisation. Good balance of content and headroom."),
                ("0.85 < τ ≤ 0.95", "WARN", "#d97706", "Budget nearly full. Adding a source risks starvation."),
                ("τ > 0.95", "WARN", "#dc2626", "Critical headroom. Next schema change will cause δ > 0."),
            ],
            "examples": [
                ("📈 Agent iteration", "Multi-hop agent: Round 1 τ=0.72 (fine). Round 2 injects conversation history — τ jumps to 0.97. Round 3 adds new source — δ=1 fires. Context rot in action."),
                ("🌍 Multilingual pipeline", "English docs: τ=0.71. Team adds French docs (same content, 30% longer due to language). τ → 0.98. German docs added next week → immediate starvation."),
                ("🏦 Regulatory pipeline", "Compliance doc gets added to context. τ goes from 0.68 to 0.96. Next quarterly update adds one more compliance section → customer data source starves silently."),
            ],
            "fix": "Increase LIMIT TOKENS global budget. Add per-source limits to control allocation. Keep τ < 0.85 to leave headroom for future sources.",
            "fix_code": """-- Before: τ = 0.97, W = 185
  LIMIT TOKENS 185

-- After: τ = 0.15, headroom preserved
  LIMIT TOKENS 2000
  -- per-source limits ensure fair allocation""",
        },
    ]

    for p in PARAMS:
        st.markdown(f"""
        <div style="background:{p['bg']};border:2px solid {p['bdr']};border-radius:12px;
             padding:20px 24px;margin-bottom:20px">

          <div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:14px">
            <div style="background:{p['color']};color:#fff;border-radius:8px;width:44px;height:44px;
                 display:flex;align-items:center;justify-content:center;font-size:22px;
                 font-family:Georgia,serif;font-weight:700;flex-shrink:0">{p['sym']}</div>
            <div>
              <div style="font-family:Syne,sans-serif;font-weight:800;font-size:16px;color:#0f172a">{p['name']}</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:13px;color:{p['color']};
                   margin-top:2px;font-weight:600">{p['formula']}</div>
            </div>
          </div>

          <div style="font-size:11.5px;color:#374151;line-height:1.8;margin-bottom:14px">{p['what']}</div>
        </div>""", unsafe_allow_html=True)

        col_ranges, col_examples = st.columns([4, 5], gap="medium")

        with col_ranges:
            st.markdown(f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#94a3b8;font-family:Syne,sans-serif;font-weight:700;margin-bottom:8px">Suitable Values</div>', unsafe_allow_html=True)
            for val, status, col, desc in p["ranges"]:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:10px;padding:7px 10px;
                     border-left:3px solid {col};background:#fff;border-radius:0 6px 6px 0;margin-bottom:4px">
                  <div style="min-width:90px">
                    <div style="font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;color:{col}">{val}</div>
                    <div style="font-size:9px;color:{col};font-weight:600">{status}</div>
                  </div>
                  <div style="font-size:10px;color:#64748b;line-height:1.6">{desc}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#94a3b8;font-family:Syne,sans-serif;font-weight:700;margin:12px 0 6px">Fix</div>', unsafe_allow_html=True)
            st.code(p["fix_code"], language="sql")

        with col_examples:
            st.markdown(f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#94a3b8;font-family:Syne,sans-serif;font-weight:700;margin-bottom:8px">Real-World Examples</div>', unsafe_allow_html=True)
            for icon_name, desc in p["examples"]:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
                     padding:10px 13px;margin-bottom:7px">
                  <div style="font-family:Syne,sans-serif;font-weight:700;font-size:11px;
                       color:#0f172a;margin-bottom:4px">{icon_name}</div>
                  <div style="font-size:10.5px;color:#475569;line-height:1.7">{desc}</div>
                </div>""", unsafe_allow_html=True)

    # Other problems section
    st.divider()
    st.markdown('<div style="font-family:Syne,sans-serif;font-weight:800;font-size:16px;color:#0f172a;margin-bottom:4px">Other Context Quality Problems</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11.5px;color:#64748b;margin-bottom:16px">Beyond the four Q metrics, these problems also degrade RAG quality.</div>', unsafe_allow_html=True)

    OTHER_PROBLEMS = [
        {
            "icon":"👻","name":"Lost in the Middle (LITM)","tag":"P5_LITM",
            "color":"#0369a1","bg":"#f0f9ff","bdr":"#7dd3fc",
            "what":"LLMs pay 40% less attention to chunks in the middle of the context window compared to first and last positions. Critical information placed in positions 5-10 of a 16-chunk context is effectively invisible to the model.",
            "formula":"A(i,n) = 0.40·e^(-0.25i) + 0.38·e^(-0.25·(n-1-i)) + 0.30",
            "formula_note":"Attention score for chunk at position i in a context of n chunks. Fitted from Liu et al. TACL 2024.",
            "threshold":"A < 0.45 AND n ≥ 5 → WARN per affected chunk",
            "example":"16 benefits loaded. Positions 5–10 have A ≈ 0.42. LLM answers 'The Platinum Card offers airline credits and lounge access' — missing TSA PreCheck, hotel credits, extended warranty (all in the dead zone).",
            "fix":"Use TOP N (5–7) to keep only the most relevant chunks. All chunks stay in high-attention positions (0–4 and n-4 to n-1).",
        },
        {
            "icon":"🔄","name":"Context Rot","tag":"CONTEXT_ROT",
            "color":"#7c3aed","bg":"#f5f3ff","bdr":"#c4b5fd",
            "what":"In multi-hop agent pipelines, context accumulates stale, redundant, and contradictory content across iterations. Each hop appends more chunks without removing irrelevant ones. By iteration 3-5, δ and κ both deteriorate silently.",
            "formula":"rot(n) = Σ staleness(cᵢ) / n",
            "formula_note":"Fraction of context that is outdated or superseded by newer chunks in the same pipeline run.",
            "threshold":"No standard threshold. Use Inspector at each hop — watch for rising κ and τ across iterations.",
            "example":"Agent hop 1: τ=0.52, κ=0.0. Hop 2: previous answer injected as context — τ=0.74, κ=0.0. Hop 3: new data added — τ=0.96, κ=0.28. Hop 4: CRITICAL (δ=1). Three hops to silent failure.",
            "fix":"Use INJECT HISTORY LAST N TURNS to cap conversation history. Run EXPLAIN at each hop. Evict chunks older than K hops.",
        },
        {
            "icon":"🔁","name":"Semantic Redundancy","tag":"REDUNDANCY",
            "color":"#059669","bg":"#f0fdf4","bdr":"#bbf7d0",
            "what":"Multiple chunks saying the same thing waste token budget. If 3 of 5 admitted chunks are near-duplicates (cosine similarity > 0.90), you are using 60% of your budget to repeat information the LLM already has.",
            "formula":"redundancy = |{(i,j) : cosine(cᵢ,cⱼ) > 0.90}| / C(n,2)",
            "formula_note":"Fraction of chunk pairs that are near-duplicates. Not yet in the Q vector but planned.",
            "threshold":"redundancy > 0.30 → inefficient budget use. Consider deduplication at ingestion time.",
            "example":"Benefits source has 7 docs, 3 of which are reformulations of the airline credit (FAQ, terms page, summary page). All 3 pass similarity threshold. 120 tokens wasted. Customer profile gets 120 fewer tokens.",
            "fix":"Deduplicate at ingestion using embedding similarity. Or use LIMIT per source to force variety.",
        },
        {
            "icon":"⏰","name":"Staleness","tag":"STALENESS",
            "color":"#92400e","bg":"#fffbeb","bdr":"#fcd34d",
            "what":"Outdated documents answer questions correctly for old state but incorrectly for current state. Unlike κ (which detects explicit numeric contradictions), staleness is about temporal decay — the document was once right but is now wrong.",
            "formula":"staleness(d) = age(d) / max_age_threshold",
            "formula_note":"Documents older than a threshold should be flagged before admission.",
            "threshold":"Domain-dependent. Legal: 90 days. Medical: 30 days. Financial rates: 24 hours.",
            "example":"'Return protection covers 30 days' was correct policy in 2024. In 2025, policy changed to 90 days. Old doc is still in vector store. κ=0 (no contradictions because old doc was removed from active set) but answer is still wrong.",
            "fix":"Tag all documents with created_at metadata. Use FILTER BY to exclude old docs. Implement auto-expiry in your vector store.",
        },
    ]

    for i in range(0, len(OTHER_PROBLEMS), 2):
        cols = st.columns(2, gap="medium")
        for j, p in enumerate(OTHER_PROBLEMS[i:i+2]):
            with cols[j]:
                st.markdown(f"""
                <div style="background:{p['bg']};border:1.5px solid {p['bdr']};border-radius:10px;padding:16px;height:100%">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                    <span style="font-size:20px">{p['icon']}</span>
                    <div>
                      <div style="font-family:Syne,sans-serif;font-weight:800;font-size:13px;color:#0f172a">{p['name']}</div>
                      <div style="font-size:9px;color:{p['color']};font-weight:700;letter-spacing:1px">{p['tag']}</div>
                    </div>
                  </div>
                  <div style="font-size:10.5px;color:#374151;line-height:1.7;margin-bottom:10px">{p['what']}</div>
                  <div style="background:#fff;border-radius:6px;padding:8px 10px;margin-bottom:8px">
                    <div style="font-family:JetBrains Mono,monospace;font-size:10px;color:{p['color']};font-weight:600">{p['formula']}</div>
                    <div style="font-size:9px;color:#94a3b8;margin-top:3px">{p['formula_note']}</div>
                  </div>
                  <div style="font-size:9px;color:#64748b;border-left:3px solid {p['color']};padding-left:8px;margin-bottom:8px"><b>Threshold:</b> {p['threshold']}</div>
                  <div style="font-size:10px;color:#475569;background:#fff;border-radius:6px;padding:8px;line-height:1.6"><b>Example:</b> {p['example']}</div>
                  <div style="font-size:10px;color:{p['color']};margin-top:8px;font-weight:600">Fix: <span style="font-weight:400;color:#475569">{p['fix']}</span></div>
                </div>""", unsafe_allow_html=True)

    # Context Rot live demo
    st.divider()
    st.markdown('<div style="font-family:Syne,sans-serif;font-weight:800;font-size:15px;color:#0f172a;margin-bottom:4px">🔄 Context Rot — Live Multi-Hop Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#64748b;margin-bottom:14px">Watch how context quality degrades across agent iterations as stale and redundant content accumulates.</div>', unsafe_allow_html=True)

    if st.button("▶ Simulate 4-Hop Agent Pipeline", key="rot_sim"):
        import math, re as _re
        try:
            from opencql_inspector.inspector import ContextInspector as _CI
            _rot_insp = _CI()
            def _tok(t): return max(1,math.ceil(len(t or "")/3.8))
            def _sim(q,t):
                a=set(_re.findall(r"[a-z]{3,}",(q or "").lower()))
                b=_re.findall(r"[a-z]{3,}",(t or "").lower())
                return round(sum(1 for w in b if w in a)/len(a|set(b)),4) if a and b else 0.0

            BASE = [
                {"text":"Amex Platinum airline lounge credit benefit travel.","source":"amex.benefits"},
                {"text":"Priya Sharma Platinum Card travel spending profile.","source":"amex.customer"},
                {"text":"Policy 2025 return ninety days extended coverage.","source":"amex.policy","ver":"v2025"},
            ]

            HOP_ADDITIONS = [
                [],  # hop 1: clean baseline
                [    # hop 2: inject previous answer as context
                    {"text":"Previous answer: Priya has airline lounge lounge benefit airline travel credit airline.","source":"amex.history"},
                ],
                [    # hop 3: inject more history + old policy
                    {"text":"Answer hop 2: Platinum airline credit lounge benefit travel airline credit airline.","source":"amex.history"},
                    {"text":"Policy 2024 return thirty days old coverage.","source":"amex.policy","ver":"v2024"},
                ],
                [    # hop 4: more redundancy + another old policy
                    {"text":"Recap: Priya has airline credit lounge benefit airline travel credit airline lounge.","source":"amex.history"},
                    {"text":"Policy 2024 thirty days return theft damage old.","source":"amex.policy","ver":"v2024"},
                    {"text":"Summary: airline lounge credit platinum benefit airline credit lounge travel.","source":"amex.history"},
                ],
            ]

            st.markdown('<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px">', unsafe_allow_html=True)
            for hop_i, extra in enumerate(HOP_ADDITIONS):
                chunks = BASE + sum(HOP_ADDITIONS[:hop_i+1],[])
                r = _rot_insp.inspect(chunks, query="airline lounge credit benefit",
                                      token_budget=2000, sources_expected=["amex.benefits","amex.customer","amex.policy"])
                eta   = getattr(r,"eta",0)
                sigma = getattr(r,"sigma",0)
                litm_count = sum(1 for w in r.warnings if "LITM" in w.get("type",""))
                status_col = "#dc2626" if r.status=="CRITICAL" else "#d97706" if r.status=="WARN" else "#16a34a"

                st.markdown(f"""
                <div style="background:#fff;border:1.5px solid {'#dc2626' if r.status=='CRITICAL' else '#f59e0b' if r.status=='WARN' else '#bbf7d0'};
                     border-radius:8px;padding:12px">
                  <div style="font-family:Syne,sans-serif;font-weight:800;font-size:11px;color:#0f172a;margin-bottom:6px">
                    Hop {hop_i+1}</div>
                  <div style="font-size:9px;font-weight:700;color:{status_col};margin-bottom:8px">{r.status} · {len(chunks)} chunks</div>
                  {"".join(f'<div style="display:flex;justify-content:space-between;font-size:10px;padding:2px 0;border-bottom:1px solid #f1f5f9"><span style="color:#64748b">{sym}</span><span style="font-family:JetBrains Mono,monospace;font-weight:700;color:{col}">{val}</span></div>'
                  for sym,val,col in [
                    ("δ starvation", r.delta, "#dc2626" if r.delta>0 else "#16a34a"),
                    ("τ utilisation", f"{r.tau:.2f}", "#d97706" if r.tau>0.85 else "#16a34a"),
                    ("κ contradiction", f"{r.kappa:.2f}", "#d97706" if r.kappa>0.3 else "#16a34a"),
                    ("η redundancy", f"{eta:.2f}", "#d97706" if eta>0.3 else "#16a34a"),
                    ("σ staleness", f"{sigma:.2f}", "#d97706" if sigma>0.3 else "#16a34a"),
                  ])}
                  <div style="font-size:9px;color:#94a3b8;margin-top:6px">{len(r.warnings)} warning(s)</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("Hop 1 is clean (PASS). By Hop 3-4, redundancy and staleness accumulate → WARN/CRITICAL. This is context rot.")
        except Exception as e:
            st.error(f"Error: {e}")

    # Summary table
    st.divider()
    st.markdown('<div style="font-family:Syne,sans-serif;font-weight:800;font-size:14px;color:#0f172a;margin-bottom:12px">Quick Reference — All Parameters</div>', unsafe_allow_html=True)
    st.markdown("""
    <table style="width:100%;border-collapse:collapse;font-size:11px">
      <tr style="background:#f1f5f9">
        <th style="padding:8px 12px;text-align:left;color:#64748b;font-weight:700;letter-spacing:1px;font-size:9px;text-transform:uppercase">Metric</th>
        <th style="padding:8px 12px;text-align:left;color:#64748b;font-weight:700;letter-spacing:1px;font-size:9px;text-transform:uppercase">Formula</th>
        <th style="padding:8px 12px;text-align:left;color:#64748b;font-weight:700;letter-spacing:1px;font-size:9px;text-transform:uppercase">Ideal</th>
        <th style="padding:8px 12px;text-align:left;color:#64748b;font-weight:700;letter-spacing:1px;font-size:9px;text-transform:uppercase">WARN threshold</th>
        <th style="padding:8px 12px;text-align:left;color:#64748b;font-weight:700;letter-spacing:1px;font-size:9px;text-transform:uppercase">Failure mode</th>
      </tr>
      <tr style="border-bottom:1px solid #e2e8f0">
        <td style="padding:8px 12px;color:#dc2626;font-family:Georgia,serif;font-weight:700;font-size:14px">δ</td>
        <td style="padding:8px 12px;font-family:JetBrains Mono,monospace;color:#374151">|{i : Aᵢ = ∅}|</td>
        <td style="padding:8px 12px;color:#16a34a;font-weight:700">= 0</td>
        <td style="padding:8px 12px;color:#dc2626">δ ≥ 1 → CRITICAL</td>
        <td style="padding:8px 12px;color:#64748b">LLM missing entire data source</td>
      </tr>
      <tr style="border-bottom:1px solid #e2e8f0;background:#fafafa">
        <td style="padding:8px 12px;color:#d97706;font-family:Georgia,serif;font-weight:700;font-size:14px">κ</td>
        <td style="padding:8px 12px;font-family:JetBrains Mono,monospace;color:#374151">ΣΦ(cₐ,cᵦ) / C(n,2)</td>
        <td style="padding:8px 12px;color:#16a34a;font-weight:700">= 0.0</td>
        <td style="padding:8px 12px;color:#d97706">κ > 0.30 → WARN</td>
        <td style="padding:8px 12px;color:#64748b">LLM synthesises contradicting facts</td>
      </tr>
      <tr style="border-bottom:1px solid #e2e8f0">
        <td style="padding:8px 12px;color:#7c3aed;font-family:Georgia,serif;font-weight:700;font-size:14px">ρ</td>
        <td style="padding:8px 12px;font-family:JetBrains Mono,monospace;color:#374151">|A| / Σ|Rᵢ|</td>
        <td style="padding:8px 12px;color:#16a34a;font-weight:700">≥ 0.40</td>
        <td style="padding:8px 12px;color:#d97706">ρ < 0.20 → WARN</td>
        <td style="padding:8px 12px;color:#64748b">Relevant content silently excluded</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;color:#0369a1;font-family:Georgia,serif;font-weight:700;font-size:14px">τ</td>
        <td style="padding:8px 12px;font-family:JetBrains Mono,monospace;color:#374151">Σtok(A) / W</td>
        <td style="padding:8px 12px;color:#16a34a;font-weight:700">0.60 – 0.85</td>
        <td style="padding:8px 12px;color:#d97706">τ > 0.95 → WARN</td>
        <td style="padding:8px 12px;color:#64748b">Next source addition → immediate δ</td>
      </tr>
    </table>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# RUN HANDLER
# ════════════════════════════════════════════════════════════════════════════════
if run_btn:
    cql  = st.session_state.cql
    logs = []
    def log(m, k="info"): logs.append((k, m))

    log("Parsing CQL parameters…")

    # Parse from CQL text
    # Strip -- comments before parsing so keywords in comments don't confuse regex
    cql_clean = re.sub(r'--[^\n]*', '', cql)

    gl_m = re.findall(r'\bLIMIT\s+TOKENS\s+(\d+)', cql_clean, re.I)
    global_limit = int(gl_m[-1]) if gl_m else P["f_gl"]

    per_limits = {}
    for m in re.finditer(r'RETRIEVE\s+FROM\s+([\w.]+)(.*?)(?=RETRIEVE|INFER|\Z)', cql_clean, re.I|re.S):
        src, blk = m.group(1), m.group(2)
        lm = re.search(r'LIMIT\s+TOKENS\s+(\d+)', blk, re.I)
        if lm: per_limits[src] = int(lm.group(1))

    thr_m = re.search(r'WHERE\s+similarity\s*[><=!]+\s*([\d.]+)', cql_clean, re.I)
    threshold = float(thr_m.group(1)) if thr_m else 0.05

    flt_m = re.search(r'FILTER\s+BY\s+(\w+)\s*=\s*"([^"]+)"', cql_clean, re.I)
    doc_filter = {flt_m.group(1): flt_m.group(2)} if flt_m else None

    goal_m = re.search(r'GOAL\s+"([^"]+)"', cql_clean, re.I)
    goal_txt = goal_override.strip() if 'goal_ov' in st.session_state and st.session_state.goal_ov else (goal_m.group(1) if goal_m else P["goal"])

    log(f"global={global_limit}tok · thr={threshold} · per={per_limits or 'none'}{' · filter='+str(doc_filter) if doc_filter else ''}")

    # Use all available docs so CQL can reference any source
    # (problem docs take priority, then global pool, then user-added extras)
    # Detect all sources: RETRIEVE FROM + JOIN ... SEMANTIC
    _cql_sources = list(dict.fromkeys(
        re.findall(r'(?:RETRIEVE\s+FROM|JOIN)\s+([\w.]+)', cql_clean, re.I)
    ))
    # Parse per-source FILTER BY
    source_filter = {}
    for m in re.finditer(r'RETRIEVE\s+FROM\s+([\w.]+)(.*?)(?=RETRIEVE|JOIN|INFER|\Z)', cql_clean, re.I|re.S):
        src, blk = m.group(1), m.group(2)
        fm = re.search(r'FILTER\s+BY\s+(\w+)\s*=\s*"([^"]+)"', blk, re.I)
        if fm:
            source_filter[src] = {fm.group(1): fm.group(2)}
    if source_filter:
        doc_filter = None  # use per-source filter instead of global
    _prob_sources = set(d["source"] for d in P["docs"])
    _extra_sources = set(d["source"] for d in st.session_state.extra)
    _need_global = any(s not in _prob_sources and s not in _extra_sources
                       for s in _cql_sources)
    all_docs = P["docs"] + st.session_state.extra
    if _need_global:
        # Add docs for sources referenced in CQL but not in current problem
        _already = set(d["source"] for d in all_docs)
        for d in ALL_DOCS:
            if d["source"] in _cql_sources and d["source"] not in _already:
                all_docs.append(d)
                _already.add(d["source"])  # avoid per-doc check after first add
        # Actually add ALL docs for missing sources cleanly:
        _missing_srcs = [s for s in _cql_sources if s not in set(d["source"] for d in P["docs"]+st.session_state.extra)]
        all_docs = P["docs"] + st.session_state.extra + [d for d in ALL_DOCS if d["source"] in _missing_srcs]
    smap = assemble(all_docs, P["query"], global_limit,
                    per_limits if per_limits else None,
                    threshold, doc_filter, source_filter)

    for src in P["sources"]:
        cs = smap.get(src, [])
        tt = sum(tok(c["text"]) for c in cs)
        log(f"{'✓' if cs else '✗'} {src} → {len(cs)} chunks · {tt} tok", "ok" if cs else "warn")

    # Inspector sources: use CQL sources if specified, else fall back to problem sources
    _insp_sources = _cql_sources if _cql_sources else P["sources"]
    _ds = {k: [d for d in all_docs if d["source"]==k] for k in _insp_sources}
    _max_age = P.get("max_age_days", None)
    report = run_inspector(smap, global_limit, P["query"], _insp_sources, _ds, _max_age)
    if report:
        lv = "ok" if report.status=="PASS" else "err" if report.status=="CRITICAL" else "warn"
        log(f"Inspector: {report.status} · ρ={report.rho} τ={report.tau:.2f} δ={report.delta} κ={report.kappa}", lv)
        for w in report.warnings:
            log(f"  ⚠ {warn_type(w)}: {warn_msg(w)}", "warn")

    st.session_state.logs = logs

    answer = None; extra = {}
    if inspect_only:
        extra = {"inspect_only": True}
    elif not api_key:
        extra = {"no_key": True}
    else:
        try:
            log("Calling Claude…")
            answer = call_llm(provider, P["system"], smap, goal_txt, api_key)
            log(f"Done · {len(answer)} chars", "ok")
            st.session_state.logs = logs
        except Exception as e:
            extra = {"api_error": str(e)}
            log(f"API error: {e}", "err")

    st.session_state.result = {"report": report, "smap": smap, "answer": answer, "insp_sources": _insp_sources, **extra}
    st.rerun()

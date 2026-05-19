"""
OpenCQL — Comprehensive Demo
Developer · QA · Product Manager · Context Quality Problems · CI Tests
Run: streamlit run opencql_demo.py
"""
from __future__ import annotations
import sys, os, math, time, re, textwrap
from collections import defaultdict

# Bundled packages sit next to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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

def assemble(docs, query, global_limit, per_limits=None, threshold=0.0, doc_filter=None):
    if doc_filter:
        k, v = list(doc_filter.items())[0]
        docs = [d for d in docs if str(d.get(k, "")) == v or k not in d]
    by_src = defaultdict(list)
    for d in docs:
        s = sim(query, d["text"])
        if s >= threshold:
            by_src[d["source"]].append({**d, "_sc": s, "_tk": tok(d["text"])})
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

def run_inspector(smap, global_limit, query, sources_expected):
    if not PKG_OK: return None
    chunks = [{"text": c["text"], "source": src}
              for src, cs in smap.items() for c in cs]
    return _inspector.inspect(chunks, query=query,
                               token_budget=global_limit,
                               sources_expected=sources_expected)

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
    {"source":"amex.offers","text":"Amex Offer ACTIVE: Spend $50+ at Amazon.com, get $10 statement credit. Valid through Dec 31 2025. Add to card.","merchant":"Amazon","status":"active"},
    {"source":"amex.offers","text":"Amex Offer ACTIVE: Spend $150+ at Delta Air Lines, get $30 statement credit on direct ticket purchases.","merchant":"Delta","status":"active"},
    {"source":"amex.offers","text":"Amex Offer ACTIVE: Spend $75+ at Hilton Hotels, get $25 statement credit at participating Hilton properties.","merchant":"Hilton","status":"active"},
    {"source":"amex.offers","text":"Amex Offer ACTIVE: Spend $40+ at DoorDash, get $15 statement credit. One-time use, food delivery orders only.","merchant":"DoorDash","status":"active"},
    {"source":"amex.offers","text":"Amex Offer ACTIVE: Spend $200+ at Marriott Bonvoy hotels, get $40 statement credit at participating locations.","merchant":"Marriott","status":"active"},
]
CUSTOMER = [
    {"source":"amex.customer","text":"Priya Sharma: Platinum Card since Jan 2022. Annual spend $85,000. Top categories: business travel (40%), fine dining (25%)."},
    {"source":"amex.customer","text":"Priya travel pattern: London and Singapore monthly for consulting engagements. Prefers Hilton and Marriott properties."},
    {"source":"amex.customer","text":"Priya account: 180,000 Membership Rewards points accumulated. Airline credit: $180 used, only $20 remaining for 2025."},
    {"source":"amex.customer","text":"Priya offer activations: Amazon $10 ✓, Delta $30 ✓, Hilton $25 ✓ all activated this month. DoorDash $15 not yet activated."},
]
POLICY_OLD = [
    {"source":"amex.policy","text":"Purchase protection: items eligible for return within 30 days of purchase. Covers accidental damage and theft.","ver":"v2024"},
    {"source":"amex.policy","text":"Return protection: extend store return period to 30 days from purchase. Maximum $300 per item, $1000 per year.","ver":"v2024"},
]
POLICY_NEW = [
    {"source":"amex.policy","text":"Purchase protection: items eligible for return within 90 days of purchase. Extended coverage for Platinum cardholders.","ver":"v2025"},
    {"source":"amex.policy","text":"Return protection: extend store return period to 90 days from purchase. Maximum $500 per item, $2000 per year.","ver":"v2025"},
]
ALL_SOURCES = ["amex.benefits", "amex.offers", "amex.customer"]
DEFAULT_QUERY = "platinum card offers airline lounge credit Priya"
DEFAULT_GOAL  = "What Amex offers and benefits should Priya use this month?"
DEFAULT_SYSTEM = "You are Priya's personal Amex concierge. Use her profile to give specific, personalised advice."

# ════════════════════════════════════════════════════════════════════════════════
# PROBLEM DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════
PROBLEMS = [
  { "id":"p1","idx":0,"icon":"🔴","metric":"δ=1","label":"CRITICAL",
    "title":"Source Starvation","sub":"One source gets zero tokens",
    "col":"#ef4444","bg":"#200808","bdr":"#7f1d1d",
    "what":"amex.benefits + amex.offers together consume the entire 65-token global budget. amex.customer gets zero chunks. The LLM never sees Priya's profile — answers generically.",
    "fix":"Add LIMIT TOKENS per RETRIEVE. Each source is guaranteed its budget regardless of how many tokens other sources consume.",
    "docs": BENEFITS+OFFERS+CUSTOMER,
    "sources": ALL_SOURCES, "query": DEFAULT_QUERY,
    "goal": DEFAULT_GOAL, "system": DEFAULT_SYSTEM,
    "b_gl":65,  "b_lim":{},
    "f_gl":2000,"f_lim":{"amex.benefits":350,"amex.offers":300,"amex.customer":250},
    "b_thr":0.05,"f_thr":0.05,
    "broken_cql":"""\
-- ❌ PROBLEM: source starvation  δ=1
-- benefits + offers fill the 65-token budget completely
-- amex.customer → 0 chunks — LLM doesn't know who Priya is

CONTEXT broken_starvation AS (
  WITH SYSTEM "You are Priya's Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 5

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4

  LIMIT TOKENS 65
)

INFER
  USING CONTEXT broken_starvation
  GOAL "What Amex offers and benefits should Priya use this month?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: each source has a guaranteed budget
-- amex.customer ALWAYS gets 250 tokens regardless of other scores

CONTEXT fixed_starvation AS (
  WITH SYSTEM "You are Priya's personal Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 350

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 300

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 250

  LIMIT TOKENS 2000
)

INFER
  USING CONTEXT fixed_starvation
  GOAL "What Amex offers and benefits should Priya use this month?"
  EXPLAIN""",
  },
  { "id":"p2","idx":1,"icon":"🔴","metric":"δ=2","label":"CRITICAL",
    "title":"Multi-Source Starvation","sub":"Two sources starved at once",
    "col":"#f87171","bg":"#1a0505","bdr":"#7f1d1d",
    "what":"With a 45-token global budget and no per-source limits, amex.benefits alone exhausts the budget. Both amex.offers AND amex.customer get zero chunks. The LLM sees only card policy.",
    "fix":"Three separate LIMIT TOKENS clauses — one per RETRIEVE block. Sum of per-source limits must be less than the global limit.",
    "docs": BENEFITS+OFFERS+CUSTOMER,
    "sources": ALL_SOURCES, "query": DEFAULT_QUERY,
    "goal": DEFAULT_GOAL, "system": DEFAULT_SYSTEM,
    "b_gl":45,  "b_lim":{},
    "f_gl":2000,"f_lim":{"amex.benefits":350,"amex.offers":300,"amex.customer":250},
    "b_thr":0.05,"f_thr":0.05,
    "broken_cql":"""\
-- ❌ PROBLEM: multi-source starvation  δ=2
-- benefits alone fills the 45-token budget
-- BOTH offers AND customer → 0 chunks

CONTEXT broken_multi AS (
  WITH SYSTEM "You are an Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 5

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4

  LIMIT TOKENS 45
)

INFER
  USING CONTEXT broken_multi
  GOAL "What Amex offers and benefits should Priya use this month?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: all three sources guaranteed their own budgets

CONTEXT fixed_multi AS (
  WITH SYSTEM "You are Priya's personal Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 350

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 300

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 250

  LIMIT TOKENS 2000
)

INFER
  USING CONTEXT fixed_multi
  GOAL "What Amex offers and benefits should Priya use this month?"
  EXPLAIN""",
  },
  { "id":"p3","idx":2,"icon":"⚡","metric":"κ>0.3","label":"WARN",
    "title":"Contradictions","sub":"Conflicting facts in context",
    "col":"#f59e0b","bg":"#1a1100","bdr":"#92400e",
    "what":"Old policy says 30-day return protection. New policy says 90 days. Both versions are loaded. The LLM synthesises a wrong answer — often '30-90 days' or picks the wrong one.",
    "fix":"FILTER BY ver = 'v2025' to load only current policy. Or deduplicate your knowledge base — never have two versions of the same fact in the same source.",
    "docs": BENEFITS+OFFERS+POLICY_OLD+POLICY_NEW+CUSTOMER,
    "sources": ["amex.benefits","amex.policy","amex.customer"],
    "query": "amex purchase return protection policy days",
    "goal": "What is the return protection period for Priya's Platinum Card?",
    "system": "You are a billing support agent. Be precise about dates and durations.",
    "b_gl":1500,"b_lim":{"amex.benefits":400,"amex.policy":600,"amex.customer":200},
    "f_gl":1500,"f_lim":{"amex.benefits":400,"amex.policy":600,"amex.customer":200},
    "b_thr":0.05,"f_thr":0.05,
    "b_filter":None,"f_filter":{"ver":"v2025"},
    "broken_cql":"""\
-- ❌ PROBLEM: contradictions  κ>0.3
-- old policy (30-day) AND new policy (90-day) both loaded
-- LLM will give wrong or confused answer

CONTEXT broken_contra AS (
  WITH SYSTEM "You are a billing support agent."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 400

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 6
    LIMIT TOKENS 600

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 200

  LIMIT TOKENS 1500
)

INFER
  USING CONTEXT broken_contra
  GOAL "What is the return protection period for Priya's Platinum Card?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: FILTER BY ver = "v2025" loads only current policy
-- No contradictions → κ=0

CONTEXT fixed_contra AS (
  WITH SYSTEM "You are a billing support agent."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 400

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 600
  FILTER BY ver = "v2025"

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 200

  LIMIT TOKENS 1500
)

INFER
  USING CONTEXT fixed_contra
  GOAL "What is the return protection period for Priya's Platinum Card?"
  EXPLAIN""",
  },
  { "id":"p4","idx":3,"icon":"📉","metric":"ρ<0.3","label":"WARN",
    "title":"Low Coverage","sub":"Threshold too strict",
    "col":"#a78bfa","bg":"#0e0a1e","bdr":"#4c1d95",
    "what":"WHERE similarity > 0.25 is above the max score any document achieves for this query. Zero chunks are admitted. The LLM gets empty context and hallucinates.",
    "fix":"Lower the threshold to 0.05–0.15 for typical RAG. Always run EXPLAIN and check ρ — if ρ < 0.3, your threshold is too strict.",
    "docs": BENEFITS+OFFERS+CUSTOMER,
    "sources": ALL_SOURCES, "query": DEFAULT_QUERY,
    "goal": DEFAULT_GOAL, "system": DEFAULT_SYSTEM,
    "b_gl":1500,"b_lim":{"amex.benefits":400,"amex.offers":300,"amex.customer":250},
    "f_gl":1500,"f_lim":{"amex.benefits":400,"amex.offers":300,"amex.customer":250},
    "b_thr":0.25,"f_thr":0.05,
    "broken_cql":"""\
-- ❌ PROBLEM: low coverage  ρ=0
-- WHERE similarity > 0.25 is above every document's score
-- All 3 sources → 0 chunks → LLM gets nothing

CONTEXT broken_coverage AS (
  WITH SYSTEM "You are Priya's Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.25
    TOP 7
    LIMIT TOKENS 400

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.25
    TOP 5
    LIMIT TOKENS 300

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.25
    TOP 4
    LIMIT TOKENS 250

  LIMIT TOKENS 1500
)

INFER
  USING CONTEXT broken_coverage
  GOAL "What Amex offers and benefits should Priya use this month?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: threshold 0.05 admits relevant chunks
-- ρ rises — good coverage of available content

CONTEXT fixed_coverage AS (
  WITH SYSTEM "You are Priya's personal Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 400

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 300

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 250

  LIMIT TOKENS 1500
)

INFER
  USING CONTEXT fixed_coverage
  GOAL "What Amex offers and benefits should Priya use this month?"
  EXPLAIN""",
  },
  { "id":"p5","idx":4,"icon":"⚠️","metric":"τ>0.95","label":"WARN",
    "title":"Budget Exhaustion","sub":"Context window almost full",
    "col":"#fbbf24","bg":"#141000","bdr":"#78350f",
    "what":"Global LIMIT TOKENS 190 is consumed to 99% by benefits + offers alone. τ approaches 1.0. No headroom left — if any source adds one more chunk, starvation is immediate.",
    "fix":"Increase global LIMIT TOKENS to give all sources breathing room. Target τ around 0.70–0.80 for safety headroom.",
    "docs": BENEFITS+OFFERS+CUSTOMER,
    "sources": ALL_SOURCES, "query": DEFAULT_QUERY,
    "goal": DEFAULT_GOAL, "system": DEFAULT_SYSTEM,
    "b_gl":190, "b_lim":{},
    "f_gl":2000,"f_lim":{"amex.benefits":400,"amex.offers":300,"amex.customer":250},
    "b_thr":0.05,"f_thr":0.05,
    "broken_cql":"""\
-- ❌ PROBLEM: budget exhaustion  τ=0.99
-- 190-token global budget is almost entirely consumed
-- Any source change → immediate starvation

CONTEXT broken_exhaustion AS (
  WITH SYSTEM "You are Priya's Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 7

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 5

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4

  LIMIT TOKENS 190
)

INFER
  USING CONTEXT broken_exhaustion
  GOAL "What Amex offers and benefits should Priya use this month?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ FIX: generous global budget, τ drops to safe range
-- Target τ 0.70–0.80 for safety headroom

CONTEXT fixed_exhaustion AS (
  WITH SYSTEM "You are Priya's personal Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 5
    LIMIT TOKENS 400

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 300

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 250

  LIMIT TOKENS 2000
)

INFER
  USING CONTEXT fixed_exhaustion
  GOAL "What Amex offers and benefits should Priya use this month?"
  EXPLAIN""",
  },
  { "id":"p6","idx":5,"icon":"💀","metric":"δ+κ+ρ","label":"CRITICAL",
    "title":"The Worst Case","sub":"All three problems firing at once",
    "col":"#94a3b8","bg":"#090910","bdr":"#334155",
    "what":"Threshold 0.25 → ρ=0 (nothing admitted). Budget 45 → starvation. Old + new policy → contradictions. This is what an untested production pipeline running for 3 months looks like.",
    "fix":"Three independent fixes — lower threshold, add per-source LIMIT TOKENS, add FILTER BY version. Each one independently improves one metric.",
    "docs": BENEFITS+OFFERS+POLICY_OLD+POLICY_NEW+CUSTOMER,
    "sources": ["amex.benefits","amex.offers","amex.policy","amex.customer"],
    "query": "amex offers benefits credits policy Priya",
    "goal": "What offers, benefits, and return protection does Priya have?",
    "system": DEFAULT_SYSTEM,
    "b_gl":45,  "b_lim":{},
    "f_gl":2000,"f_lim":{"amex.benefits":350,"amex.offers":280,"amex.policy":300,"amex.customer":250},
    "b_thr":0.25,"f_thr":0.05,
    "b_filter":None,"f_filter":{"ver":"v2025"},
    "broken_cql":"""\
-- ❌ THE WORST CASE: all three problems firing at once
-- threshold 0.25 → ρ=0 (nothing admitted)
-- budget 45 → δ>0 (starvation even if threshold lowered)
-- old+new policy → κ>0.3 (contradictions)
-- 3-month production bug. No error. HTTP 200.

CONTEXT disaster AS (
  WITH SYSTEM "You are a support agent."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.25
    TOP 7

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.25
    TOP 5

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.25
    TOP 6

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.25
    TOP 4

  LIMIT TOKENS 45
)

INFER
  USING CONTEXT disaster
  GOAL "What offers, benefits, and return protection does Priya have?"
  EXPLAIN""",
    "fixed_cql":"""\
-- ✅ THREE FIXES: each addresses one metric
-- (1) threshold 0.05 → ρ rises
-- (2) per-source LIMIT TOKENS → δ=0
-- (3) FILTER BY ver = "v2025" → κ=0

CONTEXT all_fixed AS (
  WITH SYSTEM "You are Priya's personal Amex concierge."

  RETRIEVE FROM amex.benefits
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 350

  RETRIEVE FROM amex.offers
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 280

  RETRIEVE FROM amex.policy
    WHERE similarity > 0.05
    TOP 3
    LIMIT TOKENS 300
  FILTER BY ver = "v2025"

  RETRIEVE FROM amex.customer
    WHERE similarity > 0.05
    TOP 4
    LIMIT TOKENS 250

  LIMIT TOKENS 2000
)

INFER
  USING CONTEXT all_fixed
  GOAL "What offers, benefits, and return protection does Priya have?"
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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,600;0,700;1,400&family=Syne:wght@600;700;800&display=swap');
:root{--bg:#060912;--panel:#0b0f1a;--p2:#0f1520;--bdr:#181f30;--b2:#232d45;
      --txt:#b8cce0;--mid:#4a5a7a;--dim:#1a2135;--blue:#3b82f6;
      --green:#22c55e;--red:#ef4444;--amber:#f59e0b;--purple:#a78bfa;--code:#040810}
.stApp,.main,section.main{background:var(--bg)!important}
html,body{background:var(--bg)}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:#e2e8f0!important}
p,li,label,div,span{font-family:'JetBrains Mono',monospace!important;font-size:12px;color:var(--txt)}
[data-testid="collapsedControl"]{display:none}
.stTabs [data-baseweb="tab-list"]{background:var(--panel);border-bottom:1px solid var(--bdr);padding:0 4px;gap:0}
.stTabs [data-baseweb="tab"]{font-family:'Syne',sans-serif!important;font-size:12.5px!important;font-weight:600!important;color:var(--mid)!important;padding:11px 18px!important;border-bottom:2px solid transparent!important;transition:all .15s}
.stTabs [aria-selected="true"]{color:#e2e8f0!important;font-weight:800!important;border-bottom:2px solid var(--blue)!important;background:transparent!important}
.stTabs [data-baseweb="tab-panel"]{padding-top:0!important;background:transparent}
.stButton>button{background:var(--blue)!important;color:#fff!important;border:none!important;border-radius:6px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:11.5px!important;padding:7px 16px!important;transition:all .15s}
.stButton>button:hover{filter:brightness(1.15)!important}
.stTextArea textarea{background:var(--code)!important;color:#d8e8f8!important;border:1px solid var(--b2)!important;border-radius:6px!important;font-family:'JetBrains Mono',monospace!important;font-size:11.5px!important;line-height:1.8!important}
.stTextArea textarea:focus{border-color:var(--blue)!important;box-shadow:0 0 0 2px #3b82f620!important}
.stTextInput input,.stSelectbox>div>div{background:var(--panel)!important;color:var(--txt)!important;border:1px solid var(--b2)!important;border-radius:6px!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important}
[data-testid="stMetric"]{background:var(--panel)!important;border:1px solid var(--bdr)!important;border-radius:8px!important;padding:10px 14px!important}
[data-testid="stMetricLabel"]{color:var(--mid)!important;font-size:9px!important;text-transform:uppercase;letter-spacing:1.5px;font-family:'Syne',sans-serif!important}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-family:'JetBrains Mono',monospace!important;font-size:20px!important}
.stExpander{background:var(--panel)!important;border:1px solid var(--bdr)!important;border-radius:8px!important}
details summary{color:var(--txt)!important;font-size:11px!important}
hr{border-color:var(--bdr)!important}
.stCheckbox label,.stRadio label{font-size:11px!important;color:var(--mid)!important}
#MainMenu,footer,.stDeployButton{display:none!important}
.sl{font-size:8px;letter-spacing:2.5px;text-transform:uppercase;color:var(--mid);font-family:'Syne',sans-serif;font-weight:700;margin-bottom:6px;display:block}
.qbar{height:4px;background:var(--dim);border-radius:2px;overflow:hidden;margin-top:2px}
.qfill{height:100%;border-radius:2px;transition:width .4s ease}
.chip{background:var(--panel);border:1px solid var(--bdr);border-radius:5px;padding:5px 10px;margin:3px 0;font-size:10px}
.banner{border-radius:10px;padding:12px 16px;margin:6px 0}
.ans{background:var(--panel);border:1px solid var(--bdr);border-radius:8px;padding:14px;font-size:12.5px;line-height:1.9;color:var(--txt)}
.log{background:#030609;border:1px solid #0c1220;border-radius:6px;padding:8px 12px;font-family:'JetBrains Mono',monospace;font-size:10px;line-height:1.9;max-height:110px;overflow-y:auto;margin:5px 0}
.pytest-line{font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.8;padding:2px 0}
.tag{display:inline-block;padding:1px 7px;border-radius:3px;font-size:9px;font-weight:700;margin:1px}
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
        bg  = {"PASS":"#071407","CRITICAL":"#1a0707","WARN":"#1a1100"}.get(rep.status,"#0b0f1a")
        em  = {"PASS":"✅","CRITICAL":"🚨","WARN":"⚠️"}.get(rep.status,"")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 14px;background:{bg};border:1px solid {sc}44;border-radius:8px;margin:3px 0">
          <span style="font-size:15px">{em}</span>
          <span style="font-weight:800;color:{sc};font-size:13px;font-family:Syne,sans-serif">{rep.status}</span>
          <span style="color:#2a3a55;font-size:9px">ρ={rep.rho} · τ={rep.tau:.2f} · δ={rep.delta} · κ={rep.kappa}</span>
          <span style="color:#1e2840;font-size:9px">{rep.tokens_used}/{rep.token_budget} tok</span>
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
    c = "#22c55e" if api_key else "#2a3a55"
    model_name = provider.split(" (")[0].replace("Groq — ","")
    lbl = f"● {model_name} connected" if api_key else "○ No key — Inspector only"
    st.markdown(f'<div style="color:{c};font-size:9px">{lbl}</div>', unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
t_inspect, t_ci, t_pm, t_integrate = st.tabs([
    "🔍  Inspect — Developer / QA",
    "🧪  CI Tests — GitHub Actions",
    "📊  PM View — Business Impact",
    "💻  Integrate — Add to Pipeline",
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
        st.markdown(f'<div style="background:#071407;border:1px solid #14532d;border-radius:7px;padding:8px 11px;font-size:10px;line-height:1.7;color:#4ade80">{P["fix"]}</div>', unsafe_allow_html=True)
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
                c = {"ok":"#22c55e","warn":"#f59e0b","err":"#ef4444"}.get(k,"#2a3a55")
                lh += f'<div style="color:{c}">{m}</div>'
            lh += "</div>"
            st.markdown(lh, unsafe_allow_html=True)

        st.divider()

        # ── CQL Reference ─────────────────────────────────────────────────────
        with st.expander("📖  CQL Reference"):
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
        with st.expander("🧩  Full DSL Examples — Joins · MapReduce · Aggregate"):
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
        with st.expander("📂  Sources"):
            all_d = P["docs"] + st.session_state.extra
            by_s = defaultdict(list)
            for d in all_d: by_s[d["source"]].append(d)
            for src, docs in by_s.items():
                st.markdown(f'<div style="font-size:10px;font-weight:700;color:#60a5fa;margin:5px 0 3px">{src} <span style="color:#2a3a55;font-weight:400">· {len(docs)} docs</span></div>', unsafe_allow_html=True)
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
        ri1, ri2, ri3 = st.tabs(["📊 Inspector", "💬 Answer", "📐 Math"])

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
                    bg={"PASS":"#071407","CRITICAL":"#1a0707","WARN":"#141000"}.get(rep.status,"#0b0f1a")
                    bd={"PASS":"#14532d","CRITICAL":"#7f1d1d","WARN":"#78350f"}.get(rep.status,"#1e2840")
                    st.markdown(f"""
                    <div class="banner" style="background:{bg};border:2px solid {bd}">
                      <div style="display:flex;align-items:center;gap:10px">
                        <span style="font-size:22px">{em}</span>
                        <div>
                          <div style="font-size:17px;font-weight:800;color:{sc};font-family:Syne,sans-serif">{rep.status}</div>
                          <div style="font-size:9px;color:#2a3a55">{rep.tokens_used}/{rep.token_budget} tok</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    for sym,name,val,ok,cb,cg,note in [
                        ("ρ","Coverage",rep.rho,rep.rho>=0.20,"#7c3aed","#a78bfa","% retrieved → admitted"),
                        ("τ","Utilisation",rep.tau,rep.tau<0.90,"#1d4ed8","#60a5fa","tokens used / budget"),
                        ("δ","Starvation",min(rep.delta/max(len(P["sources"]),1),1),rep.delta==0,"#991b1b","#ef4444","sources with 0 chunks"),
                        ("κ","Contradiction",rep.kappa,rep.kappa<0.30,"#92400e","#f59e0b","conflicting fact pairs"),
                    ]:
                        bc=cg if ok else cb
                        av={"ρ":rep.rho,"τ":rep.tau,"δ":rep.delta,"κ":rep.kappa}[sym]
                        st.markdown(f"""
                        <div style="margin:7px 0">
                          <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <div style="display:flex;align-items:baseline;gap:5px">
                              <span style="color:{bc};font-size:14px;font-family:Georgia,serif;font-weight:700">{sym}</span>
                              <span style="color:#2a3a55;font-size:9px">{name}</span>
                            </div>
                            <span style="color:{bc};font-size:12px;font-family:JetBrains Mono,monospace;font-weight:700">{av if sym=='δ' else f'{av:.3f}'}</span>
                          </div>
                          <div class="qbar"><div class="qfill" style="width:{min(100,val*100):.1f}%;background:{bc}"></div></div>
                          <div style="font-size:8px;color:#1e2840;margin-top:1px">{note}</div>
                        </div>""", unsafe_allow_html=True)
                    for w in rep.warnings:
                        wt=warn_type(w); wm=warn_msg(w)
                        wc="#ef4444" if "STARV" in wt else "#f59e0b"
                        st.markdown(f'<div style="background:#0f0a0a;border:1px solid {wc}33;border-radius:5px;padding:7px 10px;margin:5px 0"><span style="color:{wc};font-size:9px;font-weight:700">{wt}</span><br><span style="color:#3a4a5a;font-size:9px">{wm}</span></div>', unsafe_allow_html=True)
                    st.divider()
                    st.markdown('<span class="sl">Source Breakdown</span>', unsafe_allow_html=True)
                    smap = r.get("smap", {})
                    for src in P["sources"]:
                        cs=smap.get(src,[]); ok_s=len(cs)>0
                        sc2="#22c55e" if ok_s else "#ef4444"
                        tt=sum(tok(c["text"]) for c in cs)
                        avg=round(sum(c.get("_sc",0) for c in cs)/max(len(cs),1),3) if cs else 0
                        st.markdown(f"""
                        <div class="chip" style="border-color:{'#14532d' if ok_s else '#7f1d1d'}">
                          <div style="display:flex;align-items:center;gap:7px">
                            <span style="color:{sc2};font-size:13px">{"✓" if ok_s else "✗"}</span>
                            <div style="flex:1">
                              <div style="color:{sc2};font-weight:700;font-size:10px">{src}</div>
                              <div style="color:#2a3a55;font-size:9px">{len(cs)} chunks · {tt} tok · sim̄={avg}</div>
                            </div>
                            {"<span style='color:#ef4444;font-size:9px;font-weight:700'>STARVED</span>" if not ok_s else ""}
                          </div>
                        </div>""", unsafe_allow_html=True)
                        for c in cs[:2]:
                            st.markdown(f'<div style="padding:2px 4px 2px 24px;font-size:9px;color:#1e2840;border-left:1px solid #0f1520;margin:1px 0">{c["text"][:72]}…</div>', unsafe_allow_html=True)
                        if len(cs)>2: st.markdown(f'<div style="padding:1px 4px 1px 24px;font-size:9px;color:#151d2e">+{len(cs)-2} more</div>', unsafe_allow_html=True)

        # ── Answer ────────────────────────────────────────────────────────────
        with ri2:
            r=st.session_state.result
            if not r: st.markdown('<div style="text-align:center;padding:50px 10px;color:#1a2535;font-size:11px">Run a query to see the answer</div>', unsafe_allow_html=True)
            elif r.get("inspect_only"): st.info("Inspector only — uncheck to get LLM answer.")
            elif r.get("no_key"): st.markdown('<div style="background:#0a0e1a;border:1px solid #1a2035;border-radius:8px;padding:18px;font-size:11px;color:#2a3a55;text-align:center">Add API key (top-right) to see the LLM answer</div>', unsafe_allow_html=True)
            elif r.get("api_error"): st.error(r["api_error"])
            else:
                rep=r.get("report"); ans=r.get("answer","")
                if rep:
                    sc={"PASS":"#22c55e","CRITICAL":"#ef4444","WARN":"#f59e0b"}.get(rep.status,"#64748b")
                    bg={"PASS":"#071407","CRITICAL":"#1a0707","WARN":"#141000"}.get(rep.status,"#0b0f1a")
                    st.markdown(f'<div style="margin-bottom:7px"><span style="background:{bg};color:{sc};border:1px solid {sc}33;padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700">{rep.status}</span> <span style="color:#1e2840;font-size:9px">δ={rep.delta}</span></div>', unsafe_allow_html=True)
                    if rep.delta>0:
                        st.markdown(f'<div style="background:#1a0707;border:1px solid #7f1d1d33;border-radius:6px;padding:9px 12px;font-size:10px;color:#f87171;margin-bottom:8px">⚠ Starvation δ={rep.delta} — missing: <b>{", ".join(rep.sources_starved)}</b><br><span style="color:#3a2020">Answer lacks customer/offer context.</span></div>', unsafe_allow_html=True)
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
                '<div style="background:#0a0616;border:2px solid #4c1d95;border-radius:8px;'
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
                '<div style="background:#140e04;border:1px solid #f59e0b44;border-radius:6px;'
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
            <div style="background:#030609;border:1px solid #0c1220;border-radius:8px;padding:12px 16px;font-family:'JetBrains Mono',monospace">
              <div style="font-size:10px;color:#2a3a55;margin-bottom:6px">platform linux — Python 3.9 — pytest 7.x  ·  {pipeline_label} pipeline</div>
              <div style="font-size:10px;color:#2a3a55;margin-bottom:8px">collected {len(results)} items</div>
              <div style="border-top:1px solid #0c1220;margin:6px 0;padding-top:8px">""", unsafe_allow_html=True)

            for r in results:
                p = r["passed"]
                status_html = '<span style="color:#22c55e;font-weight:700">PASSED</span>' if p else '<span style="color:#ef4444;font-weight:700">FAILED</span>'
                st.markdown(f"""
                <div class="pytest-line" style="color:{'#3a4a5a' if p else '#e2e8f0'}">
                  {status_html} &nbsp; {r['name']} &nbsp;
                  <span style="color:#1e2840;font-size:9px">({r['elapsed']:.2f}ms)</span>
                </div>
                {f'<div style="color:#fbbf24;font-size:9px;padding-left:12px;margin-bottom:3px">    AssertionError: {r["fail_msg"]}</div>' if not p and r.get("fail_msg") else ""}
                """, unsafe_allow_html=True)

            st.markdown(f"""
              </div>
              <div style="border-top:1px solid #0c1220;margin-top:8px;padding-top:8px;font-size:11px;font-weight:700;color:{label_col}">
                {label} &nbsp; in {total_ms:.1f}ms &nbsp; <span style="color:#1e2840;font-size:9px">(no LLM calls)</span>
              </div>
            </div>""", unsafe_allow_html=True)

            if n_fail:
                st.markdown(f"""
                <div style="background:#1a0707;border:1px solid #7f1d1d44;border-radius:8px;padding:12px;margin-top:10px;font-size:11px;color:#f87171">
                  <b>Broken pipeline failed {n_fail} tests.</b><br>
                  Load the ✅ Fixed CQL template and re-run — all tests will pass.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#071407;border:1px solid #14532d44;border-radius:8px;padding:12px;margin-top:10px;font-size:11px;color:#4ade80">
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

        with st.spinner("Running broken pipeline…"):
            smap_b = assemble(docs, query, P_pm["b_gl"], P_pm["b_lim"], P_pm["b_thr"])
            rep_b  = run_inspector(smap_b, P_pm["b_gl"], query, sources)
            ans_b  = call_llm(provider, system, smap_b, goal, api_key) if api_key else None

        with st.spinner("Running fixed pipeline…"):
            smap_f = assemble(docs, query, P_pm["f_gl"], P_pm["f_lim"], P_pm["f_thr"])
            rep_f  = run_inspector(smap_f, P_pm["f_gl"], query, sources)
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
            <div style="background:#200808;border:2px solid #7f1d1d;border-radius:10px;padding:14px 18px;margin-bottom:12px">
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
                st.markdown(f'<div style="background:#1a0707;border:1px solid #7f1d1d44;border-radius:6px;padding:10px;font-size:10px;color:#f87171;margin-top:10px">Inspector: <b>{rep_b.status if rep_b else "N/A"}</b> δ={rep_b.delta if rep_b else "?"}<br><br>With API key: see the generic, wrong answer the customer would receive.</div>', unsafe_allow_html=True)

        with pc2:
            rep_f = pm["rep_f"]
            st.markdown("""
            <div style="background:#071407;border:2px solid #14532d;border-radius:10px;padding:14px 18px;margin-bottom:12px">
              <div style="font-size:18px;margin-bottom:6px">✅</div>
              <div style="font-family:Syne,sans-serif;font-weight:800;font-size:14px;color:#22c55e">With OpenCQL</div>
              <div style="font-size:10px;color:#1a4020;margin-top:3px">Managed context — all sources guaranteed</div>
            </div>""", unsafe_allow_html=True)

            if rep_f:
                for src in sources:
                    cs = pm["smap_f"].get(src,[])
                    ok = bool(cs)
                    c = "#22c55e" if ok else "#ef4444"
                    tt = sum(tok(c["text"]) for c in cs)
                    st.markdown(f'<div style="font-size:10px;color:{c};padding:3px 0">{"✓" if ok else "✗"} {src} → {len(cs)} chunks · {tt} tok</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:10px;color:#1a4020;margin-top:6px">δ={rep_f.delta} · τ={rep_f.tau:.2f} · {rep_f.tokens_used}/{rep_f.token_budget} tok</div>', unsafe_allow_html=True)

            if pm.get("ans_f"):
                st.markdown('<div style="font-size:10px;color:#1a4020;margin:10px 0 4px">Customer sees:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ans" style="border-color:#14532d33;font-size:12px">{pm["ans_f"]}</div>', unsafe_allow_html=True)
                st.markdown('<div style="color:#22c55e;font-size:10px;margin-top:6px">✓ Personalised — mentions Priya by name, her specific offers, her remaining credits.</div>', unsafe_allow_html=True)
            elif not api_key:
                st.markdown(f'<div style="background:#071407;border:1px solid #14532d44;border-radius:6px;padding:10px;font-size:10px;color:#4ade80;margin-top:10px">Inspector: <b>{rep_f.status if rep_f else "N/A"}</b> δ={rep_f.delta if rep_f else "?"}<br><br>With API key: see the personalised answer mentioning Priya\'s specific offers and credits.</div>', unsafe_allow_html=True)

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
# RUN HANDLER
# ════════════════════════════════════════════════════════════════════════════════
if run_btn:
    cql  = st.session_state.cql
    logs = []
    def log(m, k="info"): logs.append((k, m))

    log("Parsing CQL parameters…")

    # Parse from CQL text
    gl_m = re.findall(r'\bLIMIT\s+TOKENS\s+(\d+)', cql, re.I)
    global_limit = int(gl_m[-1]) if gl_m else P["f_gl"]

    per_limits = {}
    for m in re.finditer(r'RETRIEVE\s+FROM\s+([\w.]+)(.*?)(?=RETRIEVE|INFER|\Z)', cql, re.I|re.S):
        src, blk = m.group(1), m.group(2)
        lm = re.search(r'LIMIT\s+TOKENS\s+(\d+)', blk, re.I)
        if lm: per_limits[src] = int(lm.group(1))

    thr_m = re.search(r'WHERE\s+similarity\s*[><=!]+\s*([\d.]+)', cql, re.I)
    threshold = float(thr_m.group(1)) if thr_m else 0.05

    flt_m = re.search(r'FILTER\s+BY\s+(\w+)\s*=\s*"([^"]+)"', cql, re.I)
    doc_filter = {flt_m.group(1): flt_m.group(2)} if flt_m else None

    goal_m = re.search(r'GOAL\s+"([^"]+)"', cql, re.I)
    goal_txt = goal_override.strip() if 'goal_ov' in st.session_state and st.session_state.goal_ov else (goal_m.group(1) if goal_m else P["goal"])

    log(f"global={global_limit}tok · thr={threshold} · per={per_limits or 'none'}{' · filter='+str(doc_filter) if doc_filter else ''}")

    all_docs = P["docs"] + st.session_state.extra
    smap = assemble(all_docs, P["query"], global_limit,
                    per_limits if per_limits else None,
                    threshold, doc_filter)

    for src in P["sources"]:
        cs = smap.get(src, [])
        tt = sum(tok(c["text"]) for c in cs)
        log(f"{'✓' if cs else '✗'} {src} → {len(cs)} chunks · {tt} tok", "ok" if cs else "warn")

    report = run_inspector(smap, global_limit, P["query"], P["sources"])
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

    st.session_state.result = {"report": report, "smap": smap, "answer": answer, **extra}
    st.rerun()

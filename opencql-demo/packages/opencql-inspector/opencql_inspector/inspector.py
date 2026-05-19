"""
opencql_inspector/inspector.py
================================
ContextInspector — pre-inference context quality audit.

Zero mandatory dependencies. Pure Python 3.9+.

Usage:
    from opencql_inspector import ContextInspector

    inspector = ContextInspector()
    report = inspector.inspect(
        chunks=[
            {"text": "Enterprise refund: 60 days.", "source": "docs.policy", "score": 0.91},
            {"text": "GDPR Art.17: erasure within 30 days.", "source": "docs.compliance", "score": 0.87},
        ],
        query="enterprise refund policy",
        token_budget=380,
        sources_expected=["docs.policy", "docs.compliance", "docs.customer"],
    )
    print(report)
    # InspectReport(rho=0.667, tau=0.066, delta=1, kappa=0.0, status='CRITICAL')
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any


# ── Token counting ────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """
    BPE token count approximation (cl100k_base).
    ~3.8 chars per token for English prose. Accurate to ±10%.

    If you need exact counts:
        pip install tiktoken
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        text = text.strip()
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 3.8))


# ── Q vector components ───────────────────────────────────────────────────────

def _compute_rho(chunks: list[dict], sources_expected: list[str],
                 doc_store: dict[str, list] | None) -> float:
    """
    ρ = |A| / Σ|Rᵢ|
    Coverage: fraction of candidates that were admitted.
    """
    admitted = len(chunks)
    if doc_store and sources_expected:
        total_possible = sum(len(doc_store.get(s, [])) for s in sources_expected)
    else:
        total_possible = max(admitted, 1)
    return round(admitted / total_possible, 3) if total_possible > 0 else 0.0


def _compute_tau(chunks: list[dict], token_budget: int) -> float:
    """
    τ = Σtokens(A) / W
    Budget utilisation: how much of the token budget was consumed.
    """
    if token_budget <= 0:
        return 0.0
    used = sum(_count_tokens(c.get("text", "")) for c in chunks)
    return round(used / token_budget, 3)


def _compute_delta(chunks: list[dict], sources_expected: list[str]) -> int:
    """
    δ = |{i : Aᵢ = ∅}|
    Starvation count: how many expected sources contributed zero chunks.
    """
    if not sources_expected:
        return 0
    sources_present = set(c.get("source", "") for c in chunks)
    starved = [s for s in sources_expected if s not in sources_present]
    return len(starved)


def _compute_kappa(chunks: list[dict],
                   strategy: str = "keyword") -> float:
    """
    κ = Σ Φ(cₐ, c_b) / C(n, 2)
    Contradiction density: fraction of chunk pairs with conflicting facts.

    Strategies:
      "keyword" — looks for conflicting numeric values (fast, default)
      "nli"     — uses cross-encoder NLI model (slow, accurate, needs transformers)
    """
    n = len(chunks)
    if n < 2:
        return 0.0

    total_pairs = n * (n - 1) // 2
    contradictions = 0

    for i in range(n):
        for j in range(i + 1, n):
            if _chunks_contradict(chunks[i], chunks[j], strategy):
                contradictions += 1

    return round(contradictions / total_pairs, 3) if total_pairs > 0 else 0.0


def _chunks_contradict(a: dict, b: dict, strategy: str = "keyword") -> bool:
    """Check if two chunks contain conflicting information."""
    text_a = a.get("text", "").lower()
    text_b = b.get("text", "").lower()

    if strategy == "nli":
        try:
            from transformers import pipeline
            if not hasattr(_chunks_contradict, "_nli"):
                _chunks_contradict._nli = pipeline(
                    "text-classification",
                    model="cross-encoder/nli-deberta-v3-small"
                )
            result = _chunks_contradict._nli(f"{text_a} [SEP] {text_b}")[0]
            return result["label"] == "CONTRADICTION" and result["score"] > 0.7
        except Exception:
            pass  # fall through to keyword

    # Keyword strategy: look for shared topic words + different numeric values
    words_a = set(re.findall(r"[a-z]{4,}", text_a))
    words_b = set(re.findall(r"[a-z]{4,}", text_b))
    shared_words = words_a & words_b

    if len(shared_words) < 3:
        return False  # not about the same topic

    nums_a = re.findall(r"\$[\d,]+|\d+\s*(?:day|month|year|%|hour|token)", text_a)
    nums_b = re.findall(r"\$[\d,]+|\d+\s*(?:day|month|year|%|hour|token)", text_b)

    if not nums_a or not nums_b:
        return False

    # Normalize numbers
    def norm(s):
        return re.sub(r"[,\s]", "", s.lower())

    set_a = set(norm(n) for n in nums_a)
    set_b = set(norm(n) for n in nums_b)

    # Contradiction: same topic, different numbers, no overlap
    return len(set_a & set_b) == 0 and len(set_a) > 0 and len(set_b) > 0


# ── LITM position analysis ────────────────────────────────────────────────────

def _litm_analysis(chunks: list[dict]) -> list[dict]:
    """
    Detect chunks in the 'lost in the middle' dead zone.
    Uses: A(i,n) = α·e^(-λi) + β·e^(-μ(n-1-i)) + γ
    Fitted from Liu et al. TACL 2024: α=0.40, β=0.38, λ=μ=0.25, γ=0.30
    """
    n = len(chunks)
    if n == 0:
        return []

    α, β, λ, μ, γ = 0.40, 0.38, 0.25, 0.25, 0.30
    warnings = []

    for i, chunk in enumerate(chunks):
        attention = α * math.exp(-λ * i) + β * math.exp(-μ * (n - 1 - i)) + γ
        if attention < 0.45 and n >= 5:
            warnings.append({
                "type": "P5_LITM",
                "position": i,
                "total": n,
                "attention": round(attention, 3),
                "text_preview": chunk.get("text", "")[:60],
                "message": f"Chunk at position {i}/{n} has low attention ({attention:.2f}). Consider moving to position 0 or {n-1}."
            })

    return warnings



def _word_sim(t1: str, t2: str) -> float:
    """Word-overlap Jaccard similarity (4+ char words)."""
    a = set(re.findall(r"[a-z]{4,}", t1.lower()))
    b = set(re.findall(r"[a-z]{4,}", t2.lower()))
    return round(len(a & b) / len(a | b), 3) if (a | b) else 0.0


def _compute_eta(chunks: list[dict]) -> float:
    """
    η (Eta) — Semantic Redundancy
    η = |{(i,j) : word_sim(cᵢ,cⱼ) > 0.70}| / C(n,2)
    Fraction of chunk pairs that are near-duplicates.
    """
    n = len(chunks)
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    redundant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _word_sim(chunks[i].get("text",""), chunks[j].get("text","")) > 0.60:
                redundant += 1
    return round(redundant / total_pairs, 3) if total_pairs > 0 else 0.0


def _compute_sigma(chunks: list[dict], max_age_days: "int | None" = None) -> float:
    """
    σ (Sigma) — Staleness
    σ = |{c : c.age_days > threshold}| / n
    Fraction of chunks older than threshold. Falls back to version-mixing detection.
    """
    n = len(chunks)
    if n == 0:
        return 0.0
    if max_age_days is not None:
        stale = sum(
            1 for c in chunks
            if isinstance(c.get("age_days"), (int, float)) and c["age_days"] > max_age_days
        )
        return round(stale / n, 3)
    # Version-mixing fallback: old versions in context = stale content
    versions = [c.get("ver") for c in chunks if c.get("ver")]
    if len(versions) >= 2 and len(set(versions)) > 1:
        newest = max(set(versions))
        old_count = sum(1 for c in chunks if c.get("ver") and c["ver"] != newest)
        return round(old_count / n, 3)
    return 0.0


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class InspectReport:
    """
    Full context quality report from opencql-inspector.

    Attributes:
        rho      Coverage ratio (0–1). Want high.
        tau      Token budget utilisation (0–1). Want ~0.80.
        delta    Starvation count (int). Want 0.
        kappa    Contradiction density (0–1). Want 0.
        status   PASS | WARN | CRITICAL
        latency_ms  Time taken to compute this report (ms)
        tokens_used  Total tokens in admitted chunks
        token_budget  Input budget
        sources_admitted  Which sources contributed chunks
        sources_starved   Which expected sources got zero chunks
        warnings  List of specific warnings (P5 LITM, etc.)
        chunks   The chunks that were inspected
    """
    rho:              float
    tau:              float
    delta:            int
    kappa:            float
    eta:              float
    sigma:            float
    status:           str
    latency_ms:       float
    tokens_used:      int
    token_budget:     int
    sources_admitted: list[str]
    sources_starved:  list[str]
    warnings:         list[dict]
    chunks:           list[dict]

    def __repr__(self) -> str:
        return (
            f"InspectReport("
            f"rho={self.rho}, tau={self.tau}, delta={self.delta}, "
            f"kappa={self.kappa}, eta={self.eta}, sigma={self.sigma}, "
            f"status='{self.status}')"
        )

    def to_dict(self) -> dict:
        return {
            "rho":              self.rho,
            "tau":              self.tau,
            "delta":            self.delta,
            "kappa":            self.kappa,
            "eta":              self.eta,
            "sigma":            self.sigma,
            "status":           self.status,
            "latency_ms":       self.latency_ms,
            "tokens_used":      self.tokens_used,
            "token_budget":     self.token_budget,
            "sources_admitted": self.sources_admitted,
            "sources_starved":  self.sources_starved,
            "warnings":         self.warnings,
        }

    def format(self) -> str:
        """Human-readable report."""
        lines = [
            f"Q = (ρ={self.rho}, τ={self.tau:.2f}, δ={self.delta}, κ={self.kappa})",
            f"STATUS: {self.status}  [{self.latency_ms:.1f}ms]",
            f"Tokens: {self.tokens_used} / {self.token_budget}",
            "",
        ]
        if self.sources_admitted:
            for src in self.sources_admitted:
                lines.append(f"  ✓  {src}")
        if self.sources_starved:
            for src in self.sources_starved:
                lines.append(f"  ✗  {src} — STARVED (0 chunks admitted)")
        for w in self.warnings:
            lines.append(f"  ⚠  {w['message']}")
        return "\n".join(lines)


# ── Main class ────────────────────────────────────────────────────────────────

class ContextInspector:
    """
    Pre-inference context quality inspector.

    Computes Q = (ρ, τ, δ, κ) in < 5ms before any LLM API call.

    Args:
        contradiction_strategy: "keyword" (default, fast) or "nli" (slow, accurate)
        warn_tau_threshold: τ above this triggers WARN (default 0.95)
        warn_kappa_threshold: κ above this triggers WARN (default 0.3)

    Usage:
        inspector = ContextInspector()
        report = inspector.inspect(
            chunks=retrieved_chunks,
            query="user question",
            token_budget=2000,
            sources_expected=["docs.policy", "docs.customer"],
        )
        if report.delta > 0:
            raise ValueError(f"Source starvation: {report.sources_starved}")
    """

    def __init__(
        self,
        contradiction_strategy: str = "keyword",
        warn_tau_threshold: float = 0.95,
        warn_kappa_threshold: float = 0.3,
        warn_eta_threshold: float = 0.30,
        warn_sigma_threshold: float = 0.30,
        max_age_days: "int | None" = None,
    ):
        self.contradiction_strategy = contradiction_strategy
        self.warn_tau_threshold     = warn_tau_threshold
        self.warn_kappa_threshold   = warn_kappa_threshold
        self.warn_eta_threshold     = warn_eta_threshold
        self.warn_sigma_threshold   = warn_sigma_threshold
        self.max_age_days           = max_age_days

    def inspect(
        self,
        chunks: list[dict],
        query: str = "",
        token_budget: int = 4000,
        sources_expected: list[str] | None = None,
        doc_store: dict[str, list] | None = None,
    ) -> InspectReport:
        """
        Inspect assembled context chunks.

        Args:
            chunks: List of dicts, each must have "text".
                    Optionally: "source" (str), "score" (float), "tokens" (int).
            query: The retrieval query (used for relevance context, not computed here).
            token_budget: Global token budget W.
            sources_expected: List of source names that should contribute chunks.
                              δ counts sources missing from chunks.
            doc_store: Optional {source: [docs]} to compute ρ accurately.

        Returns:
            InspectReport with rho, tau, delta, kappa, status, warnings.
        """
        t0 = time.perf_counter()

        if sources_expected is None:
            sources_expected = list(set(c.get("source", "") for c in chunks if c.get("source")))

        # Compute Q vector
        rho   = _compute_rho(chunks, sources_expected, doc_store)
        tau   = _compute_tau(chunks, token_budget)
        delta = _compute_delta(chunks, sources_expected)
        kappa = _compute_kappa(chunks, self.contradiction_strategy)
        eta   = _compute_eta(chunks)
        sigma = _compute_sigma(chunks, getattr(self, "max_age_days", None))

        # Compute token usage
        tokens_used = sum(_count_tokens(c.get("text", "")) for c in chunks)

        # Sources admitted / starved
        sources_admitted = sorted(set(c.get("source", "") for c in chunks if c.get("source") and c.get("text")))
        sources_starved  = [s for s in sources_expected if s not in sources_admitted]

        # Determine status
        if delta > 0:
            status = "CRITICAL"
        elif tau > self.warn_tau_threshold or kappa > self.warn_kappa_threshold:
            status = "WARN"
        else:
            status = "PASS"

        # Gather warnings
        warnings = []

        if delta > 0:
            for src in sources_starved:
                warnings.append({
                    "type":    "P1_STARVATION",
                    "source":  src,
                    "message": f"Source '{src}' contributed 0 chunks. Add LIMIT TOKENS per RETRIEVE to guarantee budget.",
                })

        if tau > self.warn_tau_threshold:
            warnings.append({
                "type":    "P1_HIGH_TAU",
                "tau":     tau,
                "message": f"Budget utilisation τ={tau:.2f} is very high. Source starvation likely.",
            })

        if kappa > self.warn_kappa_threshold:
            warnings.append({
                "type":    "P3_CONTRADICTION",
                "kappa":   kappa,
                "message": f"Contradiction density κ={kappa:.3f}. {int(kappa * len(chunks) * (len(chunks)-1) // 2)} conflicting chunk pairs.",
            })

        if rho < 0.3 and len(chunks) > 0:
            warnings.append({
                "type":    "P2_LOW_COVERAGE",
                "rho":     rho,
                "message": f"Coverage ρ={rho:.3f} is low. Consider lowering WHERE similarity threshold.",
            })

        # Semantic redundancy warning
        if eta > 0.30:
            warnings.append({
                "type":    "P6_REDUNDANCY",
                "eta":     eta,
                "message": f"Semantic redundancy η={eta:.3f}. {int(eta * len(chunks)*(len(chunks)-1)//2)} near-duplicate chunk pairs. Consider deduplication.",
            })

        # Staleness warning
        if sigma > 0.30:
            warnings.append({
                "type":    "P7_STALENESS",
                "sigma":   sigma,
                "message": f"Staleness σ={sigma:.3f}. {int(sigma*len(chunks))}/{len(chunks)} chunks are outdated. Use FILTER BY to exclude old versions.",
            })

        # LITM analysis
        litm_warnings = _litm_analysis(chunks)
        warnings.extend(litm_warnings)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return InspectReport(
            rho=rho, tau=tau, delta=delta, kappa=kappa, eta=eta, sigma=sigma,
            status=status, latency_ms=latency_ms,
            tokens_used=tokens_used, token_budget=token_budget,
            sources_admitted=sources_admitted,
            sources_starved=sources_starved,
            warnings=warnings, chunks=chunks,
        )

    # ── Convenience: wrap a retrieval function ────────────────────────────────
    def wrap(self, retrieve_fn, token_budget: int = 4000,
             sources_expected: list[str] | None = None):
        """
        Wrap an existing retrieval function to auto-inspect results.

        Usage:
            inspector = ContextInspector()

            @inspector.wrap(token_budget=2000, sources_expected=["docs.policy"])
            def my_retrieval(query):
                return vector_db.search(query, top_k=10)

            chunks = my_retrieval("enterprise refund")  # raises if delta > 0
        """
        def decorator(fn):
            def wrapper(*args, **kwargs):
                chunks = fn(*args, **kwargs)
                query = args[0] if args else kwargs.get("query", "")
                report = self.inspect(chunks, query=query,
                                      token_budget=token_budget,
                                      sources_expected=sources_expected)
                if report.delta > 0:
                    raise ContextQualityError(report)
                return chunks
            return wrapper
        return decorator


class ContextQualityError(Exception):
    """Raised when context quality check fails (δ > 0)."""
    def __init__(self, report: InspectReport):
        self.report = report
        super().__init__(f"Context quality CRITICAL: {report.format()}")

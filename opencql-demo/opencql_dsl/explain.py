"""
opencql_explain / context_inspector.py
=======================================
A zero-dependency observation layer that works with ANY existing
RAG pipeline \u2014 LangChain, LlamaIndex, raw OpenAI, custom code.

No DSL. No grammar. No lark. No compiler.
Drop it into existing code in four lines.

USAGE \u2014 explicit recording:
    from context_inspector import ContextInspector

    inspector = ContextInspector()

    with inspector.trace("support_pipeline", goal="What is the refund policy?") as ctx:
        docs = my_vectordb.search(query, k=5)
        ctx.record_retrieve("docs.policy", docs, threshold=0.7)

        history = db.get_history(user_id)
        ctx.record_history(history, requested=5)

        ctx.record_system("You are a helpful support agent.")
        ctx.record_limit(token_budget=2000)

    print(inspector.explain())

USAGE \u2014 wrapper pattern:
    inspector = ContextInspector()

    @inspector.watch("docs.policy", threshold=0.7)
    def get_policy_docs(query):
        return pinecone.query(query, top_k=5)

    docs = get_policy_docs("refund policy")  # automatically recorded
    print(inspector.explain())

INPUT FORMAT for results in record_retrieve / record_join:
    Accepts any of these formats \u2014 pick whatever your system produces:

    # Format 1: list of (text, score) tuples
    [("chunk text here", 0.91), ("another chunk", 0.84)]

    # Format 2: list of dicts with 'text' and optional 'score'
    [{"text": "chunk text", "score": 0.91, "metadata": {...}}]

    # Format 3: list of plain strings (score assumed 1.0)
    ["chunk text here", "another chunk"]

    # Format 4: LangChain Documents (duck-typed)
    [Document(page_content="...", metadata={"score": 0.91})]
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from opencql_dsl.explain import (
    ExplainReport, SourceRecord, ChunkRecord, ChunkStatus,
    BudgetRecord, ContradictionRecord, Warning, WarnLevel,
)
from opencql_dsl.explain_formatter import ExplainFormatter


# \u2500\u2500 Token counting \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. No tiktoken dependency."""
    return max(1, len(text) // 4)


def _preview(text: str, max_len: int = 120) -> str:
    return text.replace("\
", " ").strip()[:max_len]


# \u2500\u2500 Input normaliser \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

@dataclass
class _NormalisedChunk:
    text:  str
    score: float


def _normalise(results: list[Any]) -> list[_NormalisedChunk]:
    """
    Accept any of the four input formats and return a uniform list.
    Handles LangChain Documents, dicts, tuples, and plain strings.
    """
    out = []
    for item in results:
        # LangChain / LlamaIndex Document (duck-typed \u2014 no import needed)
        if hasattr(item, "page_content"):
            score = (item.metadata or {}).get("score", 1.0)
            out.append(_NormalisedChunk(text=item.page_content, score=float(score)))

        # Dict with 'text' key
        elif isinstance(item, dict) and "text" in item:
            score = float(item.get("score", item.get("similarity", 1.0)))
            out.append(_NormalisedChunk(text=item["text"], score=score))

        # (text, score) tuple
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            text, score = item
            out.append(_NormalisedChunk(text=str(text), score=float(score)))

        # Plain string
        elif isinstance(item, str):
            out.append(_NormalisedChunk(text=item, score=1.0))

        else:
            # Best effort
            out.append(_NormalisedChunk(text=str(item), score=1.0))

    return out


# \u2500\u2500 Contradiction detector \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

# Pairs of opposing signals. If both words from a pair appear in the
# same chunk cluster on the same topic keyword, flag as a contradiction.
_OPPOSING_PAIRS = [
    ({"refund", "refunds", "reimburs"}, {"final", "no refund", "non-refundable", "not refundable"}),
    ({"available", "allowed", "permitted", "eligible"}, {"not available", "prohibited", "forbidden", "ineligible"}),
    ({"required", "mandatory", "must"}, {"optional", "not required", "voluntary"}),
    ({"always", "guaranteed"}, {"never", "not guaranteed", "no guarantee"}),
    ({"free", "no cost", "complimentary"}, {"paid", "fee", "charge", "cost"}),
]

def _detect_contradictions(chunks: list[ChunkRecord]) -> list[ContradictionRecord]:
    """
    Scan all included chunk pairs for opposing signals on the same topic.
    Simple heuristic \u2014 fast, no LLM needed, catches the obvious cases.
    """
    included = [c for c in chunks if c.status == ChunkStatus.INCLUDED]
    found = []

    for i, ca in enumerate(included):
        for cb in included[i + 1:]:
            combined = (ca.full_text + " " + cb.full_text).lower()

            for positive_set, negative_set in _OPPOSING_PAIRS:
                # Check if both sides of the pair appear across the two chunks
                ca_lower = ca.full_text.lower()
                cb_lower = cb.full_text.lower()

                has_pos_a = any(p in ca_lower for p in positive_set)
                has_neg_b = any(n in cb_lower for n in negative_set)
                has_pos_b = any(p in cb_lower for p in positive_set)
                has_neg_a = any(n in ca_lower for n in negative_set)

                conflict = (has_pos_a and has_neg_b) or (has_pos_b and has_neg_a)

                if conflict:
                    # Find the topic keyword \u2014 the word that appears in both
                    topic = next(
                        (w for w in positive_set if w in combined),
                        "policy"
                    )
                    confidence = 0.75 + (0.15 * min(ca.score, cb.score))
                    found.append(ContradictionRecord(
                        chunk_a_rank=ca.rank,
                        chunk_b_rank=cb.rank,
                        chunk_a_preview=ca.text_preview,
                        chunk_b_preview=cb.text_preview,
                        topic_hint=topic,
                        confidence=round(min(confidence, 0.97), 2),
                    ))
                    break  # one contradiction per pair is enough

    return found


# \u2500\u2500 Trace context object \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class TraceContext:
    """
    The object yielded by inspector.trace(). User calls record_* on this.
    Collects all observations and builds the ExplainReport on __exit__.
    """

    def __init__(
        self,
        context_name: str,
        goal: str,
        model: str,
        dry_run: bool,
    ):
        self._report = ExplainReport(
            context_name=context_name,
            goal=goal,
            model=model,
            dry_run=dry_run,
        )
        # Running totals used to enforce token budget at finalize time
        self._token_budget:   Optional[int] = None
        self._chunk_budget:   Optional[int] = None
        self._all_chunks:     list[ChunkRecord] = []   # flat, across all sources
        self._source_map:     dict[str, SourceRecord] = {}

    # \u2500\u2500 Recording API \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def record_system(self, prompt: str):
        """Record the system prompt."""
        self._report.system_prompt = prompt

    def record_retrieve(
        self,
        source: str,
        results: list[Any],
        *,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        all_candidates: Optional[list[Any]] = None,
    ):
        """
        Record a retrieval pass.

        source      \u2014 name of the source e.g. "docs.policy"
        results     \u2014 the chunks that PASSED your threshold (any format)
        threshold   \u2014 the similarity threshold you applied (0.0 = none)
        top_k       \u2014 how many you asked for (inferred from len(results) if omitted)
        all_candidates \u2014 if you have the full candidate list before threshold
                         filtering, pass it here for richer reporting
        """
        self._record_source(
            source=source,
            operation="RETRIEVE",
            results=results,
            threshold=threshold,
            top_k=top_k,
            all_candidates=all_candidates,
        )

    def record_join(
        self,
        source: str,
        results: list[Any],
        *,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
    ):
        """Record a JOIN pass \u2014 same as record_retrieve but labelled JOIN."""
        self._record_source(
            source=source,
            operation="JOIN",
            results=results,
            threshold=threshold,
            top_k=top_k,
        )

    def record_history(
        self,
        turns: list[Any],
        *,
        requested: int = 0,
    ):
        """
        Record conversation history injection.

        turns     \u2014 the turns actually included (list of anything)
        requested \u2014 how many turns were requested (e.g. LAST 5 TURNS)
        """
        self._report.history_turns_included  = len(turns)
        self._report.history_turns_requested = requested or len(turns)

    def record_limit(
        self,
        *,
        token_budget: Optional[int] = None,
        chunk_budget: Optional[int] = None,
    ):
        """
        Declare the token or chunk budget you are enforcing.
        Call this AFTER all record_retrieve/record_join calls so the
        inspector can correctly mark which chunks got trimmed.
        """
        self._token_budget = token_budget
        self._chunk_budget = chunk_budget

    # \u2500\u2500 Internal \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _record_source(
        self,
        source: str,
        operation: str,
        results: list[Any],
        threshold: float,
        top_k: Optional[int],
        all_candidates: Optional[list[Any]] = None,
    ):
        normalised = _normalise(results)
        candidates = _normalise(all_candidates) if all_candidates else normalised
        top_k = top_k or len(normalised)

        sr = SourceRecord(
            source=source,
            operation=operation,
            threshold=threshold,
            requested_top_k=top_k,
            candidates_evaluated=len(candidates),
        )

        rank = 1
        for nc in normalised:
            tokens = _count_tokens(nc.text)
            cr = ChunkRecord(
                rank=rank,
                score=round(nc.score, 4),
                tokens=tokens,
                text_preview=_preview(nc.text),
                source=source,
                status=ChunkStatus.INCLUDED,  # trimming applied later in _finalise
                full_text=nc.text,
            )
            sr.chunks.append(cr)
            self._all_chunks.append(cr)
            rank += 1

        # Record candidates that were below threshold (if caller provided them)
        if all_candidates:
            included_texts = {nc.text for nc in normalised}
            dropped_rank = rank
            for nc in candidates:
                if nc.text not in included_texts and nc.score < threshold:
                    cr = ChunkRecord(
                        rank=dropped_rank,
                        score=round(nc.score, 4),
                        tokens=_count_tokens(nc.text),
                        text_preview=_preview(nc.text),
                        source=source,
                        status=ChunkStatus.BELOW_THRESHOLD,
                        full_text=nc.text,
                    )
                    sr.chunks.append(cr)
                    dropped_rank += 1

        self._report.sources.append(sr)
        self._source_map[source] = sr

    def _finalise(self):
        """
        Called on __exit__. Applies budget trimming, detects contradictions,
        generates warnings, and populates the summary fields.
        """
        included = [c for c in self._all_chunks if c.status == ChunkStatus.INCLUDED]

        # \u2500\u2500 Apply chunk budget \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if self._chunk_budget is not None and len(included) > self._chunk_budget:
            for c in included[self._chunk_budget:]:
                c.status = ChunkStatus.TRIMMED
            included = included[:self._chunk_budget]

        # \u2500\u2500 Apply token budget \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        total_retrieved_tokens = sum(c.tokens for c in included)
        trimmed_count = 0

        if self._token_budget is not None:
            running = 0
            for c in included:
                if running + c.tokens > self._token_budget:
                    c.status = ChunkStatus.TRIMMED
                    trimmed_count += 1
                else:
                    running += c.tokens

        # Recompute included after trimming
        included_final = [c for c in self._all_chunks if c.status == ChunkStatus.INCLUDED]
        total_included_tokens = sum(c.tokens for c in included_final)

        # \u2500\u2500 Budget record \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self._report.budget = BudgetRecord(
            limit=self._token_budget,
            chunk_limit=self._chunk_budget,
            total_retrieved=total_retrieved_tokens,
            total_included=total_included_tokens,
            chunks_trimmed=trimmed_count,
        )

        # \u2500\u2500 Contradiction detection \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        all_chunk_records = []
        for sr in self._report.sources:
            all_chunk_records.extend(sr.chunks)

        self._report.contradictions = _detect_contradictions(all_chunk_records)

        # \u2500\u2500 Warnings \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self._generate_warnings(included_final, trimmed_count)

        # \u2500\u2500 Summary totals \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self._report.total_chunks_included = len(included_final)
        self._report.total_tokens_included = total_included_tokens

    def _generate_warnings(self, included: list[ChunkRecord], trimmed_count: int):
        r = self._report

        # Zero results from a source
        for sr in r.sources:
            if len(sr.included) == 0:
                best = sr.best_dropped_score
                if best is not None and sr.threshold > 0:
                    r.add_warning(
                        WarnLevel.CRITICAL,
                        f"No chunks retrieved from {sr.source}.",
                        f"Lower similarity threshold from {sr.threshold:.2f} to ~{best - 0.05:.2f}."
                    )
                elif sr.candidates_evaluated == 0:
                    r.add_warning(
                        WarnLevel.CRITICAL,
                        f"Source '{sr.source}' appears to be empty.",
                        f"Check that documents have been loaded into '{sr.source}'."
                    )
                else:
                    r.add_warning(
                        WarnLevel.CRITICAL,
                        f"No chunks retrieved from {sr.source}.",
                    )

        # High-scoring chunk got trimmed
        for sr in r.sources:
            for c in sr.trimmed:
                if c.score >= 0.7:
                    r.add_warning(
                        WarnLevel.WARN,
                        f"Chunk with score {c.score:.2f} from {sr.source} was trimmed.",
                        "Increase token budget or reduce TOP to avoid losing relevant chunks."
                    )
                    break  # one warning per source is enough

        # Entire source got zeroed by trimming
        for sr in r.sources:
            if len(sr.included) == 0 and len(sr.trimmed) > 0:
                r.add_warning(
                    WarnLevel.CRITICAL,
                    f"All chunks from {sr.source} were lost to token budget.",
                    f"Allocate more tokens or reduce TOP for other sources."
                )

        # Token budget exceeded
        if r.budget and r.budget.over_budget:
            r.add_warning(
                WarnLevel.WARN,
                f"{trimmed_count} chunk(s) dropped due to token limit.",
                "Increase LIMIT TOKENS or raise similarity threshold."
            )

        # Contradictions
        for c in r.contradictions:
            r.add_warning(
                WarnLevel.WARN,
                f"Contradiction detected on topic '{c.topic_hint}' "
                f"(confidence {c.confidence:.0%}).",
                "Review source documents or use FILTER to restrict to one authoritative source."
            )

        # History requested but nothing available
        if r.history_turns_requested > 0 and r.history_turns_included == 0:
            r.add_warning(
                WarnLevel.WARN,
                "History was requested but no turns were provided.",
                "Pass history turns to record_history()."
            )


# \u2500\u2500 Main inspector class \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class ContextInspector:
    """
    Drop-in observation layer for any RAG pipeline.

    One inspector instance can be reused across multiple traces.
    Each call to .trace() starts a fresh observation session.
    The last completed trace is always available via .explain().
    """

    def __init__(self, model: str = "unknown"):
        self._model = model
        self._last_report: Optional[ExplainReport] = None
        self._formatter = ExplainFormatter()

    @contextmanager
    def trace(
        self,
        context_name: str,
        *,
        goal: str = "",
        model: Optional[str] = None,
        dry_run: bool = False,
    ) -> Iterator[TraceContext]:
        """
        Context manager. Everything recorded inside the with-block
        becomes part of one ExplainReport.

        with inspector.trace("support", goal="refund question") as ctx:
            ctx.record_retrieve("docs.policy", results)
            ctx.record_limit(token_budget=2000)
        """
        tc = TraceContext(
            context_name=context_name,
            goal=goal,
            model=model or self._model,
            dry_run=dry_run,
        )
        try:
            yield tc
        finally:
            tc._finalise()
            self._last_report = tc._report

    def explain(self, report: Optional[ExplainReport] = None) -> str:
        """
        Render the last trace (or a specific report) as a human-readable string.
        """
        r = report or self._last_report
        if r is None:
            return "No trace recorded yet. Use inspector.trace() first."
        return self._formatter.render(r)

    def report(self) -> Optional[ExplainReport]:
        """Return the raw ExplainReport object for programmatic access."""
        return self._last_report

    def watch(
        self,
        source: str,
        *,
        threshold: float = 0.0,
        operation: str = "RETRIEVE",
    ):
        """
        Decorator that automatically records the return value of a
        retrieval function.

        @inspector.watch("docs.policy", threshold=0.7)
        def get_policy_docs(query):
            return pinecone.query(query, top_k=5)
        """
        def decorator(fn):
            def wrapper(*args, **kwargs):
                results = fn(*args, **kwargs)
                if self._last_report is not None:
                    # If a trace is active, find it \u2014 else just return
                    pass
                # Store for manual flush \u2014 decorator pattern is best used
                # alongside an active trace context
                self._watched_results = getattr(self, "_watched_results", [])
                self._watched_results.append((source, results, threshold, operation))
                return results
            wrapper.__name__ = fn.__name__
            return wrapper
        return decorator

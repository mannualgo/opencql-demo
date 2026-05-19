"""
pipeline.py
===========
End-to-end pipeline:

    Context-1 retrieval
        \u2192 OpenCQL ContextInspector (assembly + inspection)
            \u2192 Frontier LLM (generation)
                \u2192 Answer + EXPLAIN report

Usage:
    pipeline = OpenCQLPipeline(
        retriever=Context1Adapter(mode="mock"),
        model="mock",          # or "claude-sonnet-4-5" / "gpt-4o"
    )

    result = pipeline.run(
        query="What is the refund policy for enterprise customers?",
        sources=["docs.policy", "history.user"],
        history=[("user", "I'm on the Enterprise plan"), ...],
        explain=True,
        dry_run=False,
    )

    print(result.answer)
    print(result.explain_report)
"""

from __future__ import annotations
import os
import json
import traceback
from dataclasses import dataclass, field
from typing import Optional,  Any

from context1_adapter import Context1Adapter, Context1Result
from context_inspector import ContextInspector


# \u2500\u2500 Pipeline result \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

@dataclass
class PipelineResult:
    query:          str
    answer:         str
    explain_report: str           # formatted EXPLAIN string
    sources_used:   list[str]     # which sources contributed chunks
    chunks_included: int
    tokens_used:    int
    model_used:     str
    hops_performed: dict[str, int]  # source \u2192 max hop reached
    dry_run:        bool


# \u2500\u2500 LLM generation layer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class GenerationLayer:
    """
    Calls a frontier LLM with the assembled context.
    Supports: Anthropic Claude, OpenAI, mock.
    """

    def __init__(self, model: str = "mock"):
        self.model = model
        self._provider = self._detect(model)

    def _detect(self, model: str) -> str:
        m = model.lower()
        if m.startswith("claude"):    return "anthropic"
        if m.startswith(("gpt","o1","o3")): return "openai"
        return "mock"

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: str,
        history: list[tuple[str,str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        try:
            if self._provider == "anthropic":
                return self._anthropic(query, context_chunks, system_prompt,
                                       history, temperature, max_tokens)
            elif self._provider == "openai":
                return self._openai(query, context_chunks, system_prompt,
                                    history, temperature, max_tokens)
            else:
                return self._mock(query, context_chunks, system_prompt)
        except Exception as e:
            traceback.print_exc()
            return f"[Generation error: {e}]"

    def _build_prompt(
        self,
        query: str,
        context_chunks: list[str],
        history: list[tuple[str,str]],
    ) -> str:
        parts = []
        if context_chunks:
            ctx = "\
\
".join(
                f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
            )
            parts.append(f"CONTEXT:\
{ctx}")
        if history:
            hist = "\
".join(f"{role.upper()}: {msg}" for role, msg in history)
            parts.append(f"CONVERSATION HISTORY:\
{hist}")
        parts.append(f"QUESTION: {query}")
        return "\
\
".join(parts)

    def _anthropic(self, query, chunks, system, history, temp, max_tok) -> str:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        prompt = self._build_prompt(query, chunks, history)
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tok,
            temperature=temp,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _openai(self, query, chunks, system, history, temp, max_tok) -> str:
        import openai
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
        messages = [{"role": "system", "content": system}]
        for role, msg in history:
            messages.append({"role": role, "content": msg})
        prompt = self._build_prompt(query, chunks, [])
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
        )
        return resp.choices[0].message.content

    def _mock(self, query, chunks, system) -> str:
        ctx_preview = chunks[0][:80] if chunks else "no context"
        return (
            f"[{self.model.upper()} RESPONSE]\
"
            f"Based on the assembled context ({len(chunks)} chunks), "
            f"here is the answer to: '{query[:60]}'\
\
"
            f"The most relevant context was: \"{ctx_preview}...\"\
\
"
            f"Answer: This is a mock response demonstrating the full pipeline. "
            f"In production, swap model='mock' for model='claude-sonnet-4-5' "
            f"and set ANTHROPIC_API_KEY to get a real answer."
        )


# \u2500\u2500 The main pipeline \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class OpenCQLPipeline:
    """
    Orchestrates the three-layer pipeline:
        Context-1 retrieval \u2192 OpenCQL assembly \u2192 Frontier LLM generation

    Parameters
    ----------
    retriever   : Context1Adapter instance
    model       : LLM model name string
    system_prompt : default system prompt (overridable per call)
    token_budget  : default token budget for assembled context
    """

    def __init__(
        self,
        retriever: Context1Adapter,
        model: str = "mock",
        system_prompt: str = "You are a helpful, accurate assistant. "
                             "Answer based on the provided context only. "
                             "If the context does not contain the answer, say so.",
        token_budget: int = 3000,
    ):
        self._retriever  = retriever
        self._generator  = GenerationLayer(model=model)
        self._system     = system_prompt
        self._budget     = token_budget

    def run(
        self,
        query: str,
        sources: list[str],
        *,
        top_k_per_source: int = 5,
        threshold: float = 0.0,
        history: list[tuple[str, str]] | None = None,
        history_turns: int = 5,
        token_budget: int | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        explain: bool = True,
        dry_run: bool = False,
        context_name: str = "pipeline",
    ) -> PipelineResult:
        """
        Run the full pipeline.

        Parameters
        ----------
        query           : user question
        sources         : list of source names to retrieve from
        top_k_per_source: how many docs to retrieve per source
        threshold       : minimum similarity score
        history         : list of (role, message) tuples
        history_turns   : how many recent history turns to inject
        token_budget    : override default token budget
        system_prompt   : override default system prompt
        temperature     : LLM temperature
        max_tokens      : max LLM output tokens
        explain         : whether to generate EXPLAIN report
        dry_run         : assemble context but skip LLM call
        context_name    : label for the EXPLAIN report
        """

        history   = history   or []
        budget    = token_budget or self._budget
        system    = system_prompt or self._system

        print(f"\
{'\u2550'*60}")
        print(f"  OpenCQL Pipeline \u2014 E2E Run")
        print(f"{'\u2550'*60}")
        print(f"  Query:   {query[:70]}")
        print(f"  Sources: {sources}")
        print(f"  Model:   {self._generator.model}")
        print(f"  Budget:  {budget} tokens")
        print(f"  Mode:    {'DRY RUN' if dry_run else 'FULL RUN'}")
        print()

        # \u2500\u2500 Layer 1: Context-1 retrieval \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        print("  [LAYER 1] Context-1 Retrieval")
        print(f"  {'\u2500'*56}")

        retrieved = self._retriever.retrieve_per_source(
            query=query,
            sources=sources,
            top_k_per_source=top_k_per_source,
            threshold=threshold,
        )

        for src, results in retrieved.items():
            hops_seen = sorted(set(r.hop for r in results))
            print(f"    {src}: {len(results)} docs  "
                  f"(hops: {hops_seen})  "
                  f"top score: {max((r.score for r in results), default=0):.2f}")

        # \u2500\u2500 Layer 2: OpenCQL assembly + inspection \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        print()
        print("  [LAYER 2] OpenCQL Context Assembly")
        print(f"  {'\u2500'*56}")

        inspector = ContextInspector(model=self._generator.model)

        with inspector.trace(
            context_name,
            goal=query,
            model=self._generator.model,
            dry_run=dry_run,
        ) as ctx:

            ctx.record_system(system)

            # Record each source separately so EXPLAIN shows per-source detail
            for source_name, results in retrieved.items():
                if not results:
                    continue
                # Convert Context1Result \u2192 (text, score) tuples
                tuples = [(r.text, r.score) for r in results]
                ctx.record_retrieve(
                    source_name,
                    tuples,
                    threshold=threshold,
                    top_k=top_k_per_source,
                )
                print(f"    Recorded {len(tuples)} chunks from {source_name}")

            # Record history
            recent_history = history[-history_turns:] if history else []
            if recent_history:
                ctx.record_history(recent_history, requested=history_turns)
                print(f"    Injected {len(recent_history)} history turns")

            # Record token budget \u2014 this triggers trimming calculation
            ctx.record_limit(token_budget=budget)

        # Get the report and formatted explain
        report = inspector.report()
        explain_text = inspector.explain() if explain else ""

        # Extract final included chunks for generation
        final_chunks = []
        for sr in report.sources:
            for chunk in sr.included:
                final_chunks.append(chunk.full_text or chunk.text_preview)

        print(f"    Final context: {len(final_chunks)} chunks / "
              f"{report.total_tokens_included} tokens")

        if report.has_critical():
            print(f"    \u26a0 CRITICAL issues detected \u2014 see EXPLAIN report")

        # \u2500\u2500 Layer 3: Frontier LLM generation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        answer = ""
        if not dry_run:
            print()
            print("  [LAYER 3] Frontier LLM Generation")
            print(f"  {'\u2500'*56}")
            print(f"    Calling {self._generator.model}...")

            answer = self._generator.generate(
                query=query,
                context_chunks=final_chunks,
                system_prompt=system,
                history=recent_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print(f"    Response: {len(answer)} chars")
        else:
            answer = "[DRY RUN \u2014 LLM not called]"

        # \u2500\u2500 Build result \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        hops_performed = {
            src: max((r.hop for r in results), default=0)
            for src, results in retrieved.items()
        }

        return PipelineResult(
            query=query,
            answer=answer,
            explain_report=explain_text,
            sources_used=[s for s, r in retrieved.items() if r],
            chunks_included=report.total_chunks_included,
            tokens_used=report.total_tokens_included,
            model_used=self._generator.model,
            hops_performed=hops_performed,
            dry_run=dry_run,
        )

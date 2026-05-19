"""
OpenCQL explain_formatter.py
=============================
Renders an ExplainReport into a human-readable string.

Completely stateless \u2014 takes a report, returns a string.
No LLM calls, no I/O, no side effects.

The output is designed to be readable in a terminal, in a
notebook, or saved to a file. Width is capped at 62 chars.
"""

from __future__ import annotations
from opencql_dsl.explain import (
    ExplainReport, SourceRecord, ChunkRecord, ChunkStatus,
    BudgetRecord, PartitionRecord, ContradictionRecord,
    Warning, WarnLevel
)

# \u2500\u2500 Layout constants \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

W       = 62          # total line width
INDENT  = "  "        # two-space indent for sub-items
SEP     = "\u2500" * W     # section separator
HEAVY   = "\u2550" * W     # heavy separator for header/footer
TICK    = "\u2713"
CROSS   = "\u2717"
WARN_SYM = "\u26a0"
INFO_SYM = "\u2139"


def _pad(label: str, value: str, width: int = W) -> str:
    """Left-label, right-value, padded to width."""
    gap = width - len(label) - len(str(value))
    return label + (" " * max(1, gap)) + str(value)


def _truncate(text: str, max_len: int = 80) -> str:
    text = text.replace("\
", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def _status_icon(status: ChunkStatus) -> str:
    icons = {
        ChunkStatus.INCLUDED:        f"  {TICK} INCLUDED       ",
        ChunkStatus.BELOW_THRESHOLD: f"  {CROSS} BELOW THRESHOLD",
        ChunkStatus.TRIMMED:         f"  {CROSS} TRIMMED        ",
        ChunkStatus.DUPLICATE:       f"  {CROSS} DUPLICATE      ",
    }
    return icons.get(status, "  ? UNKNOWN        ")


def _warn_icon(level: WarnLevel) -> str:
    return {
        WarnLevel.INFO:     f"  {INFO_SYM} INFO    ",
        WarnLevel.WARN:     f"  {WARN_SYM} WARN    ",
        WarnLevel.CRITICAL: f"  {WARN_SYM} CRITICAL",
    }.get(level, "  ? UNKNOWN ")


# \u2500\u2500 Main formatter class \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class ExplainFormatter:

    def render(self, report: ExplainReport) -> str:
        lines = []
        a = lines.append   # shorthand

        a(HEAVY)
        a("  OpenCQL EXPLAIN \u2014 Context Assembly Report")
        a(HEAVY)

        # \u2500\u2500 Header block \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        a("")
        a(_pad("  CONTEXT :", report.context_name))
        a(_pad("  GOAL    :", f'"{_truncate(report.goal, 50)}"'))
        a(_pad("  MODEL   :", report.model))
        a(_pad("  MODE    :", "DRY RUN (LLM not called)" if report.dry_run else "FULL RUN"))

        # \u2500\u2500 System prompt \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        a("")
        a(f"  {SEP[:58]}")
        a("  SYSTEM PROMPT")
        a(f"  {SEP[:58]}")
        if report.system_prompt:
            # Word-wrap at ~56 chars
            words = report.system_prompt.split()
            line_, buf = [], []
            for w in words:
                buf.append(w)
                if len(" ".join(buf)) > 54:
                    line_.append(INDENT + "  " + " ".join(buf[:-1]))
                    buf = [w]
            if buf:
                line_.append(INDENT + "  " + " ".join(buf))
            for l in line_:
                a(l)
        else:
            a(f"  (none)")

        # \u2500\u2500 Per-source sections \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        for sr in report.sources:
            self._render_source(sr, lines)

        # \u2500\u2500 History \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if report.history_turns_requested > 0:
            a("")
            a(f"  {SEP[:58]}")
            a("  HISTORY")
            a(f"  {SEP[:58]}")
            a(_pad("  Requested :", f"last {report.history_turns_requested} turns"))
            a(_pad("  Included  :", f"{report.history_turns_included} turns"))
            if report.history_turns_included == 0:
                a(f"  {WARN_SYM} No history was available in params.")

        # \u2500\u2500 Token budget \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if report.budget:
            self._render_budget(report.budget, lines)

        # \u2500\u2500 Partition \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if report.partition:
            self._render_partition(report.partition, lines)

        # \u2500\u2500 Contradictions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if report.contradictions:
            self._render_contradictions(report.contradictions, lines)

        # \u2500\u2500 Warnings summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if report.warnings:
            self._render_warnings(report.warnings, lines)

        # \u2500\u2500 Final assembly summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        a("")
        a(f"  {SEP[:58]}")
        a("  FINAL CONTEXT WINDOW")
        a(f"  {SEP[:58]}")
        a(_pad("  Chunks included :", str(report.total_chunks_included)))
        a(_pad("  Tokens included :", str(report.total_tokens_included)))

        sources_summary = ", ".join(
            f"{sr.source} ({len(sr.included)})"
            for sr in report.sources
        )
        if sources_summary:
            a(f"  Sources         : {sources_summary}")

        if report.has_critical():
            a("")
            a(f"  {WARN_SYM} This context has CRITICAL issues.")
            a(f"    The model may hallucinate or give an empty response.")
            a(f"    Review the warnings above before running inference.")
        elif report.warnings:
            a("")
            a(f"  {WARN_SYM} {len(report.warnings)} warning(s) detected. Review above.")
        else:
            a("")
            a(f"  {TICK}  No issues detected. Context looks healthy.")

        a("")
        a(HEAVY)

        return "\
".join(lines)

    # \u2500\u2500 Source section \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _render_source(self, sr: SourceRecord, lines: list):
        a = lines.append
        a("")
        a(f"  {SEP[:58]}")
        a(f"  {sr.operation}: {sr.source}")
        a(f"  {SEP[:58]}")

        a(_pad("  Threshold      :", f"similarity > {sr.threshold}" if sr.threshold > 0 else "none"))
        a(_pad("  Requested      :", f"TOP {sr.requested_top_k}"))
        a(_pad("  Evaluated      :", f"{sr.candidates_evaluated} candidates"))
        a(_pad("  Included       :", f"{len(sr.included)} chunks  ({sr.tokens_included} tokens)"))

        if sr.below_threshold:
            a(_pad("  Below threshold:", f"{len(sr.below_threshold)} chunks dropped"))
        if sr.trimmed:
            a(_pad("  Trimmed        :", f"{len(sr.trimmed)} chunks dropped (token budget)"))

        # Included chunks
        if sr.included:
            a("")
            a(f"  Included chunks:")
            for c in sr.included:
                a(f"    [{c.rank}] score={c.score:.2f}  tokens={c.tokens:<4}")
                a(f"        \"{_truncate(c.text_preview, 56)}\"")

        # Dropped chunks (below threshold)
        if sr.below_threshold:
            a("")
            a(f"  Dropped (below threshold):")
            for c in sr.below_threshold:
                a(f"    [{c.rank}] score={c.score:.2f}  "
                  f"\"{_truncate(c.text_preview, 44)}\"")

        # Trimmed chunks (hit token budget)
        if sr.trimmed:
            a("")
            a(f"  Trimmed (token budget exhausted):")
            for c in sr.trimmed:
                a(f"    [{c.rank}] score={c.score:.2f}  tokens={c.tokens:<4}  "
                  f"\"{_truncate(c.text_preview, 36)}\"")

        # Zero-result critical warning
        if len(sr.included) == 0:
            a("")
            best = sr.best_dropped_score
            a(f"  {WARN_SYM} CRITICAL: No chunks included from {sr.source}.")
            if best is not None and sr.threshold > 0:
                a(f"    Best candidate score was {best:.2f} "
                  f"(threshold requires {sr.threshold:.2f}).")
                a(f"    Hint: lower threshold to ~{best - 0.05:.2f} to include top results.")
            elif sr.candidates_evaluated == 0:
                a(f"    Source appears to be empty. Check that documents")
                a(f"    have been loaded into '{sr.source}'.")

    # \u2500\u2500 Budget section \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _render_budget(self, b: BudgetRecord, lines: list):
        a = lines.append
        a("")
        a(f"  {SEP[:58]}")
        a("  TOKEN BUDGET")
        a(f"  {SEP[:58]}")

        if b.limit:
            util = f"{b.utilization_pct}%" if b.utilization_pct is not None else "n/a"
            a(_pad("  Limit     :", f"{b.limit} tokens"))
            a(_pad("  Used      :", f"{b.total_included} tokens  ({util} utilization)"))
            a(_pad("  Trimmed   :", f"{b.chunks_trimmed} chunks dropped"))

            if b.over_budget:
                a("")
                a(f"  {WARN_SYM} Retrieved {b.total_retrieved} tokens but budget is "
                  f"{b.limit}.")
                a(f"    {b.chunks_trimmed} chunk(s) were silently dropped.")
                a(f"    Hint: increase LIMIT TOKENS or reduce TOP to avoid loss.")
            else:
                a(f"  {TICK}  Within budget. {b.limit - b.total_included} tokens unused.")
        else:
            a(f"  (no token limit set)")
            a(_pad("  Total assembled :", f"{b.total_included} tokens"))

        if b.chunk_limit:
            a(_pad("  Chunk limit :", f"TOP {b.chunk_limit} chunks"))

    # \u2500\u2500 Partition section \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _render_partition(self, p: PartitionRecord, lines: list):
        a = lines.append
        a("")
        a(f"  {SEP[:58]}")
        a(f"  PARTITION BY {p.field}")
        a(f"  {SEP[:58]}")

        a(_pad("  Declared partitions:", str(len(p.declared_values))))
        a("")

        for val in p.declared_values:
            count = p.matched.get(val, 0)
            icon = TICK if count > 0 else CROSS
            a(f"    {icon}  {val:<20} {count} doc(s)")

        if p.unmatched_count > 0:
            a("")
            unmatched_str = ", ".join(f'"{v}"' for v in p.unmatched_values[:5])
            a(f"  {WARN_SYM} {p.unmatched_count} document(s) not routed to any partition.")
            a(f"    Unmatched values found: {unmatched_str}")
            a(f"    Hint: add these values as partitions or use PARTITION BY AUTO.")

    # \u2500\u2500 Contradictions section \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _render_contradictions(self, contradictions: list[ContradictionRecord], lines: list):
        a = lines.append
        a("")
        a(f"  {SEP[:58]}")
        a("  POTENTIAL CONTRADICTIONS DETECTED")
        a(f"  {SEP[:58]}")
        a(f"  {WARN_SYM} The following chunk pairs may conflict.")
        a(f"    This increases hallucination risk.")
        a("")

        for i, c in enumerate(contradictions, 1):
            a(f"  Conflict {i}  (topic: \"{c.topic_hint}\", "
              f"confidence={c.confidence:.0%})")
            a(f"    Chunk [{c.chunk_a_rank}]: \"{_truncate(c.chunk_a_preview, 50)}\"")
            a(f"    Chunk [{c.chunk_b_rank}]: \"{_truncate(c.chunk_b_preview, 50)}\"")
            a("")

        a(f"  Hint: review source documents for consistency, or")
        a(f"  use FILTER to restrict to a single authoritative source.")

    # \u2500\u2500 Warnings summary section \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _render_warnings(self, warnings: list[Warning], lines: list):
        a = lines.append
        a("")
        a(f"  {SEP[:58]}")
        a("  WARNINGS")
        a(f"  {SEP[:58]}")

        for w in warnings:
            a(f"{_warn_icon(w.level)}  {w.message}")
            if w.hint:
                a(f"             Hint: {w.hint}")

"""
OpenCQL Runtime v2
Executes a compiled execution plan (from compiler.py) against vector stores and LLMs.

Supports:
  - CONTEXT ... AS (...) definitions
  - RETRIEVE FROM <source> with similarity filtering
  - JOIN <source> SEMANTIC ON <key>
  - PARTITION BY <field> (parallel MapReduce inference)
  - LIMIT TOKENS / LIMIT CHUNKS
  - INJECT HISTORY
  - WITH SYSTEM
  - INFER with aggregation strategies: synthesis | vote | concat
"""

from __future__ import annotations
import os
import concurrent.futures
from typing import Optional,  Any

from lark import Lark

from opencql_dsl.compiler import CQLCompiler
from opencql_dsl.vectors import VectorStore, SourceRegistry
from opencql_dsl.llm import LLM, create_llm


# \u2500\u2500 Load grammar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")
with open(_GRAMMAR_PATH) as f:
    _GRAMMAR = f.read()


# \u2500\u2500 Token counting (approximate) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _count_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return len(text) // 4


def _trim_to_tokens(chunks: list[str], max_tokens: int) -> list[str]:
    result, total = [], 0
    for chunk in chunks:
        t = _count_tokens(chunk)
        if total + t > max_tokens:
            break
        result.append(chunk)
        total += t
    return result


# \u2500\u2500 Context Assembler \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class ContextAssembler:
    """
    Executes the clauses of a CONTEXT block and returns an assembled context dict:
    {
        "chunks": [str, ...],       # text chunks to inject
        "system_prompt": str | None,
        "history": [(role, msg), ...],
        "partitions": {field: str, values: [...]},
        "token_limit": int | None,
        "chunk_limit": int | None,
    }
    """

    def __init__(self, registry: SourceRegistry, params: dict):
        self.registry = registry
        self.params = params

    def assemble(self, clauses: list[dict], query: str = "") -> dict:
        ctx: dict[str, Any] = {
            "chunks": [],
            "system_prompt": None,
            "history": [],
            "partition": None,
            "token_limit": None,
            "chunk_limit": None,
        }

        for clause in clauses:
            t = clause.get("type")

            if t == "retrieve":
                self._do_retrieve(clause, query, ctx)

            elif t == "join":
                self._do_join(clause, query, ctx)

            elif t == "filter":
                self._do_filter(clause, ctx)

            elif t == "partition":
                ctx["partition"] = clause

            elif t == "limit":
                if clause["kind"] == "tokens":
                    ctx["token_limit"] = clause["value"]
                else:
                    ctx["chunk_limit"] = clause["value"]

            elif t == "history":
                self._do_history(clause, ctx)

            elif t == "system":
                ctx["system_prompt"] = self._resolve(clause["prompt"])

        # Apply limits
        if ctx["chunk_limit"] is not None:
            ctx["chunks"] = ctx["chunks"][: ctx["chunk_limit"]]
        if ctx["token_limit"] is not None:
            ctx["chunks"] = _trim_to_tokens(ctx["chunks"], ctx["token_limit"])

        return ctx

    def _resolve(self, val):
        if isinstance(val, dict) and "_param" in val:
            return self.params.get(val["_param"], val["_param"])
        return val

    def _do_retrieve(self, clause: dict, query: str, ctx: dict):
        source_name = clause.get("source", "default")
        store = self.registry.get(source_name)
        if store is None:
            print(f"  [WARN] Source '{source_name}' not found in registry. Skipping.")
            return

        top_k = clause.get("top", 5)
        threshold = 0.0
        filters = None

        where = clause.get("where")
        if where:
            # Support simple similarity threshold: "similarity > 0.8"
            if where.get("field") == "similarity" and where.get("op") == ">":
                threshold = float(where["value"])
            else:
                filters = {where["field"]: where["value"]}

        q = query or ""
        results = store.search(q, top_k=top_k, threshold=threshold, filters=filters)
        new_chunks = [r[0]["text"] for r in results]
        print(f"  [RETRIEVE] '{source_name}' \u2192 {len(new_chunks)} chunks (threshold={threshold})")
        ctx["chunks"].extend(new_chunks)

    def _do_join(self, clause: dict, query: str, ctx: dict):
        source_name = clause.get("source", "")
        store = self.registry.get(source_name)
        if store is None:
            print(f"  [WARN] JOIN source '{source_name}' not found. Skipping.")
            return

        join_type = clause.get("join_type", "semantic")
        on_field = clause.get("on")

        if join_type == "semantic":
            results = store.search(query or "", top_k=5)
            joined = [r[0]["text"] for r in results]
        else:
            # Exact join: find docs in joined store whose `on_field` value
            # matches a doc already in chunks (na\u00efve implementation)
            joined = [d["text"] for d in store.documents]

        print(f"  [JOIN] '{source_name}' ({join_type}) \u2192 {len(joined)} docs")
        ctx["chunks"].extend(joined)

    def _do_filter(self, clause: dict, ctx: dict):
        cond = clause.get("condition", {})
        field = cond.get("field")
        op = cond.get("op")
        val = self._resolve(cond.get("value"))
        before = len(ctx["chunks"])
        # This applies post-retrieval text filtering (simple string match for now)
        if op == "CONTAINS" and field == "text":
            ctx["chunks"] = [c for c in ctx["chunks"] if str(val).lower() in c.lower()]
        print(f"  [FILTER] '{field} {op} {val}' \u2192 kept {len(ctx['chunks'])}/{before} chunks")

    def _do_history(self, clause: dict, ctx: dict):
        if clause["mode"] == "explicit":
            ctx["history"].extend(clause["turns"])
        elif clause["mode"] == "last_n":
            # Caller is expected to inject the actual history via params
            n = clause.get("n", 5)
            history = self.params.get("history", [])
            ctx["history"].extend(history[-n:])


# \u2500\u2500 Partition Executor (MapReduce) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class PartitionExecutor:
    """
    Runs parallel inference chains over context partitions, then reduces.
    """

    def __init__(self, store: VectorStore, llm: LLM, max_workers: int = 4):
        self.store = store
        self.llm = llm
        self.max_workers = max_workers

    def run(
        self,
        partition: dict,
        base_chunks: list[str],
        goal: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        aggregate: str = "synthesis",
    ) -> str:
        field = partition["field"]
        values = partition["values"]
        auto = partition.get("auto", False)

        if auto:
            values = self.store.get_partitions(field)

        if not values:
            print("  [PARTITION] No partition values found.")
            return "(no partitions)"

        print(f"  [PARTITION] Running {len(values)} parallel inference chains on '{field}'")

        def _map(group_val):
            group_docs = self.store.search_by_field(field, group_val)
            group_chunks = [d["text"] for d in group_docs] + base_chunks
            context_str = "\
".join(group_chunks)
            prompt = f"Context:\
{context_str}\
\
Goal: {goal or 'Analyze the context.'}"
            sys = (system_prompt or "") + f"\
You are an expert in the '{group_val}' domain."
            response = self.llm.generate(
                prompt, system_prompt=sys, temperature=temperature, max_tokens=max_tokens
            )
            print(f"    [MAP] '{group_val}' complete.")
            return group_val, response

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_map, v): v for v in values}
            map_results = {}
            for fut in concurrent.futures.as_completed(futures):
                k, v = fut.result()
                map_results[k] = v

        return self._reduce(map_results, aggregate, goal, system_prompt, temperature, max_tokens)

    def _reduce(
        self,
        map_results: dict[str, str],
        strategy: str,
        goal: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        print(f"  [REDUCE] Strategy: {strategy}")

        if strategy == "concat":
            lines = [f"[{k}]\
{v}" for k, v in map_results.items()]
            return "\
\
".join(lines)

        elif strategy == "vote":
            # Return the most common conclusion (simple majority)
            from collections import Counter
            counts = Counter(map_results.values())
            return counts.most_common(1)[0][0]

        else:  # synthesis (default) \u2014 call LLM to synthesize
            combined = "\
\
".join(f"[{k}]\
{v}" for k, v in map_results.items())
            prompt = (
                f"The following are domain-specific analyses:\
\
{combined}\
\
"
                f"Synthesize these into a single cohesive response. Goal: {goal or 'Synthesize.'}"
            )
            return self.llm.generate(
                prompt,
                system_prompt=system_prompt or "You are a synthesis expert.",
                temperature=temperature,
                max_tokens=max_tokens,
            )


# \u2500\u2500 Main Runtime \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class CQLRuntime:
    """
    The main OpenCQL runtime.

    Usage:
        runtime = CQLRuntime()
        runtime.registry.get_or_create("docs.product").add_documents([...])
        result = runtime.execute(cql_code, query="my question", params={})
    """

    def __init__(self, default_model: str = "mock"):
        self.parser = Lark(_GRAMMAR, start="start", parser="earley", lexer="dynamic")
        self.compiler = CQLCompiler()
        self.registry = SourceRegistry()
        self.default_model = default_model
        self._context_defs: dict[str, list[dict]] = {}  # named CONTEXT blocks

    # \u2500\u2500 Public API \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def execute(
        self,
        cql_code: str,
        query: str = "",
        params: dict | None = None,
        history: list | None = None,
    ) -> str:
        """Parse, compile, and execute a CQL program. Returns the final response string."""
        params = params or {}
        if history:
            params["history"] = history

        print("\
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550")
        print("  OpenCQL Runtime v2  \u2014  Executing query")
        print("\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550")

        tree = self.parser.parse(cql_code)
        plan = self.compiler.transform(tree)

        results = []
        for stmt in plan.get("statements", []):
            r = self._execute_statement(stmt, query, params)
            if r is not None:
                results.append(r)

        return "\
\
".join(results) if results else "(no output)"

    # \u2500\u2500 Statement dispatch \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _execute_statement(self, stmt: dict, query: str, params: dict) -> str | None:
        t = stmt.get("type")

        if t == "context_def":
            self._context_defs[stmt["name"]] = stmt["clauses"]
            print(f"  [CONTEXT] Defined '{stmt['name']}'")
            return None

        elif t == "infer":
            return self._execute_infer(stmt, query, params)

        elif t == "legacy_query":
            return self._execute_legacy(stmt, query, params)

        else:
            print(f"  [WARN] Unknown statement type: {t}")
            return None

    # \u2500\u2500 INFER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _execute_infer(self, stmt: dict, query: str, params: dict) -> str:
        model_name = stmt.get("model", self.default_model)
        context_name = stmt.get("context")
        goal = stmt.get("goal", query)
        aggregate = stmt.get("aggregate", "synthesis")
        temperature = float(stmt.get("temperature", 0.7))
        max_tokens = int(stmt.get("max_tokens", 1024))
        fmt = stmt.get("format")

        llm = create_llm(model_name)
        print(f"\
  [INFER] model={model_name}, context={context_name}, goal={goal[:60]}")

        # Assemble context
        clauses = self._context_defs.get(context_name, []) if context_name else []
        assembler = ContextAssembler(self.registry, params)
        ctx = assembler.assemble(clauses, query=goal)

        system_prompt = ctx.get("system_prompt")
        chunks = ctx.get("chunks", [])
        history = ctx.get("history", [])
        partition = ctx.get("partition")

        if partition:
            # Pick a representative store for partition scanning
            first_retrieve = next(
                (c for c in clauses if c.get("type") == "retrieve"), None
            )
            store = (
                self.registry.get(first_retrieve["source"])
                if first_retrieve
                else VectorStore()
            )
            executor = PartitionExecutor(store, llm)
            response = executor.run(
                partition=partition,
                base_chunks=chunks,
                goal=goal,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                aggregate=aggregate,
            )
        else:
            context_str = "\
".join(chunks)
            history_str = "\
".join(f"{r}: {m}" for r, m in history) if history else ""
            full_prompt = (
                (f"History:\
{history_str}\
\
" if history_str else "")
                + (f"Context:\
{context_str}\
\
" if context_str else "")
                + f"Goal: {goal}"
            )
            response = llm.generate(
                full_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if fmt == "json":
            response = self._wrap_json(response)

        return response

    # \u2500\u2500 Legacy query (backwards compat) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _execute_legacy(self, stmt: dict, query: str, params: dict) -> str:
        steps = stmt.get("steps", {})
        model_name = stmt.get("model", self.default_model)
        llm = create_llm(model_name)

        store = VectorStore()
        store.add_documents([
            {"id": 1, "text": "GDPR requires data encryption.", "domain": "Legal"},
            {"id": 2, "text": "The budget is $50k.", "domain": "Financial"},
            {"id": 3, "text": "Kubernetes cluster is ready.", "domain": "Technical"},
            {"id": 4, "text": "HIPAA compliance is mandatory.", "domain": "Legal"},
        ])

        context_chunks = []
        if "knowledge" in steps:
            results = store.search(query or "query", top_k=10, threshold=0.0)
            context_chunks = [r[0] for r in results]

        if "group_by" in steps:
            gb = steps["group_by"]
            field = gb["column"]
            partitions = gb["partitions"]
            executor = PartitionExecutor(store, llm)
            return executor.run(
                partition={"field": field, "values": partitions},
                base_chunks=[c.get("text", "") for c in context_chunks],
                goal=query,
                system_prompt=None,
                temperature=0.7,
                max_tokens=512,
                aggregate="concat",
            )

        prompt = "Context: " + str(context_chunks)
        return llm.generate(prompt)

    # \u2500\u2500 Helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _wrap_json(self, text: str) -> str:
        import json
        try:
            json.loads(text)
            return text
        except Exception:
            return json.dumps({"response": text})

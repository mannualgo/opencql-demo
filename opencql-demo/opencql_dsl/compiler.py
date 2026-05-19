"""
OpenCQL Compiler v2
Transforms a Lark parse tree into a structured execution plan (dict).
"""
from __future__ import annotations

from lark import Transformer, v_args, Token


def _unquote(s):
    """Strip surrounding quotes from a string literal."""
    if isinstance(s, str) and len(s) >= 2 and s[0] in ('"', "'") and s[-1] in ('"', "'"):
        return s[1:-1]
    return s


class CQLCompiler(Transformer):
    """Lark Transformer: walks the parse tree and returns an execution plan."""

    # \u2500\u2500 top-level \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def start(self, items):
        return {"statements": items}

    # \u2500\u2500 CONTEXT definition \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def context_def(self, items):
        name = str(items[0])
        body = items[1]
        return {"type": "context_def", "name": name, "clauses": body}

    def context_body(self, items):
        return items  # list of clause dicts

    # \u2500\u2500 RETRIEVE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def retrieve_clause(self, items):
        source = items[0]
        opts = {}
        for opt in items[1:]:
            if isinstance(opt, dict):
                opts.update(opt)
        return {"type": "retrieve", "source": source, **opts}

    def source_ref(self, items):
        return ".".join(str(i) for i in items)

    def retrieve_where(self, items):
        return {"where": items[0]}

    def retrieve_top(self, items):
        return {"top": int(items[0])}

    def retrieve_token_limit(self, items):
        return {"token_limit": int(items[0])}

    def retrieve_opts(self, items):
        merged = {}
        for d in items:
            if isinstance(d, dict):
                merged.update(d)
        return merged

    # \u2500\u2500 JOIN \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def join_clause(self, items):
        source = items[0]
        join_type = "semantic"
        key = None
        for item in items[1:]:
            if isinstance(item, dict) and "join_type" in item:
                join_type = item["join_type"]
            elif isinstance(item, (str, Token)):
                key = str(item)
        return {"type": "join", "source": source, "join_type": join_type, "on": key}

    def semantic_join(self, _):
        return {"join_type": "semantic"}

    def exact_join(self, _):
        return {"join_type": "exact"}

    # \u2500\u2500 FILTER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def filter_clause(self, items):
        return {"type": "filter", "condition": items[0]}

    # \u2500\u2500 PARTITION \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def partition_clause(self, items):
        field = str(items[0])
        values = items[1]
        return {"type": "partition", "field": field, "values": values, "auto": False}

    def partition_auto(self, items):
        field = str(items[0])
        return {"type": "partition", "field": field, "values": [], "auto": True}

    # \u2500\u2500 LIMIT \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def limit_tokens(self, items):
        return {"type": "limit", "kind": "tokens", "value": int(items[0])}

    def limit_chunks(self, items):
        return {"type": "limit", "kind": "chunks", "value": int(items[0])}

    # \u2500\u2500 HISTORY \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def history_last_n(self, items):
        return {"type": "history", "mode": "last_n", "n": int(items[0])}

    def history_explicit(self, items):
        return {"type": "history", "mode": "explicit", "turns": items[0]}

    # \u2500\u2500 SYSTEM \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def system_clause(self, items):
        return {"type": "system", "prompt": _unquote(items[0])}

    # \u2500\u2500 INFER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def infer_stmt(self, items):
        plan = {"type": "infer"}
        for opt in items[0]:
            if isinstance(opt, dict):
                plan.update(opt)
        return plan

    def infer_opts(self, items):
        return items

    def infer_model(self, items):
        return {"model": _unquote(items[0])}

    def infer_context(self, items):
        return {"context": str(items[0])}

    def infer_goal(self, items):
        return {"goal": _unquote(items[0])}

    def infer_aggregate(self, items):
        return {"aggregate": _unquote(items[0])}

    def infer_temperature(self, items):
        return {"temperature": float(items[0])}

    def infer_max_tokens(self, items):
        return {"max_tokens": int(items[0])}

    def infer_format(self, items):
        return {"format": _unquote(items[0])}

    # \u2500\u2500 CONDITION \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def condition(self, items):
        field = str(items[0])
        op = items[1]
        val = items[2]
        cond = {"field": field, "op": op, "value": val}
        if len(items) > 4:
            cond["logical"] = str(items[3])
            cond["next"] = items[4]
        return cond

    def gt(self, _):  return ">"
    def lt(self, _):  return "<"
    def gte(self, _): return ">="
    def lte(self, _): return "<="
    def eq(self, _):  return "="
    def neq(self, _): return "!="
    def contains(self, _): return "CONTAINS"

    # \u2500\u2500 PRIMITIVES \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def value(self, items):
        v = items[0]
        if isinstance(v, dict):
            return v  # variable or identifier already handled
        return _unquote(str(v))

    def variable(self, items):
        return {"_param": str(items[0])}

    def identifier(self, items):
        return str(items[0])

    def number(self, items):
        raw = str(items[0])
        return float(raw) if "." in raw else int(raw)

    def value_list(self, items):
        return [_unquote(str(i)) if not isinstance(i, dict) else i for i in items]

    def tuple_list(self, items):
        return items

    def tuple_item(self, items):
        return (_unquote(str(items[0])), _unquote(str(items[1])))

    def kv_list(self, items):
        return dict(items)

    def kv_pair(self, items):
        return (str(items[0]), items[1])

    # \u2500\u2500 LEGACY QUERY (backwards compat) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def query_stmt(self, items):
        col = str(items[0])
        model = str(items[1])
        clauses = {}
        for clause in items[2:]:
            if isinstance(clause, dict):
                clauses.update(clause)
        return {"type": "legacy_query", "column": col, "model": model, "steps": clauses}

    def with_system(self, items):
        return {"system": items[0]}

    def join_knowledge(self, items):
        return {"knowledge": items[0]}

    def inject_history(self, items):
        return {"history": items[0]}

    def group_by(self, items):
        return {"group_by": {"column": str(items[0]), "partitions": items[1]}}

    def aggregate_with(self, items):
        return {"aggregate": _unquote(str(items[0]))}

    def where_clause(self, items):
        return {"where": items[0]}

    def INTEGER(self, token):
        return int(token)

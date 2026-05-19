"""
opencql_inspector
=================
Pre-inference context quality inspection for RAG pipelines.

Zero mandatory dependencies. Drop into any existing pipeline.

Quick start:
    from opencql_inspector import ContextInspector

    inspector = ContextInspector()
    report = inspector.inspect(
        chunks=[{"text": "...", "source": "docs.policy", "score": 0.91}],
        query="enterprise refund",
        token_budget=2000,
        sources_expected=["docs.policy", "docs.customer"],
    )
    print(report.format())
"""

from .inspector import (
    ContextInspector,
    ContextQualityError,
    InspectReport,
)

__all__ = [
    "ContextInspector",
    "ContextQualityError",
    "InspectReport",
]

__version__ = "0.3.0"

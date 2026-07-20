"""Tests for the GraphDB runtime validation contract."""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_graphdb_runtime.py"
SPEC = importlib.util.spec_from_file_location("validate_graphdb_runtime", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_graphdb_validation_requires_successful_grounding_readback():
    result = {
        "post": {"persisted": True},
        "query": {"queried": True, "gap_count": 1, "payload_count": 1},
    }

    assert MODULE._graphdb_validation_succeeded(result, query_required=True)


def test_graphdb_validation_rejects_missing_or_empty_readback():
    assert not MODULE._graphdb_validation_succeeded(
        {"post": {"persisted": True}, "query": {"queried": False}},
        query_required=True,
    )
    assert not MODULE._graphdb_validation_succeeded(
        {"post": {"persisted": True}, "query": {"queried": True, "gap_count": 0, "payload_count": 0}},
        query_required=True,
    )


def test_graphdb_validation_allows_explicit_post_only_mode():
    assert MODULE._graphdb_validation_succeeded({"post": {"persisted": True}}, query_required=False)
    assert not MODULE._graphdb_validation_succeeded({"post": {"persisted": False}}, query_required=False)


def test_graphdb_connection_refusal_has_actionable_diagnosis():
    result = MODULE.post_turtle(
        "http://127.0.0.1:1/repositories/mazu/statements",
        "@prefix mazu: <https://mazu.example.org/saudi#> .",
    )

    assert result["persisted"] is False
    assert "not listening" in result["diagnosis"]

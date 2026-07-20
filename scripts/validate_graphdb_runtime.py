"""Validate and optionally persist a MAZU runtime instance Turtle export."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.kg import validate_instance_ttl  # noqa: E402


def post_turtle(endpoint: str, payload: str) -> dict[str, Any]:
    """POST Turtle to a GraphDB statements endpoint."""

    request = urllib.request.Request(
        endpoint,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "text/turtle", "Accept": "application/json, text/plain, */*"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {"persisted": True, "http_status": response.status, "response_preview": body[:500]}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"persisted": False, "http_status": exc.code, "error": body[:500]}
    except Exception as exc:
        return {"persisted": False, "error": str(exc), "diagnosis": _connection_diagnosis(exc)}


def _connection_diagnosis(exc: Exception) -> str:
    """Turn common urllib failures into an actionable deployment diagnosis."""

    message = str(exc)
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, ConnectionRefusedError):
        return "GraphDB is not listening at the endpoint host/port; start GraphDB or correct --endpoint."
    if "timed out" in message.lower():
        return "GraphDB did not respond before the timeout; check service health and network routing."
    return "The GraphDB endpoint could not be reached; inspect the endpoint and service logs."


def query_grounding_gaps(repository_endpoint: str) -> dict[str, Any]:
    """Query persisted grounding records from a GraphDB repository endpoint."""

    query = """
PREFIX mazu: <https://mazu.example.org/saudi#>
SELECT (COUNT(?gap) AS ?gapCount) (COUNT(?payload) AS ?payloadCount)
WHERE {
  ?event mazu:hasGroundingGap ?gap .
  ?gap a mazu:GroundingGap ; mazu:groundingPayload ?payload .
}
""".strip()
    url = repository_endpoint.rstrip("/") + "?query=" + urllib.parse.quote(query)
    request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        bindings = result.get("results", {}).get("bindings", [])
        values = bindings[0] if bindings else {}
        return {
            "queried": True,
            "gap_count": int(values.get("gapCount", {}).get("value", 0)),
            "payload_count": int(values.get("payloadCount", {}).get("value", 0)),
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"queried": False, "http_status": exc.code, "error": body[:500]}
    except Exception as exc:
        return {"queried": False, "error": str(exc), "diagnosis": _connection_diagnosis(exc)}


def _default_query_endpoint(statements_endpoint: str) -> str:
    return statements_endpoint.rstrip("/").removesuffix("/statements")


def _graphdb_validation_succeeded(result: dict[str, Any], *, query_required: bool) -> bool:
    """Return whether the requested persistence and read-back checks passed."""

    if not result.get("post", {}).get("persisted"):
        return False
    if not query_required:
        return True
    query = result.get("query", {})
    return bool(
        query.get("queried")
        and query.get("gap_count", 0) > 0
        and query.get("payload_count", 0) > 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ttl", type=Path, help="Runtime instance Turtle export")
    parser.add_argument("--endpoint", required=True, help="GraphDB /repositories/<repo>/statements URL")
    parser.add_argument("--query-endpoint", help="Repository URL; defaults to --endpoint without /statements")
    parser.add_argument("--skip-query", action="store_true", help="Only validate and POST; do not query GraphDB")
    args = parser.parse_args()

    payload = args.ttl.read_text(encoding="utf-8")
    validation = validate_instance_ttl(payload)
    result: dict[str, Any] = {"ttl": str(args.ttl), "validation": validation}
    if not validation["valid"]:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    result["post"] = post_turtle(args.endpoint, payload)
    if result["post"].get("persisted") and not args.skip_query:
        result["query"] = query_grounding_gaps(args.query_endpoint or _default_query_endpoint(args.endpoint))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if _graphdb_validation_succeeded(result, query_required=not args.skip_query) else 1


if __name__ == "__main__":
    sys.exit(main())

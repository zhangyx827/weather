"""Check a local GraphDB instance and ensure the MAZU repository exists."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "graphdb" / "mazu-repository-config.ttl"


def _request(url: str, *, data: bytes | None = None, content_type: str | None = None) -> tuple[int, str]:
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def _repository_ids(payload: str) -> set[str]:
    parsed = json.loads(payload)
    if isinstance(parsed, list):
        return {str(item.get("id", "")) for item in parsed if isinstance(item, dict)}
    bindings = parsed.get("results", {}).get("bindings", []) if isinstance(parsed, dict) else []
    return {str(item.get("id", {}).get("value", "")) for item in bindings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:7200")
    parser.add_argument("--repository", default="mazu")
    parser.add_argument("--create", action="store_true", help="Create the repository when it is absent.")
    args = parser.parse_args()
    host = args.host.rstrip("/")
    try:
        status, body = _request(f"{host}/rest/repositories")
        repositories = _repository_ids(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(json.dumps({"ok": False, "stage": "list_repositories", "http_status": exc.code, "error": body[:500]}, indent=2))
        return 1
    except Exception as exc:
        print(json.dumps({"ok": False, "stage": "connect", "host": host, "error": str(exc)}, indent=2))
        return 1

    if args.repository in repositories:
        print(json.dumps({"ok": True, "repository": args.repository, "created": False, "repositories": sorted(repositories)}, indent=2))
        return 0
    if not args.create:
        print(json.dumps({"ok": False, "stage": "repository_missing", "repository": args.repository, "repositories": sorted(repositories)}, indent=2))
        return 2

    if args.repository != "mazu":
        print(json.dumps({"ok": False, "stage": "create_repository", "error": "the checked-in config is for repository 'mazu'; use --repository mazu"}, indent=2))
        return 2
    config = CONFIG_PATH.read_bytes()
    boundary = "----mazu-graphdb-config"
    multipart = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="config"; filename="mazu-repository-config.ttl"\r\n'
        "Content-Type: text/turtle\r\n\r\n"
    ).encode("ascii") + config + f"\r\n--{boundary}--\r\n".encode("ascii")
    try:
        status, body = _request(
            f"{host}/rest/repositories",
            data=multipart,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(json.dumps({"ok": False, "stage": "create_repository", "http_status": exc.code, "error": body[:500]}, indent=2))
        return 1
    except Exception as exc:
        print(json.dumps({"ok": False, "stage": "create_repository", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"ok": 200 <= status < 300, "repository": args.repository, "created": 200 <= status < 300, "http_status": status, "response_preview": body[:500]}, indent=2))
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())

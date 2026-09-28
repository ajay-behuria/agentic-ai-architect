"""server.py

Local stdio-based MCP server built on the official MCP Python SDK (FastMCP).

The server exposes exactly two tools that back the Prior Authorization
workflow:

* ``get_patient(patient_id)`` -- returns the member's name and plan status.
* ``get_rule()``             -- returns the policy text, evaluation order, and
                                numeric thresholds.

Design contract:
    Both tools read synthetic JSON seed data from the local ``data/`` directory.
    On any failure (unknown patient, missing file, malformed JSON) the tool
    raises a clean exception. FastMCP marshals the raised exception into a
    protocol-level ``isError`` tool result so the calling client can bubble up a
    faithful error message rather than acting on an invalid/default payload.

Run standalone (for manual inspection) with::

    python server.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Paths & server instance
# --------------------------------------------------------------------------- #

# Resolve data files relative to this script so the server works regardless of
# the caller's current working directory.
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
PATIENTS_FILE: Path = DATA_DIR / "patients.json"
RULES_FILE: Path = DATA_DIR / "rules.json"

mcp = FastMCP("prior-auth-data-server")


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> Any:
    """Load and parse a JSON file, raising a clean exception on failure.

    Args:
        path: Absolute path to the JSON file to read.

    Returns:
        The parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contents are not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required data file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
def get_patient(patient_id: str) -> dict[str, Any]:
    """Return a patient's name and benefit-plan status.

    Args:
        patient_id: The patient identifier, e.g. ``"P001"``.

    Returns:
        A mapping with ``patient_id``, ``name`` and ``plan_active`` keys.

    Raises:
        KeyError: If ``patient_id`` is not present in the patient roster.
        FileNotFoundError / ValueError: On data-file access or parse errors.
    """
    patients: dict[str, Any] = _load_json(PATIENTS_FILE)

    record = patients.get(patient_id)
    if record is None:
        # Cleanly signal "not found" instead of returning a default payload.
        raise KeyError(f"Patient '{patient_id}' was not found in the roster.")

    return {
        "patient_id": patient_id,
        "name": record["name"],
        "plan_active": bool(record["plan_active"]),
    }


@mcp.tool()
def get_rule() -> dict[str, Any]:
    """Return the prior-authorization policy, evaluation order and thresholds.

    Returns:
        A mapping with ``rules_text``, ``evaluation_order`` and ``thresholds``.

    Raises:
        FileNotFoundError / ValueError: On data-file access or parse errors.
    """
    rules: dict[str, Any] = _load_json(RULES_FILE)

    # Validate the shape up-front so downstream consumers get a clean error
    # rather than a KeyError deep inside the decision logic.
    for required_key in ("rules_text", "evaluation_order", "thresholds"):
        if required_key not in rules:
            raise ValueError(f"rules.json is missing required key '{required_key}'.")

    return {
        "rules_text": rules["rules_text"],
        "evaluation_order": rules["evaluation_order"],
        "thresholds": rules["thresholds"],
    }


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Communicate with clients over stdio (the transport used by app.py).
    mcp.run(transport="stdio")

"""test.py

Headless test orchestrator for the Prior Authorization Assistant.

Validates the three mandatory clinical cases by driving the LangGraph workflow
programmatically:

    * P001 -> Approve
    * P002 -> Deny
    * P003 -> Need more information

For each case the harness initializes state, invokes the graph, catches the
human-in-the-loop interrupt, injects a simulated ``yes`` confirmation (so the
run never blocks on interactive terminal input), and resumes execution on the
checkpointer thread. It then compares the proposed recommendation against the
expected outcome and prints ``PASS`` / ``FAIL`` per patient.

Run with::

    python test.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app import RECOMMEND_APPROVE, RECOMMEND_DENY, RECOMMEND_NEED_INFO, run_workflow

BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"

# (patient_id, expected_recommendation)
TEST_CASES: list[tuple[str, str]] = [
    ("P001", RECOMMEND_APPROVE),
    ("P002", RECOMMEND_DENY),
    ("P003", RECOMMEND_NEED_INFO),
]


def _auto_yes(_interrupt_payload: Any) -> str:
    """Simulated reviewer that always confirms, bypassing terminal blocking."""
    return "yes"


def run_case(patient_id: str, expected: str) -> bool:
    """Execute a single case and report whether it matches the expectation.

    Args:
        patient_id: The patient identifier to evaluate.
        expected: The expected recommendation string.

    Returns:
        ``True`` if the produced recommendation equals ``expected``.
    """
    note_path = str(DATA_DIR / f"{patient_id}.txt")

    try:
        final_state = run_workflow(
            patient_id=patient_id,
            clinical_note_path=note_path,
            reviewer_input_provider=_auto_yes,
            thread_id=f"test-{patient_id}",
        )
    except Exception as exc:  # noqa: BLE001 - a halted workflow is a failed case
        print(f"Patient {patient_id}: FAIL (error: {exc})")
        return False

    actual = final_state.get("proposed_recommendation")
    passed = actual == expected

    status = "PASS" if passed else "FAIL"
    detail = f"expected={expected!r}, actual={actual!r}"
    print(f"Patient {patient_id}: {status} ({detail})")
    return passed


def main() -> None:
    """Run all mandatory cases and exit non-zero if any case fails."""
    print("=== Prior Authorization Assistant :: Test Harness ===")

    results = [run_case(patient_id, expected) for patient_id, expected in TEST_CASES]

    total = len(results)
    passed = sum(results)
    print(f"\nSummary: {passed}/{total} cases passed.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

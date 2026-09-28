"""app.py

Prior Authorization Assistant -- LangGraph workflow.

Topology::

    START -> reader -> decision -> human_review -> END
                                       |
                                       +-- (MCP client) --> server.py tools

Nodes
-----
* ``reader``       : LLM-backed structured extraction of clinical metrics from
                     the free-text note. Falls back to a deterministic regex
                     parser when no LLM provider is configured, so the pipeline
                     remains fully runnable offline.
* ``decision``     : Deterministic (no-LLM) rule engine. Acts as an MCP client,
                     calling ``get_patient`` and ``get_rule`` on the running
                     stdio MCP server, then applies the policy logic.
* ``human_review`` : Human-in-the-loop gate. Uses LangGraph ``interrupt()`` to
                     pause execution and capture a ``yes``/``no`` confirmation.

The compiled graph uses an in-memory ``MemorySaver`` checkpointer so the
interrupt can be resumed on the same thread.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables (LLM provider keys, model name) from .env.
load_dotenv()

# --------------------------------------------------------------------------- #
# Constants & paths
# --------------------------------------------------------------------------- #

BASE_DIR: Path = Path(__file__).resolve().parent
SERVER_SCRIPT: Path = BASE_DIR / "server.py"

RECOMMEND_APPROVE = "Approve"
RECOMMEND_DENY = "Deny"
RECOMMEND_NEED_INFO = "Need more information"


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #

class PriorAuthState(TypedDict, total=False):
    """Explicit graph state shared across all nodes.

    Attributes:
        patient_id: Patient identifier, e.g. ``"P001"``.
        clinical_note_path: Absolute/relative path to the clinical note file.
        pain_weeks: Extracted weeks of back pain (``None`` if undocumented).
        physio_weeks: Extracted weeks of physiotherapy (``None`` if undocumented).
        plan_active: Whether the member's benefit plan is active.
        proposed_recommendation: One of Approve / Deny / Need more information.
        reason: Concise one-line justification for the recommendation.
        review_decision: Reviewer's confirmation, ``"yes"`` or ``"no"``.
    """

    patient_id: str
    clinical_note_path: str
    pain_weeks: Optional[int]
    physio_weeks: Optional[int]
    plan_active: Optional[bool]
    proposed_recommendation: Optional[str]
    reason: Optional[str]
    review_decision: Optional[str]


# --------------------------------------------------------------------------- #
# Reader node -- LLM structured output with deterministic fallback
# --------------------------------------------------------------------------- #

try:
    # Pydantic model powers structured LLM output when a provider is available.
    from pydantic import BaseModel, Field

    class ClinicalMetrics(BaseModel):
        """Structured metrics extracted from a clinical note."""

        pain_weeks: Optional[int] = Field(
            default=None,
            description="Documented weeks of back pain; null if not mentioned.",
        )
        physio_weeks: Optional[int] = Field(
            default=None,
            description="Documented weeks of physiotherapy; null if not mentioned.",
        )

except Exception:  # pragma: no cover - pydantic is a hard dependency
    ClinicalMetrics = None  # type: ignore[assignment]


_EXTRACTION_SYSTEM_PROMPT = (
    "You are a clinical data extraction assistant. Read the clinical note and "
    "extract exactly two integers: the number of weeks of back pain and the "
    "number of weeks of completed physiotherapy. If a value is not documented "
    "at all, return null for that field. Respond ONLY with the structured "
    "object; do not infer or fabricate values."
)


def _build_llm() -> Optional[Any]:
    """Instantiate an LLM client if a provider API key is configured.

    Returns:
        A LangChain chat model, or ``None`` when no provider/key is available.
    """
    import os

    model_name = os.getenv("LLM_MODEL")

    # Prefer OpenAI, then Anthropic, based on which key is present.
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model_name or "gpt-4o-mini", temperature=0)
        except Exception:
            return None

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-latest", temperature=0
            )
        except Exception:
            return None

    return None


def _regex_extract(note_text: str) -> tuple[Optional[int], Optional[int]]:
    """Deterministically extract metrics from the note text.

    Used as an offline fallback when no LLM provider is configured. Returns
    ``None`` for any metric that is not explicitly documented.

    Args:
        note_text: Raw clinical note text.

    Returns:
        A ``(pain_weeks, physio_weeks)`` tuple.
    """
    lowered = note_text.lower()

    def _find(keywords: tuple[str, ...]) -> Optional[int]:
        for keyword in keywords:
            # Match e.g. "back pain for 10 weeks" / "physiotherapy for 8 weeks".
            match = re.search(
                rf"{keyword}[^.\n]*?(\d+)\s*weeks?", lowered
            )
            if match:
                return int(match.group(1))
        return None

    pain_weeks = _find(("back pain", "pain"))
    physio_weeks = _find(("physiotherapy", "physio", "physical therapy"))
    return pain_weeks, physio_weeks


def _extract_metrics(note_text: str) -> tuple[Optional[int], Optional[int]]:
    """Extract clinical metrics, preferring an LLM and falling back to regex.

    Args:
        note_text: Raw clinical note text.

    Returns:
        A ``(pain_weeks, physio_weeks)`` tuple.
    """
    llm = _build_llm()
    if llm is not None and ClinicalMetrics is not None:
        try:
            structured = llm.with_structured_output(ClinicalMetrics)
            result: ClinicalMetrics = structured.invoke(
                [
                    ("system", _EXTRACTION_SYSTEM_PROMPT),
                    ("human", note_text),
                ]
            )
            return result.pain_weeks, result.physio_weeks
        except Exception:
            # Any LLM/parse failure degrades gracefully to the regex parser.
            pass

    return _regex_extract(note_text)


def reader_node(state: PriorAuthState) -> dict[str, Any]:
    """Reader Agent node: extract structured metrics from the clinical note.

    Args:
        state: Current graph state (must contain ``clinical_note_path``).

    Returns:
        A partial state update with ``pain_weeks`` and ``physio_weeks``.

    Raises:
        FileNotFoundError: If the clinical note file cannot be read.
    """
    note_path = Path(state["clinical_note_path"])
    if not note_path.exists():
        raise FileNotFoundError(f"Clinical note not found: {note_path}")

    note_text = note_path.read_text(encoding="utf-8")
    pain_weeks, physio_weeks = _extract_metrics(note_text)

    return {"pain_weeks": pain_weeks, "physio_weeks": physio_weeks}


# --------------------------------------------------------------------------- #
# MCP client helpers (used by the decision node)
# --------------------------------------------------------------------------- #

def _tool_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert an MCP ``CallToolResult`` into a plain dict.

    Args:
        result: The result object returned by ``ClientSession.call_tool``.

    Returns:
        The tool's returned mapping.

    Raises:
        RuntimeError: If the tool reported an error or returned no content.
    """
    if getattr(result, "isError", False):
        message = _first_text(result) or "Unknown MCP tool error."
        raise RuntimeError(message)

    # Prefer the already-parsed structured content when present.
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    text = _first_text(result)
    if text is None:
        raise RuntimeError("MCP tool returned no content.")
    return json.loads(text)


def _first_text(result: Any) -> Optional[str]:
    """Return the first text block from an MCP tool result, if any."""
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            return text
    return None


async def _fetch_patient_and_rules(patient_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Spawn the stdio MCP server and call both tools.

    Args:
        patient_id: The patient identifier to look up.

    Returns:
        A ``(patient, rules)`` tuple of dictionaries.

    Raises:
        RuntimeError: If either MCP tool call fails.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(BASE_DIR),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            patient_result = await session.call_tool(
                "get_patient", {"patient_id": patient_id}
            )
            patient = _tool_result_to_dict(patient_result)

            rule_result = await session.call_tool("get_rule", {})
            rules = _tool_result_to_dict(rule_result)

    return patient, rules


def _root_cause(exc: BaseException) -> BaseException:
    """Unwrap nested ExceptionGroups to the most informative leaf exception.

    ``stdio_client`` runs tool calls inside an anyio TaskGroup, so a tool
    failure surfaces as an ``ExceptionGroup`` whose real cause is nested. This
    walks into the single/first sub-exception so callers see the actual error.

    Args:
        exc: The (possibly grouped) exception raised by the async runtime.

    Returns:
        The unwrapped leaf exception.
    """
    current = exc
    # ``exceptions`` is present on ExceptionGroup / anyio's grouped errors.
    inner = getattr(current, "exceptions", None)
    while inner:
        current = inner[0]
        inner = getattr(current, "exceptions", None)
    return current


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync code, robust to existing loops.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop -> safe to use asyncio.run (default path).
        try:
            return asyncio.run(coro)
        except BaseException as exc:  # unwrap anyio TaskGroup ExceptionGroup
            raise _root_cause(exc) from None

    # A loop is already running (e.g. inside another async context): execute the
    # coroutine on a dedicated thread with its own event loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            return pool.submit(lambda: asyncio.run(coro)).result()
        except BaseException as exc:  # unwrap anyio TaskGroup ExceptionGroup
            raise _root_cause(exc) from None


# --------------------------------------------------------------------------- #
# Decision node -- deterministic policy engine (MCP client)
# --------------------------------------------------------------------------- #

def decision_node(state: PriorAuthState) -> dict[str, Any]:
    """Decision Agent node: apply deterministic policy using MCP-sourced data.

    Args:
        state: Current graph state (must contain ``patient_id``, ``pain_weeks``,
            ``physio_weeks``).

    Returns:
        A partial state update with ``plan_active``,
        ``proposed_recommendation`` and ``reason``.

    Raises:
        RuntimeError: If an MCP tool call fails; the workflow halts cleanly
            instead of emitting a faulty recommendation.
    """
    patient_id = state["patient_id"]

    # --- Fetch authoritative data via MCP (bubble up any failure) ---------- #
    try:
        patient, rules = _run_async(_fetch_patient_and_rules(patient_id))
    except Exception as exc:  # noqa: BLE001 - re-raise as a clean halt signal
        raise RuntimeError(
            f"MCP data retrieval failed for patient '{patient_id}': {exc}"
        ) from exc

    plan_active: bool = bool(patient["plan_active"])
    thresholds: dict[str, Any] = rules["thresholds"]
    min_pain: int = int(thresholds["min_pain_weeks"])
    min_physio: int = int(thresholds["min_physio_weeks"])

    pain_weeks: Optional[int] = state.get("pain_weeks")
    physio_weeks: Optional[int] = state.get("physio_weeks")

    # --- Sequential policy evaluation (order per rules.json) ---------------- #
    if not plan_active:
        recommendation = RECOMMEND_DENY
        reason = "Benefit plan is not active."
    elif (pain_weeks is not None and pain_weeks < min_pain) or (
        physio_weeks is not None and physio_weeks < min_physio
    ):
        recommendation = RECOMMEND_DENY
        reason = (
            f"Documented duration below threshold "
            f"(pain>={min_pain}w, physio>={min_physio}w required)."
        )
    elif pain_weeks is None or physio_weeks is None:
        recommendation = RECOMMEND_NEED_INFO
        missing = "pain duration" if pain_weeks is None else "physiotherapy duration"
        reason = f"Missing required clinical information: {missing}."
    else:
        recommendation = RECOMMEND_APPROVE
        reason = (
            f"Plan active; pain {pain_weeks}w and physio {physio_weeks}w "
            f"both meet thresholds."
        )

    return {
        "plan_active": plan_active,
        "proposed_recommendation": recommendation,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Human review node -- LangGraph interrupt
# --------------------------------------------------------------------------- #

def human_review_node(state: PriorAuthState) -> dict[str, Any]:
    """Human-in-the-loop node: pause for reviewer confirmation.

    Uses ``interrupt()`` to suspend the graph and surface the proposed
    recommendation. Execution resumes when the caller supplies a ``yes``/``no``
    string via ``Command(resume=...)``.

    Args:
        state: Current graph state.

    Returns:
        A partial state update with the normalized ``review_decision``.
    """
    reviewer_answer = interrupt(
        {
            "question": "Confirm this decision? (yes/no)",
            "recommendation": state.get("proposed_recommendation"),
            "reason": state.get("reason"),
        }
    )

    decision = "yes" if str(reviewer_answer).strip().lower() == "yes" else "no"
    return {"review_decision": decision}


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #

def build_graph() -> Any:
    """Build and compile the Prior Authorization LangGraph workflow.

    Returns:
        A compiled graph with an in-memory checkpointer enabling interrupts.
    """
    graph = StateGraph(PriorAuthState)

    graph.add_node("reader", reader_node)
    graph.add_node("decision", decision_node)
    graph.add_node("human_review", human_review_node)

    graph.add_edge(START, "reader")
    graph.add_edge("reader", "decision")
    graph.add_edge("decision", "human_review")
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=MemorySaver())


def initial_state(patient_id: str, clinical_note_path: str) -> PriorAuthState:
    """Construct a fresh initial state for a workflow run.

    Args:
        patient_id: The patient identifier.
        clinical_note_path: Path to the clinical note file.

    Returns:
        A populated ``PriorAuthState``.
    """
    return PriorAuthState(
        patient_id=patient_id,
        clinical_note_path=clinical_note_path,
        pain_weeks=None,
        physio_weeks=None,
        plan_active=None,
        proposed_recommendation=None,
        reason=None,
        review_decision=None,
    )


def run_workflow(
    patient_id: str,
    clinical_note_path: str,
    reviewer_input_provider: Callable[[Any], str],
    thread_id: Optional[str] = None,
) -> PriorAuthState:
    """Execute the workflow end-to-end, handling the human-review interrupt.

    Args:
        patient_id: The patient identifier.
        clinical_note_path: Path to the clinical note file.
        reviewer_input_provider: Callable given the interrupt payload and
            returning the reviewer's ``yes``/``no`` answer.
        thread_id: Optional checkpointer thread id (defaults to ``patient_id``).

    Returns:
        The final graph state after resuming past the interrupt.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or patient_id}}

    result: dict[str, Any] = graph.invoke(
        initial_state(patient_id, clinical_note_path), config
    )

    # If the graph paused at the human-review interrupt, collect input & resume.
    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"]
        answer = reviewer_input_provider(interrupt_payload)
        result = graph.invoke(Command(resume=answer), config)

    return result  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Interactive CLI entrypoint
# --------------------------------------------------------------------------- #

def _prompt_reviewer(_payload: Any) -> str:
    """Interactive reviewer-input provider using the terminal."""
    try:
        return input("Reviewer confirmation (yes/no): ").strip()
    except EOFError:
        return "no"


def main() -> None:
    """Run the assistant interactively for a single patient case."""
    # Default demo case; override via CLI args: python app.py P003 data/P003.txt
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "P001"
    note_path = (
        sys.argv[2] if len(sys.argv) > 2 else str(BASE_DIR / "data" / f"{patient_id}.txt")
    )

    print(f"=== Prior Authorization Assistant :: {patient_id} ===")

    try:
        final_state = run_workflow(patient_id, note_path, _prompt_reviewer)
    except Exception as exc:  # noqa: BLE001 - present a clean halt to the user
        print(f"Workflow halted due to error: {exc}")
        sys.exit(1)

    if final_state.get("review_decision") == "yes":
        print(f"Recommendation: {final_state.get('proposed_recommendation')}")
        print(f"Reason: {final_state.get('reason')}")
    else:
        print("Decision rejected by reviewer")


if __name__ == "__main__":
    main()

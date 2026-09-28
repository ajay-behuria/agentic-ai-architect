# Prior Authorization Assistant

A production-grade, end-to-end **Prior Authorization** workflow built with
**LangGraph** and the **Model Context Protocol (MCP)**. A Reader agent extracts
clinical metrics from a free-text note, a deterministic Decision agent applies
policy using data served over MCP, and a human-in-the-loop interrupt gates the
final outcome.

> Synthetic data only. No real PHI. API keys are read from environment
> variables via `.env` and are never committed.

---

## 1. Setup & Installation

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional — see note below)
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

`.env.example` structure:

```dotenv
OPENAI_API_KEY=        # provide ONE provider key (optional)
ANTHROPIC_API_KEY=
LLM_MODEL=             # optional model override
```

> **No API key? Still runs.** The Reader agent uses an LLM with structured
> outputs when a provider key is present, and otherwise falls back to a
> deterministic regex parser. This keeps the pipeline and tests fully runnable
> offline.

---

## 2. Running

```bash
# Run the assistant interactively for a single case (defaults to P001).
# You will be prompted for a yes/no reviewer confirmation.
python app.py                 # P001
python app.py P003 data/P003.txt

# Run the automated test suite (headless, auto-confirms the interrupt).
python test.py
```

> The MCP server (`server.py`) is launched automatically as a stdio subprocess
> by the Decision node — you do **not** need to start it separately. To inspect
> it standalone, run `python server.py`.

---

## 3. Workflow Topology

```
                         data/                       data/
                    +-----------+                +-----------+
                    | P00X.txt  |                |patients.  |
                    | (note)    |                |json,rules.|
                    +-----+-----+                |json       |
                          |                      +-----+-----+
                          v                            ^
   START            +-----------+                      | (reads)
     |              |  READER   |  LLM structured out  |
     +------------->|  (agent)  |  -> pain_weeks,       |
                    |           |     physio_weeks      |
                    +-----+-----+                       |
                          |  state                      |
                          v                             |
                    +-----------+       MCP stdio       |
                    | DECISION  |======================>+  server.py
                    |  (rules)  |<====== get_patient ===+  (MCP tools:
                    | MCP client|<====== get_rule ======+   get_patient,
                    +-----+-----+                           get_rule)
                          |  proposed_recommendation, reason
                          v
                    +-----------+
                    |  HUMAN    |  <== interrupt() pause ==> reviewer yes/no
                    |  REVIEW   |      (MemorySaver checkpointer resume)
                    +-----+-----+
                          |
                          v
                        END  --> "Approve/Deny/Need more information" + reason
                                 or "Decision rejected by reviewer"
```

---

## 4. Design Overview, Choices & Assumptions

**Components**
- `server.py` — Local **stdio MCP server** (official SDK / FastMCP) exposing
  `get_patient` and `get_rule`. Both raise clean exceptions on unknown IDs or
  file errors (surfaced to clients as protocol-level tool errors).
- `app.py` — **LangGraph** workflow with an explicit `TypedDict` graph state and
  three nodes: `reader` (LLM structured extraction), `decision` (deterministic
  MCP-client rule engine), and `human_review` (`interrupt()` gate). Compiled
  with an in-memory `MemorySaver` checkpointer so interrupts resume by thread.
- `test.py` — Headless harness that runs the three mandatory cases, auto-injects
  a `yes` confirmation, resumes via checkpointer threads, and reports PASS/FAIL.

**Decision logic (evaluation order)**
1. Plan inactive → **Deny**.
2. `pain_weeks < 6` **or** `physio_weeks < 6` (documented values only) → **Deny**.
3. `pain_weeks` **or** `physio_weeks` is `null` (undocumented) → **Need more information**.
4. Otherwise → **Approve**.

**Assumptions**
- Exactly 6 weeks meets the threshold; "no physio" = `0`; unmentioned = `null`.
- The Decision node performs **no** LLM call — it is fully deterministic.
- The Reader falls back to regex extraction when no LLM provider is configured,
  guaranteeing reproducible, offline-runnable behavior for the test suite.
- MCP tool failures bubble up as a clean error and halt the run rather than
  emitting a faulty recommendation.

**Expected outcomes**

| Patient | Plan   | Pain | Physio | Outcome               |
|---------|--------|------|--------|-----------------------|
| P001    | active | 10w  | 8w     | Approve               |
| P002    | inactive | 12w | 8w    | Deny                  |
| P003    | active | 9w   | null   | Need more information |

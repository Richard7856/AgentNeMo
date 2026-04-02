# Insurance Claims Agent

A production-grade multi-agent system for insurance claims triage, built with
**NVIDIA NeMo Agent Toolkit (NAT) v1.5** and deployed via NVIDIA NIM.

This project demonstrates:
- Custom tool registration with NAT's `@register_function` pattern
- ReAct orchestration routing queries to specialist tools
- RAG over insurance policy PDFs via FAISS + NVIDIA embeddings
- Structured Pydantic outputs for reliable downstream processing
- End-to-end claims workflow: classify → check coverage → retrieve required docs

---

## Architecture

```
User Query
    |
    v
[ReAct Orchestrator Agent]  (meta/llama-3.3-70b-instruct via NVIDIA NIM)
    |
    +-- [required_docs]      Pure lookup: required documents by claim type
    |
    +-- [classify_claim]     LLM: structured classification (type, priority, handler)
    |
    +-- [policy_search]      RAG: semantic search over 3 policy PDFs via FAISS
    |
    +-- [check_coverage]     LLM + RAG: binary coverage decision with deductible/limits
```

All agents are config-driven via `configs/config.yml`. Swapping models or adding tools
requires only a YAML change — no code modification.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | NVIDIA NeMo Agent Toolkit (NAT) v1.5 |
| LLM | meta/llama-3.3-70b-instruct via NVIDIA NIM |
| Embeddings | nvidia/nv-embedqa-e5-v5 via NVIDIA NIM |
| Vector Store | FAISS (local, no infra required) |
| PDF Parsing | pypdf + langchain |
| Package Manager | uv |
| Python | 3.12 |

---

## Quick Start

### Prerequisites
- Python 3.12
- uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/settings/api-keys)

### Setup

```bash
# 1. Clone and install
git clone <repo-url>
cd insurance-claims-agent
uv sync
uv pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env and set: NVIDIA_API_KEY=nvapi-...

# 3. Generate sample policies and build FAISS index
make generate-pdfs
make ingest

# 4. Run the agent
make run INPUT="What documents do I need for an auto claim?"
```

> **Note:** The Makefile uses `PYTHONPATH="." .venv/bin/nat run` instead of `uv run nat run`.
> This is required because the project lives in an iCloud path with spaces ("Mobile Documents"),
> which prevents Python's editable install mechanism from working correctly with `uv run`.
> See [docs/DECISIONS.md](docs/DECISIONS.md) for the full explanation.

---

## Example Queries

**Single-tool queries:**
```bash
make run INPUT="What documents do I need to file a home insurance claim?"
make run INPUT="What is the deductible for auto collision coverage?"
```

**Classification:**
```bash
make run INPUT="My car was totaled by a drunk driver and I have a broken arm. How urgent is this?"
```

**Coverage check:**
```bash
make run INPUT="I was driving drunk and crashed my car. Is this covered by my auto policy?"
# Expected: NOT covered -- DUI is an explicit exclusion in the policy

make run INPUT="Someone hit my parked car. Is this covered? What documents do I need?"
# Expected: Covered under collision, 3% deductible, + required documents list
```

**Multi-tool orchestration:**
```bash
make run INPUT="Water from upstairs flooded my apartment. Classify the claim, check coverage, and tell me what documents I need."
# Uses: classify_claim + check_coverage + required_docs in sequence
```

---

## NAT Tool Registration Pattern

Each tool follows this pattern (see `insurance_claims/tools/`):

```python
from nat.data_models.function import FunctionBaseConfig
from nat.cli.register_workflow import register_function
from nat.builder.function_info import FunctionInfo

class MyToolConfig(FunctionBaseConfig, name="my_tool"):
    """Tool description -- appears in nat info."""
    param: str = Field(default="value", description="...")

@register_function(config_type=MyToolConfig)
async def my_tool_function(config: MyToolConfig, builder: Builder):
    async def _impl(input: str) -> str:
        return f"result: {input}"
    yield FunctionInfo.from_fn(_impl, description=_impl.__doc__)
```

NAT discovers tools via the `nat.components` entry point in `pyproject.toml`:
```toml
[project.entry-points.'nat.components']
insurance_claims = "insurance_claims.register"
```

---

## Project Structure

```
insurance-claims-agent/
├── configs/
│   └── config.yml              # NAT workflow -- models, tools, orchestrator
├── insurance_claims/
│   ├── register.py             # Entry point: imports all tools for NAT discovery
│   ├── tools/
│   │   ├── required_docs.py    # Lookup: required documents by claim type
│   │   ├── classify_claim.py   # LLM: structured claim classification
│   │   ├── policy_search.py    # RAG: FAISS search over policy PDFs
│   │   └── check_coverage.py   # LLM+RAG: binary coverage decision
│   └── data_prep/
│       ├── generate_policies.py # Generates 3 sample policy PDFs
│       └── ingest_policies.py  # Embeds PDFs into FAISS index
├── data/
│   ├── policies/               # Sample insurance policy PDFs (synthetic)
│   └── faiss_index/            # Pre-built vector index (gitignored)
├── eval/
│   └── datasets/
│       └── claims_eval.json    # 15 evaluation Q&A pairs
├── docs/
│   ├── DECISIONS.md            # Architecture decision log
│   └── SETUP.md                # Detailed setup guide
├── Makefile                    # Convenience commands
└── pyproject.toml
```

---

## Domain Context

The insurance logic is based on real-world experience with P&C (property & casualty)
insurance operations. The sample policies use realistic:
- Coverage amounts in MXN (Mexican pesos)
- Standard exclusion clauses (DUI, intentional acts, gradual damage)
- Claims procedures aligned with Mexican insurance regulations
- Deductible structures common in Mexican market (% of insured value)

The system correctly handles nuanced scenarios:
- DUI exclusion (denies coverage even for collision)
- Gradual vs sudden water damage distinction
- Double indemnity for accidental death (life policy)
- Sublimits for jewelry/electronics within home theft coverage

---

## Observability — Phoenix Tracing

Every LLM call and tool invocation is traced via [Arize Phoenix](https://phoenix.arize.com/):

```bash
# Terminal 1 — start Phoenix dashboard
make phoenix

# Terminal 2 — run traced demo (3 queries across all 4 tools)
make trace
```

Open http://localhost:6006 to see the full trace hierarchy:

```
LangGraph (root)
  └── agent
        ├── RunnableSequence
        │     └── ChatNVIDIA  ← LLM call with tokens, latency, model
        └── ToolNode
              └── [tool name]  ← tool inputs/outputs
```

**How it works:** `trace_demo.py` calls `phoenix.otel.register()` to set the global OTEL
provider, then `LangChainInstrumentor().instrument()` to patch LangChain's callback system.
Both steps are required — `register()` alone does NOT capture LangChain spans.

---

## REST API — nat serve

```bash
make serve  # starts on http://localhost:8000
```

Three endpoint formats available:

```bash
# 1. Native NAT format
curl -X POST http://localhost:8000/v1/workflow \
  -H "Content-Type: application/json" \
  -d '{"value": "What documents do I need for an auto claim?"}'

# 2. Server-Sent Events (SSE) — streams intermediate tool calls
curl -X POST http://localhost:8000/v1/workflow/stream \
  -H "Content-Type: application/json" \
  -d '{"value": "Classify: my car was totaled in a flood"}'

# 3. OpenAI-compatible (drop-in replacement for GPT-4 clients)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "insurance-agent", "messages": [{"role": "user", "content": "Is DUI covered?"}]}'
```

Swagger UI available at http://localhost:8000/docs

---

## Future Roadmap

**Production Features:**
- [ ] Phoenix tracing in `nat serve` (instrument at startup, not just trace_demo.py)
- [ ] FastMCP server for CRM integration
- [ ] `nat eval` evaluation run with accuracy metrics
- [ ] Multi-tenant policy management (Milvus instead of FAISS)
- [ ] REST API authentication (API keys via NAT auth providers)

**Agent Capabilities:**
- [ ] Claims status tracking (Supabase integration)
- [ ] WhatsApp Business API frontend
- [ ] Document upload and OCR processing
- [ ] Automated reserve setting recommendations

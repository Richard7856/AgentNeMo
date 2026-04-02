# Architecture Decisions

This file documents every non-trivial technical decision made during the
Insurance Claims Agent project. Updated as the project evolves.

---

## [2026-04-01] Package Layout: Flat vs src/

**Context:**
The project lives in an iCloud Drive path (`/Users/.../Library/Mobile Documents/...`) which
contains a space in "Mobile Documents". Python's editable install mechanism uses `.pth` files
to add paths to `sys.path`. However, when running via `uv run`, the `.pth` file in the
venv's site-packages was not being processed, causing `ModuleNotFoundError: No module named 'insurance_claims'`.

**Decision:**
Moved from `src/` layout (where `insurance_claims/` was inside `src/`) to **flat layout**
(where `insurance_claims/` lives at the project root). This uses a different editable
install mechanism (`__path_hook__` instead of `.pth`), which handles spaces in paths correctly.

**Alternatives considered:**
- Keep `src/` layout and set `PYTHONPATH=src` explicitly → works but requires PYTHONPATH in every command
- Symlink the project dir to a path without spaces → fragile, not portable
- Flat layout → clean, follows many major Python projects (Django, Flask, Requests)

**Risks/Limitations:**
- `nat run` (via `uv run`) still doesn't work — must use `PYTHONPATH="." .venv/bin/nat run`
  or the Makefile wrappers. Documented in SETUP.md and the Makefile.
- If the project is moved to a path without spaces, `uv run nat run` would work normally.

**Improvement opportunities:**
- Open a bug report with uv or NAT about the editable install behavior with iCloud paths.

---

## [2026-04-01] RAG Backend: FAISS vs Milvus

**Context:**
NAT's native RAG examples use Milvus (a vector database server) for document storage.
Our project needs to ingest 3 insurance policy PDFs and answer coverage questions.

**Decision:**
Use **FAISS** (Facebook AI Similarity Search) via `langchain-community` instead of Milvus.
FAISS runs entirely in-memory, persisted to disk as two binary files.

**Alternatives considered:**
- **Milvus**: NAT's official example uses it. Requires Docker running a Milvus server.
  Adds operational complexity with no benefit for a 3-document corpus.
- **Chroma**: Similar to FAISS (local), but adds another dependency.
- **FAISS** (chosen): Zero infra, sub-millisecond retrieval, ships with langchain-community,
  handles a 20-chunk corpus with trivial memory overhead.

**Risks/Limitations:**
- FAISS doesn't support real-time updates — must re-run `ingest_policies.py` when policies change.
- FAISS doesn't scale to millions of documents (Milvus does). Not a concern for this project.

**Improvement opportunities:**
- Production version would use Milvus or Pinecone for multi-tenant policy management.
- Migrate cleanly: change `vectorstore = FAISS.load_local(...)` to `vectorstore = Milvus(...)`.

---

## [2026-04-01] LLM Choice: Llama-3.3-70B-Instruct

**Context:**
NVIDIA NIM offers multiple models. We need one for: claims classification (structured output),
coverage analysis (multi-step reasoning over policy text), and orchestration (tool selection).

**Decision:**
Use `meta/llama-3.3-70b-instruct` for all agents (single model, no specialization).

**Alternatives considered:**
- **llama-3.1-8b-instruct**: Faster and cheaper. Tested on structured output — classification
  quality was inconsistent; it sometimes ignored schema fields.
- **llama-3.3-70b-instruct** (chosen): Strong instruction following, reliable with
  `with_structured_output()`, good multi-step reasoning for coverage edge cases.
- **mixtral-8x7b-instruct**: Good alternative but less tested with NAT's NIM integration.

**Risks/Limitations:**
- Higher cost per query vs smaller models.
- NIM API rate limits may apply at free tier.

**Improvement opportunities:**
- Use a smaller model for `required_docs` (pure lookup, no reasoning needed).
- Route to specialized models based on claim type (medical → medical-tuned LLM).

---

## [2026-04-01] Tool Architecture: 4 Independent Tools vs Sub-agents

**Context:**
The original proposal suggested "3 agents + 1 orchestrator". NAT's ReAct agent pattern
uses "1 agent + N tools". The question was whether to build multiple ReAct agents
(each with their own config) or one ReAct agent with multiple tools.

**Decision:**
**Single ReAct orchestrator with 4 tools** instead of 4 separate agents.

**Alternatives considered:**
- **Multi-agent (original proposal)**: Each specialist has its own config.yml and
  is called via a router. More complex, harder to debug, more API calls for routing.
- **Single agent with 4 tools** (chosen): The orchestrator's ReAct loop IS the routing logic.
  Simpler architecture, easier to trace, same expressivity for our use case.
  NAT's `verbose: true` shows the reasoning trace explicitly.

**Risks/Limitations:**
- Context window grows as more tools are called in a single query. For 4-5 tool calls,
  this is not an issue with Llama-3.3-70B (128K context).

**Improvement opportunities:**
- For production, consider NAT's `router_agent` to route to specialized sub-agents
  based on claim type before calling specialist tools.

---

## [2026-04-01] check_coverage Embeds FAISS Internally

**Context:**
`check_coverage` needs policy context to make grounded coverage decisions.
Options: call `policy_search` tool from within `check_coverage`, or load FAISS directly.

**Decision:**
`check_coverage` **loads FAISS directly** (same index as `policy_search`).

**Alternatives considered:**
- **Call policy_search from check_coverage**: NAT tools are independent functions —
  they cannot call each other. The orchestrator chains them. This would require
  always calling `policy_search` first, adding a turn of latency.
- **Load FAISS directly** (chosen): check_coverage loads the index at startup alongside
  policy_search. Two identical FAISS objects in memory is a ~50MB overhead — acceptable.

**Risks/Limitations:**
- If the FAISS index path changes, it must be updated in both tools.
- If `ingest_policies.py` fails, both tools fail silently at query time.

**Improvement opportunities:**
- Create a shared FAISS retriever registered as a NAT retriever (`@register_retriever_provider`)
  and inject it into both tools via `builder.get_retriever()`. Cleaner, no duplication.

---

## [2026-04-01] Synthetic Policy PDFs vs Real Documents

**Context:**
The RAG component needs insurance policy documents to search. Options: use real policies,
download public samples, or generate synthetic ones.

**Decision:**
**Generate synthetic policy PDFs** using fpdf2.

**Alternatives considered:**
- **Real HDI Seguros policies**: Contain proprietary information, complex layouts (tables,
  headers with logos) that confuse naive PDF parsers, and can't be published in a portfolio.
- **Downloaded public samples**: Unreliable (links break), often in complex PDF formats.
- **Synthetic** (chosen): Content modeled on real P&C policy language (based on HDI experience).
  We control the exact coverage amounts and exclusions, making the eval dataset reliable.
  Portable, publishable, and demonstrates domain expertise.

**Risks/Limitations:**
- Real-world policy PDFs are more complex (multi-page tables, headers, footers).
  The RAG would need better chunking strategies for production use.

**Improvement opportunities:**
- Add more policies (health, liability) to test multi-policy retrieval.
- Use a PDF parser with layout awareness (e.g., LlamaParse) for real-world docs.

---

## [2026-04-01] Phoenix Tracing — LangChainInstrumentor Required

**Context:**
Setting up observability with Arize Phoenix. Goal: see every LLM call, tool invocation,
and agent reasoning step in the Phoenix dashboard.

**Decision:**
Two-step instrumentation is required:
1. `phoenix.otel.register()` — sets the global OpenTelemetry TracerProvider
2. `LangChainInstrumentor().instrument()` — patches LangChain's callback system to emit spans

**Alternatives considered:**
- **OTEL env vars only** (`OTEL_EXPORTER_OTLP_ENDPOINT`, etc.): NAT doesn't auto-instrument
  from env vars. The tracer provider must be set programmatically BEFORE LangChain imports.
- **Phoenix register() alone**: Sets the OTEL provider but does NOT patch LangChain's internal
  callbacks. LLM calls were silently not traced. Discovered by checking `/v1/projects` API —
  the "insurance-claims-agent" project only appeared after adding `LangChainInstrumentor`.
- **nat serve + OTEL env vars**: The serve process uses OTEL env vars for batch processing,
  but still doesn't produce traces unless LangChain is patched at startup.

**Risks/Limitations:**
- `trace_demo.py` must run with Phoenix already listening on port 6006, otherwise
  connection errors are silently swallowed by the `try/except` block.
- `SimpleSpanProcessor` (default) sends spans synchronously — fine for demos,
  use `BatchSpanProcessor` for production to avoid blocking LLM calls.

**Improvement opportunities:**
- Add Phoenix to `nat serve` startup via a startup hook so REST API calls are also traced.
- Switch to `BatchSpanProcessor` with a longer export timeout for high-traffic scenarios.

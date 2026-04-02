"""
trace_demo.py — runs the insurance claims agent with Phoenix tracing enabled.

WHY a separate script instead of relying on OTEL env vars:
NAT doesn't auto-instrument from OTEL env vars alone. The Phoenix tracer must be
registered as the global OpenTelemetry provider BEFORE any LangChain LLM calls run.
This script does that explicitly via arize-phoenix-otel's register(), then uses
nat.utils.run_workflow() to run queries so every LLM call and tool invocation is traced.

Usage:
    # Terminal 1 — Phoenix dashboard
    make phoenix

    # Terminal 2 — run traced demo
    PYTHONPATH="." .venv/bin/python trace_demo.py

Then open http://localhost:6006 to see the traces.
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"
PROJECT_NAME = "insurance-claims-agent"

# Register Phoenix BEFORE importing NAT or LangChain so every subsequent
# LLM call and tool invocation is automatically traced via OpenTelemetry.
try:
    from phoenix.otel import register
    tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint=PHOENIX_ENDPOINT,
    )
    print(f"Phoenix tracing registered → {PHOENIX_ENDPOINT}")

    # register() only sets the global OTEL provider — it does NOT auto-patch LangChain.
    # LangChainInstrumentor patches LangChain's callback system so every LLM call,
    # chain, and tool invocation emits OTEL spans that Phoenix can display.
    from openinference.instrumentation.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print(f"LangChain instrumentation enabled")
    print(f"Dashboard: http://localhost:6006/projects/{PROJECT_NAME}\n")
except Exception as e:
    print(f"Warning: Phoenix not available ({e}). Running without tracing.")
    tracer_provider = None

# Import NAT AFTER tracing is registered.
# PluginTypes.CONFIG_OBJECT = COMPONENT | FRONT_END | EVALUATOR | AUTHENTICATION —
# the same set the CLI loads when parsing a config file. This fires the registered
# hooks in nat/cli/type_registry.py which rebuild Config annotations automatically,
# so our custom tools (required_docs, classify_claim, policy_search, check_coverage)
# are discoverable when validate_schema() parses config.yml.
from nat.runtime.loader import PluginTypes, discover_and_register_plugins

discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

# Demo queries that exercise all 4 custom tools
DEMO_QUERIES = [
    "What documents do I need to file an auto insurance claim?",
    "My car was hit by a drunk driver at 2am. Both cars are totaled. I have a broken arm. Classify this claim and check if it's covered.",
    "What is the deductible for auto collision coverage and what are the coverage limits for third-party liability?",
]

CONFIG_FILE = "configs/config.yml"


async def run_demo():
    """Runs demo queries through the NAT workflow with Phoenix tracing."""
    # nat.utils.run_workflow is the high-level entry point — it handles:
    #   WorkflowBuilder.from_config() → SessionManager → session.run() → result
    # We just pass the config file path and the prompt. No manual builder needed.
    from nat.utils import run_workflow

    print("Running demo queries...\n")
    print("=" * 60)

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n[Query {i}/{len(DEMO_QUERIES)}]")
        print(f"Input: {query}")
        print("-" * 40)

        try:
            result = await run_workflow(config_file=CONFIG_FILE, prompt=query)
            output = result if isinstance(result, str) else str(result)
            print(f"Output: {output[:500]}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")

        print("=" * 60)

    print(f"\nDone! Open Phoenix to see traces: http://localhost:6006")
    print(f"Project: {PROJECT_NAME}")


if __name__ == "__main__":
    asyncio.run(run_demo())

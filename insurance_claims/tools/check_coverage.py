"""
check_coverage tool — determines whether a specific claim scenario is covered
by the applicable policy, citing the relevant policy sections.

WHY this tool exists separately from policy_search:
- policy_search retrieves raw text chunks — it returns "what the policy says"
- check_coverage makes a coverage DECISION — it answers "is this covered? yes/no/partial"
- The distinction matters: agents often need a binary answer with reasoning,
  not raw policy text to parse themselves
- This separation keeps each tool's responsibility narrow and testable

Architecture: uses policy_search internally to retrieve relevant context,
then applies LLM reasoning to make a structured coverage determination.
"""

import logging

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


# --- Output schema ---------------------------------------------------------- #

class CoverageDecision(BaseModel):
    """Structured coverage determination for an insurance claim scenario."""

    is_covered: bool = Field(
        description="True if the scenario is covered, False if excluded."
    )
    coverage_status: str = Field(
        description=(
            "One of: 'fully_covered', 'partially_covered', 'not_covered', "
            "'coverage_uncertain' (when policy is ambiguous or info is insufficient)."
        )
    )
    applicable_coverage: str = Field(
        description=(
            "The coverage section(s) that apply to this scenario "
            "(e.g., 'Coverage A - Collision', 'Coverage D - Theft'). "
            "State 'None' if not covered."
        )
    )
    deductible: str = Field(
        description=(
            "The applicable deductible amount or percentage, or 'N/A' if not covered. "
            "Include both the percentage and the estimated MXN amount if determinable."
        )
    )
    coverage_limit: str = Field(
        description="The applicable coverage limit, or 'N/A' if not covered."
    )
    exclusions_triggered: list[str] = Field(
        description=(
            "List of policy exclusions that limit or deny coverage. "
            "Empty list if no exclusions apply."
        )
    )
    explanation: str = Field(
        description=(
            "2-3 sentence explanation of the coverage decision, citing specific "
            "policy sections and amounts. This is the text the agent will share "
            "with the insured."
        )
    )
    recommended_next_step: str = Field(
        description=(
            "What the insured should do next. Examples: 'Submit claim with required documents', "
            "'Contact senior adjuster for review', 'Claim will be denied — consult legal counsel'."
        )
    )


# --- Config ----------------------------------------------------------------- #

class CheckCoverageConfig(FunctionBaseConfig, name="check_coverage"):
    """
    Determines whether an insurance claim scenario is covered by the applicable policy.
    Returns a structured coverage decision with applicable sections, deductible,
    limits, and exclusions.
    """

    llm_name: str = Field(
        default="nim_llm",
        description="Name of the LLM (from config.yml) to use for coverage analysis.",
    )


# --- Registration ----------------------------------------------------------- #

@register_function(
    config_type=CheckCoverageConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def check_coverage_function(config: CheckCoverageConfig, builder: Builder):
    """
    Registers the check_coverage tool with NAT.

    Acquires the LLM from builder so the model is configured in YAML, not hardcoded.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    import os

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    structured_llm = llm.with_structured_output(CoverageDecision)

    # Load the FAISS index so we can do RAG inside the tool.
    # WHY load it here instead of calling policy_search tool:
    # NAT tools are independent functions — one tool cannot directly call another.
    # The orchestrator chains them. But check_coverage needs policy context to make
    # decisions, so we load the same FAISS index directly.
    api_key = os.environ.get("NVIDIA_API_KEY")
    index_dir = "data/faiss_index"

    vectorstore = None
    if os.path.exists(index_dir) and api_key:
        try:
            embeddings = NVIDIAEmbeddings(
                model="nvidia/nv-embedqa-e5-v5",
                api_key=api_key,
                truncate="END",
            )
            vectorstore = FAISS.load_local(
                index_dir, embeddings, allow_dangerous_deserialization=True
            )
            logger.info("check_coverage loaded FAISS index from %s", index_dir)
        except Exception as e:
            logger.warning("check_coverage could not load FAISS index: %s", e)

    _SYSTEM_PROMPT = """You are a senior insurance coverage analyst with 20 years of experience.
Your job is to determine whether a specific claim scenario is covered by the applicable insurance policy.

You will receive:
1. A claim scenario description (what happened)
2. Relevant policy excerpts retrieved from the actual policy document

Make a precise coverage determination based ONLY on the provided policy excerpts.
If the excerpts don't provide enough information to make a confident determination,
set coverage_status to 'coverage_uncertain' and explain what additional information is needed.

Be specific: cite deductibles, limits, and exclusion clause numbers exactly as written in the policy."""

    async def _check_coverage(claim_scenario: str) -> str:
        """
        Determines whether an insurance claim scenario is covered by the applicable policy.

        Provide a detailed description of the claim scenario including: what happened,
        what type of policy is involved (auto/home/life), the insured's situation,
        and any relevant circumstances (e.g., was the driver licensed? was it intentional?).
        Returns a structured coverage decision with applicable deductibles, limits,
        exclusions, and recommended next steps.

        Args:
            claim_scenario: Detailed description of the claim scenario to analyze.

        Returns:
            JSON with coverage decision, applicable sections, deductible, limits,
            exclusions, explanation, and recommended next step.
        """
        # Retrieve policy context via FAISS — same as policy_search but targeted
        # to coverage determination (not just any info about the policy topic).
        policy_context = "No policy context available — making determination from general knowledge."
        if vectorstore:
            docs = vectorstore.similarity_search(claim_scenario, k=5)
            if docs:
                excerpts = []
                for i, doc in enumerate(docs, 1):
                    policy_type = doc.metadata.get("policy_type", "unknown")
                    excerpts.append(
                        f"[Policy Excerpt {i} — {policy_type}]\n{doc.page_content.strip()}"
                    )
                policy_context = "\n\n".join(excerpts)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Claim Scenario:\n{claim_scenario}\n\n"
                    f"Relevant Policy Excerpts:\n{policy_context}"
                ),
            },
        ]

        try:
            result: CoverageDecision = await structured_llm.ainvoke(messages)
            return result.model_dump_json(indent=2)
        except Exception as e:
            logger.error(
                "check_coverage failed for scenario=%r: %s",
                claim_scenario[:200],
                e,
            )
            fallback = CoverageDecision(
                is_covered=False,
                coverage_status="coverage_uncertain",
                applicable_coverage="Unable to determine",
                deductible="N/A",
                coverage_limit="N/A",
                exclusions_triggered=[],
                explanation=f"Coverage check failed due to a technical error: {e}. Please review manually.",
                recommended_next_step="Escalate to senior adjuster for manual review.",
            )
            return fallback.model_dump_json(indent=2)

    yield FunctionInfo.from_fn(_check_coverage, description=_check_coverage.__doc__)

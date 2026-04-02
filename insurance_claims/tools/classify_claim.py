"""
classify_claim tool — classifies an incoming insurance claim description
into a structured triage record: type, priority, complexity, and handler.

WHY this exists as a dedicated tool (not part of the orchestrator's system prompt):
- The orchestrator's job is routing, not triage logic
- Separating classification makes it independently testable and evaluatable
- Structured Pydantic output means downstream tools (check_coverage) get
  reliable typed data, not LLM-generated free text to re-parse

Pattern learned here: using an LLM inside a NAT tool with structured output
via LangChain's with_structured_output().
"""

import json
import logging

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


# --- Output schema ---------------------------------------------------------- #

class ClaimClassification(BaseModel):
    """Structured triage record produced for every incoming claim."""

    claim_type: str = Field(
        description=(
            "Type of insurance line. One of: auto, home, life, health, liability, unknown."
        )
    )
    priority: str = Field(
        description=(
            "Triage priority. One of: urgent (life-threatening or total loss), "
            "high (significant damage, >50% insured value), "
            "medium (partial damage, routine injury), "
            "low (minor damage, administrative)."
        )
    )
    estimated_complexity: str = Field(
        description=(
            "Estimated handling complexity. One of: simple (standard docs, clear liability), "
            "moderate (multiple parties or partial coverage questions), "
            "complex (legal exposure, contested liability, or unusual circumstances)."
        )
    )
    recommended_handler: str = Field(
        description=(
            "Who should handle this claim. One of: "
            "self_service (insured can submit online), "
            "junior_adjuster (routine claims), "
            "senior_adjuster (complex or high-value), "
            "legal_team (litigation risk), "
            "medical_specialist (personal injury focus)."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining why these classifications were chosen."
    )


# --- Config ----------------------------------------------------------------- #

class ClassifyClaimConfig(FunctionBaseConfig, name="classify_claim"):
    """
    Classifies an insurance claim description into a structured triage record.
    Returns claim type, priority, complexity, and recommended handler.
    """

    # The LLM that performs classification — must be defined in config.yml under llms:
    llm_name: str = Field(
        default="nim_llm",
        description="Name of the LLM (from config.yml llms section) to use for classification.",
    )


# --- Registration ----------------------------------------------------------- #

@register_function(
    config_type=ClassifyClaimConfig,
    # framework_wrappers tells NAT to wrap the LLM in LangChain format
    # so we can use .with_structured_output() on it
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def classify_claim_function(config: ClassifyClaimConfig, builder: Builder):
    """
    Registers the classify_claim tool with NAT.

    We get the LLM from the builder (which reads it from config.yml).
    This means swapping models is a config change, not a code change.
    """

    # Get the LLM as a LangChain-compatible object.
    # builder.get_llm() handles NIM auth, retry, and rate limiting for us.
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # with_structured_output() forces the LLM to return JSON matching our schema.
    # Why not parse free text? Because regex/JSON parsing on unstructured LLM output
    # is fragile — structured output is the correct abstraction here.
    structured_llm = llm.with_structured_output(ClaimClassification)

    _SYSTEM_PROMPT = """You are an expert insurance claims triage specialist with 15 years
of experience at a major P&C insurer. Classify the incoming claim description accurately.

Rules:
- claim_type: auto (vehicle damage/collision), home (property damage/theft),
  life (death benefit), health (medical expenses), liability (third-party claims),
  unknown (cannot determine from description)
- priority: urgent = total loss / life-threatening injury / catastrophic event;
  high = major damage >50% value / significant injury;
  medium = partial damage / minor injury / standard situation;
  low = minor damage / administrative query
- estimated_complexity: simple = clear liability + standard docs;
  moderate = multiple parties OR partial coverage ambiguity;
  complex = contested liability OR legal exposure OR unusual circumstances
- recommended_handler: match to complexity + type (life always → senior_adjuster minimum)
- reasoning: be specific, reference details from the description"""

    async def _classify(claim_description: str) -> str:
        """
        Classifies an insurance claim and returns a structured triage record.

        Provide a natural language description of the insurance claim — what
        happened, what was damaged, and any relevant context. Returns a JSON
        object with: claim_type, priority, estimated_complexity,
        recommended_handler, and reasoning.

        Args:
            claim_description: Natural language description of the insurance claim.

        Returns:
            JSON string with structured classification fields.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this claim:\n\n{claim_description}",
            },
        ]

        try:
            result: ClaimClassification = await structured_llm.ainvoke(messages)
            # Return as JSON string so the orchestrator can read it cleanly
            return result.model_dump_json(indent=2)
        except Exception as e:
            # Never silently swallow — log context and return a safe fallback
            logger.error(
                "classify_claim failed for description=%r: %s",
                claim_description[:200],
                e,
            )
            fallback = ClaimClassification(
                claim_type="unknown",
                priority="medium",
                estimated_complexity="moderate",
                recommended_handler="senior_adjuster",
                reasoning=f"Classification failed — defaulting to manual review. Error: {e}",
            )
            return fallback.model_dump_json(indent=2)

    yield FunctionInfo.from_fn(_classify, description=_classify.__doc__)

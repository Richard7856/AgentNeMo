"""
required_docs tool — returns the list of documents an insured must submit
for a given claim type. Pure lookup, no LLM call needed.

This is the simplest tool in the system: a dictionary lookup.
It exists to demonstrate the NAT registration pattern with zero external
dependencies, making it the ideal first tool to build and test.
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# Documents required per claim type.
# In production this would be fetched from a CMS or claims management system.
# We keep it static here to keep the demo self-contained.
_REQUIRED_DOCS: dict[str, list[str]] = {
    "auto": [
        "Completed claim form (SINIESTRO-001)",
        "Police report (if applicable)",
        "Photos of the damaged vehicle (minimum 4 angles)",
        "Driver's license of all involved parties",
        "Vehicle registration certificate",
        "Repair estimate from authorized workshop",
        "Medical certificates (if personal injury)",
    ],
    "home": [
        "Completed claim form (SINIESTRO-002)",
        "Proof of ownership or lease agreement",
        "Photos of the damaged property",
        "Fire department or police report (if applicable)",
        "Inventory list of damaged/stolen items with estimated values",
        "Repair or replacement estimates",
        "Receipts or proof of purchase for high-value items",
    ],
    "life": [
        "Death certificate (original or certified copy)",
        "Policy document",
        "Beneficiary identification documents",
        "Medical records (if death was illness-related)",
        "Autopsy report (if required by policy terms)",
        "Completed beneficiary claim form",
    ],
    "health": [
        "Completed claim form (SINIESTRO-004)",
        "Medical bills and receipts",
        "Doctor's diagnosis and treatment notes",
        "Prescription records",
        "Hospital discharge summary (if hospitalized)",
        "Pre-authorization reference number (if applicable)",
    ],
    "liability": [
        "Completed claim form (SINIESTRO-005)",
        "Description of the incident",
        "Third-party claimant information",
        "Witness statements (if available)",
        "Photos or evidence of the incident",
        "Any legal notices or court documents received",
    ],
}

_VALID_TYPES = ", ".join(_REQUIRED_DOCS.keys())


class RequiredDocsConfig(FunctionBaseConfig, name="required_docs"):
    """
    Returns the list of documents required to file a claim of a given type.
    Useful as a first step when an insured contacts us about a new claim.
    """

    # No config params needed — the lookup table is self-contained.
    # We include a fallback_message so the agent can explain the situation
    # when the claim type is unknown without returning a bare error.
    fallback_message: str = Field(
        default=(
            f"Unknown claim type. Valid types are: {_VALID_TYPES}. "
            "Please clarify the type of insurance claim."
        ),
        description="Message returned when the claim type is not recognized.",
    )


@register_function(config_type=RequiredDocsConfig)
async def required_docs_function(config: RequiredDocsConfig, builder: Builder):
    """
    Registers the required_docs tool with NAT.

    The tool name 'required_docs' matches the name= parameter on
    RequiredDocsConfig, which is what config.yml references via _type.
    """

    async def _get_required_docs(claim_type: str) -> str:
        """
        Returns the documents required to file an insurance claim.

        Provide the type of insurance claim (e.g., 'auto', 'home', 'life',
        'health', 'liability') and receive a formatted list of documents
        the insured must submit.

        Args:
            claim_type: Type of insurance claim. One of: auto, home, life,
                        health, liability.

        Returns:
            A formatted list of required documents, or a fallback message
            if the claim type is unrecognized.
        """
        normalized = claim_type.lower().strip()
        docs = _REQUIRED_DOCS.get(normalized)

        if docs is None:
            logger.warning("Unknown claim type requested: %s", claim_type)
            return config.fallback_message

        formatted = "\n".join(f"  {i + 1}. {doc}" for i, doc in enumerate(docs))
        return (
            f"Required documents for a '{normalized}' insurance claim:\n{formatted}"
        )

    yield FunctionInfo.from_fn(_get_required_docs, description=_get_required_docs.__doc__)

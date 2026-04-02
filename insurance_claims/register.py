# flake8: noqa
"""
NAT component registration entry point.

NAT discovers our tools via the 'nat.components' entry point in pyproject.toml:
    insurance_claims = "insurance_claims.register"

Simply importing each tool module here is enough — the @register_function
decorator runs at import time and registers the tool with NAT's registry.

Add a new import line each time you add a tool.
"""

from .tools.required_docs import required_docs_function
from .tools.classify_claim import classify_claim_function
from .tools.policy_search import policy_search_function
from .tools.check_coverage import check_coverage_function

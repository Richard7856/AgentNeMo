"""
policy_search tool — retrieves relevant excerpts from insurance policy PDFs
to answer coverage questions. This is the RAG (Retrieval-Augmented Generation)
component of the system.

WHY RAG instead of putting policies in the system prompt:
- Policy PDFs are ~10+ pages each. Stuffing all of them into context = expensive
  and slow on every query, even when only 2 sentences are relevant.
- RAG retrieves only the relevant chunks: 5 passages from 20 vs 3 full PDFs.
- Retrieval grounds the LLM's answers in source documents, reducing hallucination.
- Scales to hundreds of policy documents without changing the agent.

Architecture:
  Query → NIM embeddings → FAISS similarity search → top-k chunks → agent
"""

import logging
import os

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

INDEX_DIR = "data/faiss_index"
# Top-k retrieved chunks — 4 balances context richness vs. token cost.
# Fewer → may miss relevant clauses. More → LLM may get confused by noise.
TOP_K = 4


class PolicySearchConfig(FunctionBaseConfig, name="policy_search"):
    """
    Searches insurance policy documents for coverage information.
    Uses semantic similarity (RAG) over PDF chunks stored in a local FAISS index.
    Run data_prep/ingest_policies.py once to build the index before using this tool.
    """

    index_dir: str = Field(
        default=INDEX_DIR,
        description="Path to the FAISS index directory (built by ingest_policies.py).",
    )
    top_k: int = Field(
        default=TOP_K,
        description="Number of policy document chunks to retrieve per query.",
    )


@register_function(config_type=PolicySearchConfig)
async def policy_search_function(config: PolicySearchConfig, builder: Builder):
    """
    Registers the policy_search tool with NAT.

    FAISS index is loaded once at startup and kept in memory — retrieval is
    sub-millisecond and doesn't make any network calls, unlike re-embedding on
    every query.
    """
    # Import here (not at module level) to avoid slowing down NAT startup
    # when the tool isn't in the config.yml
    from langchain_community.vectorstores import FAISS
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. The policy_search tool requires it for embeddings."
        )

    # Load the pre-built FAISS index from disk.
    # allow_dangerous_deserialization=True is required by LangChain for pickle-based
    # FAISS loading — safe here because we built the index ourselves.
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=api_key,
        truncate="END",
    )

    if not os.path.exists(config.index_dir):
        raise RuntimeError(
            f"FAISS index not found at {config.index_dir}. "
            "Run: uv run python insurance_claims/data_prep/ingest_policies.py"
        )

    vectorstore = FAISS.load_local(
        config.index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("Policy FAISS index loaded from %s", config.index_dir)

    async def _search_policies(query: str) -> str:
        """
        Searches insurance policy documents for information relevant to the query.

        Use this tool to answer questions about what is covered or excluded by
        insurance policies, deductible amounts, coverage limits, claims procedures,
        and policy terms. Provide a specific question or topic to search for.

        Args:
            query: The coverage question or topic to search for in policy documents.
                   Example: "What is the deductible for auto collision coverage?"

        Returns:
            Relevant excerpts from policy documents with source information.
        """
        # Similarity search — FAISS compares query embedding to all chunk embeddings
        # and returns the top_k most similar chunks.
        docs = vectorstore.similarity_search(query, k=config.top_k)

        if not docs:
            return "No relevant policy information found for this query."

        # Format results with source metadata so the agent can cite the policy.
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            policy_type = doc.metadata.get("policy_type", "unknown")
            page = doc.metadata.get("page", "?")
            # Use just the filename, not the full path
            source_name = os.path.basename(source) if source != "unknown" else source
            parts.append(
                f"[Excerpt {i} — {policy_type} policy, page {page} of {source_name}]\n"
                f"{doc.page_content.strip()}"
            )

        return "\n\n".join(parts)

    yield FunctionInfo.from_fn(_search_policies, description=_search_policies.__doc__)

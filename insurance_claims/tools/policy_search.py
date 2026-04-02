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

WHY Milvus instead of FAISS:
- Milvus persists data in a Docker volume — no rebuild needed after restarts.
- Supports metadata filtering: future queries can filter by policy_type or tenant_id.
- Production-grade: designed for concurrent reads from multiple agents.
- See docs/DECISIONS.md for the full FAISS vs Milvus decision.

Architecture:
  Query → NIM embeddings → Milvus similarity search → top-k chunks → agent
"""

import logging
import os

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "insurance_policies"

# Top-k retrieved chunks — 4 balances context richness vs. token cost.
# Fewer → may miss relevant clauses. More → LLM may get confused by noise.
TOP_K = 4


class PolicySearchConfig(FunctionBaseConfig, name="policy_search"):
    """
    Searches insurance policy documents for coverage information.
    Uses semantic similarity (RAG) over PDF chunks stored in a Milvus vector collection.
    Run 'make ingest' once to build the collection before using this tool.
    """

    milvus_uri: str = Field(
        default=MILVUS_URI,
        description="Milvus server URI. Defaults to MILVUS_URI env var or http://localhost:19530.",
    )
    collection_name: str = Field(
        default=COLLECTION_NAME,
        description="Milvus collection name containing the policy vectors.",
    )
    top_k: int = Field(
        default=TOP_K,
        description="Number of policy document chunks to retrieve per query.",
    )


@register_function(config_type=PolicySearchConfig)
async def policy_search_function(config: PolicySearchConfig, builder: Builder):
    """
    Registers the policy_search tool with NAT.

    Milvus client is initialized once at startup. Unlike FAISS (which loads
    everything into RAM), Milvus keeps data server-side — the client just
    holds a connection, so startup is near-instant and memory usage is minimal.
    """
    # Import here (not at module level) to avoid slowing down NAT startup
    # when the tool isn't in the config.yml
    from langchain_milvus import Milvus
    from langchain_milvus.vectorstores.milvus import Milvus as _MilvusVS
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    from pymilvus import Collection, connections as _pym_connections

    # Compatibility patch: pymilvus 2.6.x MilvusClient stores handler in
    # _alias_handlers under a 'cm-{id}' alias that Collection() can't find.
    # We bridge the two systems so Collection(name, using=alias) works.
    # Applied here (tool startup) so it's active before any Milvus object is created.
    def _patched_col(self):
        if self._col_cache is not None:
            return self._col_cache
        alias = self.alias
        if alias not in _pym_connections._alias_handlers:
            try:
                _pym_connections._alias_handlers[alias] = self.client._handler
            except Exception:
                return None
        try:
            self._col_cache = Collection(self.collection_name, using=alias)
            self._cache_key = f"{self.collection_name}:{alias}"
            return self._col_cache
        except Exception:
            return None

    _MilvusVS.col = property(_patched_col)

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. The policy_search tool requires it for embeddings."
        )

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=api_key,
        truncate="END",
    )

    # Connect to the existing Milvus collection built by ingest_policies.py.
    # This does NOT load data into memory — Milvus keeps vectors server-side.
    # If the collection doesn't exist, similarity_search will raise a clear error.
    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=config.collection_name,
            connection_args={"uri": config.milvus_uri},
        )
        logger.info(
            "Connected to Milvus collection '%s' at %s",
            config.collection_name,
            config.milvus_uri,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to Milvus at {config.milvus_uri}. "
            "Make sure the stack is running: make infra-up && make ingest"
        ) from e

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
        # Similarity search — Milvus embeds the query and returns the top_k
        # most similar chunks from the collection. All embedding + search
        # computation happens server-side.
        docs = vectorstore.similarity_search(query, k=config.top_k)

        if not docs:
            return "No relevant policy information found for this query."

        # Format results with source metadata so the agent can cite the policy.
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            policy_type = doc.metadata.get("policy_type", "unknown")
            page = doc.metadata.get("page", "?")
            source_name = os.path.basename(source) if source != "unknown" else source
            parts.append(
                f"[Excerpt {i} — {policy_type} policy, page {page} of {source_name}]\n"
                f"{doc.page_content.strip()}"
            )

        return "\n\n".join(parts)

    yield FunctionInfo.from_fn(_search_policies, description=_search_policies.__doc__)

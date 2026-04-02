"""
ingest_policies.py — reads the sample policy PDFs and builds a Milvus vector collection.

WHY run this as a separate script (not at tool load time):
- Embedding 3 PDFs via NIM API takes ~30 seconds. Running it on every agent startup
  would make the agent feel broken.
- The Milvus collection is an artifact, not source code. It persists in the Docker volume.
- Re-run any time the PDFs change: make ingest

WHY Milvus instead of FAISS:
- FAISS stores vectors in a local file — one file per deployment, no multi-tenancy.
- Milvus is a dedicated vector database: persists across restarts, supports metadata
  filtering (WHERE policy_type = 'auto'), scales to millions of vectors, and is the
  production standard for enterprise RAG systems.
- In the CRM Agents SaaS context, each client gets their own Milvus Collection,
  and policies are partitioned by type — impossible to do cleanly with FAISS.
- See docs/DECISIONS.md for the full FAISS vs Milvus decision log.

Output: Milvus collection "insurance_policies" at localhost:19530
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Compatibility patch: langchain-milvus 0.3.x + pymilvus 2.6.x ─────────────
# pymilvus 2.6.x introduced ConnectionManager, so MilvusClient stores its handler
# in connections._alias_handlers[f"cm-{id(handler)}"] rather than the ORM
# connections dict. langchain-milvus still uses Collection(name, using=alias) which
# reads from _alias_handlers, but MilvusClient never registers itself there.
# This patch bridges the two systems so Collection() can find the active connection.
# WHY patch here and not in the library: langchain-milvus < 0.4 has this bug
# with pymilvus >= 2.6.0. Patching at import time is the least invasive fix.
from pymilvus import Collection, connections as _pym_connections
from langchain_milvus.vectorstores.milvus import Milvus as _MilvusVS


def _patched_col(self):
    """col property bridging pymilvus 2.6.x MilvusClient to the ORM registry."""
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
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()  # loads NVIDIA_API_KEY from .env

POLICIES_DIR = "data/policies"
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "insurance_policies"

# Chunk size is a critical RAG tuning parameter.
# Too large: chunks lose precision, retrieval returns too much noise.
# Too small: chunks lose context, retrieved fragments lack enough info to answer.
# 800 chars with 150 overlap is a good starting point for policy text (dense prose).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _wait_for_milvus(uri: str, retries: int = 10, delay: int = 8) -> None:
    """
    Polls Milvus until it's ready to accept connections.

    WHY: Milvus depends on etcd and MinIO initializing first. Even after Docker
    reports the container as healthy, Milvus needs ~10-30s more to load its
    internal indexes. Without this wait, the first connection attempt fails with
    a cryptic gRPC error.
    """
    from pymilvus import connections, exceptions

    print(f"Waiting for Milvus at {uri}...")
    # Parse host/port from URI — pymilvus connect() doesn't accept full URIs
    host = uri.replace("http://", "").split(":")[0]
    port = uri.split(":")[-1]

    for attempt in range(1, retries + 1):
        try:
            connections.connect(alias="health_check", host=host, port=port)
            connections.disconnect("health_check")
            print(f"Milvus ready (attempt {attempt}/{retries})")
            return
        except exceptions.MilvusException as e:
            print(f"  Attempt {attempt}/{retries}: not ready ({e}). Waiting {delay}s...")
            time.sleep(delay)

    print(f"ERROR: Milvus not reachable at {uri} after {retries} attempts.")
    print("Make sure Docker is running: make infra-up")
    sys.exit(1)


def main() -> None:
    """Ingests all PDFs from data/policies/ into a Milvus vector collection."""

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    # Verify Milvus is up before spending time on PDF parsing + embedding
    _wait_for_milvus(MILVUS_URI)

    # Collect PDF paths
    pdf_files = [f for f in os.listdir(POLICIES_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {POLICIES_DIR}. Run generate_policies.py first.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} policy PDFs: {pdf_files}")

    # Load and parse PDFs — PyPDFLoader extracts text page by page,
    # preserving metadata (source filename + page number) for citation in answers.
    all_docs = []
    for filename in pdf_files:
        path = os.path.join(POLICIES_DIR, filename)
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Tag each doc with the policy type — used for metadata filtering in Milvus.
        # Example: WHERE policy_type = 'auto' narrows search to auto policies only.
        policy_type = filename.replace("_policy_sample.pdf", "")
        for doc in docs:
            doc.metadata["policy_type"] = policy_type
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} pages from {filename}")

    print(f"\nTotal pages loaded: {len(all_docs)}")

    # Split into chunks for retrieval.
    # RecursiveCharacterTextSplitter tries to split on paragraph boundaries first,
    # then sentence boundaries — preserves policy section coherence better than
    # fixed-width splits.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Embed with NVIDIA's nv-embedqa-e5-v5 via NIM.
    # WHY this model: optimized for retrieval/QA tasks (the 'qa' in the name
    # = query-aware training). Best quality embedding on NIM for this use case.
    print("\nEmbedding chunks via nvidia/nv-embedqa-e5-v5...")
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=api_key,
        truncate="END",  # truncate long chunks at the end (not the beginning)
    )

    # Build Milvus collection from chunks + embeddings.
    # drop_old=True drops and recreates the collection on each ingest run —
    # safe for development, ensures the index stays in sync with the PDFs.
    # In production, you'd use incremental upserts instead.
    print(f"\nWriting vectors to Milvus collection '{COLLECTION_NAME}' at {MILVUS_URI}...")
    Milvus.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=True,  # wipe + recreate on re-ingest so index stays in sync with PDFs
    )

    print(f"\nMilvus collection '{COLLECTION_NAME}' built successfully.")
    print(f"Vectors stored at: {MILVUS_URI}")
    print("Run 'make run INPUT=\"...\"' to query the agent.")


if __name__ == "__main__":
    main()

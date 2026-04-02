"""
ingest_policies.py — reads the sample policy PDFs and builds a FAISS vector index.

WHY run this as a separate script (not at tool load time):
- Embedding 3 PDFs via NIM API takes ~30 seconds. Running it on every agent startup
  would make the agent feel broken.
- The FAISS index is an artifact, not source code. It belongs in data/, not memory.
- Regenerate any time the PDFs change: uv run python insurance_claims/data_prep/ingest_policies.py

Output: data/faiss_index/ (two files: index.faiss + index.pkl)
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # loads NVIDIA_API_KEY from .env

POLICIES_DIR = "data/policies"
INDEX_DIR = "data/faiss_index"

# Chunk size is a critical RAG tuning parameter.
# Too large: chunks lose precision, retrieval returns too much noise.
# Too small: chunks lose context, retrieved fragments lack enough info to answer.
# 800 chars with 150 overlap is a good starting point for policy text (dense prose).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def main() -> None:
    """Ingests all PDFs from data/policies/ into a FAISS vector store."""

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    # Collect PDF paths
    pdf_files = [
        f for f in os.listdir(POLICIES_DIR) if f.endswith(".pdf")
    ]
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
        # Tag each doc with the policy type for filtering later
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
    # WHY this model: it's the highest-quality English embedding model on NIM,
    # optimized for retrieval/QA tasks (the 'qa' in the name = query-aware training).
    print("\nEmbedding chunks via nvidia/nv-embedqa-e5-v5...")
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=api_key,
        truncate="END",  # truncate long chunks at the end (not the beginning)
    )

    # Build FAISS index from chunks + embeddings.
    # FAISS (Facebook AI Similarity Search) does exact L2 or cosine similarity search
    # in memory — no network calls at query time, sub-millisecond retrieval.
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist the index to disk so ingest_policies.py only needs to run once.
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"\nFAISS index saved to {INDEX_DIR}/")
    print("Run 'uv run nat run' to use the policy_search tool.")


if __name__ == "__main__":
    main()

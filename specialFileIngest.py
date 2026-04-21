"""
ingest_exhibits.py — Push Epstein Exhibit 1–4 PDFs into Pinecone
Run: python ingest_exhibits.py

Assumes the 4 PDFs are in ./epstein-exhibits/ (edit EXHIBITS below if different).
Uses all the infrastructure from ingest.py (ChunkAccumulator, GeminiKeyPool, etc.)
"""

import os
from pinecone import Pinecone
from dotenv import load_dotenv

from ingest import (
    KEY_POOL,
    INDEX_NAME,
    NAMESPACE,
    extract_text_from_pdf,
    chunk_text,
    setup_pinecone_index,
    ChunkAccumulator,
    _file_hash,
)

load_dotenv()

# ── Configure your exhibit PDFs here ──────────────────────────────────────────
# Each entry: (local file path, public/source URL to store in metadata, label)
EXHIBITS = [
        (
        "./test-pdfs/Exhibit123.pdf",
        "https://info.publicintelligence.net/EpsteinDocs-Batch7.pdf",   # ← replace with real URL
        "Epstein Exhibit 123",
    ),
    (
        "./test-pdfs/Exhibit4.pdf",
        "https://static.poder360.com.br/2024/01/documentos-publicos-Jeffrey-Epstein-3jan2024.pdf",   # ← replace with real URL
        "Epstein Exhibit 4",
    ),
]

DATASET = "epstein-exhibit"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Validate env vars
    for key in ["GEMINI_API_KEY", "PINECONE_API_KEY"]:
        if not os.getenv(key):
            raise ValueError(f"Missing env var: {key}")

    # Validate all files exist before starting
    missing = [path for path, _, _ in EXHIBITS if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"These exhibit files were not found:\n" +
            "\n".join(f"  • {p}" for p in missing)
        )

    # Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vector_store = setup_pinecone_index(pc)

    accumulator = ChunkAccumulator(vector_store)
    confirmed_docs: list[dict] = []

    # ── Process each exhibit ────────────────────────────────────────────────
    for pdf_path, source_url, label in EXHIBITS:
        print(f"\n{'='*60}")
        print(f"📄  {label}")
        print(f"    {pdf_path}")
        print(f"{'='*60}")

        try:
            file_hash = _file_hash(pdf_path)
            filename  = os.path.basename(pdf_path)

            # Step 1 — Extract text (disk-cached; skips re-OCR on rerun)
            raw_text = extract_text_from_pdf(pdf_path, cache_key=file_hash)
            if not raw_text.strip():
                print(f"  ⚠  No text extracted — skipping {label}")
                continue
            print(f"  ✓  Extracted {len(raw_text):,} chars")

            # Step 2 — Chunk (disk-cached)
            chunks = chunk_text(
                text=raw_text,
                filename=filename,
                source_url=source_url,
                dataset=DATASET,
                cache_key=file_hash,
            )
            print(f"  ✓  {len(chunks)} chunks created")

            # Step 3 — Queue into accumulator
            doc_meta = {
                "source_url": source_url,
                "filename":   filename,
                "dataset":    DATASET,
                "label":      label,
                "char_count": len(raw_text),
                
            }
            just_confirmed = accumulator.add(chunks, doc_meta)
            confirmed_docs.extend(just_confirmed)

            if just_confirmed:
                for d in just_confirmed:
                    print(f"  ✅  Confirmed in Pinecone: {d['label']}")

            print(f"  📦  Buffer size after queueing: {accumulator.buffer_size()} chunks")

        except Exception as e:
            print(f"  ✗  Error processing {label}: {e}")
            raise   # re-raise so you see the full traceback; remove to continue on error

    # ── Final flush — push whatever is still buffered ────────────────────
    print(f"\n{'='*60}")
    print(f"🏁  Final flush — {accumulator.buffer_size()} chunks remaining in buffer")
    print(f"{'='*60}")
    remaining = accumulator.flush_all()
    confirmed_docs.extend(remaining)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅  Ingestion complete!")
    print(f"    Total Pinecone requests : {accumulator.total_requests}")
    print(f"    Documents confirmed     : {len(confirmed_docs)}")
    for d in confirmed_docs:
        print(f"      • {d['label']} — {d['char_count']:,} chars")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
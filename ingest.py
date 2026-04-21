"""
ingest.py — PDF extraction + chunking + Pinecone upsert
Key optimisation:
  - Chunks are ACCUMULATED across multiple PDFs before embedding.
  - A batch is only flushed to Pinecone when it reaches MIN_BATCH (30).
  - Single-page docs (2-3 chunks) are grouped with neighbours so you
    never waste a full API request on a tiny document.
  - source_url is stored per-chunk in metadata — never lost even when
    chunks from different docs share a single batch.
  - Text + chunk results cached to disk (resume after crash, skip re-OCR).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from time import sleep
import pymupdf
import pytesseract
from PIL import Image
from tqdm import tqdm
import os, json, re, hashlib
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
INDEX_NAME    = "gov-transparency-index"
NAMESPACE     = "epstein-docs"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100
OCR_THRESHOLD = 50
DPI           = 300

MIN_BATCH     = 20   # accumulate until we have at least this many chunks
MAX_BATCH     = 30   # hard cap per Pinecone upsert request
CACHE_DIR     = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

from datetime import datetime, timedelta

# ── Gemini key pool ────────────────────────────────────────────────────────────
_RAW_KEYS: list[str] = [k for k in [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6"),
    os.getenv("GEMINI_API_KEY_7"),
    os.getenv("GEMINI_API_KEY_8"),
    os.getenv("GEMINI_API_KEY_9")
] if k]

class GeminiKeyPool:
    COOLDOWN = 65  # seconds before retrying a rate-limited key

    def __init__(self):
        self._keys     = _RAW_KEYS
        self._cooldown: dict[int, datetime] = {}
        self._idx      = 0
        self._cache:   dict[int, GoogleGenerativeAIEmbeddings] = {}
        print(f"🔑 {len(self._keys)} Gemini key(s) loaded")

    def get(self) -> GoogleGenerativeAIEmbeddings:
        """Return embedder for current active key."""
        i = self._active()
        if i not in self._cache:
            self._cache[i] = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=self._keys[i],
            )
        print(f"  🔑 Gemini key #{i+1}")
        return self._cache[i]

    def rotate(self) -> None:
        """Round-robin to next key on every successful push."""
        self._idx = (self._idx + 1) % len(self._keys)

    def cooldown(self) -> None:
        """Cool down current key, switch to next available immediately."""
        cur = self._active()
        self._cooldown[cur] = datetime.now() + timedelta(seconds=self.COOLDOWN)
        print(f"  ⏸  Key #{cur+1} rate-limited")

        for _ in range(len(self._keys)):
            self._idx = (self._idx + 1) % len(self._keys)
            if self._available(self._idx):
                print(f"  ↩  Switched to key #{self._idx+1}")
                return

        # all keys cooled — wait for soonest
        soonest = min(self._cooldown, key=lambda i: self._cooldown[i])
        wait    = max(0, (self._cooldown[soonest] - datetime.now()).total_seconds()) + 2
        print(f"  😴 All keys rate-limited — waiting {wait:.0f}s…")
        sleep(wait)
        self._cooldown.pop(soonest)
        self._idx = soonest

    def _active(self) -> int:
        for _ in range(len(self._keys)):
            if self._available(self._idx):
                return self._idx
            self._idx = (self._idx + 1) % len(self._keys)
        return self._idx

    def _available(self, i: int) -> bool:
        if i not in self._cooldown:
            return True
        if datetime.now() >= self._cooldown[i]:
            self._cooldown.pop(i)
            return True
        return False

KEY_POOL = GeminiKeyPool()
# ── Helpers ────────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def clean_numbers(text: str) -> str:
    fixes = {
        r'\bO\b':  '0',
        r'\bl\b':  '1',
        r'\bI\b':  '1',
        r'\$\s+':  '$',
        r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})': r'\1/\2/\3',
    }
    for pattern, replacement in fixes.items():
        text = re.sub(pattern, replacement, text)
    return text


# ── Text extraction (disk-cached) ──────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str, cache_key: str | None = None) -> str:
    """Extract text from PDF. Result cached to disk — skips OCR on re-run."""
    key        = cache_key or _file_hash(pdf_path)
    cache_path = os.path.join(CACHE_DIR, f"text_{key}.txt")

    if os.path.exists(cache_path):
        print(f"  📦 Text cache hit")
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    doc       = pymupdf.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(tqdm(doc, desc="Extracting text")):
        page_text  = page.get_text().strip()
        table_text = ""
        try:
            for table in page.find_tables():
                df = table.to_pandas()
                if not df.empty:
                    table_text += f"\n[TABLE]\n{df.to_markdown(index=False)}\n[/TABLE]\n"
        except Exception:
            pass

        if len(page_text) > OCR_THRESHOLD:
            full_text += f"\n[Page {page_num+1}]\n{page_text}{table_text}\n"
        else:
            try:
                pix      = page.get_pixmap(dpi=DPI)
                img      = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang="eng", config="--psm 3").strip()
                if ocr_text:
                    full_text += f"\n[Page {page_num+1} - OCR]\n{ocr_text}{table_text}"
                else:
                    print(f"     ⚠ Page {page_num+1}: no text found")
            except pytesseract.TesseractNotFoundError:
                pass
            except Exception as e:
                print(f"     ⚠ OCR error p{page_num+1}: {e}")

    doc.close()
    result = clean_numbers(full_text)

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(result)
    return result


# ── Chunking (disk-cached) ─────────────────────────────────────────────────────

def chunk_text(
    text: str,
    filename: str,
    source_url: str,       # stored in EVERY chunk's metadata
    dataset: str,
    cache_key: str | None = None,
) -> list[dict]:
    """Split text into chunks. source_url embedded in each chunk's metadata."""
    key        = cache_key or _url_hash(source_url)
    cache_path = os.path.join(CACHE_DIR, f"chunks_{key}.json")

    if os.path.exists(cache_path):
        print(f"  📦 Chunk cache hit ({filename})")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    raw_chunks = splitter.split_text(text)
    chunks     = []

    for i, chunk in enumerate(raw_chunks):
        prev_ctx       = raw_chunks[i-1][-100:] if i > 0 else ""
        chunk_with_ctx = f"[CONTEXT: {prev_ctx}]\n{chunk}" if prev_ctx else chunk

        chunks.append({
            "text": chunk_with_ctx,
            "metadata": {
                "filename":     filename,
                "source_url":   source_url,   # ← URL preserved per-chunk
                "dataset":      dataset,
                "chunk_index":  i,
                "total_chunks": len(raw_chunks),
                "chunk_size":   len(chunk),
                "has_numbers":  bool(re.search(r'\$[\d,]+|\d+%|\d{4}-\d{2}-\d{2}', chunk)),
                "has_table":    "[TABLE]" in chunk,
                "has_dates":    bool(re.search(r'\b(19|20)\d{2}\b', chunk)),
                "namespace":    NAMESPACE,
            },
        })

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return chunks


# ── Pinecone setup ─────────────────────────────────────────────────────────────

def setup_pinecone_index(pc: Pinecone) -> PineconeVectorStore:
    existing = pc.list_indexes().names()

    if INDEX_NAME not in existing:
        print(f"Creating index {INDEX_NAME}…")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        sleep(2)
    else:
        print(f"Index {INDEX_NAME} already exists.")

    return PineconeVectorStore(index_name=INDEX_NAME, embedding=KEY_POOL.get())


# ── Cross-document chunk accumulator ──────────────────────────────────────────
#
# Problem solved:
#   A 1-page PDF might produce only 2–3 chunks. Sending those alone = 1 wasted
#   API request. With many such docs you burn your daily quota fast.
#
# Solution:
#   ChunkAccumulator holds a shared buffer. The scraper feeds docs one by one.
#   The buffer only flushes to Pinecone when it reaches MIN_BATCH (30) chunks.
#   That way 10 tiny docs (3 chunks each = 30 total) → 1 API request instead of 10.
#
# URL safety:
#   Every chunk already carries source_url in its metadata dict, so mixing
#   chunks from different docs in one batch is completely safe — each vector
#   in Pinecone still knows exactly which document it came from.
#
# DB record safety:
#   add() returns a list of doc_meta dicts that are NOW confirmed in Pinecone.
#   The scraper saves DB records ONLY for those. Docs whose chunks are still
#   buffered are returned only after flush_all() at the end.
# ─────────────────────────────────────────────────────────────────────────────

class ChunkAccumulator:
    """
    Accumulate chunks across multiple PDFs.
    Flush to Pinecone only when buffer reaches MIN_BATCH (30) chunks.
    Each flush sends at most MAX_BATCH (40) chunks = 1 API request.
    """

    def __init__(self, vector_store: PineconeVectorStore, budget_callback=None):
        """
        vector_store    — connected PineconeVectorStore
        budget_callback — optional fn(n: int) called with requests used per flush
        """
        self.vector_store    = vector_store
        self.budget_callback = budget_callback
        self._buffer: list[dict] = []       # pending chunks (across docs)
        # Each entry: {source_url, filename, dataset, char_count, chunk_count}
        # chunk_count decrements as chunks leave the buffer
        self._pending_docs: list[dict] = []
        self.total_requests = 0

    # ── public ────────────────────────────────────────────────────────────────

    def add(self, chunks: list[dict], doc_meta: dict) -> list[dict]:
        """
        Queue a document's chunks.

        Returns list of doc_meta dicts whose chunks are NOW confirmed in Pinecone.
        The caller should write DB records for these immediately.
        Docs not yet returned are still buffered — call flush_all() at end of run.
        """
        self._buffer.extend(chunks)
        self._pending_docs.append({**doc_meta, "chunk_count": len(chunks)})

        flushed = []
        while len(self._buffer) >= MIN_BATCH:
            flushed.extend(self._flush_batch())
        return flushed

    def flush_all(self) -> list[dict]:
        """
        Force-push everything left in the buffer.
        Call once at the very end of each run.
        Returns all newly confirmed doc_meta dicts.
        """
        flushed = []
        while self._buffer:
            flushed.extend(self._flush_batch(force=True))
        return flushed

    def buffer_size(self) -> int:
        return len(self._buffer)

    # ── internals ─────────────────────────────────────────────────────────────

    def _flush_batch(self, force: bool = False) -> list[dict]:
        """
        Pull up to MAX_BATCH chunks from the buffer, push to Pinecone.
        Returns list of doc_meta dicts fully flushed in this call.
        """
        batch        = self._buffer[:MAX_BATCH]
        self._buffer = self._buffer[MAX_BATCH:]

        texts     = [c["text"]     for c in batch]
        metadatas = [c["metadata"] for c in batch]

        self._push_with_backoff(texts, metadatas)

        # ── track which documents are now fully out of the buffer ─────────
        flushed_docs       = []
        chunks_in_batch    = len(batch)

        while chunks_in_batch > 0 and self._pending_docs:
            doc = self._pending_docs[0]

            if doc["chunk_count"] <= chunks_in_batch:
                # all remaining chunks of this doc were in the batch
                chunks_in_batch -= doc["chunk_count"]
                flushed_docs.append(doc)
                self._pending_docs.pop(0)
            else:
                # doc spans into the next batch — subtract what we sent
                doc["chunk_count"] -= chunks_in_batch
                chunks_in_batch     = 0

        return flushed_docs

    def _push_with_backoff(
        self,
        texts: list[str],
        metadatas: list[dict],
        max_retries: int = 5,
    ) -> None:
        # Show which docs are mixed in this batch (for visibility)
        urls_in_batch = list({m["source_url"] for m in metadatas})
        print(f"\n  🚀 Pushing {len(texts)} chunks from {len(urls_in_batch)} doc(s):")
        for u in urls_in_batch:
            print(f"       • {u.split('/')[-1]}")

        for attempt in range(max_retries):
            try:
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    namespace=NAMESPACE,
                )
                self.total_requests += 1
                if self.budget_callback:
                    self.budget_callback(1)
                print(f"  ✓ Pushed  [request #{self.total_requests}]")
                KEY_POOL.rotate()          # ← round-robin to next key after every success
                sleep(30)
                return
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    KEY_POOL.cooldown()    # ← cool down + auto-switch key, no fixed wait
                    # rebuild vector_store with new active key
                    self.vector_store = PineconeVectorStore(
                        index_name=INDEX_NAME,
                        embedding=KEY_POOL.get(),
                    )
                else:
                    raise
        raise RuntimeError(f"Pinecone push failed after {max_retries} retries.")

    

"""
scraper.py — DOJ Epstein PDF scraper → Pinecone pipeline

Key behaviours:
  - Uses ChunkAccumulator from ingest.py so chunks from multiple small PDFs
    are grouped into one Pinecone request (20–30 chunks per API call).
  - Every chunk carries source_url in metadata — URLs are never lost.
  - DB records are written only AFTER chunks are confirmed in Pinecone.
  - Daily request budget enforced (persisted to disk, resets each day).
  - PDF link lists cached per dataset page (skip re-scraping on re-run).
  - Text + chunk disk caches mean zero re-work after a crash.
"""

import time, os, json, random, tempfile, hashlib
from tqdm import tqdm
from db import is_aldready_processed, save_pdf_record
from ingest import (
    extract_text_from_pdf,
    chunk_text,
    setup_pinecone_index,
    ChunkAccumulator,
    CACHE_DIR,
    _url_hash,
)
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from curl_cffi import requests as cf_requests
from pinecone import Pinecone

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
NAMESPACE = "epstein-docs"
DELAY     = 1.5     # seconds between PDF downloads — polite to DOJ server
TIMEOUT   = 30

# Daily cap on Pinecone upsert requests.
# Each request = 1 batch of 30–40 chunks.
# Tune to stay within your Gemini embedding quota.
# Example: 200 req/day × 35 avg chunks × 20 days ≈ 140,000 chunks total
DAILY_REQUEST_BUDGET = 200
BUDGET_FILE          = os.path.join(CACHE_DIR, "daily_budget.json")

DATASET_PAGES = [
    "https://www.justice.gov/epstein/doj-disclosures/data-set-6-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-7-files",
    # "https://www.justice.gov/epstein/doj-disclosures/data-set-8-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-9-files",
    # "https://www.justice.gov/epstein/doj-disclosures/data-set-10-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-11-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-12-files",
]

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":  "document",
    "Sec-Fetch-Mode":  "navigate",
    "Sec-Fetch-Site":  "none",
    "Sec-Fetch-User":  "?1",
}


# ── Daily request budget ───────────────────────────────────────────────────────

def _today() -> str:
    from datetime import date
    return date.today().isoformat()

def _load_budget() -> dict:
    if os.path.exists(BUDGET_FILE):
        with open(BUDGET_FILE) as f:
            data = json.load(f)
        if data.get("date") == _today():
            return data
    return {"date": _today(), "requests_used": 0}

def _save_budget(data: dict) -> None:
    with open(BUDGET_FILE, "w") as f:
        json.dump(data, f)

def budget_remaining() -> int:
    b = _load_budget()
    return max(0, DAILY_REQUEST_BUDGET - b["requests_used"])

def charge_budget(n: int) -> None:
    b = _load_budget()
    b["requests_used"] += n
    _save_budget(b)
    rem = DAILY_REQUEST_BUDGET - b["requests_used"]
    print(f"  📊 Budget: {b['requests_used']}/{DAILY_REQUEST_BUDGET} today  ({rem} remaining)")


# ── PDF link scraper (disk-cached) ─────────────────────────────────────────────

def _link_cache_path(page_url: str) -> str:
    key = hashlib.sha256(page_url.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"links_{key}.json")

def scrape_pdf_links(page_url: str) -> list[dict]:
    """Scrape all PDF links from a DOJ dataset page. Result cached to disk."""
    cache_path = _link_cache_path(page_url)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            links = json.load(f)
        print(f"  📦 Link cache hit — {len(links)} PDFs  ({page_url.split('/')[-1]})")
        return links

    dataset   = page_url.strip("/").split("/")[-1]
    session   = cf_requests.Session(impersonate="chrome120")
    all_links: list[dict] = []

    print(f"\n🏠 Loading: {page_url}")
    res = session.get(page_url, timeout=TIMEOUT)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    last_page = 0
    for a in soup.find_all("a", href=True):
        if "?page=" in a["href"]:
            try:
                last_page = max(last_page, int(a["href"].split("?page=")[-1]))
            except ValueError:
                pass
    print(f"  📋 {last_page + 1} pages (0–{last_page})")

    def _harvest(s: BeautifulSoup) -> list[dict]:
        return [
            {
                "filename": a["href"].split("/")[-1],
                "url":      urljoin("https://www.justice.gov", a["href"]),
                "dataset":  dataset,
            }
            for a in s.find_all("a", href=True)
            if a["href"].lower().endswith(".pdf")
        ]

    all_links.extend(_harvest(soup))
    print(f"  ✓ Page 0: {len(all_links)} PDFs")

    for i in range(1, last_page + 1):
        time.sleep(random.uniform(2, 4))
        res = session.get(
            f"{page_url}?page={i}",
            headers={"Referer": page_url if i == 1 else f"{page_url}?page={i-1}"},
            timeout=TIMEOUT,
        )
        res.raise_for_status()
        page_links = _harvest(BeautifulSoup(res.text, "html.parser"))
        print(f"  ✓ Page {i}: {len(page_links)} PDFs")
        with open(cache_path, "w") as f:
            json.dump(all_links, f)
        all_links.extend(page_links)

    print(f"\n📄 {dataset}: {len(all_links)} total PDFs")
    return all_links


# ── Download ───────────────────────────────────────────────────────────────────

_download_session = None

def get_download_session() -> cf_requests.Session:
    global _download_session
    if _download_session is not None:
        return _download_session
    session     = cf_requests.Session(impersonate="chrome120")
    cookie_file = os.path.join(os.path.dirname(__file__), "justice_cookies.json")
    with open(cookie_file) as f:
        cookies = json.load(f)
    for c in cookies:
        session.cookies.set(
            c["name"], c["value"],
            domain=c.get("domain", "justice.gov"),
            path=c.get("path", "/"),
        )
    print(f"✓ Loaded {len(cookies)} cookies")
    _download_session = session
    return session

def download_pdf(url: str) -> str | None:
    try:
        session    = get_download_session()
        res        = session.get(url, timeout=TIMEOUT, stream=True)
        raw_chunks = []
        first_bytes = b""

        for chunk in res.iter_content(chunk_size=8192):
            if not first_bytes:
                first_bytes = chunk
            raw_chunks.append(chunk)

        if b"age-verify" in first_bytes or b"<html" in first_bytes[:100]:
            print(f"    ✗ Age-gate not bypassed")
            return None
        if not first_bytes.startswith(b"%PDF"):
            print(f"    ✗ Not a PDF (starts: {first_bytes[:20]})")
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        for chunk in raw_chunks:
            tmp.write(chunk)
        tmp.close()
        return tmp.name

    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return None


# ── Per-PDF preparation (extract + chunk only, no push) ───────────────────────

def prepare_pdf(pdf_info: dict) -> tuple[list[dict], dict] | tuple[None, None]:
    """
    Download → extract text → chunk.
    Does NOT push to Pinecone — just returns (chunks, doc_meta).
    The caller feeds this into ChunkAccumulator.add().

    Returns (None, None) on failure.
    """
    filename  = pdf_info["filename"]
    url       = pdf_info["url"]
    dataset   = pdf_info["dataset"]
    cache_key = _url_hash(url)

    print(f"\n  📄 {filename}")
    tmp_path = download_pdf(url)
    if not tmp_path:
        return None, None

    try:
        raw_text = extract_text_from_pdf(tmp_path, cache_key=cache_key)
        if not raw_text.strip():
            print("    ⚠ No text — skipping")
            return None, None
        print(f"    ✓ {len(raw_text):,} chars")

        chunks = chunk_text(raw_text, filename, url, dataset, cache_key=cache_key)
        print(f"    ✓ {len(chunks)} chunks (buffered)")

        doc_meta = {
            "filename":   filename,
            "url":        url,
            "dataset":    dataset,
            "char_count": len(raw_text),
        }
        return chunks, doc_meta

    except Exception as e:
        import traceback
        print(f"    ✗ Error: {e}")
        traceback.print_exc()
        return None, None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_scraper(dataset_pages: list[str] | None = None) -> None:
    """
    For each dataset page:
      1. Get PDF links (cached).
      2. For each PDF: download + extract + chunk → feed into accumulator.
      3. Accumulator auto-flushes every time buffer hits 30 chunks.
         Many small PDFs are batched together → fewer API requests.
      4. At end of page (or run), flush remaining buffer.
      5. DB records written only after Pinecone confirms the push.
    """
    pages = dataset_pages or DATASET_PAGES

    print("Loading vector store…")
    pc           = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vector_store = setup_pinecone_index(pc)
    print("✓ Vector store ready\n")

    # accumulator shared across ALL pages in this run
    accumulator = ChunkAccumulator(
        vector_store=vector_store,
        budget_callback=charge_budget,      # charge daily budget on each flush
    )

    total_processed = total_skipped = total_failed = 0

    for page_url in pages:
        if budget_remaining() == 0:
            print("\n🛑 Daily budget exhausted — run again tomorrow.")
            break

        pdf_links = scrape_pdf_links(page_url)
        if not pdf_links:
            continue

        print(f"\n⚙  Processing {len(pdf_links)} PDFs from {page_url.split('/')[-1]}")

        for pdf_info in tqdm(pdf_links, desc=page_url.split("/")[-1]):

            # ── budget guard ─────────────────────────────────────────────────
            if budget_remaining() == 0:
                print("\n🛑 Daily budget hit — flushing buffer then stopping.")
                break

            # ── skip already-done ─────────────────────────────────────────────
            if is_aldready_processed(pdf_info["filename"]):
                total_skipped += 1
                continue

            # ── extract + chunk (no API call yet) ────────────────────────────
            chunks, doc_meta = prepare_pdf(pdf_info)
            if chunks is None:
                total_failed += 1
                time.sleep(DELAY)
                continue

            # ── feed accumulator; get back any docs NOW confirmed in Pinecone ─
            # The accumulator only pushes when buffer >= MIN_BATCH (30 chunks).
            # Multiple small PDFs are grouped into one API request automatically.
            confirmed_docs = accumulator.add(chunks, doc_meta)

            for doc in confirmed_docs:
                save_pdf_record(
                    filename=doc["filename"],
                    url=doc["url"],
                    dataset=doc["dataset"],
                    chunk_count=doc["chunk_count"],
                    char_count=doc["char_count"],
                )
                total_processed += 1

            time.sleep(DELAY)

    # ── end-of-run: flush whatever is left in the buffer ─────────────────────
    print(f"\n💾 Flushing remaining {accumulator.buffer_size()} buffered chunks…")
    final_docs = accumulator.flush_all()
    for doc in final_docs:
        save_pdf_record(
            filename=doc["filename"],
            url=doc["url"],
            dataset=doc["dataset"],
            chunk_count=doc["chunk_count"],
            char_count=doc["char_count"],
        )
        total_processed += 1

    print(f"\n{'─'*52}")
    print(f"✅  Run complete!")
    print(f"   Processed  : {total_processed} docs pushed to Pinecone")
    print(f"   Skipped    : {total_skipped}  (already in DB)")
    print(f"   Failed     : {total_failed}")
    print(f"   API calls  : {accumulator.total_requests}")
    print(f"   Budget     : {DAILY_REQUEST_BUDGET - budget_remaining()}/{DAILY_REQUEST_BUDGET} requests used today")


if __name__ == "__main__":
    run_scraper()




# import time
# import os
# import tempfile
# import random
# from tqdm import tqdm
# from db import is_aldready_processed,save_pdf_record,get_db
# from ingest import extract_text_from_pdf,chunk_text,setup_pinecone_index
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from dotenv import load_dotenv
# from curl_cffi import requests as cf_requests
# import time,random
# import json
# from pinecone import Pinecone
# import os
# # import browser_cookie3
# load_dotenv()
# BASE_URL   = "https://www.justice.gov/epstein/doj-disclosures"
# NAMESPACE  = "epstein-docs"
# DELAY      = 1.5    # seconds between requests — be polite to DOJ server
# TIMEOUT    = 30     # seconds before giving up on a download
# # HEADERS    = {
# #     "User-Agent": (
# #         "Mozilla/5.0 (compatible; GovTransparencyRAG/1.0; "
# #         "educational research project)"
# #     )
# # }
# HEADERS = {
#     "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
#     "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
#     "Accept-Language": "en-US,en;q=0.5",
#     "Accept-Encoding": "gzip, deflate, br",
#     "Connection":      "keep-alive",
#     "Upgrade-Insecure-Requests": "1",
#     "Sec-Fetch-Dest":  "document",
#     "Sec-Fetch-Mode":  "navigate",
#     "Sec-Fetch-Site":  "none",
#     "Sec-Fetch-User":  "?1",
# }


# DATASET_PAGES = [
#     # "https://www.justice.gov/epstein/doj-disclosures/data-set-1-files",
#     # "https://www.justice.gov/epstein/doj-disclosures/data-set-2-files",
#     # "https://www.justice.gov/epstein/doj-disclosures/data-set-3-files",
#     # "https://www.justice.gov/epstein/doj-disclosures/data-set-4-files",
#     # "https://www.justice.gov/epstein/doj-disclosures/data-set-5-files",
#     "https://www.justice.gov/epstein/doj-disclosures/data-set-6-files",
#     "https://www.justice.gov/epstein/doj-disclosures/data-set-7-files",
#     "https://www.justice.gov/epstein/doj-disclosures/data-set-8-files",
#     "https://www.justice.gov/epstein/doj-disclosures/data-set-9-files",
#     "https://www.justice.gov/epstein/doj-disclosures/data-set-10-files",
#     "https://www.justice.gov/epstein/doj-disclosures/data-set-11-files",
#     "https://www.justice.gov/epstein/doj-disclosures/data-set-12-files",
# ]
# # def scrape_pdf_links(page_url:str)->list[dict]:
# #     """
# #     Scrape all PDF links from a DOJ dataset page.
# #     Returns list of {filename, url, dataset} dicts.
# #     """
# #     full_page_url=page_url
# #     print(f"\n🔍 Scraping: {full_page_url}")
# #     try:
# #         res=requests.get(full_page_url,headers=HEADERS,timeout=TIMEOUT)
# #         res.raise_for_status()
# #     except Exception as e:
# #         print(f"  ✗ Failed to fetch page: {e}")
# #         return []
# #     soup=BeautifulSoup(res.text,"html.parser") #used to fetch href links exactly
# #     links=[]
# #     dataset=full_page_url.strip("/").split("/")[-1] #extracts the last part of url eg. dataset-1
        
# #     for a in soup.find_all("a",href=True):
# #         href=a["href"]
# #         if href.lower().endswith(".pdf"):
# #             full_url=urljoin("https://www.justice.gov",href)
# #             filename=href.split("/")[-1]
# #             links.append({
# #                 "filename":filename,
# #                 "url":full_url,
# #                 "dataset":dataset
# #             })
# #     print(f"  ✓ Found {len(links)} PDF links")
# #     time.sleep(DELAY)

# def scrape_pdf_links(page_url: str) -> list[dict]:
#     all_links=[]
#     dataset=page_url.strip("/").split("/")[-1]
#     #session exactly looks like chrome at TLS layer
#     session=cf_requests.Session(impersonate="chrome120")
#     print(f"\n🏠 Loading: {page_url}")
#     res = session.get(page_url, timeout=TIMEOUT)
#     res.raise_for_status()
#     soup=BeautifulSoup(res.text,"html.parser")
#     last_page=0
#     for a in soup.find_all("a",href=True):
#         if "?page=" in a["href"]:
#             try:
#                 n = int(a["href"].split("?page=")[-1])
#                 last_page = max(last_page, n)
#             except ValueError:
#                 pass
#     print(f"  📋 {last_page + 1} pages (0–{last_page})")
#     #page 0 
#     for a in soup.find_all("a",href=True):
#         if a["href"].lower().endswith(".pdf"):
#             all_links.append({
#                 "filename":a["href"].split("/")[-1],
#                 "url":      urljoin("https://www.justice.gov", a["href"]),
#                 "dataset":dataset
#             })
#     print(f"  ✓ Page 0: {len(all_links)} PDFs")
#     #page 1 to last page
#     for i in range(1,last_page+1):
#         time.sleep(random.uniform(2,4))
#         full_url=f"{page_url}?page={i}"
#         print(f"Scraping  🔍 {full_url}..")
#         #different sessions for different pages
#         res=session.get(
#             full_url,
#             headers={"Referer": page_url if i == 1 else f"{page_url}?page={i-1}"},
#             timeout=TIMEOUT
#         )
#         res.raise_for_status()
#         soup=BeautifulSoup(res.text,"html.parser")
#         links = [
#             {
#                 "filename": a["href"].split("/")[-1],
#                 "url":      urljoin("https://www.justice.gov", a["href"]),
#                 "dataset":  dataset,
#             }
#             for a in soup.find_all("a", href=True)
#             if a["href"].lower().endswith(".pdf")
#         ]
#         print(f"  ✓ Page {i}: {len(links)} PDFs")
#         all_links.extend(links)
#     print(f"\n📄 {dataset}: {len(all_links)} total PDFs")
#     return all_links
# # scrape_pdf_links("https://www.justice.gov/epstein/doj-disclosures/data-set-8-files")
# # test passed successfully

# # def download_pdf(url:str)->str | None:
# #     """Download a PDF into a temp file, return path."""
# #     try:
# #         session=cf_requests.Session(impersonate="chrome120")
# #         res=session.get(url,timeout=TIMEOUT,stream=True)
# #         # setting session cookie for age restriction
# #         session.cookies.set("justiceGate", "true", domain="www.justice.gov", path="/")
# #         res.raise_for_status()
# #         tmp=tempfile.NamedTemporaryFile(suffix=".pdf",delete=False)
# #         for chunk in res.iter_content(chunk_size=8192):
# #             tmp.write(chunk)
# #         tmp.close()
# #         return tmp.name
# #     except Exception as e:
# #         print(f"    ✗ Download failed: {e}")
# #         return None
# _download_session = None

# def get_download_session() -> cf_requests.Session:
#     """
#     Initiating the session with all the cookies needed for downloading pdf
#     """
#     global _download_session
#     if _download_session is not None:
#         return _download_session
#     session=cf_requests.Session(impersonate="chrome120")
#     cookie_file=os.path.join(os.path.dirname(__file__),"justice_cookies.json")
#     with open(cookie_file) as f:
#         cookies=json.load(f) #loading cookies from file
#     for c in cookies: #setting all the cookies to the session
#         session.cookies.set(
#             c["name"],
#             c["value"],
#             domain=c.get("domain","justice.gov"),
#             path=c.get("path","/")
#         )
#     print(f"✓ Loaded {len(cookies)} cookies from file")
#     _download_session=session
#     return session


# def download_pdf(url: str) -> str | None:
#     """Download a PDF into a temp file, return path."""
#     try:
#         session = get_download_session()
#         res = session.get(url, timeout=TIMEOUT, stream=True)

#         # Detect HTML instead of PDF
#         content_type = res.headers.get("Content-Type", "")
#         first_bytes  = b""
#         chunks       = []

#         for chunk in res.iter_content(chunk_size=8192):
#             if not first_bytes:
#                 first_bytes = chunk
#             chunks.append(chunk)

#         if b"age-verify" in first_bytes or b"<html" in first_bytes[:100]:
#             print(f"    ✗ Still getting HTML — age gate not bypassed")
#             return None

#         if not first_bytes.startswith(b"%PDF"):
#             print(f"    ✗ Not a valid PDF (starts with: {first_bytes[:20]})")
#             return None

#         tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
#         for chunk in chunks:
#             tmp.write(chunk)
#         tmp.close()
#         return tmp.name

#     except Exception as e:
#         print(f"    ✗ Download failed: {e}")
#         return None
# # print(download_pdf("https://www.justice.gov/epstein/files/DataSet%204/EFTA00008320.pdf"))
# #download test passed
# from time import sleep

# def embed_with_retry(vector_store, texts, metadatas, namespace, max_retries=5):
#     """Upload chunks with exponential backoff on rate limit."""
#     batch_size = 25  # smaller batches = less likely to hit quota

#     for i in range(0, len(texts), batch_size):
#         batch_texts     = texts[i:i+batch_size]
#         batch_metadatas = metadatas[i:i+batch_size]

#         for attempt in range(max_retries):
#             try:
#                 vector_store.add_texts(
#                     texts=batch_texts,
#                     metadatas=batch_metadatas,
#                     namespace=namespace
#                 )
#                 print(f"    ✓ Batch {i//batch_size + 1}: {len(batch_texts)} chunks pushed")
#                 sleep(5)  # polite delay between every batch
#                 break      # success — move to next batch

#             except Exception as e:
#                 if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
#                     wait = 60 * (attempt + 1)  # 60s, 120s, 180s...
#                     print(f"    ⏳ Rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})")
#                     sleep(wait)
#                 else:
#                     raise  # non-rate-limit error — re-raise immediately
#         else:
#             raise Exception(f"Failed after {max_retries} retries on batch {i//batch_size + 1}")
# def process_pdf(pdf_info:dict,vector_store)->bool:
#     """
#     Full pipeline for one PDF:
#     download → extract → chunk → embed → push to Pinecone → delete

#     Returns True if successful, False if failed.
#     """
#     filename=pdf_info["filename"]
#     url=pdf_info["url"]
#     dataset=pdf_info["dataset"]
#     print(f"\n  📄 {filename}")
#     tmp_path=download_pdf(url)
#     if not tmp_path:
#         return False
#     try:
#         raw_text=extract_text_from_pdf(tmp_path)
#         if not raw_text.strip():
#             return False
#         print(f"    ✓ Extracted {len(raw_text):,} chars")
#         #chunk text
#         chunk_data=chunk_text(raw_text,filename,url,dataset)
#         texts=[c["text"] for c in chunk_data]
#         metadatas=[c["metadata"] for c in chunk_data]
#         embed_with_retry(vector_store, texts, metadatas, NAMESPACE)
#         print(f"    ✓ Pushed {len(texts)} chunks to Pinecone")
#         save_pdf_record(
#             filename=filename,
#             url=url,
#             dataset=dataset,
#             chunk_count=len(texts),
#             char_count=len(raw_text)
#         )
#         return True

#     except Exception as e:
#         import traceback
#         print(f"    ✗ Processing error: {e}")
#         traceback.print_exc()  # ← add this to see full traceback
#         return False
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)
# def run_scraper(dataset_pages:list[str] | None):
#     """
#     Main entry point.
#     Scrapes all dataset pages, processes each PDF one at a time.
#     Skips already-processed files using Postgres record.
#     """
#     pages=dataset_pages or DATASET_PAGES
#     print("Loading vector store...")
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     vector_store=setup_pinecone_index(pc)
#     print("✓ Vector store ready\n")
#     total_processed = 0
#     total_skipped   = 0
#     total_failed    = 0
#     for page_url in pages:
#         pdf_links = scrape_pdf_links(page_url)
#         if not pdf_links:
#             continue
#         for pdf_info in tqdm(pdf_links,desc=f"Processing {page_url}!.."):
#             filename=pdf_info["filename"]
#             url=pdf_info["url"]
#             #Skip if aldready processed (stored in postgres)
#             if is_aldready_processed(filename):
#                 total_skipped+=1
#                 continue
#             #process the current pdf
#             success=process_pdf(pdf_info,vector_store)
#             if success:
#                 total_processed += 1
#             else:
#                 total_failed+=1
#             time.sleep(DELAY)
#         print(f"\n{'─'*50}")
#         print(f"✅ Scraping complete!")
#         print(f"   Processed: {total_processed}")
#         print(f"   Skipped:   {total_skipped} (already done)")
#         print(f"   Failed:    {total_failed}")
    

# if __name__ == "__main__":
#     run_scraper(None)
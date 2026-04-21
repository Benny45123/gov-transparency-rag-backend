import json
import os
from dotenv import load_dotenv
from supabase import create_client,Client
load_dotenv()
supabase: Client=create_client(
    os.getenv("SUPABASE_URL") or "",
    os.getenv("SUPABASE_KEY") or ""
)
def get_db():
    yield supabase


def is_aldready_processed(filename:str)->bool:
    try:
        res=supabase.table("processed_pdfs").select("id").eq("filename",filename).limit(1).execute()
        return len(res.data)>0
    except Exception as e:
        print(f"Database Error:{e}")
        return False
        

def save_pdf_record(filename:str,url:str,dataset:str,chunk_count:int,char_count:int):
    supabase.table("processed_pdfs").upsert({
        "filename":    filename,
        "url":         url,
        "dataset":     dataset,
        "chunk_count": chunk_count,
        "char_count":  char_count,
    }, on_conflict="filename").execute()
def save_query(question: str, answer: str, sources: str, namespace: str="epstein-docs"):
    supabase.table("query_history").insert({
        "question":  question,
        "answer":    answer,
        "sources":   json.dumps(sources) if not isinstance(sources, str) else sources,  
        "namespace": namespace,
    }).execute()
def fetch_history(limit: int = 10) -> list:
    res = supabase.table("query_history").select("*").order("id", desc=True).limit(limit).execute()
    return [
        {
            "id":         r["id"],
            "question":   r["question"],
            "answer":     r["answer"],
            "sources":    json.loads(r["sources"] or "[]"),  
            "namespace":  r["namespace"],
            "created_at": r["created_at"],
        }
        for r in res.data  
    ]
def get_stats() -> dict:
    pdfs   = supabase.table("processed_pdfs").select("id", count="exact").execute()
    chunks = supabase.table("processed_pdfs").select("chunk_count").execute()
    queries = supabase.table("query_history").select("id", count="exact").execute()

    return {
        "pdfs_processed": pdfs.count or 0,
        "total_chunks":   sum(r["chunk_count"] or 0 for r in chunks.data),
        "total_queries":  queries.count or 0,
    }
# print(get_stats())
# save_pdf_record("xyz","xyz.pdf","epstein",10,20)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pymupdf
from tqdm import tqdm
from langchain_pinecone import PineconeVectorStore
import getpass
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
def text_extract(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page_num,page in enumerate(tqdm(doc, desc="Extracting text from PDF")):
        page_text = page.get_text().strip()
        if len(page_text)>50:
            full_text += f"\n[Page {page_num+1}]\n{page_text}\n"

pdf_path = "../test-pdfs/epstein_test.pdf"
extracted_text = text_extract(pdf_path)
print(extracted_text)
splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
chunks = splitter.split_text(extracted_text)
print(chunks)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google API key: ")
else:
    print("Using existing GEMINI_API_KEY environment variable.")
embedding_model=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectors=embedding_model.embed_documents(chunks)
print(len(vectors))
print(len(vectors[0]))
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "gov-transparency-index"
vector_store = PineconeVectorStore.from_texts(
    texts=chunks,
    embedding=embedding_model,
    index_name=index_name,
    namespace="epstein-docs",  
    metadatas=[                 
        {
            "source": pdf_path,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }
        for i, chunk in enumerate(chunks)
    ]
)

print(f"✓ {len(chunks)} chunks stored in Pinecone")
print(f"  Index: {index_name}")
print(f"  Namespace: epstein-docs")
results = vector_store.similarity_search(
    query="Who did Epstein fly on his private jet?",
    k=3,                        # top 3 most relevant chunks
    namespace="epstein-docs"
)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:300])
    print(f"Source: {doc.metadata['source']}")



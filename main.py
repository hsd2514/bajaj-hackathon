from google import genai
from dotenv import load_dotenv
import os
# --- Load environment variables from .env file ---
load_dotenv()

# --- Gemini Client ---
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# --- Imports ---
import os, time, requests, fitz, numpy as np, faiss, asyncio
from functools import lru_cache
import hashlib
from typing import List

# HuggingFace sentence-transformers for local embeddings
from sentence_transformers import SentenceTransformer

# Load the embedding model once (MiniLM is fast and accurate for most RAG use)
hf_model = SentenceTransformer('all-MiniLM-L6-v2')


# --- Batch QA Function ---
def batch_answer_questions(questions: List[str], chunks, idx, top_k: int = 6, max_workers: int = 1):
    try:
        # Use HuggingFace model for question embeddings
        q_embs = hf_model.encode(questions, show_progress_bar=False, convert_to_numpy=True)
        D, I = idx.search(q_embs, top_k)
        prompts = []
        for q_idx, question in enumerate(questions):
            context = "\n\n".join(chunks[i] for i in I[q_idx])
            prompt = f"""You are analyzing the National Parivar Mediclaim Plus Policy document. Answer the question based on the provided context.\n\nPolicy Document Context:\n{context}\n\nQuestion: {question}\n\nInstructions:\n- Answer based ONLY on the context provided.\n- Be specific with numbers, timeframes, and conditions.\n- Limit your answer to one or two sentences.\n- Do not list exclusions, definitions, or repeat the question.\n- If the answer is in the context, state it directly.'\n\nAnswer:"""
            prompts.append(prompt)
        # Use Gemini for answer generation
        answers = [None] * len(questions)
        import time as _time
        def answer_one(idx_):
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompts[idx_]
                )
                _time.sleep(.66)  # Wait 0.66 seconds between requests to avoid rate limits
                return resp.text.strip()
            except Exception as e:
                return f"Error: {str(e)}"
        for i in range(len(questions)):
            answers[i] = answer_one(i)
        return answers
    except Exception as e:
        return [f"Error: {str(e)}"] * len(questions)



# --- Global Cache ---
document_cache = {}

# --- Utility Functions ---
def get_document_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

@lru_cache(maxsize=5)
def load_and_chunk_pdf_cached(url: str, chunk_size: int = 500, overlap: int = 100):
    print(f"Loading PDF from URL...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    doc = fitz.open(stream=resp.content, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size-overlap)]
    print(f"[+] {len(chunks)} chunks created")
    return chunks

import time as _time
def create_embeddings_batch(chunks_batch, max_retries=5):
    # HuggingFace model is local, no rate limits or retries needed
    try:
        return hf_model.encode(chunks_batch, show_progress_bar=False, convert_to_numpy=True).tolist()
    except Exception as e:
        print(f"[ERROR] Embedding batch failed: {e}")
        raise

def get_or_create_embeddings_parallel(doc_hash: str, chunks):
    if doc_hash in document_cache:
        print(f"[+] Using cached embeddings")
        return document_cache[doc_hash]['index'], document_cache[doc_hash]['chunks']
    print(f"[+] Creating embeddings with HuggingFace model...")
    batch_size = 32  # HuggingFace can handle larger batches
    batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    all_embeddings = []
    for i, batch in enumerate(batches):
        try:
            batch_embeddings = create_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"  Completed batch {i + 1}/{len(batches)}")
        except Exception as e:
            print(f"  Error in batch {i}: {e}")
    embs = np.array(all_embeddings, dtype="float32")
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    document_cache[doc_hash] = {'index': idx, 'chunks': chunks}
    print(f"[+] Embeddings cached")
    return idx, chunks


# --- Main ---
async def main():
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    print("=== STREAMLINED FAST RAG SYSTEM ===")
    start_time = time.time()
    doc_hash = get_document_hash(pdf_url)
    t1 = time.time()
    chunks = load_and_chunk_pdf_cached(pdf_url)
    load_time = time.time() - t1
    t2 = time.time()
    idx, chunks = get_or_create_embeddings_parallel(doc_hash, chunks)
    embed_time = time.time() - t2
    t3 = time.time()
    answers = batch_answer_questions(questions, chunks, idx, top_k=10, max_workers=6)
    qa_time = time.time() - t3
    total_time = time.time() - start_time
    for i, a in enumerate(answers, 1):
        print(f"{i}. {a}")


if __name__ == "__main__":
    asyncio.run(main())

# --- RAG Pipeline Implementation ---
import os
import httpx
import time
import fitz
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv
load_dotenv()

PDF_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre‑existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check‑ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub‑limits on room rent and ICU charges for Plan A?"
]

def chunk_pdf(url, chunk_size=350, overlap=50):
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    doc = fitz.open(stream=resp.content, filetype="pdf")
    text = " ".join(page.get_text() for page in doc)
    doc.close()
    # Improved chunking: split by sentences, then group sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = []
    total_len = 0
    for sent in sentences:
        chunk.append(sent)
        total_len += len(sent.split())
        if total_len >= chunk_size:
            chunks.append(" ".join(chunk))
            # Overlap: keep last few sentences
            chunk = chunk[-max(1, overlap//20):]
            total_len = sum(len(s.split()) for s in chunk)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def embed_chunks(chunks, client):
    # Gemini embedding API
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunks,
        config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 3072}
    )
    arr = np.array([ce.values for ce in resp.embeddings], dtype="float32")
    return arr

def build_faiss_index(embs):
    d = embs.shape[1]
    nlist = max(10, len(embs)//10)
    quant = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
    idx.train(embs)
    idx.nprobe = max(1, nlist//10)
    idx.add(embs)
    return idx

def embed_query(questions, client):
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=questions,
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 3072}
    )
    arr = np.array([ce.values for ce in resp.embeddings], dtype="float32")
    return arr

def answer_with_context(question, context, client):
    system_prompt = (
        "You are an expert insurance policy Q&A assistant. Given a question and a context from a policy document, answer as accurately, concisely, and as short and crisp as possible, using only the provided context.\n"
        "- Use only the provided context.\n"
        "- If the answer is a fact, state it clearly and directly.\n"
        "- If the answer requires conditions, eligibility, or limits, summarize the key points precisely.\n"
        "- Do not quote, cite, or mention the source, section, page, or clause.\n"
        "- Do not invent or assume information not present in the context.\n"
        "- Use clear, professional language suitable for insurance customers.\n"
        "- Do not repeat the question in your answer.\n"
        "Return only the answer as a string."
    )
    prompt = f"{system_prompt}\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return resp.text.strip()
    except Exception as e:
        print(f"[RAG] Error in answer_with_context: {e}")
        return f"[Error: {e}]"

def rag_answer_questions(pdf_url, questions, top_k=4, api_key=None):
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable. Please set GEMINI_API_KEY in your .env or environment.")
    client = genai.Client(api_key=api_key)
    print("[RAG] Chunking PDF...")
    chunks = chunk_pdf(pdf_url)
    print(f"[RAG] {len(chunks)} chunks created.")
    print("[RAG] Embedding chunks...")
    embs = embed_chunks(chunks, client)
    print("[RAG] Building FAISS index...")
    idx = build_faiss_index(embs)
    print("[RAG] Embedding questions...")
    q_embs = embed_query(questions, client)
    top_k = 6
    D, I = idx.search(q_embs, top_k)
    print(f"[RAG] Retrieved top-{top_k} chunks per question.")
    answers = []
    for i, q in enumerate(questions):
        # Optionally filter out duplicate chunks
        unique_idxs = []
        seen = set()
        for j in I[i]:
            if j not in seen:
                unique_idxs.append(j)
                seen.add(j)
        context = "\n\n".join(chunks[j] for j in unique_idxs)
        ans = answer_with_context(q, context, client)
        print(f"A{i+1}: {ans}\n")
        answers.append(ans)
    return answers

def main():
    print("=== RAG PDF Q&A ===")
    t0 = time.time()
    answers = rag_answer_questions(PDF_URL, QUESTIONS)
    print(f"[+] Total runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

# --- RAG Pipeline Implementation ---
import os
import httpx
import time
import fitz
import zipfile
import docx
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv
import io
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

def chunk_pdf(url_or_bytes, chunk_size=350, overlap=50):
    import re
    def extract_text_from_pdf(pdf_bytes):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        return text

    def extract_text_from_docx(docx_bytes):
        with open("_temp.docx", "wb") as f:
            f.write(docx_bytes)
        doc = docx.Document("_temp.docx")
        text = " ".join([p.text for p in doc.paragraphs])
        os.remove("_temp.docx")
        return text

    def extract_text_from_txt(txt_bytes):
        return txt_bytes.decode(errors="ignore")

    def extract_text_from_binary(bin_bytes, ext_hint=None):
        # Fallback: try to decode as text
        try:
            return bin_bytes.decode(errors="ignore")
        except Exception:
            return ""

    # Handle URL or bytes (PDF, ZIP, DOCX, TXT, etc.)
    if isinstance(url_or_bytes, str):
        resp = httpx.get(url_or_bytes, timeout=30)
        resp.raise_for_status()
        data = resp.content
        ext = url_or_bytes.split("?")[0].split(".")[-1].lower()
    else:
        data = url_or_bytes
        ext = None

    text = ""
    if ext == "zip" or (ext is None and zipfile.is_zipfile(io.BytesIO(data))):
        all_texts = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for fname in z.namelist():
                fext = fname.split(".")[-1].lower()
                fbytes = z.read(fname)
                if fext == "pdf":
                    all_texts.append(extract_text_from_pdf(fbytes))
                elif fext == "docx":
                    all_texts.append(extract_text_from_docx(fbytes))
                elif fext == "txt":
                    all_texts.append(extract_text_from_txt(fbytes))
        text = " ".join(all_texts)
    elif ext == "pdf" or (ext is None and fitz.open(stream=data, filetype="pdf")):
        text = extract_text_from_pdf(data)
    elif ext == "docx":
        text = extract_text_from_docx(data)
    elif ext == "txt":
        text = extract_text_from_txt(data)
    else:
        # Try PDF first, fallback to TXT, then binary extraction
        try:
            text = extract_text_from_pdf(data)
        except Exception:
            try:
                text = extract_text_from_txt(data)
            except Exception:
                text = extract_text_from_binary(data, ext)

    # Improved chunking: split by paragraphs, clean and deduplicate
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    # Remove duplicates and very short chunks
    seen = set()
    clean_paragraphs = []
    for p in paragraphs:
        pnorm = p.lower().strip()
        if len(pnorm) < 30 or pnorm in seen:
            continue
        seen.add(pnorm)
        clean_paragraphs.append(p)
    # Group paragraphs into chunks of ~chunk_size words
    chunks = []
    chunk = []
    total_len = 0
    for para in clean_paragraphs:
        chunk.append(para)
        total_len += len(para.split())
        if total_len >= chunk_size:
            chunks.append("\n\n".join(chunk))
            # Overlap: keep last paragraph
            chunk = chunk[-1:]
            total_len = sum(len(s.split()) for s in chunk)
    if chunk:
        chunks.append("\n\n".join(chunk))
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
        "- List all relevant facts, exclusions, and limits.\n"
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
    top_k = 12
    D, I = idx.search(q_embs, top_k)
    print(f"[RAG] Retrieved top-{top_k} chunks per question.")
    answers = []
    for i, q in enumerate(questions):
        unique_idxs = []
        seen = set()
        for j in I[i]:
            if j not in seen:
                unique_idxs.append(j)
                seen.add(j)
        candidate_chunks = [chunks[j] for j in unique_idxs]
        # Semantic re-ranking: use Gemini to score each chunk for relevance
        scored_chunks = []
        for chunk in candidate_chunks:
            score_prompt = (
                f"Given the following insurance policy context and question, rate the relevance of the context to answering the question on a scale of 1 (not relevant) to 5 (highly relevant).\n"
                f"Context: {chunk}\nQuestion: {q}\nRelevance score:"
            )
            try:
                score_resp = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=score_prompt
                )
                score_text = score_resp.text.strip()
                score = int(next((s for s in score_text if s.isdigit()), '1'))
            except Exception:
                score = 1
            scored_chunks.append((score, chunk))
        # Select top N chunks by score, up to ~3500 words (to fit Gemini context)
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        top_chunks = []
        total_words = 0
        for score, chunk in scored_chunks:
            nwords = len(chunk.split())
            if total_words + nwords > 3500:
                break
            top_chunks.append(chunk)
            total_words += nwords
        context = "\n\n".join(top_chunks)
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

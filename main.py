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
import math
import re
from typing import List, Tuple
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
    elif ext == "pdf":
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
    # Group paragraphs into chunks of ~chunk_size words with simple overlap
    chunks = []
    chunk = []
    total_len = 0
    for para in clean_paragraphs:
        chunk.append(para)
        total_len += len(para.split())
        if total_len >= chunk_size:
            chunks.append("\n\n".join(chunk))
            # Overlap: keep last paragraph to maintain continuity
            if overlap > 0 and len(chunk) > 0:
                chunk = chunk[-1:]
            else:
                chunk = []
            total_len = sum(len(s.split()) for s in chunk)
    if chunk:
        chunks.append("\n\n".join(chunk))
    return chunks

def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms

def embed_chunks(chunks, client):
    # Gemini embedding API
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunks,
        config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 3072}
    )
    arr = np.array([ce.values for ce in resp.embeddings], dtype="float32")
    return _l2_normalize_rows(arr)

def build_faiss_index(embs: np.ndarray):
    """Build a cosine-similarity index using inner product on L2-normalized vectors.
    For small collections, use a flat index for accuracy. For larger, use IVF.
    """
    d = embs.shape[1]
    num = embs.shape[0]
    if num < 5000:
        idx = faiss.IndexFlatIP(d)
        idx.add(embs)
        return idx
    # IVF for scalability
    nlist = max(32, num // 100)
    quant = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(embs)
    idx.nprobe = max(1, nlist // 10)
    idx.add(embs)
    return idx

def embed_query(questions, client):
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=questions,
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 3072}
    )
    arr = np.array([ce.values for ce in resp.embeddings], dtype="float32")
    return _l2_normalize_rows(arr)

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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def _build_tfidf(chunks: List[str]):
    vocab: dict = {}
    doc_term_counts: List[dict] = []
    df_counts: dict = {}
    for chunk in chunks:
        tokens = _tokenize(chunk)
        counts = {}
        for tok in tokens:
            idx = vocab.setdefault(tok, len(vocab))
            counts[idx] = counts.get(idx, 0) + 1
        doc_term_counts.append(counts)
        for idx in counts.keys():
            df_counts[idx] = df_counts.get(idx, 0) + 1
    num_docs = len(chunks)
    idf = np.zeros(len(vocab), dtype=np.float32)
    for idx, df in df_counts.items():
        idf[idx] = math.log((1 + num_docs) / (1 + df)) + 1.0
    # Build dense tf-idf matrix
    mat = np.zeros((num_docs, len(vocab)), dtype=np.float32)
    for i, counts in enumerate(doc_term_counts):
        if not counts:
            continue
        max_tf = max(counts.values())
        for j, tf in counts.items():
            mat[i, j] = (0.5 + 0.5 * tf / max_tf) * idf[j]
    mat = _l2_normalize_rows(mat)
    return vocab, idf, mat

def _tfidf_vectorize_query(question: str, vocab: dict, idf: np.ndarray) -> np.ndarray:
    tokens = _tokenize(question)
    counts = {}
    for tok in tokens:
        if tok in vocab:
            j = vocab[tok]
            counts[j] = counts.get(j, 0) + 1
    vec = np.zeros((len(idf),), dtype=np.float32)
    if not counts:
        return vec
    max_tf = max(counts.values())
    for j, tf in counts.items():
        vec[j] = (0.5 + 0.5 * tf / max_tf) * idf[j]
    norm = np.linalg.norm(vec) + 1e-12
    vec /= norm
    return vec

def _offline_answer(question: str, context: str) -> str:
    # Extract top 2 sentences most similar to the question by token overlap
    sentences = re.split(r"(?<=[.!?])\s+", context)
    if not sentences:
        return "Information not found in document."
    q_tokens = set(_tokenize(question))
    scored = []
    for s in sentences:
        s_tokens = set(_tokenize(s))
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens) / (len(q_tokens) + 1e-9)
        scored.append((overlap, s.strip()))
    if not scored:
        return sentences[0][:300]
    scored.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [s for _, s in scored[:2]]
    ans = " ".join(top_sentences).strip()
    return ans[:800]

def rag_answer_questions(pdf_url, questions, top_k=12, api_key=None):
    # Try to initialize Gemini client if API key is provided
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    client = None
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"[RAG] Could not init Gemini client, falling back offline: {e}")
            client = None

    print("[RAG] Chunking document...")
    chunks = chunk_pdf(pdf_url)
    print(f"[RAG] {len(chunks)} chunks created.")
    if not chunks:
        return ["No content found in document."] * len(questions)

    # ONLINE PATH: Gemini embeddings + FAISS cosine
    if client is not None:
        print("[RAG] Embedding chunks (Gemini)...")
        embs = embed_chunks(chunks, client)
        print("[RAG] Building FAISS index (cosine)...")
        idx = build_faiss_index(embs)
        print("[RAG] Embedding questions (Gemini)...")
        q_embs = embed_query(questions, client)
        D, I = idx.search(q_embs, top_k)
        print(f"[RAG] Retrieved top-{top_k} chunks per question.")
        answers = []
        for i, q in enumerate(questions):
            # Unique candidate chunk ids in order of retrieval
            seen = set()
            unique_idxs = []
            for j in I[i]:
                if j not in seen:
                    unique_idxs.append(j)
                    seen.add(j)
            candidate_chunks = [chunks[j] for j in unique_idxs]

            # Semantic re-ranking with LLM scoring
            scored_chunks = []
            for chunk in candidate_chunks:
                score_prompt = (
                    "Rate relevance of the context to the question on 1-5.\n"
                    f"Context: {chunk}\nQuestion: {q}\nRelevance score:"
                )
                try:
                    score_resp = client.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=score_prompt
                    )
                    score_text = score_resp.text.strip()
                    digits = [int(ch) for ch in score_text if ch.isdigit()]
                    score = digits[0] if digits else 1
                except Exception:
                    score = 1
                scored_chunks.append((score, chunk))

            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            # Limit total words to keep prompt within context
            top_chunks = []
            total_words = 0
            for score, chunk in scored_chunks:
                nwords = len(chunk.split())
                if total_words + nwords > 3000:
                    break
                top_chunks.append(chunk)
                total_words += nwords
            context = "\n\n".join(top_chunks)
            ans = answer_with_context(q, context, client)
            print(f"A{i+1}: {ans}\n")
            answers.append(ans)
        return answers

    # OFFLINE PATH: TF-IDF cosine retrieval + extractive answer
    print("[RAG] Building TF-IDF index (offline)...")
    vocab, idf, doc_matrix = _build_tfidf(chunks)
    answers = []
    for q in questions:
        q_vec = _tfidf_vectorize_query(q, vocab, idf)
        sims = doc_matrix @ q_vec
        top_idx = np.argsort(-sims)[: top_k]
        # Build concise context from top chunks (limit words)
        context_chunks = []
        total_words = 0
        for j in top_idx:
            chunk = chunks[int(j)]
            nwords = len(chunk.split())
            if total_words + nwords > 1200:
                break
            context_chunks.append(chunk)
            total_words += nwords
        context = "\n\n".join(context_chunks)
        ans = _offline_answer(q, context)
        answers.append(ans)
    return answers

def main():
    print("=== RAG PDF Q&A ===")
    t0 = time.time()
    answers = rag_answer_questions(PDF_URL, QUESTIONS)
    print(f"[+] Total runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

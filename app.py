from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import main  # Import your RAG logic from main.py
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

API_KEY = os.getenv("HACKRX_API_KEY")  # Use env var for API key

@app.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(
    req: RunRequest,
    authorization: Optional[str] = Header(None)
):
    # Auth check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Run the RAG pipeline (reuse your main logic, but async)
    pdf_url = req.documents
    questions = req.questions
    doc_hash = main.get_document_hash(pdf_url)
    chunks = main.load_and_chunk_pdf_cached(pdf_url)
    idx, chunks = main.get_or_create_embeddings_parallel(doc_hash, chunks)
    answers = await asyncio.to_thread(main.batch_answer_questions, questions, chunks, idx, 10, 1)
    return {"answers": answers}

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import os
from dotenv import load_dotenv
import main  # Uses your RAG logic

load_dotenv()

app = FastAPI()

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

API_KEY = os.getenv("HACKRX_API_KEY")

@app.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(req: RunRequest, authorization: Optional[str] = Header(None)):
    # --- API Key Auth ---
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # # --- RAG Execution ---
    # try:
    #     idx, chunks = main.process_document(req.documents)
    #     answers = await asyncio.to_thread(
    #         main.batch_answer_questions, req.questions, idx, chunks, 10
    #     )
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

    # return {"answers": answers}
    # --- Gemini PDF Q&A Execution ---
    try:
        answers = await asyncio.to_thread(
            main.answer_questions_with_gemini, req.documents, req.questions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answers": answers}

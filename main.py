

from google import genai
from google.genai import types
import httpx
import time
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

import os
def answer_questions_with_gemini(pdf_url, questions):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable. Please set GEMINI_API_KEY in your .env or environment.")
    client = genai.Client(api_key=api_key)
    doc_data = httpx.get(pdf_url).content
    system_prompt = (
        "You are an expert insurance policy Q&A assistant. Given a question and a policy document (as PDF), your job is to answer the question as accurately, concisely, and as short and crisp as possible, using only information from the document.\n"
        "- Be short and crisp.\n"
        "- If the answer is a fact, state it clearly and directly.\n"
        "- If the answer requires conditions, eligibility, or limits, summarize the key points.\n"
        "- Do not quote, cite, or mention the source, section, page, or clause.\n"
        "- If the answer is not found in the document, reply: 'Information not available in the provided document.'\n"
        "- Do not invent or assume information not present in the document.\n"
        "- Use clear, professional language suitable for insurance customers.\n"
        "- For multi-part or complex answers, use bullet points or short paragraphs for clarity.\n"
        "- Do not repeat the question in your answer.\n"
        "Return only the answer as a string."
    )
    answers = []
    for i, q in enumerate(questions, 1):
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(
                    data=doc_data,
                    mime_type='application/pdf',
                ),
                system_prompt + "\nQuestion: " + q
            ]
        )
        print(f"{i}. {response.text.strip()}")
        answers.append(response.text.strip())
        time.sleep(0.2)
    return answers

def main():
    print("=== Gemini PDF Q&A ===")
    t0 = time.time()
    answers = answer_questions_with_gemini(PDF_URL, QUESTIONS)
    print(f"[+] Total runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

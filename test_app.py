import os
import time
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_hackrx_run():
    api_key = os.getenv("HACKRX_API_KEY", "testkey")
    url = "/hackrx/run"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?"
        ]
    }
    start = time.time()
    response = client.post(url, json=data, headers=headers)
    elapsed = time.time() - start
    print(f"test_hackrx_run: {elapsed:.2f}s")
    assert response.status_code == 200
    assert "answers" in response.json()
    assert isinstance(response.json()["answers"], list)
    assert len(response.json()["answers"]) == 2

def test_hackrx_run_full():
    api_key = "916bc63e6ba116bde9c394dae5a1c755571929ad2b4e7d54d3a410fc700a1968"
    url = "/hackrx/run"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
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
    }
    start = time.time()
    response = client.post(url, json=data, headers=headers)
    elapsed = time.time() - start
    print(f"test_hackrx_run_full: {elapsed:.2f}s")
    assert response.status_code == 200
    assert "answers" in response.json()
    assert isinstance(response.json()["answers"], list)
    assert len(response.json()["answers"]) == 10

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Spell & Grammar Check API"


def test_health_basic():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_check_requires_api_key():
    response = client.post(
        "/v1/check",
        json={"text": "This is a testt sentence.", "type": "full", "model": "finetune"},
    )
    assert response.status_code == 401

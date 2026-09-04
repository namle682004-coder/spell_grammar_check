from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.dependencies import get_current_user
from src.main import app

client = TestClient(app)


def test_check_quota_exceeded_returns_429():
    async def mock_user():
        return {"user_id": "user1", "api_key_id": "key1"}

    app.dependency_overrides[get_current_user] = mock_user

    try:
        with patch("src.api.routes.check.SpellGrammarService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.check = AsyncMock(
                return_value={
                    "success": False,
                    "error": "Monthly request limit exceeded: 100/100",
                }
            )

            response = client.post(
                "/v1/check",
                json={"text": "test sentence", "type": "full", "model": "finetune"},
                headers={"X-API-Key": "sg_test_key"},
            )

            assert response.status_code == 429
    finally:
        app.dependency_overrides.clear()


def test_check_stream_returns_sse_events():
    async def mock_user():
        return {"user_id": "user1", "api_key_id": "key1"}

    app.dependency_overrides[get_current_user] = mock_user

    try:
        with patch("src.api.routes.check.SpellGrammarService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.check = AsyncMock(
                return_value={
                    "success": True,
                    "request_id": "req_12345",
                    "original_text": "test sentence",
                    "corrected_text": "Test sentence.",
                    "corrections": [],
                    "correction_count": 0,
                    "model_used": "finetune",
                }
            )

            response = client.post(
                "/v1/check/stream",
                json={"text": "test sentence", "type": "full", "model": "finetune"},
                headers={"X-API-Key": "sg_test_key"},
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            assert "event: metadata" in response.text
            assert "event: chunk" in response.text
            assert "event: complete" in response.text
    finally:
        app.dependency_overrides.clear()


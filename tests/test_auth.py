from src.services.auth_service import AuthService


def test_password_hash_and_verify():
    hashed = AuthService.hash_password("secure-password")
    assert AuthService.verify_password("secure-password", hashed)
    assert not AuthService.verify_password("wrong-password", hashed)


def test_jwt_roundtrip():
    token = AuthService.generate_jwt("user123")
    payload = AuthService.verify_jwt(token)
    assert payload is not None
    assert payload["user_id"] == "user123"


def test_jwt_invalid_token():
    assert AuthService.verify_jwt("not-a-valid-token") is None
